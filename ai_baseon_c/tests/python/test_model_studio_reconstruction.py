"""Structured reconstruction admission, dependency and label-isolation tests."""

from copy import deepcopy
import itertools
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from test_model_studio_compression import make_session
from model_converter.schema import ConversionError, parse_spec
from model_studio.analysis import analyze_session, load_session
from model_studio.compression import prune_session, save_checkpoint
from model_studio import reconstruction_pruning as reconstruction
from model_studio.validation import build_model


TEST_ROOT = Path(__file__).resolve().parents[2]


class ReconstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def session(self, bias=True, hidden=10):
        session, _ = make_session()
        generator = torch.Generator().manual_seed(313)
        for index in (0, 4, 8):
            key = f"model.{index}.weight"
            session.state[key] = torch.randn(session.state[key].shape, generator=generator) * 0.15
            if not bias:
                del session.state[f"model.{index}.bias"]
        session.state["fc1.weight"] = torch.randn((hidden, session.spec.dimensions["flatten_size"]), generator=generator) * 0.1
        session.state["fc1.bias"] = session.state["fc1.bias"][:hidden]
        session.state["fc2.weight"] = torch.randn((3, hidden), generator=generator) * 0.1
        session.spec = parse_spec(session.state, session.metadata)
        return session

    def data(self, directory, labels=None, count=12):
        path = Path(directory) / "calibration.npz"
        inputs = np.random.default_rng(17).integers(0, 256, size=(count, 1, 64), dtype=np.uint8)
        np.savez(path, inputs=inputs, labels=np.arange(count) % 3 if labels is None else labels)
        return path, inputs

    def assert_state_equal(self, left, right):
        self.assertEqual(left.keys(), right.keys())
        for key in left:
            self.assertTrue(torch.equal(left[key], right[key]), key)

    def test_zero_identity_without_calibration_or_advanced_allocation(self):
        session = self.session()
        with patch.object(reconstruction, "_calibration", side_effect=AssertionError("must not load")), \
                patch.object(torch.linalg, "inv", side_effect=AssertionError("must not invert")):
            candidate, report = prune_session(session, 0, "reconstruction")
        self.assert_state_equal(candidate.state, session.state)
        self.assertIsNone(report["calibration"])
        self.assertEqual(report["removed"], 0)

    def test_physical_dependencies_and_nonincreasing_calibration_fit(self):
        session = self.session()
        snapshot = deepcopy(session.state)
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as directory:
            path, inputs = self.data(directory)
            candidate, report = prune_session(session, 0.5, "reconstruction", data_path=path)
        self.assertEqual([shape[0] for shape in candidate.spec.dimensions["conv_shapes"]], [2, 3, 4])
        self.assertEqual(candidate.state["model.4.weight"].shape, (3, 2, 5))
        self.assertEqual(candidate.state["model.8.weight"].shape, (4, 3, 5))
        self.assertEqual(candidate.state["fc1.weight"].shape, (3, 16))
        self.assertEqual(candidate.state["fc2.weight"].shape, (3, 3))
        self.assertFalse(report["calibration"]["labels_used"])
        for layer in report["layers"]:
            fit = layer.get("next_layer_reconstruction", layer.get("classifier_reconstruction"))
            self.assertLessEqual(fit["calibration_rmse_after"], fit["calibration_rmse_before"] + 1e-8)
        head = report["layers"][-1]["classifier_reconstruction"]
        self.assertEqual(len(head["damping_candidates"]), 6)
        self.assertIn(head["damping"], reconstruction.HEAD_DAMPING_CANDIDATES)
        output = build_model(candidate)(torch.from_numpy(inputs.astype(np.float32) / 255))
        self.assertEqual(output.shape, (12, 3))
        self.assertTrue(bool(torch.isfinite(output).all()))
        self.assertLess(analyze_session(candidate)["summary"]["macs"], analyze_session(session)["summary"]["macs"])
        self.assert_state_equal(session.state, snapshot)

    def test_calibration_labels_do_not_affect_selection_fit_or_hash(self):
        session = self.session()
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as directory:
            path, _ = self.data(directory)
            a, ar = prune_session(session, 0.5, "reconstruction", data_path=path)
            self.data(directory, np.full(12, 2))
            b, br = prune_session(session, 0.5, "reconstruction", data_path=path)
        self.assert_state_equal(a.state, b.state)
        self.assertEqual(ar["calibration"]["selected_inputs_sha256"], br["calibration"]["selected_inputs_sha256"])
        self.assertEqual(ar["layers"], br["layers"])

    def test_full_ratio_bias_free_and_rank_cannot_expand_original_head(self):
        session = self.session(bias=False, hidden=1)
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as directory:
            path, inputs = self.data(directory)
            candidate, report = prune_session(session, 1.0, "reconstruction", data_path=path)
        self.assertEqual([shape[0] for shape in candidate.spec.dimensions["conv_shapes"]], [1, 1, 1])
        self.assertEqual(candidate.spec.dimensions["hidden_size"], 1)
        self.assertFalse(any(f"model.{index}.bias" in candidate.state for index in (0, 4, 8)))
        self.assertGreaterEqual(report["units"]["fc1_neurons"]["removed"], 0)
        self.assertTrue(bool(torch.isfinite(build_model(candidate)(torch.from_numpy(inputs.astype(np.float32) / 255))).all()))

    def test_non_cnn_and_budget_rejected_before_calibration(self):
        with patch.object(reconstruction, "_calibration", side_effect=AssertionError("must not load")):
            for kind in ("rnn", "lstm"):
                with self.assertRaisesRegex(ConversionError, "requires cnn_voice_v1"):
                    prune_session(make_session(kind)[0], 0.2, "reconstruction")
            with patch.object(reconstruction, "MAX_FIT_OPERATIONS", 1):
                with self.assertRaisesRegex(ConversionError, "budget"):
                    prune_session(self.session(), 0.2, "reconstruction")

    def test_dead_channels_ties_and_tiny_calibration_are_finite(self):
        session = self.session()
        for index in (1, 5, 9):
            session.state[f"model.{index}.weight"].zero_()
            session.state[f"model.{index}.bias"].zero_()
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as directory:
            path, _ = self.data(directory, count=1)
            candidate, report = prune_session(session, 0.5, "reconstruction", data_path=path, max_samples=1)
        self.assertEqual(report["layers"][0]["kept_channels"], [2, 3])
        self.assertTrue(all(bool(torch.isfinite(value).all()) for value in candidate.state.values()))
        self.assertEqual(report["layers"][-1]["classifier_reconstruction"]["damping"], 1e-2)

    def test_invalid_data_does_not_mutate_source_and_save_history_is_reproducible(self):
        session = self.session()
        snapshot = deepcopy(session.state)
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as directory:
            invalid = Path(directory) / "invalid.npy"
            np.save(invalid, np.full((2, 1, 64), np.nan, dtype=np.float32))
            with self.assertRaisesRegex(ConversionError, "NaN|finite"):
                prune_session(session, 0.5, "reconstruction", data_path=invalid)
            path, _ = self.data(directory)
            candidate, report = prune_session(session, 0.5, "reconstruction", data_path=path)
            output = Path(directory) / "reconstructed.pth"
            save_checkpoint(candidate, output)
            reloaded = load_session(output)
            self.assert_state_equal(reloaded.state, candidate.state)
            self.assertEqual(reloaded.metadata["studio_history"][0]["layers"], report["layers"])
        self.assert_state_equal(session.state, snapshot)

    def test_covariance_selection_preserves_redundant_signal_over_small_weight(self):
        # Removing a small unique coefficient loses information; removing one
        # of two correlated, individually larger coefficients can be repaired.
        u = torch.linspace(-2, 2, 100, dtype=torch.float64)
        v = torch.sin(3 * u)
        features = torch.stack((u, u, v), 1)
        weight = torch.tensor([[1.0, 1.0, 0.1]], dtype=torch.float64)
        target = features @ weight.t() + 0.3
        kept, trace = reconstruction._obs_groups(features, weight, 1, 2)
        self.assertEqual(kept.tolist(), [1, 2])
        self.assertEqual(trace[0]["channel"], 0)
        self.assertEqual(trace[0]["tied_channels"], [0, 1])
        _, _, fitted = reconstruction._affine_fit(features[:, kept], target,
            weight[:, kept], torch.tensor([0.3]), reconstruction.CONV_DAMPING)
        _, _, magnitude_fit = reconstruction._affine_fit(features[:, :2], target,
            weight[:, :2], torch.tensor([0.3]), reconstruction.CONV_DAMPING)
        self.assertLess(fitted["calibration_rmse_after"], magnitude_fit["calibration_rmse_after"] * 0.05)

    def test_obs_duplicate_choice_is_stable_across_valid_inverse_orders_and_numpy(self):
        # Equivalent SPD inverses can differ by one ULP in their diagonal.
        # Exercise actual alternative solves rather than changing the expected
        # channel to whichever exact floating minimum a particular BLAS finds.
        u = torch.linspace(-2, 2, 100, dtype=torch.float64)
        features = torch.stack((u, u, torch.sin(3 * u)), 1)
        weight = torch.tensor([[1.0, 1.0, 0.1]], dtype=torch.float64)
        centered = features - features.mean(0)
        gram = centered.t() @ centered / len(features)
        ridge = float(gram.diag().mean()) * reconstruction.CONV_DAMPING
        regularized = gram + ridge * torch.eye(3, dtype=torch.float64)
        inverses = []
        for order in itertools.permutations(range(3)):
            indices = torch.tensor(order)
            reverse = torch.argsort(indices)
            permuted = torch.linalg.inv(regularized[indices[:, None], indices])
            inverses.append((str(order), permuted[reverse[:, None], reverse]))
        inverses.append(("numpy.linalg.inv", torch.from_numpy(np.linalg.inv(regularized.numpy()))))
        for name, inverse in inverses:
            with self.subTest(solver=name):
                residual = regularized @ inverse - torch.eye(3, dtype=torch.float64)
                self.assertLess(float(residual.abs().max()), 1e-9)
                with patch.object(torch.linalg, "inv", return_value=inverse.clone()):
                    kept, trace = reconstruction._obs_groups(features, weight, 1, 2)
                self.assertEqual(kept.tolist(), [1, 2])
                self.assertEqual(trace[0]["tied_channels"], [0, 1])
                self.assertGreaterEqual(trace[0]["quadratic_loss"], trace[0]["minimum_quadratic_loss"])
                self.assertLessEqual(trace[0]["quadratic_loss"] - trace[0]["minimum_quadratic_loss"],
                                     trace[0]["tie_absolute_tolerance"])

    def test_obs_roundoff_ties_are_relative_minimum_anchored_and_preserve_zero(self):
        relative = reconstruction.OBS_TIE_EPS_MULTIPLIER * math.ulp(1.0) * 3
        for scale in (1e-200, 1.0, 1e200):
            candidates = [(scale * (1 + relative * 1.5), 0),
                          (scale * (1 + relative * 0.75), 1), (scale, 2)]
            for order in (candidates, list(reversed(candidates))):
                selected, report = reconstruction._select_obs_candidate(order, 3)
                self.assertEqual(selected[1], 1)
                self.assertEqual(report["tied_channels"], [1, 2])
                self.assertEqual(report["minimum_quadratic_loss"], scale)
            # A distinguishable lower cost must win, regardless of channel ID.
            selected, _ = reconstruction._select_obs_candidate(
                [(scale * (1 + relative * 4), 0), (scale, 2)], 3)
            self.assertEqual(selected[1], 2)
        selected, report = reconstruction._select_obs_candidate([(1e-300, 0), (0.0, 2)], 3)
        self.assertEqual(selected[1], 2)
        self.assertEqual(report["tie_absolute_tolerance"], 0.0)
        self.assertEqual(report["tied_channels"], [2])
        selected, _ = reconstruction._select_obs_candidate([(0.0, 2), (0.0, 0)], 3)
        self.assertEqual(selected[1], 0)

    def test_obs_two_tap_groups_and_genuinely_lower_loss(self):
        generator = torch.Generator().manual_seed(722)
        unique = torch.randn((128, 4), generator=generator, dtype=torch.float64)
        features = torch.cat((unique[:, :2], unique[:, :2], unique[:, 2:]), 1)
        weight = torch.tensor([[1.0, 0.3, 1.0, 0.3, 0.1, 0.05]], dtype=torch.float64)
        kept, trace = reconstruction._obs_groups(features, weight, 2, 2)
        self.assertEqual(kept.tolist(), [1, 2])
        self.assertEqual(trace[0]["tied_channels"], [0, 1])
        # Break the mathematical symmetry by a gap well outside roundoff.
        weight[:, :2] *= 1.00001
        kept, trace = reconstruction._obs_groups(features, weight, 2, 2)
        self.assertEqual(kept.tolist(), [0, 2])
        self.assertEqual(trace[0]["tied_channels"], [1])

    def test_obs_multistep_losses_match_independent_constrained_primal_residual(self):
        generator = torch.Generator().manual_seed(723)
        features = torch.randn((80, 8), generator=generator, dtype=torch.float64)
        weight = torch.randn((3, 8), generator=generator, dtype=torch.float64)
        centered = features - features.mean(0)
        gram = centered.t() @ centered / len(features)
        regularized = gram + float(gram.diag().mean()) * reconstruction.CONV_DAMPING * torch.eye(8, dtype=torch.float64)

        def constrained_loss(deleted):
            # Independently minimize the ORIGINAL primal quadratic with all
            # deleted coefficients fixed to zero; no inverse/Schur updates.
            removed = torch.tensor([column for channel in sorted(deleted)
                                    for column in (2 * channel, 2 * channel + 1)])
            retained = torch.tensor([column for channel in range(4) if channel not in deleted
                                     for column in (2 * channel, 2 * channel + 1)])
            right = weight[:, removed] @ regularized[removed[:, None], retained]
            fitted = weight[:, retained] + torch.linalg.solve(
                regularized[retained[:, None], retained], right.t()).t()
            approximation = torch.zeros_like(weight)
            approximation[:, retained] = fitted
            delta = approximation - weight
            return float((delta @ regularized * delta).sum())

        kept, trace = reconstruction._obs_groups(features, weight, 2, 2)
        deleted, previous = set(), 0.0
        for step in trace:
            candidate_losses = {channel: constrained_loss(deleted | {channel}) - previous
                                for channel in range(4) if channel not in deleted}
            self.assertAlmostEqual(step["minimum_quadratic_loss"], min(candidate_losses.values()), delta=1e-10)
            deleted.add(step["channel"])
            total = constrained_loss(deleted)
            self.assertAlmostEqual(step["quadratic_loss"], total - previous, delta=1e-10)
            previous = total
        self.assertEqual(kept.tolist(), [channel for channel in range(4) if channel not in deleted])

    def test_returned_checkpoint_tensors_remain_mutable_for_further_work(self):
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as directory:
            path, _ = self.data(directory)
            candidate, _ = prune_session(self.session(), 0.5, "reconstruction", data_path=path)
        candidate.state["fc1.weight"].add_(0)
        self.assertFalse(candidate.state["fc1.weight"].is_inference())

    def test_convolution_scratch_budget_is_admitted_before_calibration(self):
        # The fixture's activation plus head estimate alone is below 2 MiB;
        # explicit double patch/Gram/solve scratch makes the convolution exceed it.
        with patch.object(reconstruction, "MAX_ADVANCED_WORKING_BYTES", 2 * 1024 * 1024), \
                patch.object(reconstruction, "_calibration", side_effect=AssertionError("must not load")):
            with self.assertRaisesRegex(ConversionError, "convolution CPU/memory budget"):
                prune_session(self.session(), 0.5, "reconstruction")

    def test_many_class_small_feature_factorization_avoids_class_squared_identity(self):
        fitted = torch.arange(4096 * 4, dtype=torch.float64).reshape(4096, 4) / 10000
        state = {}
        with patch.object(torch, "eye", side_effect=AssertionError("must not form classes squared identity")):
            report = reconstruction._factor_head(state, fitted, torch.zeros(4096), 1)
        self.assertEqual(state["fc1.weight"].shape, (1, 4))
        self.assertEqual(state["fc2.weight"].shape, (4096, 1))
        self.assertFalse(report["exact_linear_rank_retained"])

    def test_head_regularization_reuses_training_gram_before_final_fit(self):
        generator = torch.Generator().manual_seed(318)
        features = torch.randn((12, 7), generator=generator, dtype=torch.float64)
        weight = torch.randn((3, 7), generator=generator, dtype=torch.float64)
        bias = torch.randn(3, generator=generator, dtype=torch.float64)
        target = features @ weight.t() + bias + torch.randn((12, 3), generator=generator, dtype=torch.float64) * 0.1
        with patch.object(reconstruction, "_affine_context", wraps=reconstruction._affine_context) as context:
            _, _, report = reconstruction._head_fit(features, target, weight, bias)
        # Six lambda candidates share one statistics/Gram context, followed by
        # exactly one context for the final refit on all calibration inputs.
        self.assertEqual(context.call_count, 2)
        self.assertEqual([call.args[0].shape[0] for call in context.call_args_list], [9, 12])
        self.assertEqual(len(report["damping_candidates"]), 6)

    def test_last_channel_obs_budget_not_only_final_head_gram(self):
        from dataclasses import replace
        session = self.session()
        # A narrow stem and 1000 final channels has a modest 4000-wide head,
        # but its 1000x1000 group-OBS inverse updates exceed the CPU budget.
        dimensions = dict(session.spec.dimensions, hidden_size=1, flatten_size=4000,
            conv_shapes=[(1, 1, 5, 60, 30), (1, 1, 5, 26, 13), (1000, 1, 5, 9, 4)])
        with self.assertRaisesRegex(ConversionError, "classifier CPU/memory budget"):
            reconstruction.validate_reconstruction_budget(replace(session.spec, dimensions=dimensions), 0.5)


if __name__ == "__main__":
    unittest.main()
