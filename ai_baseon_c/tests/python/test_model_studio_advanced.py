"""Numerical, dependency and admission tests for advanced dense compression."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from test_model_studio_compression import make_session, LIBRARY
from model_converter.schema import ConversionError
from model_studio.compression import prune_session, save_checkpoint
from model_studio.analysis import load_session
from model_studio.validation import build_model
from model_studio.advanced_pruning import geometric_scores, _distance_score, validate_advanced_budget


class AdvancedCompressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def unchanged(self, original, session):
        for key in original:
            self.assertTrue(torch.equal(original[key], session.state[key]), key)

    def test_fpgm_removes_geometric_center_not_minimum_norm(self):
        score = _distance_score(torch.tensor([[0.0], [2.0], [3.0]]))
        torch.testing.assert_close(score, torch.tensor([5.0, 3.0, 4.0], dtype=torch.float64))
        self.assertEqual(int(score.argmin()), 1)

    def test_fpgm_bn_folded_and_dependency_scores(self):
        session, _ = make_session("cnn")
        scores, report = geometric_scores(session.state, session.spec)
        self.assertEqual([len(scores[key]) for key in ("conv0", "conv1", "conv2", "fc1")], [4, 6, 8, 10])
        expected = torch.from_numpy(session.spec.arrays["conv0_weight"]).double().flatten(1)
        bias = torch.from_numpy(session.spec.arrays["conv0_bias"]).double()[:, None]
        torch.testing.assert_close(scores["conv0"], _distance_score(torch.cat((expected, bias), 1)))
        self.assertTrue(report["bn_folded"])

    def test_fpgm_all_schemas_deterministic_boundaries_and_original(self):
        for kind in ("cnn", "rnn", "lstm"):
            session, _ = make_session(kind)
            original = deepcopy(session.state)
            for amount in (0, 0.4, 1):
                with self.subTest(kind=kind, amount=amount):
                    first, report = prune_session(session, amount, "fpgm")
                    second, _ = prune_session(session, amount, "fpgm")
                    self.unchanged(first.state, second)
                    if amount == 0:
                        self.unchanged(original, first)
                    if amount == 1:
                        self.assertEqual(first.spec.dimensions["hidden_size"], 1)
                    shape = (1, 1, 64) if kind == "cnn" else (1, 5, 3)
                    self.assertEqual(build_model(first)(torch.zeros(shape)).shape, (1, 3))
                    json.dumps(report, allow_nan=False)
            self.unchanged(original, session)

    def test_fpgm_rejects_budget_and_nonfinite_before_pairwise_allocation(self):
        with patch("model_studio.advanced_pruning.MAX_DISTANCE_OPERATIONS", 1):
            with patch("model_studio.advanced_pruning.torch.cdist") as distance:
                with self.assertRaisesRegex(ConversionError, "budget"):
                    _distance_score(torch.ones(2, 2))
                distance.assert_not_called()
        session, _ = make_session()
        session.state["model.0.weight"][0, 0, 0] = float("inf")
        with self.assertRaises(ConversionError):
            prune_session(session, 0.2, "fpgm")

    def test_svd_full_effective_rank_preserves_affine_logits(self):
        session, _ = make_session()
        # Random matrices avoid the rank-one fixture's trivial low-rank head.
        generator = torch.Generator().manual_seed(706)
        session.state["fc1.weight"].normal_(generator=generator)
        session.state["fc2.weight"].normal_(generator=generator)
        session.spec = replace(session.spec)
        original = deepcopy(session.state)
        values = torch.randn(9, 1, 64, generator=generator)
        candidate, report = prune_session(session, 0.2, "svd")
        with torch.inference_mode():
            expected, actual = build_model(session)(values), build_model(candidate)(values)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        self.assertEqual(candidate.spec.dimensions["hidden_size"], 3)
        self.assertTrue(report["exact_linear_rank_retained"])
        self.assertEqual(report["discarded_singular_energy_fraction"], 0)
        self.unchanged(original, session)

    def test_svd_zero_identity_truncated_rank_and_schema_rejection(self):
        session, _ = make_session()
        identity, _ = prune_session(session, 0, "svd")
        self.unchanged(session.state, identity)
        minimal, report = prune_session(session, 1, "svd")
        self.assertEqual(minimal.spec.dimensions["hidden_size"], 1)
        self.assertFalse(report["exact_linear_rank_retained"])
        self.assertTrue(torch.isfinite(build_model(minimal)(torch.zeros(2, 1, 64))).all())
        for kind in ("rnn", "lstm"):
            with self.assertRaisesRegex(ConversionError, "FC1 and FC2"):
                prune_session(make_session(kind)[0], 0.2, "svd")

    def test_svd_budget_rejected_before_svd(self):
        session, _ = make_session()
        with patch("model_studio.advanced_pruning.MAX_SVD_OPERATIONS", 1):
            with patch("model_studio.advanced_pruning.torch.linalg.svd") as svd:
                with self.assertRaisesRegex(ConversionError, "budget"):
                    prune_session(session, 0.3, "svd")
                svd.assert_not_called()

    def test_svd_product_budget_and_small_hidden_rank_reporting(self):
        session, _ = make_session()
        # The effective matrix is tiny, but forming CxH times HxF can be large.
        huge = replace(session.spec, dimensions=dict(session.spec.dimensions,
                       hidden_size=10000000, num_classes=3, flatten_size=100))
        with self.assertRaisesRegex(ConversionError, "budget"):
            validate_advanced_budget(huge, "svd", 0.2)
        for key in ("fc1.weight", "fc1.bias"):
            session.state[key] = session.state[key][:2].clone()
        session.state["fc2.weight"] = session.state["fc2.weight"][:, :2].clone()
        _, report = prune_session(session, 0.2, "svd")
        self.assertEqual(report["rank"], 2)
        self.assertTrue(report["exact_linear_rank_retained"])

    def test_zero_ratio_does_not_allocate_advanced_matrices(self):
        session, _ = make_session()
        with patch("model_studio.advanced_pruning.MAX_DISTANCE_OPERATIONS", 1):
            with patch("model_studio.advanced_pruning.torch.cdist") as distance:
                identity, _ = prune_session(session, 0, "fpgm")
                self.unchanged(session.state, identity)
                distance.assert_not_called()
        with patch("model_studio.advanced_pruning.MAX_SVD_OPERATIONS", 1):
            with patch("model_studio.advanced_pruning.torch.linalg.svd") as svd:
                identity, _ = prune_session(session, 0, "svd")
                self.unchanged(session.state, identity)
                svd.assert_not_called()

    def test_fpgm_preflight_rejects_before_vector_cast_or_concatenation(self):
        session, _ = make_session()
        with patch("model_studio.advanced_pruning.MAX_ADVANCED_WORKING_BYTES", 1):
            with patch("model_studio.advanced_pruning.torch.from_numpy") as cast:
                with self.assertRaisesRegex(ConversionError, "budget"):
                    prune_session(session, 0.2, "fpgm")
                cast.assert_not_called()

    def test_taylor_all_schemas_physical_dependencies_reconstruction_and_identity(self):
        with tempfile.TemporaryDirectory(prefix=".studio-test-", dir=LIBRARY) as directory:
            path = Path(directory) / "calibration.npz"
            for kind in ("cnn", "rnn", "lstm"):
                with self.subTest(kind=kind):
                    session, _ = make_session(kind)
                    original = deepcopy(session.state)
                    elements = session.spec.dimensions["input_elements"]
                    np.savez(path, inputs=np.random.default_rng(707).normal(size=(7, elements)).astype(np.float32),
                             labels=np.zeros(7, np.int64))
                    candidate, report = prune_session(session, 0.4, "taylor", data_path=path, max_samples=7)
                    self.assertEqual(report["calibration"]["samples"], 7)
                    self.assertFalse(report["calibration"]["labels_used"])
                    self.assertEqual(len(report["calibration"]["selected_inputs_sha256"]), 64)
                    ridge = report["head_reconstruction"]
                    self.assertLessEqual(ridge["calibration_rmse_after"], ridge["calibration_rmse_before"] + 1e-6)
                    self.assertLess(candidate.spec.dimensions["hidden_size"], session.spec.dimensions["hidden_size"])
                    shape = (1, 1, 64) if kind == "cnn" else (1, 5, 3)
                    self.assertEqual(build_model(candidate)(torch.zeros(shape)).shape, (1, 3))
                    identity, _ = prune_session(session, 0, "taylor")
                    self.unchanged(original, identity)
                    self.unchanged(original, session)
                    json.dumps(report, allow_nan=False)

    def test_taylor_labels_never_influence_selection_or_reconstruction(self):
        session, _ = make_session("lstm")
        inputs = np.random.default_rng(708).normal(size=(5, 15)).astype(np.float32)
        with tempfile.TemporaryDirectory(prefix=".studio-test-", dir=LIBRARY) as directory:
            path = Path(directory) / "calibration.npz"
            np.savez(path, inputs=inputs, labels=np.zeros(5, np.int64))
            first, a = prune_session(session, 0.5, "taylor", data_path=path, max_samples=5)
            np.savez(path, inputs=inputs, labels=np.full(5, 2, np.int64))
            second, b = prune_session(session, 0.5, "taylor", data_path=path, max_samples=5)
            self.unchanged(first.state, second)
            self.assertEqual(a["calibration"]["selected_inputs_sha256"], b["calibration"]["selected_inputs_sha256"])
            np.savez(path, inputs=inputs + 0.25)
            _, c = prune_session(session, 0.5, "taylor", data_path=path, max_samples=5)
            self.assertNotEqual(a["calibration"]["selected_inputs_sha256"], c["calibration"]["selected_inputs_sha256"])

    def test_taylor_full_ratio_retains_one_unit_all_schemas(self):
        with tempfile.TemporaryDirectory(prefix=".studio-test-", dir=LIBRARY) as directory:
            path = Path(directory) / "calibration.npy"
            for kind in ("cnn", "rnn", "lstm"):
                session, _ = make_session(kind)
                np.save(path, np.zeros((2, session.spec.dimensions["input_elements"]), np.float32))
                candidate, _ = prune_session(session, 1, "taylor", data_path=path, max_samples=2)
                self.assertEqual(candidate.spec.dimensions["hidden_size"], 1)

    def test_taylor_data_and_preprocessing_diagnostics(self):
        session, _ = make_session()
        with tempfile.TemporaryDirectory(prefix=".studio-test-", dir=LIBRARY) as directory:
            path = Path(directory) / "calibration.npy"
            np.save(path, np.zeros((2, 64), np.float32))
            for kwargs in ({"data_path": None}, {"max_samples": 129}, {"max_samples": True},
                           {"input_scale": float("nan")}, {"input_scale": 0},
                           {"input_zero_point": 256}, {"input_zero_point": False}):
                options = dict(data_path=path, max_samples=2)
                options.update(kwargs)
                with self.subTest(kwargs=kwargs), self.assertRaises(ConversionError):
                    prune_session(session, 0.2, "taylor", **options)
            np.save(path, np.full((2, 64), float("nan"), np.float32))
            with self.assertRaises(ConversionError):
                prune_session(session, 0.2, "taylor", data_path=path, max_samples=2)

    def test_taylor_budget_rejects_before_dataset_or_model_allocation(self):
        session, _ = make_session("rnn")
        with patch("model_studio.advanced_pruning.MAX_FIT_OPERATIONS", 1):
            with patch("model_studio.validation._numpy_data") as reader:
                with patch("model_studio.validation.build_model") as builder:
                    with self.assertRaisesRegex(ConversionError, "budget"):
                        prune_session(session, 0.3, "taylor", data_path="nonexistent.npy", max_samples=2)
                    reader.assert_not_called()
                    builder.assert_not_called()

    def test_taylor_uint8_preprocessing_and_teacher_failure_do_not_mutate(self):
        session, _ = make_session("rnn")
        original = deepcopy(session.state)
        inputs = np.arange(45, dtype=np.uint8).reshape(3, 15)
        with tempfile.TemporaryDirectory(prefix=".studio-test-", dir=LIBRARY) as directory:
            path = Path(directory) / "calibration.npy"
            np.save(path, inputs)
            quantized, _ = prune_session(session, 0.5, "taylor", data_path=path, max_samples=3,
                                         input_scale=0.01, input_zero_point=20)
            np.save(path, (inputs.astype(np.float32) - 20) * np.float32(0.01))
            floating, _ = prune_session(session, 0.5, "taylor", data_path=path, max_samples=3)
            self.unchanged(quantized.state, floating)
            with patch("model_studio.advanced_pruning.F.cross_entropy", side_effect=RuntimeError("simulated backward failure")):
                with self.assertRaises(RuntimeError):
                    prune_session(session, 0.5, "taylor", data_path=path, max_samples=3)
            self.unchanged(original, session)

    def test_taylor_enables_required_gradients_inside_outer_no_grad(self):
        session, _ = make_session("rnn")
        with tempfile.TemporaryDirectory(prefix=".studio-test-", dir=LIBRARY) as directory:
            path = Path(directory) / "calibration.npy"
            np.save(path, np.ones((2, 15), np.float32))
            with torch.no_grad():
                candidate, report = prune_session(session, 0.2, "taylor", data_path=path, max_samples=2)
            self.assertEqual(report["calibration"]["samples"], 2)
            self.assertTrue(torch.isfinite(candidate.state["head.weight"]).all())

    def test_saved_taylor_history_preserves_hash_without_restoring_current_edits(self):
        session, _ = make_session("rnn")
        session.metadata["studio_history"] = [{"method": "prior", "amount": 0.1}]
        with tempfile.TemporaryDirectory(prefix=".studio-test-", dir=LIBRARY) as directory:
            path = Path(directory) / "calibration.npy"
            np.save(path, np.ones((2, 15), np.float32))
            candidate, report = prune_session(session, 0.2, "taylor", data_path=path, max_samples=2)
            target = Path(directory) / "candidate.pth"
            save_checkpoint(candidate, target)
            imported = load_session(target)
            history = imported.metadata["studio_history"]
            self.assertEqual(len(history), 2)
            self.assertEqual(history[0], session.metadata["studio_history"][0])
            self.assertEqual(history[1]["calibration"]["selected_inputs_sha256"],
                             report["calibration"]["selected_inputs_sha256"])
            self.assertEqual(history[1]["calibration"]["path"], str(path.resolve()))
            self.assertEqual(imported.changes, [])
            self.assertEqual(len(candidate.metadata["studio_history"]), 1)
            self.unchanged(candidate.state, imported)
            next_target = Path(directory) / "candidate-again.pth"
            save_checkpoint(imported, next_target)
            self.assertEqual(load_session(next_target).metadata["studio_history"], history)

    def test_checkpoint_rejects_invalid_history_and_current_edit_records(self):
        session, _ = make_session()
        with tempfile.TemporaryDirectory(prefix=".studio-test-", dir=LIBRARY) as directory:
            target = Path(directory) / "candidate.pth"
            for history in ({}, ({"method": "bad"},), ["bad"], [{"tensor": torch.ones(1)}]):
                session.metadata["studio_history"] = history
                with self.subTest(history=history), self.assertRaises(ConversionError):
                    save_checkpoint(session, target)
                self.assertFalse(target.exists())
            session.metadata.pop("studio_history")
            session.changes = ["bad"]
            with self.assertRaises(ConversionError):
                save_checkpoint(session, target)
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
