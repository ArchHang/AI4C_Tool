"""Pure Python, independently hand-counted model-studio accounting tests (no gcc).

Run from ai_baseon_c with:
  python -s -B -X utf8 -m unittest discover -s tests/python -p test_model_studio_analysis.py -v
"""

import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import warnings

import numpy as np
import torch

LIBRARY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LIBRARY / "tools"))

from model_converter.schema import ConversionError, parse_spec
from model_studio.analysis import ModelSession, analyze_session, load_session


def cnn_state(length=64, conv_bias=True):
    state, inputs, width = {}, 1, length
    for index, outputs in zip((0, 4, 8), (2, 3, 4)):
        state[f"model.{index}.weight"] = torch.full((outputs, inputs, 5), 0.25)
        if conv_bias:
            state[f"model.{index}.bias"] = torch.zeros(outputs)
        for suffix, value in (("weight", 1.0), ("bias", 0.0),
                              ("running_mean", 0.0), ("running_var", 1.0)):
            state[f"model.{index + 1}.{suffix}"] = torch.full((outputs,), value)
        state[f"model.{index + 1}.num_batches_tracked"] = torch.tensor(3)
        width, inputs = (width - 4) // 2, outputs
    state.update({"fc1.weight": torch.full((5, inputs * width), 0.25),
                  "fc1.bias": torch.zeros(5), "fc2.weight": torch.full((3, 5), 0.25),
                  "fc2.bias": torch.zeros(3)})
    return state


def recurrent_state(kind="rnn", inputs=3, hidden=2, classes=2, prefix="rnn.", head="head."):
    gates = 4 if kind == "lstm" else 1
    return {prefix + "weight_ih_l0": torch.full((gates * hidden, inputs), 0.25),
            prefix + "weight_hh_l0": torch.full((gates * hidden, hidden), 0.25),
            prefix + "bias_ih_l0": torch.zeros(gates * hidden),
            prefix + "bias_hh_l0": torch.zeros(gates * hidden),
            head + "weight": torch.full((classes, hidden), 0.25),
            head + "bias": torch.zeros(classes)}


def session(state, architecture="auto", **metadata):
    return ModelSession(None, "", state, metadata, parse_spec(state, metadata, architecture))


class AnalysisTests(unittest.TestCase):
    def test_cnn_hand_counted_shapes_macs_and_parameter_buffers(self):
        result = analyze_session(session(cnn_state(), input_length=64))
        summary = result["summary"]
        # Conv MACs: 2*60*1*5 + 3*26*2*5 + 4*9*3*5; FCs: 5*16 + 3*5.
        self.assertEqual(summary["macs"], 2015)
        self.assertEqual(summary["mac_flops"], 4030)
        # Trainable: Conv=109, BN gamma/beta=18, FC=103. Buffers: 18 floats + 3 counters.
        self.assertEqual(summary["total_parameters"], 230)
        self.assertEqual(summary["state_elements"], 251)
        self.assertEqual(summary["state_bytes"], 248 * 4 + 3 * 8)
        self.assertEqual(result["dimensions"]["flatten_size"], 16)
        layers = result["layers"]
        self.assertEqual(len(layers), 15)
        self.assertEqual(layers[0]["input_shape"], [1, 1, 64])
        self.assertEqual(layers[0]["output_shape"], [1, 2, 60])
        self.assertEqual(layers[0]["macs"], 600)
        self.assertEqual(layers[1]["parameters"], 4)
        self.assertEqual(layers[1]["weight_bytes"], 0)
        self.assertIn("model.1.running_var", layers[1]["tensors"])
        self.assertEqual(layers[1]["attributes"]["folded_into"], "model.0")
        self.assertEqual(layers[12]["output_shape"], [1, 16])
        self.assertEqual(layers[13]["attributes"]["activation"], "none")
        self.assertIn("no ReLU", " ".join(layers[13]["notes"]))
        self.assertTrue(all(row["macs"] == 0 for row in layers if row["op"] in
                            ("BatchNorm1d", "ReLU", "MaxPool1d", "Flatten")))
        self.assertEqual(summary["max_activation_bytes"], 120)
        self.assertTrue(result["supported"])
        self.assertEqual(result["schema"], "cnn_voice_v1")
        json.dumps(result, allow_nan=False)

    def test_cnn_payloads_match_actual_constant_array_accounting(self):
        known = session(cnn_state(), input_length=64)
        # 195 matrix weights, 17 deployed biases, 17 row scales. FP32 also emits 18 helper floats.
        expected = {"int8": (331, 428, 0), "fp32": (920, 812, 72), "both": (1183, 1240, 72)}
        with mock.patch("model_converter.emitter.emit_model", side_effect=AssertionError("must not emit")):
            for precision, (payload, workspace, helper) in expected.items():
                with self.subTest(precision=precision):
                    result = analyze_session(known, precision)
                    self.assertEqual(result["summary"]["deployed_parameter_bytes"], payload)
                    self.assertEqual(result["summary"]["workspace_bytes_lower_bound"], workspace)
                    self.assertEqual(result["deployment"]["fp32_bn_helper_bytes"], helper)
                    self.assertEqual(sum(layer["weight_bytes"] for layer in result["layers"]), payload)
                    self.assertEqual(result["deployment"]["parameter_array_bytes"] + helper, payload)
        self.assertEqual(analyze_session(known, "fp32")["summary"]["max_activation_bytes"], 480)

    def test_absent_conv_bias_is_not_a_trainable_parameter_but_export_has_folded_bias(self):
        result = analyze_session(session(cnn_state(conv_bias=False), input_length=64))
        self.assertEqual(result["summary"]["total_parameters"], 230 - 9)
        self.assertEqual(result["summary"]["deployed_parameter_bytes"], 331)
        self.assertEqual(result["layers"][0]["tensors"], ["model.0.weight"])

    def test_weight_sparsity_uses_matrix_weights_not_bn_or_bias(self):
        state = cnn_state()
        state["model.0.weight"].zero_()
        result = analyze_session(session(state, input_length=64))
        self.assertAlmostEqual(result["summary"]["weight_sparsity"], 10 / 195)

    def test_rnn_hand_counts_and_sequence_length(self):
        state = recurrent_state()
        known = session(state, "rnn", seq_length=4)
        result = analyze_session(known)
        self.assertEqual(result["summary"]["total_parameters"], 20)
        self.assertEqual(result["summary"]["macs"], 4 * 2 * (3 + 2) + 2 * 2)
        self.assertEqual(result["summary"]["deployed_parameter_bytes"], 78)
        self.assertEqual(result["summary"]["workspace_bytes_lower_bound"], 60)
        self.assertEqual(result["layers"][0]["input_shape"], [1, 4, 3])
        self.assertEqual(result["layers"][0]["output_shape"], [1, 4, 2])
        self.assertEqual(result["layers"][1]["macs"], 4)
        longer = analyze_session(session(state, "rnn", seq_length=7))
        self.assertEqual(longer["summary"]["macs"], 7 * 10 + 4)
        self.assertEqual(longer["summary"]["deployed_parameter_bytes"], 78)
        self.assertEqual(longer["summary"]["workspace_bytes_lower_bound"], 99)
        self.assertEqual(analyze_session(known, "fp32")["summary"]["deployed_parameter_bytes"], 80)
        self.assertEqual(analyze_session(known, "both")["summary"]["deployed_parameter_bytes"], 134)

    def test_lstm_gates_merged_bias_and_alternative_prefixes(self):
        known = session(recurrent_state("lstm", prefix="encoder.", head="classifier."),
                        seq_length=4)
        result = analyze_session(known)
        self.assertEqual(result["architecture"], "lstm")
        self.assertEqual(result["summary"]["total_parameters"], 62)
        self.assertEqual(result["summary"]["macs"], 4 * 4 * 2 * (3 + 2) + 2 * 2)
        self.assertEqual(result["summary"]["mac_flops"], 328)
        self.assertEqual(result["summary"]["deployed_parameter_bytes"], 220)
        self.assertEqual(result["summary"]["workspace_bytes_lower_bound"], 100)
        self.assertEqual(result["layers"][0]["attributes"]["gates"], 4)
        self.assertEqual(result["layers"][1]["tensors"], ["classifier.weight", "classifier.bias"])
        self.assertEqual(analyze_session(known, "fp32")["summary"]["deployed_parameter_bytes"], 216)
        self.assertEqual(analyze_session(known, "both")["summary"]["deployed_parameter_bytes"], 396)

    def test_unknown_inventory_never_infers_graph_or_trainability(self):
        known = ModelSession(None, "", {"weight": torch.tensor([[0.0, 2.0], [-2.0, 0.0]])}, {}, None,
                             ["original graph rejection"], [{"step": "test", "path": Path("sample")}])
        result = analyze_session(known)
        self.assertFalse(result["supported"])
        self.assertEqual(result["architecture"], "unknown")
        self.assertIsNone(result["schema"])
        self.assertEqual(result["layers"], [])
        self.assertEqual(result["dimensions"], {})
        for name in ("total_parameters", "deployed_parameter_bytes", "workspace_bytes_lower_bound",
                     "macs", "mac_flops", "max_activation_bytes", "weight_sparsity"):
            self.assertIsNone(result["summary"][name], name)
        item = result["tensors"][0]
        self.assertEqual(item["zero_fraction"], 0.5)
        self.assertEqual(item["mean"], 0)
        self.assertAlmostEqual(item["std"], math.sqrt(2))
        self.assertIn("original graph rejection", result["warnings"])
        self.assertEqual(result["changes"][0]["path"], "sample")
        json.dumps(result, allow_nan=False)

    def test_inventory_statistics_are_finite_for_extreme_float64(self):
        largest = np.finfo(np.float64).max
        state = {"values": torch.tensor([-largest, 0, largest], dtype=torch.float64),
                 "flag": torch.tensor([True, False]),
                 "counter": torch.tensor([2 ** 63 - 1], dtype=torch.int64),
                 "bfloat": torch.tensor([1, 3], dtype=torch.bfloat16),
                 "view": torch.arange(12, dtype=torch.float32).reshape(3, 4).T}
        result = analyze_session(ModelSession(None, "", state, {}, None))
        values = result["tensors"][0]
        self.assertEqual(values["mean"], 0.0)
        self.assertTrue(math.isfinite(values["std"]))
        self.assertAlmostEqual(values["std"] / largest, math.sqrt(2 / 3))
        self.assertEqual(result["tensors"][2]["max"], 2 ** 63 - 1)
        json.dumps(result, allow_nan=False)

    def test_invalid_tensors_rejected_before_inventory_statistics(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Sparse invariant checks are implicitly disabled.*",
                                    category=UserWarning)
            sparse = torch.sparse_coo_tensor(torch.tensor([[0]]), torch.tensor([1.0]), (2,),
                                            check_invariants=True)
        cases = {"nan": torch.tensor([float("nan")]), "inf": torch.tensor([float("inf")]),
                 "empty": torch.empty(0), "complex": torch.tensor([1 + 2j]),
                 "meta": torch.empty(2, device="meta"),
                 "sparse": sparse,
                 "quantized": torch.quantize_per_tensor(torch.ones(2), 0.1, 0, torch.qint8)}
        for name, tensor in cases.items():
            with self.subTest(name=name), self.assertRaises(ConversionError):
                analyze_session(ModelSession(None, "", {"value": tensor}, {}, None))
        for invalid in ({}, {"not_tensor": 3}, {1: torch.ones(1)}):
            with self.subTest(state=invalid), self.assertRaises(ConversionError):
                analyze_session(ModelSession(None, "", invalid, {}, None))

    def test_option_validation_and_no_mutation(self):
        known = session(cnn_state(), input_length=64)
        for options in ({"precision": "int4"}, {"weight_scaling": "percentile"}):
            with self.subTest(options=options), self.assertRaises(ConversionError):
                analyze_session(known, **options)
        before_warnings, before_changes = list(known.warnings), list(known.changes)
        analyze_session(known, weight_scaling="mse")
        self.assertEqual(known.warnings, before_warnings)
        self.assertEqual(known.changes, before_changes)
        other = ModelSession(None, "", {}, {}, None)
        other.warnings.append("independent default")
        self.assertEqual(known.warnings, [])

    def test_large_sequence_accounting_does_not_allocate_activations(self):
        state = recurrent_state(inputs=1, hidden=1, classes=1)
        known = session(state, "rnn", seq_length=100_000_000)
        result = analyze_session(known)
        self.assertEqual(result["summary"]["macs"], 200_000_001)
        self.assertEqual(result["summary"]["max_activation_bytes"], 100_000_000)
        # Individual arrays satisfy schema limits, but the combined emitter workspace does not.
        too_big = session(state, "rnn", seq_length=220_000_000)
        with self.assertRaisesRegex(ConversionError, "Combined workspace"):
            analyze_session(too_big)


class LoadSessionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="studio_analysis_", dir=LIBRARY / "tests" / "python")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def save(self, state, metadata=None, name="weights.pth"):
        path = self.root / name
        torch.save({"state_dict": state, "metadata": metadata or {}}, path)
        return path

    def test_load_reuses_safe_loader_normalizes_keys_and_hashes_source(self):
        state = {"module." + name: value for name, value in cnn_state().items()}
        path = self.save(state)
        known = load_session(path, input_length=64)
        self.assertEqual(known.source, path.resolve())
        self.assertEqual(known.sha256, hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertIn("model.0.weight", known.state)
        self.assertEqual(known.spec.architecture, "cnn")
        self.assertEqual(analyze_session(known)["source"]["path"], str(path.resolve()))
        self.assertEqual(list(self.root.iterdir()), [path])

    def test_original_parse_error_is_retained_and_inventory_can_be_disabled(self):
        path = self.save({"projection.weight": torch.ones(2, 3)})
        with self.assertRaises(ConversionError) as caught:
            load_session(path, allow_inventory=False)
        inventory = load_session(path)
        self.assertIsNone(inventory.spec)
        self.assertIn(str(caught.exception), inventory.warnings)
        self.assertIsNone(analyze_session(inventory)["summary"]["macs"])

    def test_rnn_activation_ambiguity_stays_inventory_until_explicit(self):
        path = self.save(recurrent_state(), {"seq_length": 4})
        inventory = load_session(path)
        self.assertIsNone(inventory.spec)
        self.assertIn("does not encode its activation", inventory.warnings[0])
        self.assertEqual(load_session(path, architecture="rnn").spec.architecture, "rnn")

    def test_minimum_cnn_length_and_invalid_flatten_dimensions(self):
        path = self.save(cnn_state(36))
        self.assertEqual(load_session(path, input_length=36, allow_inventory=False).spec.dimensions["flatten_size"], 4)
        with self.assertRaisesRegex(ConversionError, "too short"):
            load_session(path, input_length=35, allow_inventory=False)
        with self.assertRaisesRegex(ConversionError, "fc1 input"):
            load_session(path, input_length=64, allow_inventory=False)
        self.assertIsNone(load_session(path, input_length=64).spec)

    def test_invalid_values_do_not_degrade_to_inventory(self):
        for name, value in (("nan", float("nan")), ("inf", float("inf"))):
            with self.subTest(name=name):
                path = self.save({"unknown": torch.tensor([value])}, name=name + ".pth")
                with self.assertRaisesRegex(ConversionError, "NaN/Inf"):
                    load_session(path)
        path = self.save({"unknown": torch.ones(1)})
        for options in ({"architecture": "transformer"}, {"input_length": True},
                        {"input_length": 0}, {"seq_length": -1}, {"seq_length": 0x80000000}):
            with self.subTest(options=options), self.assertRaises(ConversionError):
                load_session(path, **options)

    def test_safe_loader_cannot_be_bypassed_by_inventory_mode(self):
        path = self.root / "full_module.pth"
        torch.save(torch.nn.Linear(2, 2), path)
        with self.assertRaisesRegex(ConversionError, "Safe checkpoint loading failed"):
            load_session(path)
        with self.assertRaisesRegex(ConversionError, "known CNNVoiceClassifier"):
            load_session(path, trusted_module=True)
        with self.assertRaisesRegex(ConversionError, "Checkpoint not found"):
            load_session(self.root / "missing.pth")


if __name__ == "__main__":
    unittest.main()
