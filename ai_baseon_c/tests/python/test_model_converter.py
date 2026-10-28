"""Compile exported graphs and compare them with PyTorch and independent integer oracles.

Run with a Python environment containing torch/numpy and gcc on PATH:
  python -m unittest discover -s tests/python -p test_model_converter.py -v
"""

import _ctypes
import ctypes as ct
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

LIBRARY = Path(__file__).resolve().parents[2]
REPOSITORY = LIBRARY.parent
sys.path.insert(0, str(LIBRARY / "tools"))

from model_converter import ConversionError, convert_checkpoint
from model_converter.emitter import quantize_parameters, workspace_members
from model_converter.loaders import CNNVoiceClassifier, load_checkpoint
from model_converter.schema import ModelSpec, parse_spec, validate_portable_storage
from model_converter.quantization import quantize_rows
import verify_cnn_model as cnn_oracle
import verify_models as recurrent_oracle


def make_cnn(length=64, channels_per_layer=(2, 3, 4)):
    torch.manual_seed(1701)
    model = CNNVoiceClassifier()
    layers, channels, width = [], 1, length
    for outputs in channels_per_layer:
        layers.extend((nn.Conv1d(channels, outputs, 5, bias=True), nn.BatchNorm1d(outputs),
                       nn.ReLU(), nn.MaxPool1d(2)))
        width = (width - 4) // 2
        channels = outputs
    layers.append(nn.Flatten())
    model.model = nn.Sequential(*layers)
    model.dropout = nn.Dropout(0.4)
    model.fc1 = nn.Linear(width * channels, 5)
    model.fc2 = nn.Linear(5, 3)
    with torch.no_grad():
        for index in (1, 5, 9):
            model.model[index].running_mean.uniform_(-0.2, 0.2)
            model.model[index].running_var.uniform_(0.25, 1.5)
            model.model[index].weight.uniform_(0.8, 1.2)
            model.model[index].bias.uniform_(-0.2, 0.2)
    return model.eval()


def make_recurrent(kind, hidden=5, width=3, classes=3):
    torch.manual_seed(1702)
    recurrent = nn.RNN(width, hidden, batch_first=True) if kind == "rnn" else nn.LSTM(width, hidden, batch_first=True)
    head = nn.Linear(hidden, classes)
    state = {"rnn." + key: value for key, value in recurrent.state_dict().items()}
    state.update({"head." + key: value for key, value in head.state_dict().items()})
    return state, recurrent.eval(), head.eval()


def rebuild_cnn(state, metadata):
    model = CNNVoiceClassifier()
    layers = []
    for layer_index, index in enumerate((0, 4, 8)):
        outputs, inputs, _ = state[f"model.{index}.weight"].shape
        epsilon = metadata.get("bn_eps", [1e-5] * 3)[layer_index]
        layers.extend((nn.Conv1d(inputs, outputs, 5, bias=f"model.{index}.bias" in state),
                       nn.BatchNorm1d(outputs, eps=epsilon), nn.ReLU(), nn.MaxPool1d(2)))
    model.model = nn.Sequential(*layers, nn.Flatten())
    hidden, flattened = state["fc1.weight"].shape
    classes = state["fc2.weight"].shape[0]
    model.dropout = nn.Dropout(0.4)
    model.fc1, model.fc2 = nn.Linear(flattened, hidden), nn.Linear(hidden, classes)
    model.load_state_dict(state)
    return model.eval()


class ExportedModel:
    def __init__(self, directory, spec, precision="both", extra_flags=(), cache_policy="none"):
        suffix = ".dll" if os.name == "nt" else ".so"
        destination = directory / ("inference" + suffix)
        cmd = ["gcc", "-std=c99", "-O3", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror",
               "-shared", "-I" + str(LIBRARY / "include"), "-I" + str(directory)]
        if os.name != "nt":
            cmd.append("-fPIC")
        cmd.extend(extra_flags)
        cmd.extend([str(directory / "model.c"), str(LIBRARY / "src" / "cnn.c"),
                    str(LIBRARY / "src" / "platform.c")])
        if spec.architecture != "cnn":
            cmd.append(str(LIBRARY / "src" / (spec.architecture + ".c")))
        subprocess.run(cmd + ["-lm", "-o", str(destination)], check=True, capture_output=True, text=True)
        self.library = ct.CDLL(str(destination))
        types = {"uint8_t": ct.c_uint8, "int16_t": ct.c_int16, "int32_t": ct.c_int32,
                 "int": ct.c_int, "float": ct.c_float}
        members = workspace_members(spec, precision, cache_policy)
        self.workspace_type = type("Workspace", (ct.Structure,),
                                   {"_fields_": [(name, types[ctype] * size) for ctype, name, size in members]})
        self.workspace = self.workspace_type()
        self.dimensions = spec.dimensions
        for name in ("workspace_size", "workspace_alignment"):
            function = getattr(self.library, "ai_model_" + name)
            function.restype = ct.c_size_t
            function.argtypes = []
        assert self.library.ai_model_workspace_size() == ct.sizeof(self.workspace)
        assert self.library.ai_model_workspace_alignment() == ct.alignment(self.workspace_type)
        self.library.ai_model_init.argtypes = [ct.c_void_p, ct.c_size_t]
        self.library.ai_model_init.restype = ct.c_int32
        assert self.library.ai_model_init(ct.byref(self.workspace), ct.sizeof(self.workspace)) == 0
        for name in ("run", "run_fp32", "run_int8", "run_quantized"):
            function = getattr(self.library, "ai_model_" + name, None)
            if function:
                function.argtypes = [ct.c_void_p, ct.c_size_t, ct.c_void_p, ct.c_size_t, ct.c_void_p]
                function.restype = ct.c_int32

    def run(self, input_values, name="run_int8"):
        values = np.ascontiguousarray(input_values)
        logits = np.zeros(self.dimensions["num_classes"], np.float32)
        status = getattr(self.library, "ai_model_" + name)(values.ctypes.data, values.size,
                                                          logits.ctypes.data, logits.size,
                                                          ct.byref(self.workspace))
        assert status == 0, status
        return logits

    def close(self):
        if self.library is not None:
            handle = self.library._handle
            self.library = None
            if os.name == "nt":
                _ctypes.FreeLibrary(handle)
            else:
                _ctypes.dlclose(handle)


def oracle_parameters(spec):
    q, _ = quantize_parameters(spec)
    p = dict(spec.arrays, **q, **spec.dimensions)
    for key in ("weight", "weight_q", "weight_scale", "bias"):
        p["out_" + ("w" if key == "weight" else "w_q" if key == "weight_q" else
                     "w_scale" if key == "weight_scale" else "b")] = p["out_" + key]
    p["w_ih_q"] = p.pop("w_ih_q")
    return p


def cnn_reference(spec, q_input, weight_scaling="maxabs"):
    quantized, _ = quantize_parameters(spec, weight_scaling=weight_scaling)
    arrays = dict(spec.arrays, **quantized)
    feature = q_input.reshape(spec.dimensions["input_channels"], spec.dimensions["input_length"])
    scale = np.float32(1 / 255)
    for layer in range(3):
        feature, scale = cnn_oracle.dyn_conv_quant(feature, arrays[f"conv{layer}_weight_q"],
                          arrays[f"conv{layer}_weight_scale"], arrays[f"conv{layer}_bias"], scale)
        feature = cnn_oracle.pool_u8(feature)
    hidden = ((feature.reshape(-1).astype(np.int64) @ arrays["fc1_weight_q"].astype(np.int64).T)
              .astype(np.float32) * (arrays["fc1_weight_scale"] * scale) + arrays["fc1_bias"])
    fc_scale = max(np.float32((hidden.max() - hidden.min()) / np.float32(255)), np.float32(1e-9))
    if hidden.max() - hidden.min() < np.float32(255) * np.float32(1e-9):
        magnitude = max(abs(hidden.min()), abs(hidden.max()))
        fc_scale = np.float32(magnitude / np.float32(128)) if magnitude > 1e-9 else np.float32(1e-9)
        zp = 255 if hidden.min() < 0 else 0
    else:
        minimum = min(np.float32(hidden.min()), np.float32(0))
        maximum = max(np.float32(hidden.max()), np.float32(0))
        fc_scale = max(np.float32((maximum - minimum) / np.float32(255)), np.float32(1e-9))
        zp = int(np.rint(np.clip(np.float32(-minimum) / fc_scale, 0, 255)))
    hidden_q = cnn_oracle.quantize_u8_py(hidden, fc_scale, zp)
    logits = ((hidden_q.astype(np.int64) - zp) @ arrays["fc2_weight_q"].astype(np.int64).T).astype(np.float32)
    return logits * (arrays["fc2_weight_scale"] * fc_scale) + arrays["fc2_bias"]


@unittest.skipUnless(shutil.which("gcc"), "gcc is required for compiled generated-model validation")
class ConverterTests(unittest.TestCase):
    def setUp(self):
        temporary_root = LIBRARY / "build" / "converter-tests"
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="model_converter_", dir=temporary_root)
        self.directory = Path(self.temp.name)
        self.native_models = []

    def tearDown(self):
        for model in self.native_models:
            model.close()
        self.temp.cleanup()

    def export(self, state, metadata, precision="both", **options):
        checkpoint = self.directory / (str(len(self.native_models)) + ".pth")
        torch.save({"state_dict": state, "metadata": metadata}, checkpoint)
        destination = self.directory / ("export" + str(len(self.native_models)))
        manifest = convert_checkpoint(checkpoint, destination, precision=precision, **options)
        spec = parse_spec(state, metadata, options.get("architecture", "auto"),
                          options.get("input_length"), options.get("seq_length"))
        native = ExportedModel(destination, spec, precision)
        self.native_models.append(native)
        return native, spec, manifest

    def test_cnn_float_and_integer_oracles(self):
        model = make_cnn()
        native, spec, _ = self.export(model.state_dict(), {"schema": "cnn_voice_v1", "input_length": 64})
        rng = np.random.default_rng(73)
        for _ in range(12):
            q = rng.integers(0, 256, 64, dtype=np.uint8)
            x = q.astype(np.float32) / np.float32(255)
            with torch.no_grad():
                expected = model(torch.from_numpy(x.reshape(1, 1, 64))).numpy().reshape(-1)
            np.testing.assert_allclose(native.run(x, "run_fp32"), expected, rtol=2e-6, atol=2e-6)
            np.testing.assert_allclose(native.run(q), cnn_reference(spec, q), rtol=0, atol=1e-7)
            np.testing.assert_array_equal(native.run(x, "run_quantized"), native.run(q))

    def test_recurrent_float_hidden_integer_oracles(self):
        for kind in ("rnn", "lstm"):
            state, recurrent, head = make_recurrent(kind)
            metadata = {"schema": "rnn_tanh_v1" if kind == "rnn" else "lstm_ifgo_v1", "seq_length": 7}
            native, spec, _ = self.export(state, metadata)
            p = oracle_parameters(spec)
            rng = np.random.default_rng(74)
            for _ in range(12):
                q = rng.integers(0, 256, (7, 3), dtype=np.uint8)
                x = q.astype(np.float32) / np.float32(255)
                with torch.no_grad():
                    hidden, _ = recurrent(torch.from_numpy(x[None]))
                    expected = head(hidden[:, -1]).numpy().reshape(-1)
                np.testing.assert_allclose(native.run(x, "run_fp32"), expected, rtol=2e-6, atol=2e-6)
                c_hidden = np.ctypeslib.as_array(native.workspace.hidden).reshape(7, 5)
                np.testing.assert_allclose(c_hidden, hidden.numpy()[0], rtol=2e-6, atol=2e-6)
                expected_q, expected_scale, expected_zp, _ = getattr(recurrent_oracle, kind + "_int8_ref")(q, p)
                expected_logits = recurrent_oracle.linear_int8_ref(expected_q[-1], expected_scale[-1], expected_zp[-1], p)
                actual_logits = native.run(q)
                np.testing.assert_allclose(actual_logits, expected_logits, rtol=0, atol=2e-7)
                np.testing.assert_array_equal(np.ctypeslib.as_array(native.workspace.hidden_q).reshape(7, 5), expected_q)
                np.testing.assert_allclose(np.ctypeslib.as_array(native.workspace.hidden_scales), expected_scale,
                                           rtol=2e-6, atol=2e-9)
                np.testing.assert_array_equal(np.ctypeslib.as_array(native.workspace.hidden_zero_points), expected_zp)

    def test_api_rejects_invalid_buffers_and_nonfinite(self):
        model = make_cnn()
        native, _, _ = self.export(model.state_dict(), {"input_length": 64})
        values, outputs = np.zeros(64, np.float32), np.zeros(3, np.float32)
        call = native.library.ai_model_run
        valid = [values.ctypes.data, 64, outputs.ctypes.data, 3, ct.byref(native.workspace)]
        for index, replacement in ((0, None), (1, 63), (2, None), (3, 2), (4, None)):
            invalid = valid.copy()
            invalid[index] = replacement
            self.assertNotEqual(call(*invalid), 0)
        self.assertNotEqual(native.library.ai_model_init(ct.byref(native.workspace), ct.sizeof(native.workspace) - 1), 0)
        self.assertNotEqual(native.library.ai_model_init(ct.addressof(native.workspace) + 1, ct.sizeof(native.workspace)), 0)
        alias = valid.copy()
        alias[2] = values.ctypes.data
        self.assertNotEqual(call(*alias), 0)
        alias[2] = ct.addressof(native.workspace)
        self.assertNotEqual(call(*alias), 0)
        for value in (float("nan"), float("inf"), -float("inf")):
            values[0] = value
            self.assertNotEqual(call(*valid), 0)

    def test_single_pass_input_quantization_matches_scalar_edges(self):
        # Odd zero points distinguish round(x/scale)+zp from round(x/scale+zp).
        # Compare the generated entry with both the public scalar quantizer and
        # NumPy ties-to-even, including overflow-to-infinity after finite input.
        state, _, _ = make_recurrent("rnn", hidden=1, width=1025, classes=1)
        metadata = {"schema": "rnn_tanh_v1", "seq_length": 1}
        for scale, zero_point in ((0.125, 127), (1 / 255, 1), (1e-37, 255)):
            native, _, _ = self.export(state, metadata, precision="int8",
                                       input_scale=scale, input_zero_point=zero_point)
            scale = np.float32(scale)
            values = (np.arange(-512, 513, dtype=np.float32) + np.float32(0.5)) * scale
            values[0:4] = [np.finfo(np.float32).max, -np.finfo(np.float32).max, 0.0, -0.0]
            values[4:6] = [np.nextafter(scale * np.float32(0.5), np.float32(0.0)),
                            np.nextafter(scale * np.float32(0.5), np.float32(1.0))]
            native.run(values, "run_quantized")
            actual = np.ctypeslib.as_array(native.workspace.input_q).copy()
            scalar = np.empty(values.size, dtype=np.uint8)
            quantize = native.library.quantize_u8
            quantize.argtypes = [ct.c_void_p, ct.c_uint32, ct.c_float, ct.c_int, ct.c_void_p]
            quantize.restype = None
            quantize(values.ctypes.data, values.size, scale, zero_point, scalar.ctypes.data)
            with np.errstate(over="ignore"):
                oracle = np.clip(np.rint(values / scale) + zero_point, 0, 255).astype(np.uint8)
            np.testing.assert_array_equal(actual, scalar)
            np.testing.assert_array_equal(actual, oracle)
            logits = np.full(1, 123.0, np.float32)
            valid = [values.ctypes.data, values.size, logits.ctypes.data, 1,
                     ct.byref(native.workspace)]
            call = native.library.ai_model_run_quantized
            invalid_alignment = valid.copy()
            invalid_alignment[0] += 1
            self.assertNotEqual(call(*invalid_alignment), 0)
            for value in (float("nan"), float("inf"), -float("inf")):
                values[-1] = value
                self.assertNotEqual(call(*valid), 0)
                self.assertEqual(logits[0], 123.0)

    def test_invalid_tensor_and_batch_norm_arithmetic_rejected(self):
        state = make_cnn().state_dict()
        for value in (state["fc1.weight"].to_sparse(), torch.empty(5, 16, device="meta")):
            invalid = dict(state)
            invalid["fc1.weight"] = value
            with self.assertRaisesRegex(ConversionError, "dense, materialized"):
                parse_spec(invalid, {"input_length": 64})
        for epsilon in (1e100, 1e-100, None, "invalid", True, float("nan"), -1.0):
            with self.assertRaisesRegex(ConversionError, "epsilon.*float32"):
                parse_spec(state, {"input_length": 64, "bn_eps": [epsilon, 1e-5, 1e-5]})
        # Finite operands may overflow before final weights/bias are formed.
        # Previously var+eps became inf, zeroed the BN factor, and looked valid.
        invalid = dict(state)
        invalid["model.1.running_var"] = torch.full((2,), torch.finfo(torch.float32).max)
        with self.assertRaisesRegex(ConversionError, "BatchNorm folding overflowed"):
            parse_spec(invalid, {"input_length": 64,
                                "bn_eps": [float(np.finfo(np.float32).max), 1e-5, 1e-5]})

    def test_quantize_rows_rejects_invalid_numeric_arrays(self):
        for values in (np.empty((0, 2)), np.empty((2, 0)), np.array(1.0), np.ones(3),
                       np.array([[float("nan")]]), np.array([[float("inf")]]),
                       np.array([[1e100]]), np.array([[1.0 + 2.0j]])):
            with self.assertRaises(ValueError):
                quantize_rows(values)
        # Extreme finite data and deliberately tiny candidates must stay finite;
        # overflow during normalization denotes intentional endpoint clipping.
        values = np.array([[np.finfo(np.float32).max, 0.0]], np.float32)
        with np.errstate(all="raise"):
            quantized, scale, _, metrics = quantize_rows(values, weight_scaling="mse",
                                                        mse_min_ratio=1e-40, mse_steps=2)
        self.assertTrue(np.all(np.isfinite(scale)))
        self.assertTrue(np.isfinite(metrics["mse"]))
        self.assertEqual(quantized[0, 0], 127)

    def test_checkpoint_and_schema_errors(self):
        model = make_cnn()
        state = model.state_dict()
        for key, value in (("fc1.weight", torch.ones(5, 17)),
                           ("model.1.running_var", torch.full((2,), -1.0)),
                           ("fc1.bias", torch.full((5,), float("nan")))):
            malformed = dict(state)
            malformed[key] = value
            with self.assertRaises(ConversionError):
                parse_spec(malformed, {"input_length": 64})
        with self.assertRaises(ConversionError):
            parse_spec(state, {"input_length": 64}, input_length=0)
        recurrent_state, _, _ = make_recurrent("rnn")
        with self.assertRaises(ConversionError):
            parse_spec(recurrent_state, {"seq_length": 7})
        with self.assertRaises(ConversionError):
            parse_spec(recurrent_state, {"seq_length": 7, "schema": "rnn_tanh_v1", "nonlinearity": "relu"})
        stacked = dict(recurrent_state, **{"rnn.weight_hh_l1": torch.ones(5, 5)})
        with self.assertRaises(ConversionError):
            parse_spec(stacked, {"seq_length": 7}, "rnn")
        with self.assertRaises(ConversionError):
            parse_spec(recurrent_state, {}, "rnn")
        path = self.directory / "invalid.pth"
        torch.save({"state_dict": {"weight": "not a tensor"}}, path)
        with self.assertRaises(ConversionError):
            load_checkpoint(path)

    def test_mse_weight_search_never_increases_row_error(self):
        rng = np.random.default_rng(93)
        arrays = [np.zeros((3, 9), np.float32), -np.ones((3, 9), np.float32),
                  -rng.uniform(0, 2, (5, 127)).astype(np.float32),
                  rng.normal(size=(12, 513)).astype(np.float32)]
        outliers = rng.normal(0, 0.02, (4, 257)).astype(np.float32)
        outliers[:, 0] = 1.0
        arrays.append(outliers)
        for array in arrays:
            baseline_q, baseline_s, _, _ = quantize_rows(array)
            optimized_q, optimized_s, rowsum, metrics = quantize_rows(array, weight_scaling="mse")
            baseline_error = np.mean((baseline_q.astype(np.float64) * baseline_s[:, None] - array) ** 2, axis=1)
            optimized_error = np.mean((optimized_q.astype(np.float64) * optimized_s[:, None] - array) ** 2, axis=1)
            self.assertTrue(np.all(optimized_error <= baseline_error))
            self.assertTrue(np.all(np.isfinite(optimized_s)))
            self.assertTrue(np.all(optimized_s > 0))
            np.testing.assert_array_equal(rowsum, optimized_q.astype(np.int32).sum(axis=1))
            self.assertLessEqual(metrics["mse"], metrics["maxabs_baseline_mse"])
        for options in ({"mse_min_ratio": 0}, {"mse_min_ratio": float("nan")}, {"mse_steps": 1}):
            with self.assertRaises(ValueError):
                quantize_rows(arrays[-1], weight_scaling="mse", **options)

    def test_mse_export_matches_integer_oracle(self):
        model = make_cnn()
        native, spec, manifest = self.export(model.state_dict(), {"input_length": 64},
                                             weight_scaling="mse")
        q = np.arange(64, dtype=np.uint8) * 4
        np.testing.assert_allclose(native.run(q), cnn_reference(spec, q, "mse"), rtol=0, atol=1e-7)
        self.assertEqual(manifest["weight_quantization"]["scale_selection"], "mse")
        for metrics in manifest["weight_quantization_error"].values():
            self.assertLessEqual(metrics["mse"], metrics["maxabs_baseline_mse"])

    def test_all_precision_choices_compile(self):
        for kind in ("cnn", "rnn", "lstm"):
            if kind == "cnn":
                state, metadata = make_cnn().state_dict(), {"input_length": 64}
            else:
                state, _, _ = make_recurrent(kind)
                metadata = {"schema": "rnn_tanh_v1" if kind == "rnn" else "lstm_ifgo_v1", "seq_length": 7}
            for precision in ("fp32", "int8"):
                native, spec, manifest = self.export(state, metadata, precision)
                values = np.zeros(spec.dimensions["input_elements"], np.float32)
                self.assertTrue(np.all(np.isfinite(native.run(values, "run"))))
                self.assertEqual(manifest["precision"], precision)

    def test_bundled_runtime_compiles(self):
        state, _, _ = make_recurrent("lstm")
        checkpoint = self.directory / "bundle.pth"
        torch.save({"state_dict": state, "metadata": {"schema": "lstm_ifgo_v1", "seq_length": 7}}, checkpoint)
        destination = self.directory / "bundle"
        manifest = convert_checkpoint(checkpoint, destination, bundle_runtime=True)
        executable = destination / ("example.exe" if os.name == "nt" else "example")
        runtime = manifest["bundled_runtime"]["sources"]
        subprocess.run(["gcc", "-std=c99", "-O3", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror",
                        "-I" + str(destination / "runtime"), str(destination / "model.c"),
                        str(destination / "example.c")] + [str(destination / "runtime" / source) for source in runtime]
                       + ["-lm", "-o", str(executable)], check=True, capture_output=True, text=True)
        result = subprocess.run([str(executable)], check=True, capture_output=True, text=True)
        self.assertEqual(len(result.stdout.split()), 3)

    def test_constant_fc_quantization_preserves_nonzero(self):
        for value in (-2.0, 0.0, 2.0):
            model = make_cnn()
            with torch.no_grad():
                model.fc1.weight.zero_()
                model.fc1.bias.fill_(value)
            native, spec, _ = self.export(model.state_dict(), {"input_length": 64})
            q = np.zeros(64, np.uint8)
            np.testing.assert_allclose(native.run(q), cnn_reference(spec, q), rtol=0, atol=1e-7)
            x = q.astype(np.float32)
            np.testing.assert_allclose(native.run(q), native.run(x, "run_fp32"), rtol=0, atol=0.02)

    def test_same_sign_fc_ranges_preserve_endpoints(self):
        # An identity head exposes activation reconstruction directly, without
        # reproducing the converter's scale/zero-point formulas in the oracle.
        for sign in (-1.0, 1.0):
            model = make_cnn()
            expected = sign * np.array([1.0, 2.0, 5.0], np.float32)
            with torch.no_grad():
                model.fc1.weight.zero_()
                model.fc1.bias.copy_(torch.arange(1, 6, dtype=torch.float32) * sign)
                model.fc2.weight.zero_()
                model.fc2.bias.zero_()
                for output, hidden in enumerate((0, 1, 4)):
                    model.fc2.weight[output, hidden] = 1.0
            native, _, _ = self.export(model.state_dict(), {"input_length": 64})
            q = np.zeros(64, np.uint8)
            np.testing.assert_array_equal(native.run(q.astype(np.float32), "run_fp32"), expected)
            # Both same-sign ranges span magnitude five, hence nearest rounding
            # is bounded by half one uint8 step plus float arithmetic error.
            np.testing.assert_allclose(native.run(q), expected, rtol=0,
                                       atol=5.0 / (2.0 * 255.0) + 2e-6)
            self.assertAlmostEqual(float(native.run(q)[-1]), float(expected[-1]), places=5)

    def test_portable32_storage_limits_reject_without_large_allocations(self):
        for kind in ("rnn", "lstm"):
            state, _, _ = make_recurrent(kind, hidden=1, width=1000, classes=1)
            metadata = {"schema": "rnn_tanh_v1" if kind == "rnn" else "lstm_ifgo_v1",
                        "seq_length": 600000}
            with self.assertRaisesRegex(ConversionError, "float input.*portable 32-bit"):
                parse_spec(state, metadata)
            metadata["seq_length"] = (0x7fffffff // 4) // 1000
            spec = parse_spec(state, metadata)
            self.assertLessEqual(spec.dimensions["input_elements"], 0x7fffffff // 4)
        # Synthetic dimensions exercise matrix guards without allocating a
        # multi-gigabyte tensor just to check its logical storage requirement.
        spec = ModelSpec("lstm", "lstm_ifgo_v1",
                         {"input_elements": 1, "num_classes": 1, "hidden_size": 12000,
                          "input_size": 1, "seq_length": 1}, {}, {})
        with self.assertRaisesRegex(ConversionError, "recurrent_weights.*portable 32-bit"):
            validate_portable_storage(spec)

    def test_trained_recurrent_checkpoints_5000_samples(self):
        from train_models import make_dataset
        report = {}
        torch.set_num_threads(1)
        for kind in ("rnn", "lstm"):
            path = LIBRARY / "model" / (kind + "_params.npz")
            if not path.is_file():
                self.skipTest("Trained recurrent NPZ fixtures unavailable")
            parameters = np.load(path)
            recurrent, head = recurrent_oracle.build_torch_model(kind, parameters)
            state = {"rnn." + key: value for key, value in recurrent.state_dict().items()}
            state.update({"head." + key: value for key, value in head.state_dict().items()})
            metadata = {"schema": "rnn_tanh_v1" if kind == "rnn" else "lstm_ifgo_v1", "seq_length": 16}
            native, spec, _ = self.export(state, metadata)
            x, labels = make_dataset(5000, 16, int(parameters["input_size"]),
                                      int(parameters["num_classes"]), seed=20260808)
            q = np.rint(x * np.float32(255)).astype(np.uint8)
            with torch.no_grad():
                hidden, _ = recurrent(torch.from_numpy(x))
                expected = head(hidden[:, -1]).numpy()
            fp32 = np.stack([native.run(frame, "run_fp32") for frame in x])
            int8 = np.stack([native.run(frame, "run_int8") for frame in q])
            np.testing.assert_allclose(fp32, expected, rtol=0, atol=5e-4)
            np.testing.assert_array_equal(int8.argmax(1), expected.argmax(1))
            self.assertTrue(np.all(np.isfinite(int8)))
            report[kind] = {"samples": 5000, "fp32_correct": int(np.sum(fp32.argmax(1) == labels)),
                            "int8_correct": int(np.sum(int8.argmax(1) == labels)),
                            "prediction_mismatches": int(np.sum(fp32.argmax(1) != int8.argmax(1))),
                            "fp32_torch_max_abs_error": float(np.max(np.abs(fp32 - expected))),
                            "quantized_vs_fp32_max_abs_error": float(np.max(np.abs(int8 - fp32)))}
        (LIBRARY / "build" / "converter_recurrent_accuracy.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")

    def test_train_export_uses_unified_converter(self):
        from types import SimpleNamespace
        import train_models
        previous_root, previous_model_dir = train_models.ROOT, train_models.MODEL_DIR
        try:
            train_models.ROOT = str(self.directory)
            train_models.MODEL_DIR = str(self.directory / "model")
            _, recurrent, head = make_recurrent("rnn")
            args = SimpleNamespace(input_dim=3, hidden=5, num_classes=3, seq_len=7)
            train_models.export_params("rnn", recurrent, head, args)
            destination = self.directory / "model" / "generated" / "rnn"
            self.assertTrue((self.directory / "model" / "rnn.pth").is_file())
            self.assertTrue((self.directory / "examples" / "models" / "rnn_model.h").is_file())
            manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema"], "rnn_tanh_v1")
            self.assertEqual(manifest["dimensions"]["seq_length"], 7)
            executable = destination / ("example.exe" if os.name == "nt" else "example")
            subprocess.run(["gcc", "-std=c99", "-O3", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror",
                            "-I" + str(destination / "runtime"), str(destination / "model.c"),
                            str(destination / "example.c")] +
                           [str(destination / "runtime" / source) for source in manifest["bundled_runtime"]["sources"]]
                           + ["-lm", "-o", str(executable)], check=True, capture_output=True, text=True)
            subprocess.run([str(executable)], check=True, capture_output=True, text=True)
        finally:
            train_models.ROOT, train_models.MODEL_DIR = previous_root, previous_model_dir

    def test_real_cnn_all_dataset_frames(self):
        checkpoint = REPOSITORY / "model" / "best_voice_classifier.pth"
        manifest_path = REPOSITORY / "datasets" / "test_int8" / "manifest.csv"
        if not checkpoint.exists() or not manifest_path.exists():
            self.skipTest("Repository CNN/dataset unavailable")
        state, metadata, _ = load_checkpoint(checkpoint, trusted_module=True)
        native, spec, default_manifest = self.export(state, metadata)
        optimized, _, mse_manifest = self.export(state, metadata, weight_scaling="mse")
        self.assertEqual(default_manifest["parameter_bytes"], mse_manifest["parameter_bytes"])
        self.assertEqual(default_manifest["workspace_members"], mse_manifest["workspace_members"])
        features, labels = [], []
        with manifest_path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                feature_path = manifest_path.parent / Path(row["feat_bin"].replace("\\", "/"))
                values = np.fromfile(feature_path, dtype=np.uint8)
                self.assertEqual(values.size % 2048, 0)
                frames = values.reshape(-1, 2048)
                features.extend(frames)
                labels.extend([int(row["label_id"])] * len(frames))
        q = np.stack(features)
        x = q.astype(np.float32) / np.float32(255)
        torch.set_num_threads(1)
        model = rebuild_cnn(state, metadata)
        with torch.no_grad():
            expected = np.concatenate([model(torch.from_numpy(chunk[:, None])).numpy()
                                       for chunk in np.array_split(x, 16)])
        fp32 = np.stack([native.run(frame, "run_fp32") for frame in x])
        int8 = np.stack([native.run(frame, "run_int8") for frame in q])
        int8_mse = np.stack([optimized.run(frame, "run_int8") for frame in q])
        self.assertTrue(np.all(np.isfinite(fp32)))
        self.assertTrue(np.all(np.isfinite(int8)))
        self.assertTrue(np.all(np.isfinite(int8_mse)))
        np.testing.assert_allclose(fp32, expected, rtol=2e-5, atol=2e-5)
        report = {"samples": len(labels), "fp32_correct": int(np.sum(fp32.argmax(1) == labels)),
                  "int8_correct": int(np.sum(int8.argmax(1) == labels)),
                  "prediction_mismatches": int(np.sum(fp32.argmax(1) != int8.argmax(1))),
                  "fp32_torch_max_abs_error": float(np.max(np.abs(fp32 - expected))),
                  "quantized_vs_fp32_max_abs_error": float(np.max(np.abs(int8 - fp32))),
                  "quantized_vs_fp32_rmse": float(np.sqrt(np.mean((int8.astype(np.float64) - fp32) ** 2))),
                  "mse_int8_correct": int(np.sum(int8_mse.argmax(1) == labels)),
                  "mse_prediction_mismatches_vs_fp32": int(np.sum(fp32.argmax(1) != int8_mse.argmax(1))),
                  "mse_quantized_vs_fp32_max_abs_error": float(np.max(np.abs(int8_mse - fp32))),
                  "mse_quantized_vs_fp32_rmse": float(np.sqrt(np.mean((int8_mse.astype(np.float64) - fp32) ** 2))),
                  "mse_weight_metrics": mse_manifest["weight_quantization_error"]}
        (LIBRARY / "build" / "converter_cnn_accuracy.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")

    def test_real_repository_cnn_checkpoint(self):
        checkpoint = REPOSITORY / "model" / "best_voice_classifier.pth"
        if not checkpoint.exists():
            self.skipTest("Repository CNN checkpoint unavailable")
        with self.assertRaises(ConversionError):
            load_checkpoint(checkpoint)
        state, metadata, _ = load_checkpoint(checkpoint, trusted_module=True)
        native, spec, manifest = self.export(state, metadata)
        self.assertEqual(manifest["dimensions"]["input_elements"], 2048)
        self.assertEqual(manifest["dimensions"]["flatten_size"], 4032)
        model = rebuild_cnn(state, metadata)
        rng = np.random.default_rng(75)
        for _ in range(6):
            q = rng.integers(0, 256, 2048, dtype=np.uint8)
            x = q.astype(np.float32) / np.float32(255)
            with torch.no_grad():
                expected = model(torch.from_numpy(x.reshape(1, 1, 2048))).numpy().reshape(-1)
            np.testing.assert_allclose(native.run(x, "run_fp32"), expected, rtol=2e-5, atol=2e-5)
            np.testing.assert_allclose(native.run(q), cnn_reference(spec, q), rtol=0, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
