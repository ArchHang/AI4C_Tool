"""End-to-end candidate checkpoint/export/C regressions for CNN, RNN and LSTM.

Requires torch/numpy and gcc on PATH; without gcc all three tests explicitly skip.
Run from ai_baseon_c after module-level tests have passed:
  python -s -B -X utf8 -m unittest discover -s tests/python -p test_model_studio_system.py -v

Both native builds use only the exported model and its bundled runtime. Fixtures,
checkpoints, bundles, executables and loaded DLLs are owned by one automatically
cleaned project-local TemporaryDirectory per test. No hardware is involved.
"""

import ctypes as ct
import hashlib
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
sys.path.insert(0, str(LIBRARY / "tools"))

from model_converter.emitter import workspace_members
from model_studio.analysis import analyze_session, load_session
from model_studio.compression import prune_session, save_checkpoint
from model_studio.services import export_session, state_digest
from test_model_converter import ExportedModel, cnn_reference, make_cnn, make_recurrent
from test_model_converter import oracle_parameters, rebuild_cnn
import verify_models as recurrent_oracle

COMPILER = shutil.which("gcc")


def _checked(command, directory):
    result = subprocess.run(command, cwd=directory, capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise AssertionError(f"Command failed ({result.returncode}): {command!r}\n"
                             f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    return result


class BundledExportedModel(ExportedModel):
    """Reuse the converter test runner, compiling exclusively the exported sources."""

    def __init__(self, directory, spec, manifest):
        self.library = None
        self.dimensions = spec.dimensions
        extension = ".dll" if os.name == "nt" else ".so"
        destination = directory / ("inference" + extension)
        command = [COMPILER, "-std=c99", "-O3", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror",
                   "-shared", "-I" + str(directory / "runtime")]
        if os.name != "nt":
            command.append("-fPIC")
        command += [str(directory / "model.c")]
        command += [str(directory / "runtime" / source)
                    for source in manifest["bundled_runtime"]["sources"]]
        _checked(command + ["-lm", "-o", str(destination)], directory)
        self.library = ct.CDLL(str(destination))
        types = {"uint8_t": ct.c_uint8, "int32_t": ct.c_int32, "int": ct.c_int, "float": ct.c_float}
        members = workspace_members(spec, "both")
        self.workspace_type = type("StudioSystemWorkspace", (ct.Structure,),
                                   {"_fields_": [(name, types[ctype] * size)
                                                 for ctype, name, size in members]})
        self.workspace = self.workspace_type()
        for name in ("workspace_size", "workspace_alignment"):
            function = getattr(self.library, "ai_model_" + name)
            function.restype = ct.c_size_t
            function.argtypes = []
        if self.library.ai_model_workspace_size() != ct.sizeof(self.workspace):
            raise AssertionError("Generated workspace does not match the actual target ABI size")
        if self.library.ai_model_workspace_alignment() != ct.alignment(self.workspace_type):
            raise AssertionError("Generated workspace does not match the actual target ABI alignment")
        self.library.ai_model_init.argtypes = [ct.c_void_p, ct.c_size_t]
        self.library.ai_model_init.restype = ct.c_int32
        status = self.library.ai_model_init(ct.byref(self.workspace), ct.sizeof(self.workspace))
        if status:
            raise AssertionError(f"Generated C initialization failed: {status}")
        for name in ("run", "run_fp32", "run_int8", "run_quantized"):
            function = getattr(self.library, "ai_model_" + name)
            function.argtypes = [ct.c_void_p, ct.c_size_t, ct.c_void_p, ct.c_size_t, ct.c_void_p]
            function.restype = ct.c_int32


@unittest.skipUnless(COMPILER, "gcc is required; C system regressions were not executed")
class ModelStudioSystemTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="studio_system_", dir=LIBRARY / "tests" / "python")
        self.directory = Path(self.temporary.name)
        self.native_models = []

    def tearDown(self):
        try:
            for native in reversed(self.native_models):
                native.close()
        finally:
            # The DLL handles have been released before Windows removes the owned tree.
            self.temporary.cleanup()

    def _fixture(self, kind):
        if kind == "cnn":
            model = make_cnn(length=64)
            return model.state_dict(), {"schema": "cnn_voice_v1", "input_length": 64}
        state, _, _ = make_recurrent(kind, hidden=5, width=3, classes=3)
        return state, {"schema": "rnn_tanh_v1" if kind == "rnn" else "lstm_ifgo_v1", "seq_length": 7}

    def _candidate_torch(self, candidate):
        d = candidate.spec.dimensions
        if candidate.spec.architecture == "cnn":
            return rebuild_cnn(candidate.state, candidate.metadata)
        recurrent_type = nn.RNN if candidate.spec.architecture == "rnn" else nn.LSTM
        recurrent = recurrent_type(d["input_size"], d["hidden_size"], batch_first=True)
        recurrent.load_state_dict({name.removeprefix("rnn."): value for name, value in candidate.state.items()
                                   if name.startswith("rnn.")})
        head = nn.Linear(d["hidden_size"], d["num_classes"])
        head.load_state_dict({name.removeprefix("head."): value for name, value in candidate.state.items()
                             if name.startswith("head.")})
        return recurrent.eval(), head.eval()

    def _open_native(self, directory, spec, manifest):
        # Register first so even an exception after CDLL loading is cleaned up.
        native = BundledExportedModel.__new__(BundledExportedModel)
        native.library = None
        self.native_models.append(native)
        BundledExportedModel.__init__(native, directory, spec, manifest)
        return native

    def _standalone_example(self, directory, manifest, classes):
        executable = directory / ("standalone_example.exe" if os.name == "nt" else "standalone_example")
        command = [COMPILER, "-std=c99", "-O3", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror",
                   "-I" + str(directory / "runtime"), str(directory / "model.c"),
                   str(directory / "example.c")]
        command += [str(directory / "runtime" / source)
                    for source in manifest["bundled_runtime"]["sources"]]
        _checked(command + ["-lm", "-o", str(executable)], directory)
        result = _checked([str(executable)], directory)
        logits = np.asarray([float(value) for value in result.stdout.split()], dtype=np.float64)
        self.assertEqual(logits.size, classes)
        self.assertTrue(np.all(np.isfinite(logits)))
        return logits

    def _compare_native(self, kind, native, candidate):
        d = candidate.spec.dimensions
        model = self._candidate_torch(candidate)
        parameters = oracle_parameters(candidate.spec) if kind != "cnn" else None
        rng = np.random.default_rng({"cnn": 701, "rnn": 702, "lstm": 703}[kind])
        max_fp32_error, max_int8_error = 0.0, 0.0
        for _ in range(12):
            shape = (d["input_elements"],) if kind == "cnn" else (d["seq_length"], d["input_size"])
            quantized = rng.integers(0, 256, shape, dtype=np.uint8)
            values = quantized.astype(np.float32) / np.float32(255)
            with torch.no_grad():
                if kind == "cnn":
                    expected = model(torch.from_numpy(values.reshape(1, d["input_channels"],
                                                                        d["input_length"]))).numpy().reshape(-1)
                else:
                    recurrent, head = model
                    hidden, _ = recurrent(torch.from_numpy(values[None]))
                    expected = head(hidden[:, -1]).numpy().reshape(-1)
            fp32_logits = native.run(values, "run_fp32")
            np.testing.assert_allclose(fp32_logits, expected, rtol=2e-6, atol=2e-6)
            max_fp32_error = max(max_fp32_error, float(np.max(np.abs(fp32_logits - expected))))
            if kind == "cnn":
                oracle_logits = cnn_reference(candidate.spec, quantized)
            else:
                c_hidden = np.ctypeslib.as_array(native.workspace.hidden).reshape(d["seq_length"], d["hidden_size"])
                np.testing.assert_allclose(c_hidden, hidden.numpy()[0], rtol=2e-6, atol=2e-6)
                expected_q, expected_scale, expected_zp, _ = getattr(recurrent_oracle, kind + "_int8_ref")(
                    quantized, parameters)
                oracle_logits = recurrent_oracle.linear_int8_ref(expected_q[-1], expected_scale[-1],
                                                                 expected_zp[-1], parameters)
            int8_logits = native.run(quantized)
            np.testing.assert_allclose(int8_logits, oracle_logits, rtol=0, atol=2e-7)
            max_int8_error = max(max_int8_error, float(np.max(np.abs(int8_logits - oracle_logits))))
            if kind != "cnn":
                np.testing.assert_array_equal(np.ctypeslib.as_array(native.workspace.hidden_q).reshape(
                    d["seq_length"], d["hidden_size"]), expected_q)
                np.testing.assert_allclose(np.ctypeslib.as_array(native.workspace.hidden_scales), expected_scale,
                                           rtol=2e-6, atol=2e-9)
                np.testing.assert_array_equal(np.ctypeslib.as_array(native.workspace.hidden_zero_points), expected_zp)
            np.testing.assert_array_equal(native.run(values, "run_quantized"), int8_logits)
        return {"samples": 12, "max_fp32_abs_error": max_fp32_error,
                "max_int8_oracle_abs_error": max_int8_error}

    def _exercise(self, kind, method="structured"):
        state, metadata = self._fixture(kind)
        source_path = self.directory / "original.pth"
        torch.save({"state_dict": state, "metadata": metadata}, source_path)
        source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
        original = load_session(source_path, allow_inventory=False)
        original_tensor_digest = state_digest(original)
        original_state = {name: value.clone() for name, value in original.state.items()}
        original_dimensions = dict(original.spec.dimensions)
        before = analyze_session(original, precision="both")["summary"]

        options = {}
        if method == "taylor":
            rng = np.random.default_rng(714)
            shape = (24, original.spec.dimensions["input_elements"])
            calibration = self.directory / "calibration.npz"
            np.savez(calibration, inputs=rng.integers(0, 256, shape, dtype=np.uint8))
            options = {"data_path": calibration, "max_samples": 24}
        candidate, pruning = prune_session(original, amount=0.4, method=method, **options)
        after = analyze_session(candidate, precision="both")["summary"]
        self.assertEqual(candidate.sha256, source_sha)
        self.assertEqual(candidate.source, source_path.resolve())
        self.assertLess(after["total_parameters"], before["total_parameters"])
        self.assertLess(after["macs"], before["macs"])
        self.assertLess(after["deployed_parameter_bytes"], before["deployed_parameter_bytes"])
        self.assertEqual(candidate.spec.dimensions["hidden_size"], 3)
        if kind == "cnn" and method != "svd":
            self.assertEqual([shape[0] for shape in candidate.spec.dimensions["conv_shapes"]], [2, 2, 3])
            self.assertEqual(candidate.spec.dimensions["flatten_size"], 12)
            self.assertEqual(list(candidate.state["fc1.weight"].shape), [3, 12])
            self.assertEqual(list(candidate.state["fc2.weight"].shape), [3, 3])
        elif kind != "cnn":
            gates = 4 if kind == "lstm" else 1
            self.assertEqual(list(candidate.state["rnn.weight_hh_l0"].shape), [gates * 3, 3])
            self.assertEqual(list(candidate.state["head.weight"].shape), [3, 3])
        self.assertNotEqual(state_digest(candidate), original_tensor_digest)

        saved = save_checkpoint(candidate, self.directory / "candidate.pth")
        reloaded = load_session(saved["path"], allow_inventory=False)
        self.assertEqual(reloaded.sha256, saved["sha256"])
        self.assertEqual(reloaded.spec.dimensions, candidate.spec.dimensions)
        self.assertEqual(state_digest(reloaded), state_digest(candidate))
        for name, value in candidate.state.items():
            self.assertTrue(torch.equal(value, reloaded.state[name]), name)

        destination = self.directory / "candidate_bundle"
        exported = export_session(reloaded, destination, precision="both", bundle_runtime=True)
        manifest = exported["manifest"]
        on_disk_manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest, on_disk_manifest)
        self.assertEqual(manifest["dimensions"], candidate.spec.dimensions)
        self.assertNotEqual(manifest["dimensions"], original_dimensions)
        self.assertEqual(manifest["checkpoint_sha256"], reloaded.sha256)
        self.assertEqual(manifest["studio_provenance"]["source_checkpoint_sha256"], reloaded.sha256)
        self.assertEqual(manifest["studio_provenance"]["candidate_state_sha256"], state_digest(candidate))
        self.assertEqual(manifest["studio_analysis"], after)
        self.assertEqual(manifest["precision"], "both")
        self.assertTrue(manifest["bundled_runtime"]["sources"])
        self.assertNotIn('"ai_baseon_c/', (destination / "model.c").read_text(encoding="utf-8"))
        native = self._open_native(destination, reloaded.spec, manifest)
        comparisons = self._compare_native(kind, native, reloaded)
        example_logits = self._standalone_example(destination, manifest, candidate.spec.dimensions["num_classes"])

        # Pruning, checkpoint saving, generation, compilation and inference must preserve the source.
        self.assertEqual(hashlib.sha256(source_path.read_bytes()).hexdigest(), source_sha)
        self.assertEqual(original.sha256, source_sha)
        self.assertEqual(state_digest(original), original_tensor_digest)
        self.assertEqual(original.spec.dimensions, original_dimensions)
        for name, value in original_state.items():
            self.assertTrue(torch.equal(value, original.state[name]), name)
        summary = {"architecture": kind, "method": method, "structured_pruning_amount": pruning["amount"],
                   "before_dimensions": original_dimensions, "candidate_dimensions": candidate.spec.dimensions,
                   "parameters_before": before["total_parameters"], "parameters_after": after["total_parameters"],
                   "macs_before": before["macs"], "macs_after": after["macs"],
                   "deployed_bytes_before": before["deployed_parameter_bytes"],
                   "deployed_bytes_after": after["deployed_parameter_bytes"],
                   "original_checkpoint_sha256": source_sha, "saved_candidate_sha256": reloaded.sha256,
                   "bundled_example_logits": example_logits.tolist(), **comparisons}
        print("MODEL_STUDIO_SYSTEM " + json.dumps(summary, ensure_ascii=False, sort_keys=True, allow_nan=False))

    def test_cnn_pruned_checkpoint_and_bundled_c(self):
        self._exercise("cnn")

    def test_rnn_pruned_checkpoint_and_bundled_c(self):
        self._exercise("rnn")

    def test_lstm_pruned_checkpoint_and_bundled_c(self):
        self._exercise("lstm")

    def test_cnn_svd_bundled_c(self):
        self._exercise("cnn", "svd")

    def test_cnn_fpgm_bundled_c(self):
        self._exercise("cnn", "fpgm")

    def test_rnn_fpgm_bundled_c(self):
        self._exercise("rnn", "fpgm")

    def test_lstm_fpgm_bundled_c(self):
        self._exercise("lstm", "fpgm")

    def test_cnn_taylor_bundled_c(self):
        self._exercise("cnn", "taylor")

    def test_rnn_taylor_bundled_c(self):
        self._exercise("rnn", "taylor")

    def test_lstm_taylor_bundled_c(self):
        self._exercise("lstm", "taylor")


if __name__ == "__main__":
    unittest.main()
