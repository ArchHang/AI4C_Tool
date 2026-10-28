"""Exercise optional packed cache export, real ABI layout, fallback and exactness."""

import ctypes as ct
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from test_model_converter import (ExportedModel, LIBRARY, cnn_reference, make_cnn,
                                  make_recurrent)
from model_converter.emitter import emit_model, weight_cache_metadata, workspace_members
from model_converter.schema import ConversionError, parse_spec


@unittest.skipUnless(shutil.which("gcc"), "gcc is required")
class WeightCacheTests(unittest.TestCase):
    def setUp(self):
        directory = LIBRARY / "build" / "weight-cache-tests"
        directory.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=directory)
        self.directory = Path(self.temporary.name)
        self.models = []

    def tearDown(self):
        for model in self.models:
            model.close()
        self.temporary.cleanup()

    def export(self, spec, policy, flags=()):
        destination = self.directory / str(len(self.models))
        manifest = emit_model(spec, destination, precision="both", cache_policy=policy)
        model = ExportedModel(destination, spec, "both", flags, cache_policy=policy)
        self.models.append(model)
        return model, manifest

    def test_packed_export_abi_initialization_and_integer_oracle(self):
        model = make_cnn()
        spec = parse_spec(model.state_dict(), {"schema": "cnn_voice_v1", "input_length": 64})
        rng = np.random.default_rng(8128)
        for flags in (("-DNN_BACKEND=1",), ("-DNN_BACKEND=4", "-mavx2")):
            plain, _ = self.export(spec, "none", flags)
            packed, manifest = self.export(spec, "packed", flags)
            metadata = manifest["weight_cache"]
            self.assertEqual(metadata, weight_cache_metadata(spec, "both", "packed"))
            self.assertEqual(metadata["bytes"], (8 * 2 + 12 * 3 + 16 * 4) * 2)
            self.assertEqual(metadata["state_bytes"], 4)
            for layer in metadata["layers"]:
                field = f"conv{layer['layer']}_packed"
                offset = getattr(packed.workspace_type, field).offset
                self.assertEqual(offset % 4, 0)
                values = np.ctypeslib.as_array(getattr(packed.workspace, field))
                self.assertEqual(values.nbytes, layer["bytes"])
            snapshot = bytes(packed.workspace)[getattr(packed.workspace_type, "conv0_packed").offset:]
            for _ in range(16):
                q = rng.integers(0, 256, 64, dtype=np.uint8)
                np.testing.assert_array_equal(packed.run(q), plain.run(q))
                np.testing.assert_allclose(packed.run(q), cnn_reference(spec, q), rtol=0, atol=1e-7)
            self.assertEqual(snapshot, bytes(packed.workspace)[
                getattr(packed.workspace_type, "conv0_packed").offset:])
            q = np.zeros(64, np.uint8)
            logits = np.zeros(3, np.float32)
            packed.workspace.weight_cache_ready[0] = 0
            status = packed.library.ai_model_run_int8(q.ctypes.data, q.size,
                logits.ctypes.data, logits.size, ct.byref(packed.workspace))
            self.assertNotEqual(status, 0)
            self.assertEqual(packed.library.ai_model_init(ct.byref(packed.workspace),
                                                       ct.sizeof(packed.workspace)), 0)
            np.testing.assert_array_equal(packed.run(q), plain.run(q))

    def test_policy_validation_and_recurrent_no_cache(self):
        cnn = make_cnn()
        spec = parse_spec(cnn.state_dict(), {"schema": "cnn_voice_v1", "input_length": 64})
        self.assertEqual(workspace_members(spec, "both"), workspace_members(spec, "both", "none"))
        with self.assertRaisesRegex(ConversionError, "cache_policy"):
            emit_model(spec, self.directory / "bad", cache_policy="unknown")
        self.assertEqual(weight_cache_metadata(spec, "fp32", "packed")["bytes"], 0)
        spec.dimensions["conv_shapes"] = [(40, 20, 5, 60, 30)] * 3
        self.assertEqual(weight_cache_metadata(spec, "both", "packed")["layers"], [])
        for kind in ("rnn", "lstm"):
            state, _, _ = make_recurrent(kind)
            schema = "rnn_tanh_v1" if kind == "rnn" else "lstm_ifgo_v1"
            recurrent = parse_spec(state, {"schema": schema, "seq_length": 4})
            plain, _ = self.export(recurrent, "none")
            packed, manifest = self.export(recurrent, "packed")
            self.assertEqual(manifest["weight_cache"]["bytes"], 0)
            values = np.zeros(recurrent.dimensions["input_elements"], np.uint8)
            np.testing.assert_array_equal(plain.run(values), packed.run(values))

    def test_mixed_large_layer_fallback(self):
        # Only the last Conv has an 80-tap dot and must retain the ordinary backend.
        model = make_cnn(channels_per_layer=(4, 16, 4))
        spec = parse_spec(model.state_dict(), {"schema": "cnn_voice_v1", "input_length": 64})
        plain, _ = self.export(spec, "none")
        packed, manifest = self.export(spec, "packed")
        self.assertEqual([layer["layer"] for layer in manifest["weight_cache"]["layers"]], [0, 1])
        self.assertFalse(hasattr(packed.workspace, "conv2_packed"))
        rng = np.random.default_rng(8129)
        for _ in range(12):
            q = rng.integers(0, 256, 64, dtype=np.uint8)
            np.testing.assert_array_equal(packed.run(q), plain.run(q))
            np.testing.assert_allclose(packed.run(q), cnn_reference(spec, q), rtol=0, atol=1e-7)

    def test_native_kernel_oracle_and_rejections(self):
        for name, flags in (("portable_o0", ["-O0", "-DNN_BACKEND=1"]),
                            ("portable_o3", ["-O3", "-DNN_BACKEND=1"]),
                            ("avx2_o3", ["-O3", "-DNN_BACKEND=4", "-mavx2"])):
            executable = self.directory / (name + ".exe")
            command = ["gcc", "-std=c11", "-Wall", "-Wextra", "-Werror",
                       "-ffp-contract=off", "-fno-fast-math", "-I" + str(LIBRARY / "include"),
                       str(LIBRARY / "src/cnn.c"), str(LIBRARY / "src/platform.c"),
                       str(LIBRARY / "tests/test_weight_cache.c"), "-lm", *flags,
                       "-o", str(executable)]
            subprocess.run(command, check=True, capture_output=True, text=True)
            output = subprocess.run([str(executable)], check=True, capture_output=True,
                                    text=True).stdout
            self.assertIn("failures=0", output)


if __name__ == "__main__":
    unittest.main()
