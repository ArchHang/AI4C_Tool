"""CPU-only pruning, safe checkpoint persistence, and local-data validation tests."""

from copy import deepcopy
from contextlib import ExitStack
import csv
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch
from torch import nn

LIBRARY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LIBRARY / "tools"))

from model_converter.loaders import CNNVoiceClassifier, load_checkpoint
from model_converter.schema import ConversionError, parse_spec
from model_studio.analysis import ModelSession
from model_studio.compression import prune_session, save_checkpoint
from model_studio.validation import build_model, evaluate_sessions
from model_studio.validation import _evaluation_plan, _inspect_numpy_header, _numpy_data


def make_session(kind="cnn", recurrent_prefix="rnn.", head_prefix="head."):
    torch.manual_seed(401)
    if kind == "cnn":
        model = CNNVoiceClassifier()
        layers = []
        channels, width = 1, 64
        for outputs in (4, 6, 8):
            layers.extend((nn.Conv1d(channels, outputs, 5), nn.BatchNorm1d(outputs),
                           nn.ReLU(), nn.MaxPool1d(2)))
            channels, width = outputs, (width - 4) // 2
        model.model = nn.Sequential(*layers, nn.Flatten())
        model.fc1, model.fc2 = nn.Linear(width * channels, 10), nn.Linear(10, 3)
        model.dropout = nn.Dropout(0.4)
        with torch.no_grad():
            for index in (1, 5, 9):
                model.model[index].running_mean.uniform_(-0.3, 0.3)
                model.model[index].running_var.uniform_(0.5, 1.5)
                model.model[index].weight.uniform_(0.5, 1.5)
            # Distinct ascending row magnitudes make channel selection inspectable.
            for index in (0, 4, 8):
                weight = model.model[index].weight
                weight.copy_(torch.arange(1, weight.shape[0] + 1)[:, None, None] * 0.001)
            model.fc1.weight.copy_(torch.arange(1, 11)[:, None] * 0.001)
        metadata = {"schema": "cnn_voice_v1", "input_length": 64,
                    "bn_eps": [1e-5, 2e-5, 3e-5]}
        for ordinal, index in enumerate((1, 5, 9)):
            model.model[index].eps = metadata["bn_eps"][ordinal]
        state = model.state_dict()
        reference = model.eval()
    else:
        recurrent = (nn.RNN(3, 6, batch_first=True) if kind == "rnn"
                     else nn.LSTM(3, 6, batch_first=True))
        head = nn.Linear(6, 3)
        state = {recurrent_prefix + key: value for key, value in recurrent.state_dict().items()}
        state.update({head_prefix + key: value for key, value in head.state_dict().items()})
        metadata = {"schema": "rnn_tanh_v1" if kind == "rnn" else "lstm_ifgo_v1", "seq_length": 5}
        reference = (recurrent.eval(), head.eval())
    state = {key: value.detach().clone() for key, value in state.items()}
    spec = parse_spec(state, metadata)
    return ModelSession(None, "test-digest", state, metadata, spec), reference


class CompressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def assert_state_equal(self, left, right):
        self.assertEqual(left.keys(), right.keys())
        for key in left:
            self.assertTrue(torch.equal(left[key], right[key]), key)

    def test_cnn_physical_pruning_tracks_bn_flatten_and_head(self):
        session, _ = make_session()
        snapshot = deepcopy(session)
        result, report = prune_session(session, 0.5)
        self.assertEqual([shape[0] for shape in result.spec.dimensions["conv_shapes"]], [2, 3, 4])
        self.assertEqual(result.state["model.4.weight"].shape, (3, 2, 5))
        self.assertEqual(result.state["model.8.weight"].shape, (4, 3, 5))
        self.assertEqual(result.state["fc1.weight"].shape, (5, 16))
        self.assertEqual(result.state["fc2.weight"].shape, (3, 5))
        self.assertTrue(torch.equal(result.state["model.1.running_mean"], session.state["model.1.running_mean"][-2:]))
        self.assertTrue(torch.equal(result.state["fc1.weight"], session.state["fc1.weight"][-5:, -16:]))
        self.assertEqual(result.spec.dimensions["input_elements"], 64)
        self.assertEqual(result.spec.dimensions["num_classes"], 3)
        self.assertEqual(result.metadata["input_length"], 64)
        self.assertLess(report["weights_after"], report["weights_before"])
        self.assertEqual(report["quality_status"], "requires_validation")
        self.assert_state_equal(session.state, snapshot.state)
        self.assertEqual(session.metadata, snapshot.metadata)
        self.assertEqual(session.changes, [])
        json.dumps(result.changes, allow_nan=False)
        result.state["fc2.bias"].zero_()
        self.assert_state_equal(session.state, snapshot.state)

    def test_recurrent_gate_rows_columns_and_head_are_synchronized(self):
        for kind in ("rnn", "lstm"):
            with self.subTest(kind=kind):
                session, _ = make_session(kind, "encoder.", "classifier.")
                snapshot = deepcopy(session.state)
                result, _ = prune_session(session, 0.5)
                gates = 4 if kind == "lstm" else 1
                self.assertEqual(result.state["encoder.weight_ih_l0"].shape, (gates * 3, 3))
                self.assertEqual(result.state["encoder.weight_hh_l0"].shape, (gates * 3, 3))
                self.assertEqual(result.state["encoder.bias_ih_l0"].shape, (gates * 3,))
                self.assertEqual(result.state["classifier.weight"].shape, (3, 3))
                self.assertEqual(result.spec.dimensions["seq_length"], 5)
                self.assertEqual(result.spec.dimensions["input_size"], 3)
                self.assertEqual(result.spec.dimensions["num_classes"], 3)
                w_ih = session.state["encoder.weight_ih_l0"].double().abs().reshape(gates, 6, 3)
                w_hh = session.state["encoder.weight_hh_l0"].double().abs().reshape(gates, 6, 6)
                score = w_ih.sum((0, 2)) + w_hh.sum((0, 2)) + w_hh.sum((0, 1))
                score += session.state["classifier.weight"].double().abs().sum(0)
                selected = torch.sort(torch.argsort(score, descending=True, stable=True)[:3]).values
                rows = torch.cat([selected + gate * 6 for gate in range(gates)])
                self.assertTrue(torch.equal(result.state["encoder.weight_hh_l0"],
                    snapshot["encoder.weight_hh_l0"][rows][:, selected]))
                self.assertTrue(torch.equal(result.state["encoder.bias_ih_l0"], snapshot["encoder.bias_ih_l0"][rows]))
                self.assert_state_equal(session.state, snapshot)
                self.assertEqual(build_model(result)(torch.zeros(2, 5, 3)).shape, (2, 3))

    def test_zero_and_full_ratio_boundaries_all_graphs(self):
        for kind in ("cnn", "rnn", "lstm"):
            with self.subTest(kind=kind):
                session, _ = make_session(kind)
                identity, report = prune_session(session, 0)
                self.assert_state_equal(identity.state, session.state)
                self.assertEqual(report["removed"], 0)
                minimal, _ = prune_session(session, 1)
                self.assertEqual(minimal.spec.dimensions["hidden_size"], 1)
                if kind == "cnn":
                    self.assertEqual([shape[0] for shape in minimal.spec.dimensions["conv_shapes"]], [1, 1, 1])
                self.assertEqual(minimal.spec.dimensions["num_classes"], 3)
                self.assertEqual(build_model(minimal)(torch.zeros(1, 1, 64) if kind == "cnn" else torch.zeros(1, 5, 3)).shape, (1, 3))

    def test_unstructured_exact_global_count_retains_shapes(self):
        for kind in ("cnn", "rnn", "lstm"):
            session, _ = make_session(kind)
            original = deepcopy(session.state)
            with torch.no_grad():
                for key, value in session.state.items():
                    if "weight" in key and value.ndim >= 2:
                        value[value == 0] = 1
            result, report = prune_session(session, 0.37, "unstructured")
            self.assertEqual(report["weights_before"], report["weights_after"])
            self.assertEqual(report["zero_weights_after"], int(report["weights_before"] * 0.37))
            self.assertIn("same MACs", report["dense_execution_note"])
            for key in original:
                self.assertEqual(result.state[key].shape, original[key].shape)
            self.assertEqual(result.spec.dimensions, session.spec.dimensions)

    def test_invalid_amount_method_and_unknown_graph(self):
        session, _ = make_session()
        for amount in (-0.1, 1.1, float("nan"), float("inf"), True, "invalid", None):
            with self.subTest(amount=amount), self.assertRaises(ConversionError):
                prune_session(session, amount)
        with self.assertRaises(ConversionError):
            prune_session(session, 0.2, "mystery")
        with self.assertRaises(ConversionError):
            prune_session(replace(session, spec=None), 0)

    def test_noncontiguous_sparse_weights_are_actually_zeroed(self):
        session, _ = make_session("rnn")
        weight = session.state["rnn.weight_ih_l0"]
        session.state["rnn.weight_ih_l0"] = weight.t().contiguous().t()
        self.assertFalse(session.state["rnn.weight_ih_l0"].is_contiguous())
        result, report = prune_session(session, 1, "unstructured")
        self.assertEqual(report["zero_weights_after"], report["weights_before"])
        self.assertTrue(torch.equal(session.state["rnn.weight_ih_l0"], weight))
        self.assertEqual(torch.count_nonzero(result.state["rnn.weight_ih_l0"]), 0)

    def test_save_reload_no_overwrite_and_failure_cleanup(self):
        session, _ = make_session("lstm")
        result, _ = prune_session(session, 0.5)
        with tempfile.TemporaryDirectory(prefix=".studio-test-", dir=LIBRARY) as directory:
            base = Path(directory)
            source = base / "source.pth"
            torch.save({"state_dict": session.state, "metadata": session.metadata}, source)
            original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            result.source = source
            target = base / "pruned.pth"
            saved = save_checkpoint(result, target)
            state, metadata, digest = load_checkpoint(target)
            self.assertEqual(saved["sha256"], digest)
            self.assert_state_equal(state, result.state)
            self.assertEqual(metadata["seq_length"], 5)
            self.assertEqual(parse_spec(state, metadata).dimensions["hidden_size"], 3)
            with self.assertRaises(ConversionError):
                save_checkpoint(result, target)
            with self.assertRaises(ConversionError):
                save_checkpoint(result, source, overwrite=True)
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), original_hash)
            save_checkpoint(result, target, overwrite=True)
            with patch("model_studio.compression.torch.save", side_effect=RuntimeError("simulated disk failure")):
                with self.assertRaises(ConversionError):
                    save_checkpoint(result, base / "failed.pth")
            self.assertEqual(sorted(item.name for item in base.iterdir()), ["pruned.pth", "source.pth"])
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), original_hash)

    def test_build_matches_original_torch_semantics(self):
        for kind in ("cnn", "rnn", "lstm"):
            with self.subTest(kind=kind):
                session, original = make_session(kind)
                values = torch.randn(4, 1, 64) if kind == "cnn" else torch.randn(4, 5, 3)
                with torch.inference_mode():
                    if kind == "cnn":
                        expected = original(values)
                    else:
                        recurrent, head = original
                        expected = head(recurrent(values)[0][:, -1])
                    actual = build_model(session)(values)
                torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)

    def test_numeric_metrics_uint8_and_float_inputs(self):
        for kind in ("cnn", "rnn", "lstm"):
            session, _ = make_session(kind)
            candidate, _ = prune_session(session, 0.5)
            elements = session.spec.dimensions["input_elements"]
            inputs = np.random.default_rng(402).integers(0, 256, (6, elements), dtype=np.uint8)
            shape = (1, 64) if kind == "cnn" else (5, 3)
            values = torch.from_numpy((inputs.astype(np.float32) - 128) * np.float32(0.01)).reshape(6, *shape)
            with torch.inference_mode():
                ref_logits = build_model(session)(values).numpy()
                cand_logits = build_model(candidate)(values).numpy()
            labels = ref_logits.argmax(1)
            difference = cand_logits.astype(np.float64) - ref_logits.astype(np.float64)
            with tempfile.TemporaryDirectory(prefix=".studio-test-", dir=LIBRARY) as directory:
                dataset = Path(directory) / "data.npz"
                np.savez(dataset, inputs=inputs, labels=labels)
                report = evaluate_sessions(session, candidate, dataset, input_scale=0.01, input_zero_point=128)
                self.assertEqual(report["samples"], 6)
                self.assertAlmostEqual(report["max_abs_error"], float(np.abs(difference).max()), places=7)
                self.assertAlmostEqual(report["rmse"], float(np.sqrt(np.mean(difference ** 2))), places=7)
                self.assertEqual(report["prediction_agreement"], float(np.mean(ref_logits.argmax(1) == cand_logits.argmax(1))))
                self.assertEqual(report["reference_accuracy"], 1)
                self.assertEqual(report["candidate_accuracy"], float(np.mean(cand_logits.argmax(1) == labels)))
                self.assertEqual(report["device"], "cpu")
                json.dumps(report, allow_nan=False)
                floating = Path(directory) / "float.npy"
                np.save(floating, values.numpy())
                identity = evaluate_sessions(session, session, floating, max_samples=2, input_scale=123)
                self.assertEqual(identity["samples"], 2)
                self.assertEqual(identity["max_abs_error"], 0)
                self.assertIsNone(identity["candidate_accuracy"])
                self.assertIn("accuracy_unknown", identity["quality_status"])
                self.assertEqual(identity["input_interpretation"], "preprocessed_float")

    def test_manifest_fixed_frame_split_and_rejection(self):
        session, _ = make_session("rnn")
        with tempfile.TemporaryDirectory(prefix=".studio-test-", dir=LIBRARY) as directory:
            base = Path(directory)
            feature = base / "feature.bin"
            np.arange(45, dtype=np.uint8).tofile(feature)
            manifest = base / "manifest.csv"
            def write_manifest(label=1, frames=3, feature_name="feature.bin"):
                with manifest.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, ["feat_bin", "label_id", "frames"])
                    writer.writeheader()
                    writer.writerow({"feat_bin": feature_name, "label_id": label, "frames": frames})
            write_manifest()
            digest = hashlib.sha256(feature.read_bytes()).hexdigest()
            result = evaluate_sessions(session, session, base, max_samples=2)
            self.assertEqual(result["samples"], 2)
            self.assertEqual(result["max_abs_error"], 0)
            self.assertIsNotNone(result["reference_accuracy"])
            self.assertEqual(hashlib.sha256(feature.read_bytes()).hexdigest(), digest)
            for kwargs in ({"label": 3}, {"frames": 2}, {"feature_name": "../escape.bin"}, {"feature_name": ""}):
                write_manifest(**kwargs)
                with self.subTest(kwargs=kwargs), self.assertRaises(ConversionError):
                    evaluate_sessions(session, session, manifest)
            write_manifest()
            feature.write_bytes(b"\x00" * 16)
            with self.assertRaises(ConversionError):
                evaluate_sessions(session, session, manifest)

    def test_bad_datasets_and_options_have_diagnostics(self):
        session, _ = make_session("lstm")
        with tempfile.TemporaryDirectory(prefix=".studio-test-", dir=LIBRARY) as directory:
            dataset = Path(directory) / "data.npz"
            good = np.zeros((2, 15), np.float32)
            cases = [dict(inputs=np.zeros((0, 15), np.float32)),
                     dict(inputs=np.zeros((2, 14), np.float32)),
                     dict(inputs=np.zeros((2, 15), np.int16)),
                     dict(inputs=np.full((2, 15), np.nan, np.float32)),
                     dict(inputs=good, labels=np.array([0, 3])),
                     dict(inputs=good, labels=np.array([0.0, 1.0])),
                     dict(inputs=good, labels=np.array([1])),
                     dict(inputs=np.array([[object()]], dtype=object)),
                     dict(other=good)]
            for arrays in cases:
                np.savez(dataset, **arrays)
                with self.subTest(keys=list(arrays)), self.assertRaises(ConversionError):
                    evaluate_sessions(session, session, dataset)
            np.savez(dataset, inputs=good)
            for kwargs in ({"input_scale": float("nan")}, {"input_scale": 0},
                           {"input_scale": float("inf")}, {"input_scale": 1e-300},
                           {"input_zero_point": -1}, {"input_zero_point": True},
                           {"max_samples": 0}, {"max_samples": 100001}, {"max_samples": 1.5}):
                with self.subTest(kwargs=kwargs), self.assertRaises(ConversionError):
                    evaluate_sessions(session, session, dataset, **kwargs)
            with patch("model_studio.validation.MAX_INPUT_ELEMENTS", 10), self.assertRaises(ConversionError):
                evaluate_sessions(session, session, dataset)

    def test_huge_numpy_headers_rejected_before_loading(self):
        session, _ = make_session("rnn")
        header = io.BytesIO()
        np.lib.format.write_array_header_1_0(header,
            {"descr": "<f4", "fortran_order": False, "shape": (1000000000, 15)})
        with tempfile.TemporaryDirectory(prefix=".studio-test-", dir=LIBRARY) as directory:
            for suffix in (".npy", ".npz"):
                path = Path(directory) / ("huge" + suffix)
                if suffix == ".npy":
                    path.write_bytes(header.getvalue())
                else:
                    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                        archive.writestr("inputs.npy", header.getvalue())
                with self.subTest(suffix=suffix), patch("model_studio.validation.np.load") as loader:
                    with self.assertRaisesRegex(ConversionError, "capacity"):
                        evaluate_sessions(session, session, path)
                    loader.assert_not_called()

    def test_npz_decompressed_capacity_checked_before_loading(self):
        session, _ = make_session("rnn")
        with tempfile.TemporaryDirectory(prefix=".studio-test-", dir=LIBRARY) as directory:
            dataset = Path(directory) / "compressed.npz"
            np.savez_compressed(dataset, inputs=np.zeros((50, 15), np.float32))
            self.assertLess(dataset.stat().st_size, 1000)
            with patch("model_studio.validation.MAX_DATA_BYTES", 1000), patch("model_studio.validation.np.load") as loader:
                with self.assertRaisesRegex(ConversionError, "Decompressed"):
                    evaluate_sessions(session, session, dataset)
                loader.assert_not_called()

    @staticmethod
    def _npy_bytes(array, version):
        stream = io.BytesIO()
        np.lib.format.write_array(stream, array, version=version, allow_pickle=False)
        return stream.getvalue()

    @staticmethod
    def _raw_npy(header, version=(3, 0), payload=b""):
        if isinstance(header, str):
            header = header.encode("utf-8" if version == (3, 0) else "latin1")
        length_size = 2 if version == (1, 0) else 4
        return b"\x93NUMPY" + bytes(version) + len(header).to_bytes(length_size, "little") + header + payload

    def test_npy_versions_and_npz_members_preserve_numeric_data(self):
        session, _ = make_session("rnn")
        arrays = [np.arange(30, dtype=np.uint8).reshape(2, 15),
                  np.asfortranarray(np.arange(30, dtype=np.float32).reshape(2, 5, 3) / 32),
                  (np.arange(30, dtype=np.float64).reshape(2, 15) / 32).astype(">f8")]
        labels = np.array([0, 2], dtype=">i8")
        with tempfile.TemporaryDirectory(prefix=".studio-test-", dir=LIBRARY) as directory:
            for version in ((1, 0), (2, 0), (3, 0)):
                for array in arrays:
                    for suffix in (".npy", ".npz"):
                        with self.subTest(version=version, dtype=str(array.dtype), suffix=suffix):
                            path = Path(directory) / ("data" + suffix)
                            if suffix == ".npy":
                                path.write_bytes(self._npy_bytes(array, version))
                            else:
                                with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                                    archive.writestr("inputs.npy", self._npy_bytes(array, version))
                                    archive.writestr("labels.npy", self._npy_bytes(labels, version))
                            actual, actual_labels = _numpy_data(path, session.spec, 2)
                            np.testing.assert_array_equal(actual, array.reshape(2, 5, 3))
                            self.assertTrue(actual.flags.c_contiguous)
                            self.assertEqual(actual.dtype, np.uint8 if array.dtype == np.uint8 else np.float32)
                            if suffix == ".npz":
                                np.testing.assert_array_equal(actual_labels, labels)
                            else:
                                self.assertIsNone(actual_labels)

    def test_numpy_header_inspection_requires_only_public_numpy_apis(self):
        session, _ = make_session("rnn")
        array = np.zeros((2, 15), np.float32)
        fixtures = [(version, self._npy_bytes(array, version)) for version in ((1, 0), (2, 0), (3, 0))]
        # Old NumPy may still export the private name removed in NumPy 2.5.
        # Supply only documented functions; the readers' own implementation is
        # NumPy's responsibility and may internally use private module globals.
        public_format = SimpleNamespace(**{name: getattr(np.lib.format, name) for name in
            ("read_magic", "read_array_header_1_0", "read_array_header_2_0", "descr_to_dtype")})
        with patch("model_studio.validation.np.lib.format", public_format):
            for version, content in fixtures:
                with self.subTest(version=version):
                    stream = io.BytesIO(content)
                    self.assertEqual(_inspect_numpy_header(stream, len(content), session.spec), (2, 15))
                    self.assertEqual(len(content) - stream.tell(), array.nbytes)

    def test_malformed_numpy_headers_rejected_before_payload_loading(self):
        session, _ = make_session("rnn")
        good = {"descr": "|u1", "fortran_order": False, "shape": (2, 15)}
        for version in ((1, 0), (2, 0), (3, 0)):
            cases = [b"not an NPY file", b"\x93NUMPY\x04\x00",
                     b"\x93NUMPY" + bytes(version) + b"\x01",
                     self._raw_npy("{", version),
                     self._raw_npy("__import__('os').getcwd()", version),
                     self._raw_npy(repr({"descr": "|u1", "shape": (2, 15)}), version),
                     self._raw_npy(repr(dict(good, unexpected=1)), version),
                     self._raw_npy(repr(dict(good, fortran_order=0)), version),
                     self._raw_npy(repr(dict(good, shape=[2, 15])), version),
                     self._raw_npy(repr(dict(good, shape=(2, -15))), version),
                     self._raw_npy(repr(dict(good, shape=(True, 15))), version),
                     self._raw_npy(repr(dict(good, descr="nonsense")), version),
                     self._raw_npy(repr(good), version, b"\x00" * 29),
                     self._raw_npy(repr(good), version, b"\x00" * 31)]
            if version == (3, 0):
                cases += [self._raw_npy(b"\xff", version),
                          self._raw_npy("[" * 300 + "]" * 300, version)]
            with tempfile.TemporaryDirectory(prefix=".studio-test-", dir=LIBRARY) as directory:
                for index, content in enumerate(cases):
                    for suffix in (".npy", ".npz"):
                        with self.subTest(version=version, case=index, suffix=suffix):
                            path = Path(directory) / ("bad" + suffix)
                            if suffix == ".npy":
                                path.write_bytes(content)
                            else:
                                with zipfile.ZipFile(path, "w") as archive:
                                    archive.writestr("inputs.npy", content)
                            with patch("model_studio.validation.np.load") as loader:
                                with self.assertRaises(ConversionError):
                                    evaluate_sessions(session, session, path)
                                loader.assert_not_called()

    def test_numpy_header_size_is_bounded_before_read_or_parse(self):
        session, _ = make_session("rnn")
        header = repr({"descr": "|u1", "fortran_order": False, "shape": (2, 15)}).encode()
        for version in ((1, 0), (2, 0), (3, 0)):
            content = self._raw_npy(header.ljust(10000, b" "), version, b"\x00" * 30)
            self.assertEqual(_inspect_numpy_header(io.BytesIO(content), len(content), session.spec), (2, 15))
            for advertised in (10001, 65535 if version == (1, 0) else 2 ** 32 - 1):
                prefix = b"\x93NUMPY" + bytes(version) + advertised.to_bytes(2 if version == (1, 0) else 4, "little")
                stream = io.BytesIO(prefix)
                with self.subTest(version=version, advertised=advertised):
                    with patch.object(stream, "read", wraps=stream.read) as reader:
                        with self.assertRaisesRegex(ConversionError, "capacity"):
                            _inspect_numpy_header(stream, advertised + len(prefix), session.spec)
                        self.assertLessEqual(max(call.args[0] for call in reader.call_args_list), 8)
            truncated = self._raw_npy(header, version)[:-1]
            with self.subTest(version=version), self.assertRaisesRegex(ConversionError, "Truncated"):
                _inspect_numpy_header(io.BytesIO(truncated), len(truncated), session.spec)

    def test_numpy_header_object_dtype_unicode_fields_and_byte_capacity(self):
        session, _ = make_session("rnn")
        for version in ((1, 0), (2, 0), (3, 0)):
            for dtype in ("|O", "|b1", "<i2", [("field", "<f4")]):
                header = {"descr": dtype, "fortran_order": False, "shape": (2, 15)}
                content = self._raw_npy(repr(header), version)
                with self.subTest(version=version, dtype=dtype), self.assertRaises(ConversionError):
                    _inspect_numpy_header(io.BytesIO(content), len(content), session.spec)
            content = self._npy_bytes(np.zeros((2, 15), np.float64), version)
            with patch("model_studio.validation.MAX_DATA_BYTES", 200):
                with self.subTest(version=version), self.assertRaisesRegex(ConversionError, "byte capacity"):
                    _inspect_numpy_header(io.BytesIO(content), len(content), session.spec)
        # A real UTF-8 descriptor is decoded correctly as 3.0, then rejected by
        # the application's numeric-input contract rather than treated as latin1.
        content = self._npy_bytes(np.zeros((2, 15), dtype=[("频域", "<f4")]), (3, 0))
        with self.assertRaisesRegex(ConversionError, "inputs must be uint8"):
            _inspect_numpy_header(io.BytesIO(content), len(content), session.spec)

    def test_npz_v3_label_header_checks_precede_payload_loading(self):
        session, _ = make_session("rnn")
        inputs = self._npy_bytes(np.zeros((2, 15), np.uint8), (3, 0))
        with tempfile.TemporaryDirectory(prefix=".studio-test-", dir=LIBRARY) as directory:
            path = Path(directory) / "labels.npz"
            for labels in (np.zeros(1, np.int64), np.zeros(2, np.bool_), np.zeros(2, np.float32)):
                with self.subTest(dtype=str(labels.dtype), shape=labels.shape):
                    with zipfile.ZipFile(path, "w") as archive:
                        archive.writestr("inputs.npy", inputs)
                        archive.writestr("labels.npy", self._npy_bytes(labels, (3, 0)))
                    with patch("model_studio.validation.np.load") as loader:
                        with self.assertRaisesRegex(ConversionError, "labels"):
                            evaluate_sessions(session, session, path)
                        loader.assert_not_called()

    def test_cpu_large_graph_budgets_refuse_before_data_or_model_allocation(self):
        session, _ = make_session("rnn")
        def recurrent_spec(hidden, sequence):
            dimensions = dict(session.spec.dimensions, input_size=1, input_elements=sequence,
                              hidden_size=hidden, seq_length=sequence)
            return replace(session.spec, dimensions=dimensions)
        # Synthetic shape-only specifications exercise the budget arithmetic;
        # no large tensors, sequence arrays or module weights are allocated.
        cases = [
            ("sequence length", recurrent_spec(512, 500000), {}),
            ("activation/workspace", recurrent_spec(512, 500000),
             {"MAX_CPU_SEQUENCE_LENGTH": 500000, "MAX_CPU_SAMPLE_MACS": 10 ** 15}),
            ("MACs per sample", recurrent_spec(512, 1000), {}),
            ("model reconstruction", recurrent_spec(4096, 1), {}),
        ]
        cnn, _ = make_session("cnn")
        cnn_dimensions = dict(cnn.spec.dimensions, input_length=200000, input_elements=200000,
                              hidden_size=1, flatten_size=24996,
                              conv_shapes=[[4096, 1, 5, 199996, 99998],
                                           [1, 4096, 5, 99994, 49997],
                                           [1, 1, 5, 49993, 24996]])
        cases.append(("activation/workspace", replace(cnn.spec, dimensions=cnn_dimensions),
                      {"MAX_CPU_SAMPLE_MACS": 10 ** 15}))
        for diagnostic, spec, overrides in cases:
            with self.subTest(diagnostic=diagnostic, architecture=spec.architecture), ExitStack() as stack:
                stack.enter_context(patch("model_studio.validation._validated_spec", return_value=spec))
                for name, value in overrides.items():
                    stack.enter_context(patch("model_studio.validation." + name, value))
                builder = stack.enter_context(patch("model_studio.validation.build_model"))
                numpy_reader = stack.enter_context(patch("model_studio.validation._numpy_data"))
                manifest_reader = stack.enter_context(patch("model_studio.validation._manifest_data"))
                with self.assertRaisesRegex(ConversionError, diagnostic):
                    evaluate_sessions(session, session, "dataset_does_not_need_to_exist.npz")
                builder.assert_not_called()
                numpy_reader.assert_not_called()
                manifest_reader.assert_not_called()
        with patch("model_studio.validation._validated_spec", return_value=recurrent_spec(512, 500000)):
            with patch("model_studio.validation.nn.RNN") as factory:
                with self.assertRaisesRegex(ConversionError, "sequence length"):
                    build_model(session)
                factory.assert_not_called()

    def test_cpu_dynamic_batch_limit_is_reported_and_used(self):
        session, _ = make_session("cnn")
        per_sample = _evaluation_plan(session.spec, session.spec)["estimated_working_bytes_per_sample"]
        batches = []
        original_builder = build_model
        def recording_builder(model_session):
            model = original_builder(model_session)
            model.register_forward_pre_hook(lambda _, arguments: batches.append(len(arguments[0])))
            return model
        with tempfile.TemporaryDirectory(prefix=".studio-test-", dir=LIBRARY) as directory:
            path = Path(directory) / "small.npy"
            np.save(path, np.zeros((7, 64), np.float32))
            with patch("model_studio.validation.MAX_CPU_WORKING_BYTES", 3 * per_sample):
                with patch("model_studio.validation.build_model", side_effect=recording_builder):
                    result = evaluate_sessions(session, session, path)
        self.assertEqual(result["batch_size"], 3)
        self.assertEqual(batches, [3, 3, 3, 3, 1, 1])
        budget = result["cpu_evaluation_budget"]
        self.assertEqual(budget["estimated_batch_working_bytes"], 3 * per_sample)
        self.assertLessEqual(budget["estimated_batch_working_bytes"], budget["working_bytes_limit"])
        self.assertIn("not measured", budget["accounting"])
        self.assertEqual(budget["workspace_factor"], 4)
        json.dumps(result, allow_nan=False)

    def test_cpu_total_macs_checked_from_header_before_loading_or_building(self):
        session, _ = make_session("rnn")
        per_sample = _evaluation_plan(session.spec, session.spec)["estimated_macs_per_sample"]
        with tempfile.TemporaryDirectory(prefix=".studio-test-", dir=LIBRARY) as directory:
            base = Path(directory)
            data = np.zeros((2, 15), np.uint8)
            np.save(base / "input.npy", data)
            np.savez(base / "input.npz", inputs=data)
            data.tofile(base / "input.bin")
            (base / "manifest.csv").write_text("feat_bin,label_id,frames\ninput.bin,0,2\n", encoding="utf-8")
            for name in ("input.npy", "input.npz", "manifest.csv"):
                with self.subTest(name=name), patch("model_studio.validation.MAX_CPU_TOTAL_MACS", per_sample):
                    with patch("model_studio.validation.np.load") as loader, patch("model_studio.validation.np.fromfile") as reader:
                        with patch("model_studio.validation.build_model") as builder:
                            with self.assertRaisesRegex(ConversionError, "total budget"):
                                evaluate_sessions(session, session, base / name, max_samples=2)
                            loader.assert_not_called()
                            reader.assert_not_called()
                            builder.assert_not_called()
            # A smaller selected prefix stays within the same total-MAC budget.
            with patch("model_studio.validation.MAX_CPU_TOTAL_MACS", per_sample):
                report = evaluate_sessions(session, session, base / "input.npz", max_samples=1)
            self.assertEqual(report["samples"], 1)
            self.assertEqual(report["cpu_evaluation_budget"]["estimated_total_macs"], per_sample)


if __name__ == "__main__":
    unittest.main()
