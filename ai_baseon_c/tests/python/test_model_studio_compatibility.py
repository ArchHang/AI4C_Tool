"""Safe automatic legacy import, wrappers, and failure-isolation contracts."""

import hashlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch
from torch import nn

LIBRARY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LIBRARY / "tools"))

from model_converter.loaders import CNNVoiceClassifier
from model_converter.schema import ConversionError
from model_studio.analysis import load_session


def legacy_cnn():
    model = CNNVoiceClassifier()
    stages, inputs, length = [], 1, 64
    for outputs in (2, 3, 4):
        stages.extend((nn.Conv1d(inputs, outputs, 5), nn.BatchNorm1d(outputs),
                       nn.ReLU(), nn.MaxPool1d(2)))
        inputs, length = outputs, (length - 4) // 2
    stages.append(nn.Flatten())
    model.model = nn.Sequential(*stages)
    model.fc1 = nn.Linear(inputs * length, 5)
    model.dropout = nn.Dropout()
    model.fc2 = nn.Linear(5, 3)
    return model


class DangerousCheckpoint:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return os.system, (f'echo unsafe > "{self.marker}"',)


class CompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="studio_compatibility_",
                                                       dir=LIBRARY / "tests" / "python")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def save_module(self, model, name="legacy.pth"):
        path = self.root / name
        # Historical files identify the class by its script global. Saving a
        # fixture temporarily publishes only the local compatibility identity.
        with mock.patch.object(sys.modules["__main__"], "CNNVoiceClassifier",
                               CNNVoiceClassifier, create=True):
            torch.save(model, path)
        return path

    def test_legacy_cnn_imports_automatically_without_source_mutation(self):
        model = legacy_cnn()
        path = self.save_module(model)
        original = hashlib.sha256(path.read_bytes()).hexdigest()
        with mock.patch("torch.load", wraps=torch.load) as loader:
            candidate = load_session(path, input_length=64)
        self.assertEqual(candidate.spec.schema, "cnn_voice_v1")
        self.assertEqual(candidate.spec.dimensions["flatten_size"], 16)
        self.assertTrue(any("自动兼容" in notice for notice in candidate.warnings))
        self.assertEqual(candidate.changes, [])
        self.assertEqual(candidate.sha256, original)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), original)
        self.assertEqual(loader.call_count, 2)
        self.assertTrue(all(call.kwargs.get("weights_only") is True for call in loader.call_args_list))
        for key, expected in model.state_dict().items():
            self.assertTrue(torch.equal(candidate.state[key], expected), key)

    def test_strict_opt_out_and_existing_explicit_opt_in(self):
        path = self.save_module(legacy_cnn())
        with self.assertRaisesRegex(ConversionError, "Safe checkpoint loading failed"):
            load_session(path, input_length=64, auto_legacy=False)
        self.assertEqual(load_session(path, input_length=64, trusted_module=True,
                                      auto_legacy=False).spec.architecture, "cnn")

    def test_bad_legacy_graph_rejected_even_when_inventory_is_allowed(self):
        model = legacy_cnn()
        model.model[3] = nn.MaxPool1d(3)
        path = self.save_module(model)
        with self.assertRaisesRegex(ConversionError, "Only floor maxpool2"):
            load_session(path, input_length=64)

    def test_known_identity_with_unknown_global_cannot_trigger_automatic_retry(self):
        model = legacy_cnn()
        marker = self.root / "must_not_exist.txt"
        model.extra_payload = DangerousCheckpoint(marker)
        path = self.save_module(model)
        with mock.patch("torch.load", wraps=torch.load) as loader:
            with self.assertRaisesRegex(ConversionError, "Safe checkpoint loading failed"):
                load_session(path)
        self.assertEqual(loader.call_count, 1)
        self.assertFalse(marker.exists())

    def test_arbitrary_module_and_pickle_payload_stay_rejected(self):
        for name, payload in (("linear.pth", nn.Linear(2, 2)),
                              ("dangerous.pth", DangerousCheckpoint(self.root / "never.txt"))):
            with self.subTest(name=name):
                path = self.root / name
                torch.save(payload, path)
                with self.assertRaisesRegex(ConversionError, "Safe checkpoint loading failed"):
                    load_session(path)
        self.assertFalse((self.root / "never.txt").exists())

    def test_common_wrappers_and_dataparallel_prefix_need_no_legacy_retry(self):
        state = {"module." + key: value for key, value in legacy_cnn().state_dict().items()}
        for wrapper in ("state_dict", "model_state_dict"):
            with self.subTest(wrapper=wrapper):
                path = self.root / (wrapper + ".pth")
                torch.save({wrapper: state, "model_config": {"schema": "cnn_voice_v1",
                                                             "input_length": 64}}, path)
                with mock.patch("torch.load", wraps=torch.load) as loader:
                    candidate = load_session(path)
                self.assertEqual(candidate.spec.architecture, "cnn")
                self.assertEqual(candidate.warnings, [])
                self.assertEqual(loader.call_count, 1)

    def test_corrupt_and_nonexistent_files_keep_conversion_errors(self):
        corrupt = self.root / "broken.pth"
        corrupt.write_bytes(b"this is not a checkpoint")
        with self.assertRaisesRegex(ConversionError, "Safe checkpoint loading failed"):
            load_session(corrupt)
        with self.assertRaisesRegex(ConversionError, "Checkpoint not found"):
            load_session(self.root / "missing.pth")

    def test_unknown_or_invalid_tensor_graph_is_explicit_inventory(self):
        path = self.root / "inventory.pth"
        torch.save({"unrecognized.weight": torch.ones(2, 3)}, path)
        candidate = load_session(path)
        self.assertIsNone(candidate.spec)
        self.assertTrue(candidate.warnings)
        with self.assertRaises(ConversionError):
            load_session(path, allow_inventory=False)
        invalid = legacy_cnn().state_dict()
        invalid["fc1.weight"] = torch.ones(5, 17)
        torch.save(invalid, path)
        candidate = load_session(path, input_length=64)
        self.assertIsNone(candidate.spec)
        self.assertTrue(any("fc1 input" in warning for warning in candidate.warnings))


if __name__ == "__main__":
    unittest.main()
