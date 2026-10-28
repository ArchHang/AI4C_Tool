"""Portable release-check and untrained-example contracts; no corpus or hardware."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))
from check_repository import CORE_TEST_MODULES, EXTERNAL_TESTS, REQUIRED_FIXTURES, core_suite, _flatten
from generate_example_models import generate_examples


class RepositoryToolsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix=".repository-tools-test-", dir=Path(__file__).parent)
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def test_generated_cpu_checkpoints_are_all_supported_untrained_schemas(self):
        import torch
        from model_studio.analysis import load_session
        report = generate_examples(self.directory / "examples")
        self.assertFalse(report["trained"])
        self.assertFalse(report["contains_voice_data"])
        self.assertEqual(report["device"], "cpu")
        self.assertEqual([item["architecture"] for item in report["models"]], ["cnn", "rnn", "lstm"])
        for item in report["models"]:
            path = self.directory / "examples" / item["checkpoint"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), item["sha256"])
            state = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(state["metadata"]["training_status"], "untrained_synthetic")
            self.assertTrue(state["metadata"]["example_only"])
            self.assertTrue(all(value.device.type == "cpu" for value in state["state_dict"].values()))
            session = load_session(path, allow_inventory=False)
            self.assertEqual(session.spec.architecture, item["architecture"])
            self.assertEqual(session.spec.dimensions, item["dimensions"])

    def test_equal_seed_preserves_tensor_values_and_existing_output(self):
        import torch
        first, second = self.directory / "a", self.directory / "b"
        generate_examples(first, seed=71);generate_examples(second, seed=71)
        for kind in ("cnn", "rnn", "lstm"):
            left = torch.load(first / (kind + ".pth"), weights_only=True)["state_dict"]
            right = torch.load(second / (kind + ".pth"), weights_only=True)["state_dict"]
            self.assertEqual(set(left), set(right))
            for key in left:
                self.assertTrue(torch.equal(left[key], right[key]))
        before = (first / "examples.json").read_bytes()
        with self.assertRaises(FileExistsError):
            generate_examples(first)
        self.assertEqual((first / "examples.json").read_bytes(), before)
        for seed in (-1, 2 ** 63, True):
            with self.assertRaises(ValueError):
                generate_examples(self.directory / "invalid", seed=seed)
        self.assertFalse((self.directory / "invalid").exists())

    def test_core_suite_has_no_private_experiment_or_missing_data_skip(self):
        root = TOOLS.parents[1]
        selected = list(_flatten(core_suite(root)))
        identities = {test.id() for test in selected}
        self.assertGreater(len(identities), 100)
        self.assertFalse(identities & EXTERNAL_TESTS)
        self.assertFalse(any("speaker_" in name or "f446_" in name or "board_matrix" in name for name in identities))
        self.assertIn("test_model_studio_system", CORE_TEST_MODULES)
        self.assertNotIn("test_model_studio_gui", CORE_TEST_MODULES)
        self.assertEqual(set(REQUIRED_FIXTURES), {"examples/models/rnn_model.h", "examples/models/lstm_model.h",
                                                 "model/rnn_params.npz", "model/lstm_params.npz"})

    def test_windows_launcher_has_no_personal_environment_default(self):
        text = (TOOLS.parent / "start_model_studio.cmd").read_text(encoding="utf-8")
        self.assertNotIn("23350", text)
        self.assertNotIn("pytorch212_p310", text)
        self.assertLess(text.index("MODEL_STUDIO_PYTHON"), text.index(".venv"))
        self.assertLess(text.index(".venv"), text.index('STUDIO_PYTHON=python'))


if __name__ == "__main__":
    unittest.main()
