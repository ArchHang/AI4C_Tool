"""Static RAM accounting and end-to-end option plumbing for optional DSP caches."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from model_converter.cli import convert_checkpoint
from model_converter.schema import ConversionError
from model_studio.analysis import analyze_session, load_session
from model_studio.services import export_session
from test_model_studio_services import small_cnn_state


class CacheIntegrationTests(unittest.TestCase):
    def test_ram_budget_and_manifest_match_exported_members(self):
        with tempfile.TemporaryDirectory(prefix=".cache-plumbing-", dir=Path(__file__).parent) as name:
            root = Path(name)
            checkpoint = root / "source.pth"
            torch.save({"state_dict": small_cnn_state(), "metadata": {"input_length": 64}}, checkpoint)
            session = load_session(checkpoint)
            plain = analyze_session(session)
            cached = analyze_session(session, cache_policy="packed")
            cache = cached["deployment"]["weight_cache"]
            self.assertEqual(cache["bytes"], (8 * 3 + 16 * 4 + 20 * 5) * 2)
            self.assertEqual(cache["state_bytes"], 4)
            self.assertEqual(cached["summary"]["workspace_bytes_lower_bound"] -
                             plain["summary"]["workspace_bytes_lower_bound"], cache["bytes"] + 4)
            self.assertEqual(cached["summary"]["deployed_parameter_bytes"],
                             plain["summary"]["deployed_parameter_bytes"])
            exported = export_session(session, root / "studio", cache_policy="packed")
            manifest = exported["manifest"]
            self.assertEqual(manifest["weight_cache"], cache)
            self.assertEqual(exported["analysis"], cached)
            cli = convert_checkpoint(checkpoint, root / "cli", cache_policy="packed")
            self.assertEqual(cli["weight_cache"], cache)
            self.assertEqual((root / "cli/model.h").read_text(encoding="utf-8"),
                             (root / "studio/model.h").read_text(encoding="utf-8"))
            self.assertEqual(json.loads((root / "studio/analysis.json").read_text(encoding="utf-8")), cached)

    def test_fp32_cache_is_explicit_noop_and_invalid_policy_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix=".cache-plumbing-", dir=Path(__file__).parent) as name:
            root = Path(name)
            checkpoint = root / "source.pth"
            torch.save({"state_dict": small_cnn_state(), "metadata": {"input_length": 64}}, checkpoint)
            session = load_session(checkpoint)
            plain = analyze_session(session, precision="fp32")
            cached = analyze_session(session, precision="fp32", cache_policy="packed")
            self.assertEqual(cached["deployment"]["weight_cache"]["bytes"], 0)
            self.assertEqual(cached["summary"], plain["summary"])
            with self.assertRaises(ConversionError):
                analyze_session(session, cache_policy="invalid")
            with self.assertRaises(ConversionError):
                export_session(session, root / "bad", cache_policy="invalid")
            self.assertFalse((root / "bad").exists())


if __name__ == "__main__":
    unittest.main()
