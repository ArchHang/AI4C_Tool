"""Export boundaries and provenance, independent of a compiler or desktop."""

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))
from model_converter.schema import ConversionError
from model_studio.analysis import analyze_session, load_session
from model_studio.compression import prune_session
from model_studio.services import export_session, state_digest, write_analysis_report


def small_cnn_state():
    torch.manual_seed(541)
    stages = []
    inputs, width = 1, 64
    for outputs in (3, 4, 5):
        stages.extend((nn.Conv1d(inputs, outputs, 5), nn.BatchNorm1d(outputs), nn.ReLU(), nn.MaxPool1d(2)))
        inputs, width = outputs, (width - 4) // 2
    state = {"model." + name: value for name, value in nn.Sequential(*stages, nn.Flatten()).state_dict().items()}
    state.update({"fc1." + name: value for name, value in nn.Linear(width * inputs, 6).state_dict().items()})
    state.update({"fc2." + name: value for name, value in nn.Linear(6, 3).state_dict().items()})
    return state


class ServicesTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix=".studio-services-", dir=Path(__file__).parent)
        self.directory = Path(self.temporary.name)
        self.source = self.directory / "source.pth"
        torch.save({"state_dict": small_cnn_state(), "metadata": {"input_length": 64}}, self.source)
        self.original_bytes = self.source.read_bytes()
        self.session = load_session(self.source)

    def tearDown(self):
        self.temporary.cleanup()

    def test_exports_candidate_not_original_and_provenance(self):
        candidate, _ = prune_session(self.session, 0.5)
        destination = self.directory / "export"
        result = export_session(candidate, destination, precision="both")
        manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["dimensions"], candidate.spec.dimensions)
        self.assertNotEqual(manifest["dimensions"], self.session.spec.dimensions)
        self.assertIsNone(manifest["checkpoint_sha256"])
        provenance = manifest["studio_provenance"]
        self.assertEqual(provenance["source_checkpoint_sha256"], self.session.sha256)
        self.assertEqual(provenance["candidate_state_sha256"], state_digest(candidate))
        self.assertNotEqual(state_digest(candidate), state_digest(self.session))
        self.assertTrue(provenance["changes"])
        self.assertEqual(manifest["quality_status"], "not_evaluated")
        self.assertEqual(manifest["target_performance"], "not_measured")
        self.assertEqual(result["analysis"]["summary"], manifest["studio_analysis"])
        self.assertIn('"runtime/cnn.h"', (destination / "model.c").read_text(encoding="utf-8"))
        for name in manifest["bundled_runtime"]["sources"]:
            self.assertTrue((destination / "runtime" / name).is_file())
        self.assertEqual(self.source.read_bytes(), self.original_bytes)
        self.assertFalse(list(self.directory.glob(".studio-export-*")))

    def test_existing_directory_and_invalid_options_preserve_files(self):
        destination = self.directory / "keep"
        destination.mkdir()
        marker = destination / "model.c"
        marker.write_bytes(b"user file")
        with self.assertRaises(ConversionError):
            export_session(self.session, destination)
        self.assertEqual(marker.read_bytes(), b"user file")
        with self.assertRaises(ConversionError):
            export_session(self.session, self.directory / "invalid", prefix="Invalid!")
        self.assertFalse((self.directory / "invalid").exists())
        self.assertFalse(list(self.directory.glob(".studio-export-*")))

    def test_partial_export_failure_leaves_no_final_or_temporary_directory(self):
        def broken(spec, directory, **options):
            (Path(directory) / "partial.c").write_text("partial", encoding="utf-8")
            raise RuntimeError("interrupted")
        with patch("model_studio.services.emit_model", side_effect=broken):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                export_session(self.session, self.directory / "incomplete")
        self.assertFalse((self.directory / "incomplete").exists())
        self.assertFalse(list(self.directory.glob(".studio-export-*")))
        self.assertEqual(self.source.read_bytes(), self.original_bytes)

    def test_inventory_cannot_export_and_report_formats(self):
        inventory_path = self.directory / "unknown.pth"
        torch.save({"weight": torch.ones(2, 3)}, inventory_path)
        inventory = load_session(inventory_path)
        with self.assertRaises(ConversionError):
            export_session(inventory, self.directory / "unknown-export")
        analysis = analyze_session(self.session)
        json_path, csv_path = self.directory / "analysis.json", self.directory / "analysis.csv"
        write_analysis_report(analysis, json_path)
        self.assertEqual(json.loads(json_path.read_text(encoding="utf-8")), analysis)
        write_analysis_report(analysis, csv_path)
        self.assertIn("macs", csv_path.read_text(encoding="utf-8-sig").splitlines()[0])
        with self.assertRaises(FileExistsError):
            write_analysis_report(analysis, json_path)
        with self.assertRaises(ConversionError):
            write_analysis_report(analysis, self.directory / "bad.pth")

    def test_unmodified_source_digest_and_sample_comparison_scope(self):
        comparison = {"samples": 2, "prediction_agreement": 1.0}
        result = export_session(self.session, self.directory / "fp32", precision="fp32",
                                bundle_runtime=False, validation=comparison)
        self.assertEqual(result["manifest"]["checkpoint_sha256"], hashlib.sha256(self.original_bytes).hexdigest())
        self.assertEqual(result["manifest"]["candidate_fp32_comparison"], comparison)
        self.assertEqual(result["manifest"]["quality_status"], "sample_comparison_only")
        self.assertNotIn("bundled_runtime", result["manifest"])

    def test_report_never_overwrites_source_via_hardlink(self):
        alias = self.directory / "alias.json"
        try:
            os.link(self.source, alias)
        except OSError as error:
            self.skipTest(f"Hardlinks unavailable: {error}")
        with self.assertRaises(ConversionError):
            write_analysis_report(analyze_session(self.session), alias, overwrite=True)
        self.assertEqual(self.source.read_bytes(), self.original_bytes)


if __name__ == "__main__":
    unittest.main()
