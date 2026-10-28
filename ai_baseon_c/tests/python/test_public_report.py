"""Public aggregate parser, tampering rejection and real PNG metadata tests."""

from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

LIBRARY = Path(__file__).resolve().parents[2]
ROOT = LIBRARY.parent
sys.path.insert(0, str(LIBRARY / "tools"))
import plot_benchmark_report as plot


class PublicReportTests(unittest.TestCase):
    def setUp(self):
        self.payload = (ROOT / "docs/assets/benchmark-summary.json").read_bytes()
        self.summary = json.loads(self.payload)

    def test_real_public_counts_macro_and_weight_compression(self):
        plot.validate_summary(self.summary)
        case = self.summary["models"]["cnn"]["precisions"]["int8"]
        self.assertEqual(case["metrics"]["correct"], 1599)
        self.assertAlmostEqual(case["metrics"]["macro_recall"], (708/788 + 748/800 + 119/139 + 24/42)/4)
        for model in self.summary["models"].values():
            self.assertEqual(model["precisions"]["fp32"]["flash"]["weight_bytes"],
                             4 * model["precisions"]["int8"]["flash"]["weight_bytes"])

    def test_quality_tampering_is_rejected_from_actual_matrix_counts(self):
        for name, value in (("accuracy", .99), ("macro_recall", .99), ("correct", True),
                            ("class_support", [788, 800, 139, 0]), ("per_class_recall", [1., 1., 1., 1.])):
            changed = deepcopy(self.summary)
            changed["models"]["cnn"]["precisions"]["int8"]["metrics"][name] = value
            with self.subTest(field=name), self.assertRaises(ValueError):plot.validate_summary(changed)

    def test_flash_accounting_rejects_double_counts_and_false_compression(self):
        for name, value in (("weight_bytes", 1), ("model_constants_bytes", 1), ("other_firmware_bytes", 1),
                            ("alias_or_overlap_count", True), ("elf_nm_c_sets_equal", 1)):
            changed = deepcopy(self.summary)
            changed["models"]["cnn"]["precisions"]["int8"]["flash"][name] = value
            with self.subTest(field=name), self.assertRaises(ValueError):plot.validate_summary(changed)
        changed = deepcopy(self.summary)
        changed["models"]["cnn"]["precisions"]["int8"]["flash"]["category_bytes"]["fp32_weight"] = 4
        with self.assertRaises(ValueError):plot.validate_summary(changed)

    def test_latency_units_private_paths_and_nonfinite_are_rejected(self):
        for name, value in (("median_ms", .001), ("p95_ms", .001), ("median_cycles", False), ("host_latency_measured", True)):
            changed = deepcopy(self.summary)
            changed["models"]["rnn"]["precisions"]["fp32"]["metrics"][name] = value
            with self.subTest(field=name), self.assertRaises(ValueError):plot.validate_summary(changed)
        changed = deepcopy(self.summary);changed["private_file"] = "C:/Users/example/voice.wav"
        with self.assertRaises(ValueError):plot.validate_summary(changed)
        changed = deepcopy(self.summary);changed["conditions"]["timing"] = "interrupts masked"
        with self.assertRaises(ValueError):plot.validate_summary(changed)
        with tempfile.TemporaryDirectory(prefix=".public-report-test-", dir=Path(__file__).parent) as temporary:
            path = Path(temporary) / "bad.json";path.write_bytes(b'{"value":NaN}')
            with self.assertRaises(ValueError):plot.load_summary(path)

    def test_overwrite_refuses_hard_link_to_another_file(self):
        with tempfile.TemporaryDirectory(prefix=".public-report-test-", dir=Path(__file__).parent) as temporary:
            directory = Path(temporary);source = directory / "public.json";source.write_bytes(self.payload)
            output = directory / "figures";output.mkdir()
            preserved = directory / "preserved.bin";preserved.write_bytes(b"must not change")
            try:os.link(preserved, output / plot.OUTPUTS[0])
            except OSError as error:self.skipTest("Hard links unavailable: " + str(error))
            with self.assertRaises(ValueError):plot.render_report(source, output, overwrite=True)
            self.assertEqual(preserved.read_bytes(), b"must not change")

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is required for the rendering smoke test")
    def test_real_render_has_only_relative_public_metadata_and_preserves_input(self):
        with tempfile.TemporaryDirectory(prefix=".public-report-test-", dir=Path(__file__).parent) as temporary:
            directory = Path(temporary);source = directory / "public.json";source.write_bytes(self.payload)
            output = directory / "figures"
            report = plot.render_report(source, output)
            self.assertEqual(source.read_bytes(), self.payload)
            self.assertEqual(report["summary_sha256"], hashlib.sha256(self.payload).hexdigest())
            self.assertEqual(len(report["figures"]), 2)
            for item in report["figures"]:
                metadata = plot.png_metadata(output / item["file"])
                self.assertEqual(metadata["SummaryFile"], "public.json")
                self.assertEqual(metadata["SummarySHA256"], report["summary_sha256"])
                self.assertNotIn("SourceFile", metadata)
                self.assertNotIn(str(directory), json.dumps(metadata))
                self.assertGreater(min(metadata["pixels"]), 1000)
            with self.assertRaises(FileExistsError):plot.render_report(source, output)
            self.assertEqual(source.read_bytes(), self.payload)
            self.assertFalse(list(output.glob(".public-report-*")))


if __name__ == "__main__":
    unittest.main()
