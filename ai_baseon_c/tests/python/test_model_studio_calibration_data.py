"""Deterministic preparation, exporter arithmetic, overlap and boundary tests."""

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import wave

import numpy as np

LIBRARY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LIBRARY / "tools"))

from model_converter.schema import ConversionError
from model_studio.calibration_data import LABELS, prepare_calibration


class CalibrationDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="studio_calibration_data_",
                                                       dir=LIBRARY / "tests" / "python")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.train, self.test = self.root / "train", self.root / "test"
        self.train.mkdir()
        self.test.mkdir()
        self.write_wav(self.test / "test.wav", frequency=3)

    def write_wav(self, path, frequency=2, frames=4, rate=8000, channels=1):
        path.parent.mkdir(parents=True, exist_ok=True)
        index = np.arange(4096 * frames * (rate // 8000), dtype=np.float64)
        values = (15000 * np.sin(2 * np.pi * frequency * index / 4096)).astype("<i2")
        if channels > 1:
            values = np.stack((values, np.zeros_like(values)), axis=1)
        with wave.open(str(path), "wb") as output:
            output.setnchannels(channels)
            output.setsampwidth(2)
            output.setframerate(rate)
            output.writeframes(values.tobytes())
        return values

    def populate(self, files=3, frames=4):
        for label, label_id in LABELS.items():
            for index in range(files):
                self.write_wav(self.train / label / f"voice_{index}.wav",
                               frequency=11 + label_id * 13 + index, frames=frames)

    def prepare(self, samples=16, name="calibration.npz", **options):
        return prepare_calibration(self.train, self.test, self.root / name,
                                   max_samples=samples, **options)

    def test_balanced_diverse_reproducible_selection_excludes_byte_duplicates(self):
        self.populate()
        label = next(iter(LABELS))
        (self.train / label / "test_duplicate.wav").write_bytes((self.test / "test.wav").read_bytes())
        (self.train / label / "voice_duplicate.wav").write_bytes((self.train / label / "voice_0.wav").read_bytes())
        originals = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in self.train.rglob("*.wav")}
        first = self.prepare()
        second = self.prepare(name="second.npz")
        a, b = first["provenance"], second["provenance"]
        self.assertEqual(a, b)
        self.assertEqual(a["class_counts"], {label: 4 for label in LABELS})
        self.assertEqual(len(a["excluded_wavs"]), 2)
        self.assertEqual({row["reason"] for row in a["excluded_wavs"]},
                         {"identical_test_wav", "duplicate_train_wav"})
        self.assertEqual(len({row["wav"] for row in a["selected"]}), 12)
        self.assertEqual(hashlib.sha256(Path(first["dataset_path"]).read_bytes()).hexdigest(), a["npz_sha256"])
        with np.load(first["dataset_path"], allow_pickle=False) as data:
            self.assertEqual(data["inputs"].shape, (16, 1, 2048))
            self.assertEqual(data["inputs"].dtype, np.uint8)
            np.testing.assert_array_equal(data["labels"][:4], [0, 1, 2, 3])
        for path, expected in originals.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)
        self.assertEqual(json.loads(Path(first["provenance_path"]).read_text(encoding="utf-8")), a)

    def test_exact_existing_exporter_formula_and_evenly_spaced_frame_positions(self):
        label = next(iter(LABELS))
        values = self.write_wav(self.train / label / "one.wav", frequency=17.3, frames=9)
        result = self.prepare(samples=3)
        self.assertEqual([row["frame_index"] for row in result["provenance"]["selected"]], [0, 4, 8])
        with np.load(result["dataset_path"], allow_pickle=False) as data:
            for item, row in zip(data["inputs"], result["provenance"]["selected"]):
                frame = row["frame_index"]
                segment = values[frame * 4096:(frame + 1) * 4096].astype(np.float32) / 32768.0
                magnitude = np.abs(np.fft.rfft(segment, n=4096))[:2048]
                normalized = np.empty(2048, dtype=np.float32)
                normalized[:] = magnitude / (np.max(np.abs(magnitude)) + 1e-9)
                expected = np.clip(np.round(normalized / (1 / 255)).astype(np.int64), 0, 255).astype(np.uint8)
                np.testing.assert_array_equal(item[0], expected)

    def test_stereo_integer_decimation_matches_first_channel_exporter(self):
        label = next(iter(LABELS))
        values = self.write_wav(self.train / label / "stereo.wav", frequency=20.5, frames=2,
                                rate=16000, channels=2)
        result = self.prepare(samples=1)
        row = result["provenance"]["selected"][0]
        audio = values[:, 0].astype(np.float32) / 32768.0
        audio = audio[::2]
        start = row["frame_index"] * 4096
        magnitude = np.abs(np.fft.rfft(audio[start:start + 4096], n=4096))[:2048]
        normalized = (magnitude / (magnitude.max() + 1e-9)).astype(np.float32)
        expected = np.clip(np.round(normalized / (1 / 255)).astype(np.int64), 0, 255).astype(np.uint8)
        with np.load(result["dataset_path"], allow_pickle=False) as data:
            np.testing.assert_array_equal(data["inputs"][0, 0], expected)
        self.assertEqual(row["channels"], 2)
        self.assertEqual(row["downsample_factor"], 2)

    def test_insufficient_samples_and_missing_classes_are_explicit(self):
        label = next(iter(LABELS))
        self.write_wav(self.train / label / "short.wav", frequency=17, frames=1)
        result = self.prepare(samples=8)
        self.assertEqual(result["provenance"]["selected_samples"], 1)
        self.assertTrue(any("Only 1" in warning for warning in result["provenance"]["warnings"]))
        self.assertEqual(sum("No eligible calibration" in warning for warning in result["provenance"]["warnings"]), 3)

    def test_new_file_protection_and_no_output_for_invalid_datasets(self):
        destination = self.root / "calibration.npz"
        with self.assertRaisesRegex(ConversionError, "No eligible"):
            self.prepare()
        self.assertFalse(destination.exists())
        self.populate()
        destination.write_bytes(b"user-owned")
        with self.assertRaisesRegex(ConversionError, "new file names"):
            self.prepare()
        self.assertEqual(destination.read_bytes(), b"user-owned")
        self.assertFalse(destination.with_suffix(".json").exists())

    def test_invalid_options_rejected_without_creating_outputs(self):
        self.populate()
        for count in (0, -1, 129, True, 3.5):
            with self.subTest(count=count), self.assertRaises(ConversionError):
                self.prepare(samples=count)
        for options in ({"input_scale": 0}, {"input_scale": float("nan")},
                        {"input_scale": True}, {"input_scale": 1e-300},
                        {"input_zero_point": -1}, {"input_zero_point": 256},
                        {"input_zero_point": True}):
            with self.subTest(options=options), self.assertRaises(ConversionError):
                self.prepare(**options)
        with self.assertRaisesRegex(ConversionError, "non-nested"):
            prepare_calibration(self.train, self.train, self.root / "bad.npz")
        with self.assertRaisesRegex(ConversionError, ".npz"):
            self.prepare(name="bad.wav")
        self.assertFalse((self.root / "calibration.npz").exists())

    def test_second_file_publish_failure_cleans_owned_output(self):
        self.populate(files=1, frames=1)
        original = os.rename if os.name == "nt" else os.link
        calls = []
        def failing_publish(source, target):
            calls.append(target)
            if len(calls) == 2:
                raise PermissionError("injected provenance publication failure")
            return original(source, target)
        tool = "model_studio.calibration_data.os." + ("rename" if os.name == "nt" else "link")
        with mock.patch(tool, side_effect=failing_publish), self.assertRaises(PermissionError):
            self.prepare(samples=4)
        self.assertFalse((self.root / "calibration.npz").exists())
        self.assertFalse((self.root / "calibration.json").exists())
        self.assertEqual(list(self.root.glob(".calibration-*")), [])

    def test_tiny_scale_saturates_without_int64_overflow(self):
        self.populate(files=1, frames=1)
        result = self.prepare(samples=4, input_scale=float(np.finfo(np.float32).tiny), input_zero_point=127)
        with np.load(result["dataset_path"], allow_pickle=False) as data:
            self.assertEqual(data["inputs"].dtype, np.uint8)
            self.assertTrue(np.all(data["inputs"] >= 127))
            self.assertTrue(np.any(data["inputs"] == 255))


if __name__ == "__main__":
    unittest.main()
