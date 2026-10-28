"""Build a reproducible, class-balanced calibration set from this voice dataset.

Feature extraction and uint8 rounding mirror demo/scripts/export_testset_int8_bins.py
and verify_int8.py; no model execution, training import, or label fitting occurs.
"""

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import wave

import numpy as np

if __name__ == "__main__" and not __package__:
    # Also support the same direct-script launch convention as the GUI entry.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_converter.schema import ConversionError


LABELS = {"No Human Voice": 0, "Others": 1, "XiaoYuan": 2, "XiaoXin": 3}
FRAME_LENGTH = 4096
FEATURE_LENGTH = 2048
MAX_SAMPLES = 128


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _wav_metadata(path):
    try:
        with wave.open(str(path), "rb") as source:
            rate, channels, width = source.getframerate(), source.getnchannels(), source.getsampwidth()
            if (width != 2 or source.getcomptype() != "NONE" or channels < 1 or channels > 8
                    or rate < 8000 or rate > 192000 or rate % 8000):
                raise ConversionError("Expected PCM16, 1..8 channels and an integer multiple of 8kHz <=192kHz")
            factor = rate // 8000
            frames = (source.getnframes() + factor - 1) // factor // FRAME_LENGTH
            stat = path.stat()
            return {"sample_rate": rate, "channels": channels, "downsample_factor": factor,
                    "complete_frames": frames, "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    except (OSError, EOFError, wave.Error) as error:
        raise ConversionError(f"Cannot read PCM WAV {path}: {error}") from error


def _even_indices(total, count):
    if count == 0:
        return []
    if count == 1:
        return [total // 2]
    # Integer arithmetic avoids platform floating rounding or repeated indices.
    return [(index * (total - 1)) // (count - 1) for index in range(count)]


def _feature(path, metadata, frame_index):
    factor, channels = metadata["downsample_factor"], metadata["channels"]
    with wave.open(str(path), "rb") as source:
        source.setpos(frame_index * FRAME_LENGTH * factor)
        raw = source.readframes(FRAME_LENGTH * factor)
    audio = np.frombuffer(raw, dtype="<i2")
    if audio.size % channels:
        raise ConversionError(f"Truncated PCM samples: {path}")
    audio = audio.reshape(-1, channels)[::factor, 0].astype(np.float32) / 32768.0
    if audio.size != FRAME_LENGTH:
        raise ConversionError(f"Truncated selected frame {frame_index}: {path}")
    magnitude = np.abs(np.fft.rfft(audio, n=FRAME_LENGTH))[:FEATURE_LENGTH]
    # The existing exporter first assigns normalized spectra to a float32
    # feature array, then quantizes it; retain this intermediate rounding.
    return np.asarray(magnitude / (np.max(np.abs(magnitude)) + 1e-9), dtype=np.float32)


def _publish_pair(destination, payload, provenance_path, json_payload):
    """Publish complete new files; roll back only this call's first file on error."""
    owned = None
    with tempfile.TemporaryDirectory(prefix=".calibration-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        try:
            for name, contents, target in (("inputs.npz", payload, destination),
                                            ("provenance.json", json_payload, provenance_path)):
                source = staging / name
                source.write_bytes(contents)
                if os.name == "nt":
                    os.rename(source, target)  # Windows refuses existing target atomically.
                else:
                    os.link(source, target)  # POSIX exclusive publish of complete bytes.
                if target == destination:
                    stat = target.stat()
                    owned = (stat.st_dev, stat.st_ino)
        except Exception:
            if owned is not None and destination.exists():
                stat = destination.stat()
                if (stat.st_dev, stat.st_ino) == owned:
                    destination.unlink()
            raise


def prepare_calibration(train_root, test_root, output_path, max_samples=128,
                        input_scale=1 / 255, input_zero_point=0):
    """Write new ``.npz`` + sibling ``.json``; return their paths and provenance.

    Classes take turns up to the requested total. Within each class, uniformly
    spaced WAV files take turns before a second frame is taken from a file;
    selected frame positions span that file. Test-byte duplicates and repeated
    training WAV contents are excluded by SHA256 before selection. This detects
    exact file duplicates, not near-duplicate audio or shared recording sources.
    """
    if (isinstance(max_samples, bool) or not isinstance(max_samples, int)
            or not 1 <= max_samples <= MAX_SAMPLES):
        raise ConversionError(f"max_samples must be an integer in [1,{MAX_SAMPLES}]")
    try:
        scale = float(input_scale)
    except (TypeError, ValueError, OverflowError) as error:
        raise ConversionError("input_scale must be finite and positive") from error
    if (isinstance(input_scale, bool) or not math.isfinite(scale) or scale <= 0
            or scale > np.finfo(np.float32).max or np.float32(scale) <= 0):
        raise ConversionError("input_scale must be finite and positive float32")
    if (isinstance(input_zero_point, bool) or not isinstance(input_zero_point, int)
            or not 0 <= input_zero_point <= 255):
        raise ConversionError("input_zero_point must be an integer in [0,255]")
    train, test, destination = (Path(value).expanduser().resolve()
                                for value in (train_root, test_root, output_path))
    if not train.is_dir() or not test.is_dir():
        raise ConversionError("train_root and test_root must be existing directories")
    if train == test or train in test.parents or test in train.parents:
        raise ConversionError("Training and test roots must be distinct, non-nested directories")
    if destination.suffix.lower() != ".npz":
        raise ConversionError("Calibration output_path must end in .npz")
    provenance_path = destination.with_suffix(".json")
    if destination.exists() or provenance_path.exists():
        raise ConversionError("Calibration output and provenance must use new file names")
    test_files = sorted(test.rglob("*.wav"), key=lambda value: str(value.relative_to(test)).casefold())
    if not test_files:
        raise ConversionError("No test WAV files found; cannot check overlap")
    test_hashes = {}
    for path in test_files:
        test_hashes.setdefault(_sha256(path), []).append(str(path.relative_to(test)))
    by_class = {name: [] for name in LABELS}
    seen, exclusions, unsupported = {}, [], []
    inspected = 0
    for label, label_id in LABELS.items():
        for path in sorted((train / label).glob("*.wav"), key=lambda value: value.name.casefold()):
            inspected += 1
            digest = _sha256(path)
            relative = str(path.relative_to(train))
            if digest in test_hashes:
                exclusions.append({"wav": relative, "sha256": digest, "reason": "identical_test_wav",
                                   "test_matches": test_hashes[digest]})
                continue
            if digest in seen:
                exclusions.append({"wav": relative, "sha256": digest, "reason": "duplicate_train_wav",
                                   "first_train_wav": seen[digest]})
                continue
            seen[digest] = relative
            try:
                metadata = _wav_metadata(path)
            except ConversionError as error:
                unsupported.append({"wav": relative, "reason": str(error)})
                continue
            if metadata["complete_frames"] == 0:
                unsupported.append({"wav": relative, "reason": "no complete 4096-sample frame"})
                continue
            by_class[label].append({"path": path, "wav": relative, "label_id": label_id,
                                    "sha256": digest, **metadata})
    capacity = {label: sum(wav["complete_frames"] for wav in files) for label, files in by_class.items()}
    quotas = {label: 0 for label in LABELS}
    while sum(quotas.values()) < max_samples:
        changed = False
        for label in LABELS:
            if quotas[label] < capacity[label] and sum(quotas.values()) < max_samples:
                quotas[label] += 1
                changed = True
        if not changed:
            break
    count = sum(quotas.values())
    if not count:
        raise ConversionError("No eligible training frames remain after overlap/format checks")
    chosen = []
    for label, files in by_class.items():
        quota = quotas[label]
        indices = _even_indices(len(files), min(quota, len(files)))
        candidates = [files[index] for index in indices]
        allocations = [0] * len(candidates)
        while sum(allocations) < quota:
            for index, item in enumerate(candidates):
                if allocations[index] < item["complete_frames"] and sum(allocations) < quota:
                    allocations[index] += 1
        rows = []
        for item, wanted in zip(candidates, allocations):
            for frame in _even_indices(item["complete_frames"], wanted):
                rows.append({**item, "frame_index": frame})
        chosen.append(rows)
    # Interleave class rows so a smaller Taylor sample limit remains balanced.
    ordered = [rows[index] for index in range(max(map(len, chosen))) for rows in chosen if index < len(rows)]
    inputs = np.empty((count, 1, FEATURE_LENGTH), dtype=np.uint8)
    labels = np.empty(count, dtype=np.int64)
    selected = []
    for index, item in enumerate(ordered):
        feature = _feature(item["path"], item, item["frame_index"])
        # np.round is nearest-even, matching the existing test-set exporter.
        # Clip before integer conversion too, so tiny positive scales cannot
        # overflow int64 while computing an otherwise saturating uint8 value.
        with np.errstate(over="ignore"):
            quantized = np.clip(np.round(feature / scale), -input_zero_point,
                                255 - input_zero_point).astype(np.int64) + input_zero_point
        inputs[index, 0] = quantized.astype(np.uint8)
        labels[index] = item["label_id"]
        stat = item["path"].stat()
        if stat.st_size != item["size_bytes"] or stat.st_mtime_ns != item["mtime_ns"]:
            raise ConversionError(f"Training WAV changed during preparation: {item['path']}")
        selected.append({key: item[key] for key in ("wav", "label_id", "sha256", "sample_rate", "channels",
                                                     "downsample_factor", "frame_index", "complete_frames")})
    for path, expected in {item["path"]: item["sha256"] for item in ordered}.items():
        if _sha256(path) != expected:
            raise ConversionError(f"Training WAV contents changed during preparation: {path}")
    warnings = []
    if count < max_samples:
        warnings.append(f"Only {count} eligible frames available for {max_samples} requested samples")
    for label, quota in quotas.items():
        if not quota:
            warnings.append(f"No eligible calibration frames from class {label}")
    provenance = {
        "format": "model_studio_voice_calibration_v1", "train_root": str(train), "test_root": str(test),
        "requested_samples": max_samples, "selected_samples": count, "class_counts": quotas,
        "training_wavs_inspected": inspected, "test_wavs_hashed": len(test_files),
        "eligible_complete_frames": capacity, "excluded_wavs": exclusions, "unsupported_wavs": unsupported,
        "selected": selected, "warnings": warnings,
        "selection": "balanced classes; evenly spaced files; round-robin file allocations; evenly spaced frame positions; interleaved classes",
        "overlap_check": "SHA256 of complete WAV files; does not detect near-duplicates or common recording origins",
        "preprocessing": {"sample_rate": 8000, "pcm": "signed little-endian int16 divided by 32768 as float32",
                          "channels": "first channel", "resampling": "integer decimation only, matching existing exporter",
                          "frame_samples": FRAME_LENGTH, "tail": "drop incomplete frames",
                          "features": "abs(numpy.fft.rfft(frame))[:2048] / (max(abs(magnitude)) + 1e-9); cast float32 before quantization",
                          "quantization": "clip(round_to_nearest_even(feature / scale) + zero_point, 0, 255)",
                          "input_scale": scale, "input_zero_point": input_zero_point},
        "inputs_shape": list(inputs.shape), "inputs_dtype": str(inputs.dtype),
        "inputs_sha256": hashlib.sha256(inputs.tobytes()).hexdigest(),
        "labels_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
        "software": {"numpy": np.__version__}, "purpose": "calibration only; do not use as independent validation"}
    buffer = io.BytesIO()
    np.savez_compressed(buffer, inputs=inputs, labels=labels)
    payload = buffer.getvalue()
    provenance["npz_sha256"] = hashlib.sha256(payload).hexdigest()
    json_payload = (json.dumps(provenance, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Both final buffers are prepared before publishing. Individual files are
    # complete and exclusively created; a second-file failure rolls back ours.
    _publish_pair(destination, payload, provenance_path, json_payload)
    return {"dataset_path": str(destination), "provenance_path": str(provenance_path), "provenance": provenance}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare independent, balanced voice calibration inputs")
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--test-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--input-scale", type=float, default=1 / 255)
    parser.add_argument("--input-zero-point", type=int, default=0)
    arguments = parser.parse_args(argv)
    result = prepare_calibration(arguments.train_root, arguments.test_root, arguments.output,
                                 arguments.max_samples, arguments.input_scale, arguments.input_zero_point)
    print(json.dumps({key: value for key, value in result.items() if key != "provenance"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
