"""CPU reference evaluation using explicit graph semantics and local test data."""

import csv
import math
from pathlib import Path
import time
import zipfile

import numpy as np
import torch
from torch import nn

from model_converter.loaders import CNNVoiceClassifier
from model_converter.schema import ConversionError

from .compression import _validated_spec


MAX_DATA_BYTES = 512 * 1024 * 1024
MAX_INPUT_ELEMENTS = 64 * 1024 * 1024
MAX_SAMPLES = 100000
BATCH_SIZE = 32
# Evaluation admission limits, separate from converter/static-analysis limits.
# These guard local CPU execution; they are not measurements of process RSS.
MAX_CPU_WORKING_BYTES = 256 * 1024 * 1024
MAX_CPU_MODEL_BYTES = 256 * 1024 * 1024
CPU_WORKSPACE_FACTOR = 4
MAX_CPU_SEQUENCE_LENGTH = 8192
MAX_CPU_SAMPLE_MACS = 100000000
MAX_CPU_TOTAL_MACS = 2000000000


class _RecurrentClassifier(nn.Module):
    def __init__(self, recurrent, head):
        super().__init__()
        self.recurrent = recurrent
        self.head = head

    def forward(self, values):
        sequence, _ = self.recurrent(values)
        return self.head(sequence[:, -1, :])


def build_model(session):
    """Rebuild the known unfused PyTorch graph in float32 on CPU in eval mode."""
    spec = _validated_spec(session)
    _evaluation_plan(spec)
    state = {key: value.detach().cpu() for key, value in session.state.items()}
    if spec.architecture == "cnn":
        model = CNNVoiceClassifier()
        layers = []
        for ordinal, index in enumerate((0, 4, 8)):
            outputs, inputs, kernel = state[f"model.{index}.weight"].shape
            epsilon = session.metadata.get("bn_eps", [1e-5] * 3)[ordinal]
            layers.extend((nn.Conv1d(inputs, outputs, kernel, bias=f"model.{index}.bias" in state),
                           nn.BatchNorm1d(outputs, eps=float(epsilon)), nn.ReLU(), nn.MaxPool1d(2)))
        model.model = nn.Sequential(*layers, nn.Flatten())
        model.fc1 = nn.Linear(spec.dimensions["flatten_size"], spec.dimensions["hidden_size"])
        model.fc2 = nn.Linear(spec.dimensions["hidden_size"], spec.dimensions["num_classes"])
        # Historical graph has dropout after fc1, with no fc1 ReLU. Eval disables it.
        model.dropout = nn.Dropout(0.4)
        model.load_state_dict(state)
    else:
        prefix = next(key[:-len("weight_ih_l0")] for key in state if key.endswith("weight_ih_l0"))
        head_prefix = next(prefix for prefix in ("head.", "fc.", "classifier.")
                           if prefix + "weight" in state)
        factory = nn.LSTM if spec.architecture == "lstm" else nn.RNN
        recurrent = factory(spec.dimensions["input_size"], spec.dimensions["hidden_size"],
                            batch_first=True)
        recurrent.load_state_dict({key[len(prefix):]: value for key, value in state.items()
                                   if key.startswith(prefix) and key != head_prefix + "weight"
                                   and key != head_prefix + "bias"})
        head = nn.Linear(spec.dimensions["hidden_size"], spec.dimensions["num_classes"])
        head.load_state_dict({"weight": state[head_prefix + "weight"],
                              "bias": state[head_prefix + "bias"]})
        model = _RecurrentClassifier(recurrent, head)
    return model.cpu().float().eval()


def _shape(spec):
    dimensions = spec.dimensions
    if spec.architecture == "cnn":
        return (dimensions["input_channels"], dimensions["input_length"])
    return (dimensions["seq_length"], dimensions["input_size"])


def _graph_cpu_cost(spec):
    """Conservative shape accounting, without allocating activations or a model."""
    dimensions = spec.dimensions
    inputs, classes = dimensions["input_elements"], dimensions["num_classes"]
    hidden = dimensions["hidden_size"]
    if spec.architecture == "cnn":
        # Include conv/BN/ReLU outputs, pool outputs and a full im2col buffer
        # for every block, even though many buffers do not coexist in eval.
        intermediates = sum(3 * outputs * width + outputs * pooled
                            + channels * kernel * width
                            for outputs, channels, kernel, width, pooled in dimensions["conv_shapes"])
        intermediates += hidden + classes
        macs = sum(outputs * channels * kernel * width
                   for outputs, channels, kernel, width, _ in dimensions["conv_shapes"])
        macs += hidden * dimensions["flatten_size"] + classes * hidden
        parameters = sum(outputs * channels * kernel + 5 * outputs
                         for outputs, channels, kernel, _, _ in dimensions["conv_shapes"])
        parameters += hidden * dimensions["flatten_size"] + hidden + classes * hidden + classes
    else:
        sequence, width = dimensions["seq_length"], dimensions["input_size"]
        if sequence > MAX_CPU_SEQUENCE_LENGTH:
            raise ConversionError(f"CPU evaluation sequence length {sequence} exceeds "
                                  f"the limit {MAX_CPU_SEQUENCE_LENGTH}; static analysis remains available")
        gates = 4 if spec.architecture == "lstm" else 1
        # PyTorch returns the complete [batch, sequence, hidden] output.
        # Reserve another gate sequence plus hidden/cell scratch, then apply
        # a workspace factor for backend packing and temporary buffers.
        intermediates = sequence * hidden * (1 + gates) + hidden * (gates + 2) + classes
        macs = sequence * gates * hidden * (width + hidden) + classes * hidden
        parameters = gates * hidden * (width + hidden + 2) + classes * hidden + classes
    if macs > MAX_CPU_SAMPLE_MACS:
        raise ConversionError(f"CPU evaluation requires {macs} estimated dense MACs per sample "
                              f"for one graph; limit is {MAX_CPU_SAMPLE_MACS}; static analysis remains available")
    return {"working_bytes_per_sample": 4 * (2 * inputs + CPU_WORKSPACE_FACTOR * intermediates),
            "model_bytes": 4 * parameters, "macs_per_sample": macs}


def _evaluation_plan(*specs):
    costs = [_graph_cpu_cost(spec) for spec in specs]
    per_sample = sum(item["working_bytes_per_sample"] for item in costs)
    # Both models are resident during comparison. Account for two FP32 copies
    # while modules are initialized and weights are copied into them.
    model_bytes = 2 * sum(item["model_bytes"] for item in costs)
    if per_sample > MAX_CPU_WORKING_BYTES:
        raise ConversionError(f"CPU evaluation estimated local activation/workspace bytes per sample "
                              f"are {per_sample}; budget is {MAX_CPU_WORKING_BYTES}; "
                              "static analysis remains available")
    if model_bytes > MAX_CPU_MODEL_BYTES:
        raise ConversionError(f"CPU evaluation estimated FP32 model reconstruction bytes are {model_bytes}; "
                              f"budget is {MAX_CPU_MODEL_BYTES}; static analysis remains available")
    return {"batch_size": min(BATCH_SIZE, MAX_CPU_WORKING_BYTES // per_sample),
            "estimated_working_bytes_per_sample": per_sample,
            "estimated_model_reconstruction_bytes": model_bytes,
            "estimated_macs_per_sample": sum(item["macs_per_sample"] for item in costs),
            "working_bytes_limit": MAX_CPU_WORKING_BYTES, "model_bytes_limit": MAX_CPU_MODEL_BYTES,
            "total_macs_limit": MAX_CPU_TOTAL_MACS, "single_graph_macs_per_sample_limit": MAX_CPU_SAMPLE_MACS,
            "sequence_length_limit": MAX_CPU_SEQUENCE_LENGTH, "workspace_factor": CPU_WORKSPACE_FACTOR,
            "accounting": "Conservative FP32 shape estimate: input copies, CNN conv/BN/ReLU/pool and "
                          "im2col buffers, or full recurrent output/gate sequences and hidden/cell scratch; "
                          "intermediates use the workspace factor, and compared graphs are summed. "
                          "Model reconstruction has a separate two-copy estimate. This is an admission "
                          "budget, not measured or guaranteed peak process memory; excludes existing "
                          "sessions, dataset storage, Python and backend allocator overhead."}


def _check_evaluation_count(count, limit, plan):
    if plan is None:
        return
    selected = min(count, limit)
    macs = selected * plan["estimated_macs_per_sample"]
    if macs > MAX_CPU_TOTAL_MACS:
        raise ConversionError(f"CPU evaluation of {selected} samples requires {macs} estimated dense MACs; "
                              f"total budget is {MAX_CPU_TOTAL_MACS}; reduce max_samples")


def _capacity(count, elements):
    if count < 1:
        raise ConversionError("Evaluation dataset is empty")
    if count > MAX_SAMPLES or count * elements > MAX_INPUT_ELEMENTS:
        raise ConversionError(f"Dataset capacity exceeded: at most {MAX_SAMPLES} samples and "
                              f"{MAX_INPUT_ELEMENTS} input elements are supported")


def _labels(values, count, classes):
    if values is None:
        return None
    if (values.shape != (count,) or not np.issubdtype(values.dtype, np.integer)
            or np.issubdtype(values.dtype, np.bool_)):
        raise ConversionError("labels must be a one-dimensional integer array matching inputs")
    if np.any(values < 0) or np.any(values >= classes):
        raise ConversionError(f"labels must lie in [0, {classes - 1}]")
    return values.astype(np.int64, copy=False)


def _arrays(inputs, labels, spec, limit, plan=None):
    expected = _shape(spec)
    elements = spec.dimensions["input_elements"]
    if (inputs.ndim not in (2, 3) or tuple(inputs.shape[1:]) not in ((elements,), expected)):
        raise ConversionError(f"inputs must have shape [N, {elements}] or [N, {expected[0]}, {expected[1]}]")
    _capacity(inputs.shape[0], elements)
    if inputs.nbytes > MAX_DATA_BYTES:
        raise ConversionError(f"Input array exceeds the {MAX_DATA_BYTES}-byte capacity")
    if inputs.dtype != np.uint8 and not np.issubdtype(inputs.dtype, np.floating):
        raise ConversionError("inputs must be uint8 quantized frames or floating preprocessed values")
    if np.issubdtype(inputs.dtype, np.floating):
        # Validate the complete dataset, even when inference uses a prefix.
        scan_batch = BATCH_SIZE if plan is None else plan["batch_size"]
        for begin in range(0, inputs.shape[0], scan_batch):
            values = inputs[begin:begin + scan_batch]
            if not np.all(np.isfinite(values)) or np.any(np.abs(values) > np.finfo(np.float32).max):
                raise ConversionError("inputs contain NaN/Inf or values outside finite float32 range")
    labels = _labels(labels, inputs.shape[0], spec.dimensions["num_classes"])
    count = min(inputs.shape[0], limit)
    # Copy once in C order before closing mappings/archives. Converting float
    # data to the model's FP32 type here also avoids retaining a second float64
    # dataset or making another whole-array copy when reshaping Fortran data.
    dtype = np.uint8 if inputs.dtype == np.uint8 else np.float32
    return np.array(inputs[:count], dtype=dtype, copy=True, order="C").reshape(count, *expected), (
        None if labels is None else labels[:count].copy())


def _inspect_numpy_header(stream, size, spec, labels=False, count=None):
    """Reject huge or object NPY payloads before NumPy allocates the array."""
    version = np.lib.format.read_magic(stream)
    shape, _, dtype = np.lib.format._read_array_header(stream, version, max_header_size=10000)
    if dtype.hasobject:
        raise ConversionError("Object arrays are forbidden; pickle loading is disabled")
    if labels:
        if (shape != (count,) or not np.issubdtype(dtype, np.integer)
                or np.issubdtype(dtype, np.bool_)):
            raise ConversionError("labels must be a one-dimensional integer array matching inputs")
    else:
        expected = _shape(spec)
        elements = spec.dimensions["input_elements"]
        if len(shape) not in (2, 3) or shape[1:] not in ((elements,), expected):
            raise ConversionError(f"inputs header has incompatible shape {shape}")
        _capacity(shape[0], elements)
        if dtype != np.uint8 and not np.issubdtype(dtype, np.floating):
            raise ConversionError("inputs must be uint8 quantized frames or floating preprocessed values")
    elements = math.prod(shape)
    payload = elements * dtype.itemsize
    if payload > MAX_DATA_BYTES:
        raise ConversionError("NPY array header exceeds the configured byte capacity")
    if stream.tell() + payload != size:
        raise ConversionError("NPY array header and payload size do not match")
    return shape


def _numpy_data(path, spec, limit, plan=None):
    if path.stat().st_size > MAX_DATA_BYTES:
        raise ConversionError(f"Dataset file exceeds the {MAX_DATA_BYTES}-byte capacity")
    if path.suffix.lower() == ".npy":
        with path.open("rb") as stream:
            shape = _inspect_numpy_header(stream, path.stat().st_size, spec)
        _check_evaluation_count(shape[0], limit, plan)
        inputs = np.load(path, allow_pickle=False, mmap_mode="r")
        try:
            return _arrays(inputs, None, spec, limit, plan)
        finally:
            mapping = getattr(inputs, "_mmap", None)
            if mapping is not None:
                mapping.close()
    with zipfile.ZipFile(path) as archive:
        if sum(item.file_size for item in archive.infolist()) > MAX_DATA_BYTES:
            raise ConversionError("Decompressed npz dataset exceeds the configured capacity")
        if len(archive.namelist()) != len(set(archive.namelist())):
            raise ConversionError("npz dataset contains duplicate array names")
        if "inputs.npy" not in archive.namelist():
            raise ConversionError("npz dataset requires an inputs array (and optional labels)")
        member = archive.getinfo("inputs.npy")
        with archive.open(member) as stream:
            shape = _inspect_numpy_header(stream, member.file_size, spec)
        if "labels.npy" in archive.namelist():
            member = archive.getinfo("labels.npy")
            with archive.open(member) as stream:
                _inspect_numpy_header(stream, member.file_size, spec, labels=True, count=shape[0])
        _check_evaluation_count(shape[0], limit, plan)
    with np.load(path, allow_pickle=False) as archive:
        if "inputs" not in archive.files:
            raise ConversionError("npz dataset requires an inputs array (and optional labels)")
        return _arrays(archive["inputs"], archive["labels"] if "labels" in archive.files else None,
                       spec, limit, plan)


def _manifest_data(path, spec, limit, plan=None):
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ConversionError("Evaluation manifest exceeds the 16 MiB capacity")
    elements = spec.dimensions["input_elements"]
    classes = spec.dimensions["num_classes"]
    files = []
    total = 0
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {"feat_bin", "label_id"}.issubset(reader.fieldnames):
            raise ConversionError("manifest.csv requires feat_bin and label_id columns")
        for line, row in enumerate(reader, 2):
            try:
                feature_text = row["feat_bin"]
                if not isinstance(feature_text, str) or not feature_text.strip():
                    raise ValueError("feat_bin must be a nonempty relative feature path")
                raw_path = Path(feature_text.replace("\\", "/"))
                if raw_path.is_absolute() or not str(raw_path) or str(raw_path) == ".":
                    raise ValueError("feat_bin must be a relative feature path")
                feature_path = (path.parent / raw_path).resolve()
                if not feature_path.is_relative_to(path.parent.resolve()):
                    raise ValueError("feat_bin must remain within the manifest directory")
                label = int(row["label_id"])
                if not 0 <= label < classes:
                    raise ValueError(f"label_id must lie in [0, {classes - 1}]")
                size = feature_path.stat().st_size
                if not feature_path.is_file() or size == 0 or size % elements:
                    raise ValueError(f"feat_bin size must be a positive multiple of {elements}: {feature_path}")
                frames = size // elements
                if row.get("frames", "") and int(row["frames"]) != frames:
                    raise ValueError("frames field does not match feat_bin size")
                total += frames
                _capacity(total, elements)
                files.append((feature_path, label, frames))
            except (OSError, TypeError, ValueError) as error:
                raise ConversionError(f"Invalid manifest row {line}: {error}") from error
    _capacity(total, elements)
    _check_evaluation_count(total, limit, plan)
    count = min(total, limit)
    inputs = np.empty((count, elements), np.uint8)
    labels = np.empty(count, np.int64)
    cursor = 0
    for feature_path, label, frames in files:
        take = min(frames, count - cursor)
        if take <= 0:
            break
        values = np.fromfile(feature_path, dtype=np.uint8, count=take * elements)
        if values.size != take * elements:
            raise ConversionError(f"Feature file changed while reading: {feature_path}")
        inputs[cursor:cursor + take] = values.reshape(take, elements)
        labels[cursor:cursor + take] = label
        cursor += take
    return inputs.reshape(count, *_shape(spec)), labels


def evaluate_sessions(reference, candidate, data_path, max_samples=256,
                      input_scale=1 / 255, input_zero_point=0):
    """Compare logits/predictions on local data; accuracy describes only labeled samples."""
    started = time.perf_counter()
    if (not isinstance(max_samples, int) or isinstance(max_samples, bool)
            or not 1 <= max_samples <= MAX_SAMPLES):
        raise ConversionError(f"max_samples must be an integer in [1, {MAX_SAMPLES}]")
    try:
        scale = float(input_scale)
    except (TypeError, ValueError, OverflowError) as error:
        raise ConversionError("input_scale must be finite, positive and representable in float32") from error
    if (isinstance(input_scale, bool) or not math.isfinite(scale) or scale <= 0
            or scale > np.finfo(np.float32).max or np.float32(scale) <= 0):
        raise ConversionError("input_scale must be finite, positive and representable in float32")
    if (not isinstance(input_zero_point, int) or isinstance(input_zero_point, bool)
            or not 0 <= input_zero_point <= 255):
        raise ConversionError("input_zero_point must be an integer in [0, 255]")
    ref_spec, cand_spec = _validated_spec(reference), _validated_spec(candidate)
    if (ref_spec.architecture != cand_spec.architecture or _shape(ref_spec) != _shape(cand_spec)
            or ref_spec.dimensions["num_classes"] != cand_spec.dimensions["num_classes"]):
        raise ConversionError("Reference and candidate must have matching architecture, input shape and classes")
    plan = _evaluation_plan(ref_spec, cand_spec)
    path = Path(data_path).expanduser().resolve()
    if path.is_dir():
        path = path / "manifest.csv"
    if not path.is_file():
        raise ConversionError(f"Evaluation dataset not found: {path}")
    try:
        if path.suffix.lower() == ".csv":
            inputs, labels = _manifest_data(path, ref_spec, max_samples, plan)
        elif path.suffix.lower() in (".npy", ".npz"):
            inputs, labels = _numpy_data(path, ref_spec, max_samples, plan)
        else:
            raise ConversionError("Evaluation data must be .npy, .npz or a feat_bin manifest.csv")
    except ConversionError:
        raise
    except (OSError, ValueError, TypeError, EOFError, KeyError, RuntimeError, zipfile.BadZipFile) as error:
        raise ConversionError(f"Cannot read evaluation data {path}: {error}") from error
    is_quantized = inputs.dtype == np.uint8
    _check_evaluation_count(len(inputs), max_samples, plan)
    batch_size = min(len(inputs), plan["batch_size"])
    ref_model, cand_model = build_model(reference), build_model(candidate)
    squared_error = 0.0
    maximum_error = 0.0
    agreed = ref_correct = cand_correct = 0
    with torch.inference_mode():
        for begin in range(0, len(inputs), batch_size):
            values = inputs[begin:begin + batch_size].astype(np.float32)
            if is_quantized:
                with np.errstate(over="ignore", invalid="ignore"):
                    values = (values - input_zero_point) * np.float32(scale)
            if not np.all(np.isfinite(values)):
                raise ConversionError("Dequantized inputs overflow float32; reduce input_scale")
            tensor = torch.from_numpy(np.ascontiguousarray(values))
            ref_logits = ref_model(tensor).numpy()
            cand_logits = cand_model(tensor).numpy()
            if not np.all(np.isfinite(ref_logits)) or not np.all(np.isfinite(cand_logits)):
                raise ConversionError("CPU inference produced NaN/Inf logits")
            difference = cand_logits.astype(np.float64) - ref_logits.astype(np.float64)
            maximum_error = max(maximum_error, float(np.abs(difference).max()))
            squared_error += float(np.square(difference).sum())
            ref_prediction, cand_prediction = ref_logits.argmax(1), cand_logits.argmax(1)
            agreed += int(np.count_nonzero(ref_prediction == cand_prediction))
            if labels is not None:
                expected = labels[begin:begin + len(values)]
                ref_correct += int(np.count_nonzero(ref_prediction == expected))
                cand_correct += int(np.count_nonzero(cand_prediction == expected))
    count = len(inputs)
    return {"samples": count, "max_abs_error": maximum_error,
            "rmse": math.sqrt(squared_error / (count * ref_spec.dimensions["num_classes"])),
            "prediction_agreement": agreed / count,
            "reference_accuracy": None if labels is None else ref_correct / count,
            "candidate_accuracy": None if labels is None else cand_correct / count,
            "quality_status": ("labeled_samples_only_representativeness_unverified" if labels is not None
                               else "unlabeled_comparison_only_task_accuracy_unknown"),
            "quality_note": "Random/synthetic inputs and prediction agreement do not establish task accuracy. "
                            "Accuracy, when present, applies only to the supplied labeled samples.",
            "device": "cpu", "inference_mode": True, "batch_size": batch_size,
            "cpu_evaluation_budget": dict(plan, estimated_total_macs=count * plan["estimated_macs_per_sample"],
                                          estimated_batch_working_bytes=batch_size * plan["estimated_working_bytes_per_sample"]),
            "input_interpretation": "uint8_dequantized" if is_quantized else "preprocessed_float",
            "elapsed_seconds": time.perf_counter() - started}
