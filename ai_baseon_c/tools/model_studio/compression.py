"""Physical pruning of the explicitly supported converter graphs."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import math
import os
from pathlib import Path
import tempfile

import torch

from model_converter.schema import ConversionError, parse_spec


def _validated_spec(session):
    if session.spec is None:
        raise ConversionError("Pruning/evaluation requires a supported CNN/RNN/LSTM graph")
    dimensions = session.spec.dimensions
    return parse_spec(session.state, session.metadata, session.spec.architecture,
                      input_length=dimensions.get("input_length"),
                      seq_length=dimensions.get("seq_length"))


def _metadata(session, spec):
    metadata = deepcopy(session.metadata)
    metadata["schema"] = spec.schema
    if spec.architecture == "cnn":
        metadata["input_length"] = spec.dimensions["input_length"]
    else:
        metadata["seq_length"] = spec.dimensions["seq_length"]
        metadata["nonlinearity"] = "tanh"
    return metadata


def _select(scores, amount):
    keep = max(1, scores.numel() - math.floor(scores.numel() * amount))
    # Original order is retained after selecting the largest L1 scores. Stable
    # sorting makes ties reproducible, including all-zero trained layers.
    return torch.sort(torch.argsort(scores, descending=True, stable=True)[:keep]).values


def _slice(state, key, indices, axis=0):
    if key in state:
        state[key] = state[key].index_select(axis, indices).clone()


def _structured_cnn(state, spec, amount, scores=None):
    units = {}
    previous = None
    for index in (0, 4, 8):
        key = f"model.{index}.weight"
        if previous is not None:
            _slice(state, key, previous, 1)
        original = state[key].shape[0]
        score = (state[key].double().abs().sum(dim=(1, 2)) if scores is None
                 else scores[f"conv{index // 4}"])
        selected = _select(score, amount)
        _slice(state, key, selected)
        _slice(state, f"model.{index}.bias", selected)
        for suffix in ("weight", "bias", "running_mean", "running_var"):
            _slice(state, f"model.{index + 1}.{suffix}", selected)
        units[f"conv{index // 4}_channels"] = {"before": original,
            "kept": selected.numel(), "removed": original - selected.numel(),
            "selected_indices": selected.tolist()}
        previous = selected
    pooled_width = spec.dimensions["conv_shapes"][-1][-1]
    columns = (previous[:, None] * pooled_width + torch.arange(pooled_width)).reshape(-1)
    _slice(state, "fc1.weight", columns, 1)
    original = state["fc1.weight"].shape[0]
    selected = _select(state["fc1.weight"].double().abs().sum(dim=1)
                       if scores is None else scores["fc1"], amount)
    _slice(state, "fc1.weight", selected)
    _slice(state, "fc1.bias", selected)
    _slice(state, "fc2.weight", selected, 1)
    units["fc1_neurons"] = {"before": original, "kept": selected.numel(),
                             "removed": original - selected.numel(),
                             "selected_indices": selected.tolist()}
    return units


def _structured_recurrent(state, spec, amount, scores=None):
    prefix = next(key[:-len("weight_ih_l0")] for key in state if key.endswith("weight_ih_l0"))
    head = next(prefix for prefix in ("head.", "fc.", "classifier.") if prefix + "weight" in state)
    hidden = spec.dimensions["hidden_size"]
    gates = 4 if spec.architecture == "lstm" else 1
    w_ih = state[prefix + "weight_ih_l0"].double().abs().reshape(gates, hidden, -1)
    w_hh = state[prefix + "weight_hh_l0"].double().abs().reshape(gates, hidden, hidden)
    if scores is None:
        scores = (w_ih.sum(dim=(0, 2)) + w_hh.sum(dim=(0, 2)) + w_hh.sum(dim=(0, 1))
                  + state[head + "weight"].double().abs().sum(dim=0))
    else:
        scores = scores["hidden"]
    selected = _select(scores, amount)
    rows = torch.cat([selected + gate * hidden for gate in range(gates)])
    for suffix in ("weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0"):
        _slice(state, prefix + suffix, rows)
    _slice(state, prefix + "weight_hh_l0", selected, 1)
    _slice(state, head + "weight", selected, 1)
    return {"hidden_units": {"before": hidden, "kept": selected.numel(),
                              "removed": hidden - selected.numel(),
                              "selected_indices": selected.tolist()}}


def _weights(state):
    return [(key, value) for key, value in state.items() if "weight" in key and value.ndim >= 2]


def prune_session(session, amount, method="structured", *, data_path=None,
                  max_samples=128, input_scale=1 / 255, input_zero_point=0):
    """Return an independent session; amount is the fraction removed in [0, 1]."""
    if isinstance(amount, bool):
        raise ConversionError("Pruning amount must be a finite number in [0, 1]")
    try:
        amount = float(amount)
    except (TypeError, ValueError, OverflowError) as error:
        raise ConversionError("Pruning amount must be a finite number in [0, 1]") from error
    if not math.isfinite(amount) or not 0 <= amount <= 1:
        raise ConversionError("Pruning amount must be a finite number in [0, 1]")
    if method not in ("structured", "unstructured", "fpgm", "taylor", "svd", "reconstruction"):
        raise ConversionError("Pruning method must be structured, unstructured, fpgm, taylor, svd or reconstruction")
    spec = _validated_spec(session)
    if method in ("fpgm", "svd"):
        from .advanced_pruning import validate_advanced_budget
        validate_advanced_budget(spec, method, amount)
    elif method == "reconstruction":
        from .reconstruction_pruning import validate_reconstruction_budget
        validate_reconstruction_budget(spec, amount)
    state = {key: value.detach().cpu().contiguous().clone() for key, value in session.state.items()}
    before = sum(value.numel() for _, value in _weights(state))
    details = {}
    calibration = None
    if method == "reconstruction":
        from .reconstruction_pruning import reconstruct_channels
        units, details = reconstruct_channels(
            replace(session, spec=spec), state, amount, data_path, max_samples,
            input_scale, input_zero_point)
        sparse = "Convolution channels and the affine head are physically reduced; exported dense arrays and MACs can decrease."
    elif method == "svd":
        from .advanced_pruning import compress_linear_head
        units, details = compress_linear_head(state, spec, amount)
        sparse = "The two linear CNN heads are refactorized; dense arrays and MACs can decrease."
    elif method in ("structured", "fpgm", "taylor"):
        scores = None
        if method == "fpgm" and amount > 0:
            from .advanced_pruning import geometric_scores
            scores, details = geometric_scores(state, spec)
        elif method == "fpgm":
            details = {"criterion": "FPGM; zero ratio is identity without distance calculation"}
        elif method == "taylor":
            from .advanced_pruning import taylor_scores
            scores, calibration, details = taylor_scores(
                session, data_path, max_samples, input_scale, input_zero_point,
                skip=amount == 0)
        units = (_structured_cnn(state, spec, amount, scores) if spec.architecture == "cnn"
                 else _structured_recurrent(state, spec, amount, scores))
        sparse = "Physical dimensions are reduced; exported dense arrays and MACs can decrease."
    else:
        weights = _weights(state)
        scores = torch.cat([value.double().abs().reshape(-1) for _, value in weights])
        selected = torch.argsort(scores, stable=True)[:math.floor(before * amount)]
        offset = 0
        for _, value in weights:
            local = selected[(selected >= offset) & (selected < offset + value.numel())] - offset
            value.reshape(-1)[local] = 0
            offset += value.numel()
        units = {"weights": {"before": before, "kept": before - selected.numel(),
                              "removed": selected.numel()}}
        sparse = ("Unstructured zeros keep all tensor shapes. The current dense C kernels still "
                  "execute the same MACs and allocate the same arrays; no dense size/speed gain is claimed.")
    metadata = _metadata(session, spec)
    new_spec = parse_spec(state, metadata, spec.architecture)
    after = sum(value.numel() for _, value in _weights(state))
    report = {"method": method, "amount": amount, "units": units,
              "weights_before": before, "weights_after": after,
              "kept": sum(item["kept"] for item in units.values()),
              "removed": sum(item["removed"] for item in units.values()),
              "zero_weights_after": sum(int((value == 0).sum()) for _, value in _weights(state)),
              "quality_status": "requires_validation", "quality_note":
              "Pruning has not established task accuracy; evaluate representative labeled data and retrain if needed.",
              "dense_execution_note": sparse}
    changes = deepcopy(session.changes) + [deepcopy(report)]
    result = replace(session, state=state, metadata=metadata, spec=new_spec,
                     warnings=deepcopy(session.warnings), changes=changes)
    if method == "taylor" and amount > 0:
        from .advanced_pruning import reconstruct_head
        details["head_reconstruction"] = reconstruct_head(session, result, calibration)
        result.spec = parse_spec(result.state, result.metadata, spec.architecture)
    report.update(details)
    report["zero_weights_after"] = sum(int((value == 0).sum()) for _, value in _weights(result.state))
    result.changes[-1] = deepcopy(report)
    return result, report


def _safe_metadata(value, location="metadata"):
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, dict) and all(type(key) is str for key in value):
        return {key: _safe_metadata(item, f"{location}.{key}") for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_metadata(item, location) for item in value]
    raise ConversionError(f"{location} must contain only finite primitive checkpoint metadata")


def save_checkpoint(session, path, overwrite=False):
    """Atomically save tensors plus primitive metadata, never over the source file."""
    if not isinstance(overwrite, bool):
        raise ConversionError("overwrite must be a boolean")
    spec = _validated_spec(session)
    destination = Path(path).expanduser().resolve()
    if session.source is not None and destination == Path(session.source).resolve():
        raise ConversionError("The original source checkpoint must never be overwritten")
    if destination.exists():
        if (session.source is not None and Path(session.source).exists()
                and os.path.samefile(destination, session.source)):
            raise ConversionError("The original source checkpoint must never be overwritten")
        if not overwrite:
            raise ConversionError(f"Destination already exists (overwrite is disabled): {destination}")
        if not destination.is_file():
            raise ConversionError(f"Checkpoint destination is not a file: {destination}")
    inherited_history = session.metadata.get("studio_history", [])
    if (not isinstance(inherited_history, list)
            or not all(isinstance(item, dict) for item in inherited_history)
            or not isinstance(session.changes, list)
            or not all(isinstance(item, dict) for item in session.changes)):
        raise ConversionError("studio_history and current changes must be lists of primitive metadata dictionaries")
    metadata = _safe_metadata(_metadata(session, spec))
    # An imported checkpoint remains the new baseline. Preserve prior changes
    # as provenance, rather than restoring them as edits to that baseline.
    # This also retains calibration input hashes for leakage checks after reload.
    metadata["studio_history"] = _safe_metadata(
        inherited_history + session.changes, "metadata.studio_history")
    checkpoint = {"state_dict": {key: value.detach().cpu().clone() for key, value in session.state.items()},
                  "metadata": metadata}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".model-studio-", suffix=".tmp",
                                         dir=destination.parent, delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(checkpoint, stream)
            stream.flush()
            os.fsync(stream.fileno())
        digest = hashlib.sha256()
        with temporary.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        size = temporary.stat().st_size
        if overwrite:
            os.replace(temporary, destination)
        elif os.name == "nt":
            # Windows rename is atomic and fails if the destination exists.
            os.rename(temporary, destination)
        else:
            # A hard link publishes the fully written file with exclusive-create
            # semantics, so a concurrent writer cannot be silently overwritten.
            os.link(temporary, destination)
        return {"path": str(destination), "sha256": digest.hexdigest(), "bytes": size,
                "schema": spec.schema, "format": "state_dict_and_metadata"}
    except (OSError, RuntimeError, ValueError) as error:
        raise ConversionError(f"Cannot save checkpoint {destination}: {error}") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
