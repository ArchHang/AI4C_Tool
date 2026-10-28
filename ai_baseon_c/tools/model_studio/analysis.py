"""Read-only checkpoint inventory and accounting for the supported deployment graphs.

All operation counts describe one batch and dense theoretical MACs. They are not
device timing estimates. Memory accounting follows the emitter without emitting
files, quantizing matrices, or allocating tensors with activation-sized shapes.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from pathlib import Path

import numpy as np
import torch

from model_converter.emitter import workspace_members, weight_cache_metadata
from model_converter.schema import ConversionError, ModelSpec, parse_spec, positive_dimension
from model_converter.schema import validate_portable_storage

from .compatibility import load_studio_checkpoint


@dataclass
class ModelSession:
    source: Path | None
    sha256: str
    state: dict
    metadata: dict
    spec: ModelSpec | None
    warnings: list = field(default_factory=list)
    changes: list = field(default_factory=list)


def _validate_state(state):
    """Validate inventories too: a failed graph parse must not hide unsafe values."""
    if not isinstance(state, Mapping) or not state:
        raise ConversionError("state_dict must be a nonempty mapping")
    integer_dtypes = (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    for name, value in state.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ConversionError("state_dict must map string keys to tensors only")
        if value.layout != torch.strided or value.device.type == "meta":
            raise ConversionError(f"Tensor {name} must contain dense, materialized strided values")
        if value.is_quantized or value.is_complex():
            raise ConversionError(f"Tensor {name} must contain unquantized real numeric values")
        if not value.is_floating_point() and value.dtype not in integer_dtypes:
            raise ConversionError(f"Tensor {name} has unsupported numeric dtype {value.dtype}")
        if value.numel() == 0:
            raise ConversionError(f"Tensor {name} is empty")
        try:
            finite = bool(torch.isfinite(value.detach()).all().item())
        except (RuntimeError, TypeError, ValueError, NotImplementedError) as error:
            raise ConversionError(f"Tensor {name} cannot be inspected: {error}") from error
        if not finite:
            raise ConversionError(f"Tensor {name} contains NaN/Inf")


def load_session(path, architecture="auto", input_length=None, seq_length=None,
                 trusted_module=False, allow_inventory=True, auto_legacy=True):
    """Use restricted loading; retain unsupported graphs as tensor inventories.

    Malformed tensor values and invalid option values always raise. A supported
    graph's shape/schema rejection is retained verbatim in inventory warnings;
    callers requiring an exportable graph can set ``allow_inventory=False``.
    """
    if architecture not in ("auto", "cnn", "rnn", "lstm"):
        raise ConversionError(f"Unsupported architecture {architecture}")
    for name, value in (("input_length", input_length), ("seq_length", seq_length)):
        if value is not None:
            positive_dimension(value, name, 0x7fffffff)
    state, metadata, digest, notices = load_studio_checkpoint(
        path, trusted_module=trusted_module, auto_legacy=auto_legacy)
    _validate_state(state)
    warnings = list(notices)
    try:
        spec = parse_spec(state, metadata, architecture, input_length, seq_length)
    except ConversionError as error:
        if not allow_inventory:
            raise
        spec = None
        warnings.append(str(error))
    return ModelSession(Path(path).resolve(), digest, state, metadata, spec, warnings)


def _tensor_inventory(name, value):
    """Population statistics scaled to stay finite even for extreme float64 input."""
    detached = value.detach().cpu().resolve_neg()
    try:
        array = detached.to(dtype=torch.float64).numpy()
        minimum, maximum = detached.min().item(), detached.max().item()
        magnitude = max(abs(float(minimum)), abs(float(maximum)))
        if magnitude:
            normalized = array / magnitude
            mean_normalized = float(np.mean(normalized, dtype=np.float64))
            variance = float(np.mean((normalized - mean_normalized) ** 2, dtype=np.float64))
            mean = max(-1.0, min(1.0, mean_normalized)) * magnitude
            std = min(1.0, math.sqrt(max(0.0, variance))) * magnitude
        else:
            mean = std = 0.0
        zeros = int(np.count_nonzero(array == 0))
    except (RuntimeError, TypeError, ValueError, NotImplementedError) as error:
        raise ConversionError(f"Tensor {name} cannot be summarized: {error}") from error
    return {"name": name, "shape": [int(size) for size in value.shape],
            "dtype": str(value.dtype), "elements": int(value.numel()),
            "bytes": int(value.numel() * value.element_size()),
            "zero_fraction": zeros / value.numel(), "min": minimum, "max": maximum,
            "mean": mean, "std": std}


def _is_weight(name):
    return name.endswith("weight") or name in ("w_ih", "w_hh")


def _deployment_memory(spec, precision, cache_policy="none"):
    """Mirror emit_model arrays and the extra constants generated by emit_cnn."""
    validate_portable_storage(spec)
    by_array = {}
    for name, array in spec.arrays.items():
        size = 0
        if precision in ("fp32", "both") or not _is_weight(name):
            size += int(array.nbytes)
        if precision in ("int8", "both") and _is_weight(name):
            size += int(array.size) + int(array.shape[0]) * 4  # int8 weights + FP32 scales
            if name in ("w_ih", "w_hh"):
                size += int(array.shape[0]) * 4  # int32 affine correction row sums
        by_array[name] = size
    helper_bytes = 0
    if spec.architecture == "cnn" and precision in ("fp32", "both"):
        for index, shape in enumerate(spec.dimensions["conv_shapes"]):
            size = int(shape[0]) * 2 * 4  # zeros and identity gamma; each is a separate array
            by_array[f"conv{index}_bn_helpers"] = size
            helper_bytes += size
    members = workspace_members(spec, precision, cache_policy)
    if any(count <= 0 or count > 0x1fffffff for _, _, count in members):
        raise ConversionError("Generated buffer dimensions exceed portable 32-bit address-space bounds")
    type_bytes = {"uint8_t": 1, "int16_t": 2, "float": 4, "int32_t": 4, "int": 4}
    workspace = sum(int(count) * type_bytes[ctype]
                    for ctype, _, count in members)
    if workspace > 0x7fffff00:
        raise ConversionError("Combined workspace exceeds portable signed 32-bit size bounds")
    return by_array, {"parameter_array_bytes": sum(by_array.values()) - helper_bytes,
                      "fp32_bn_helper_bytes": helper_bytes,
                      "workspace_bytes_lower_bound": workspace,
                      "cache_policy": cache_policy,
                      "weight_cache": weight_cache_metadata(spec, precision, cache_policy),
                      "workspace_members": [{"type": ctype, "name": name, "elements": int(count)}
                                            for ctype, name, count in members]}


def _layer(name, op, input_shape, output_shape, parameters=0, weight_bytes=0,
           macs=0, activation_size=4, attributes=None, notes=None, tensors=None):
    return {"name": name, "op": op, "input_shape": list(input_shape),
            "output_shape": list(output_shape), "parameters": int(parameters),
            "weight_bytes": int(weight_bytes), "macs": int(macs),
            "activation_bytes": math.prod(output_shape) * activation_size,
            "attributes": attributes or {}, "notes": notes or [], "tensors": tensors or []}


def _parameter_count(state, names):
    return sum(int(state[name].numel()) for name in names if name in state)


def _cnn_layers(session, precision, payload):
    d, state = session.spec.dimensions, session.state
    activation_size = 1 if precision == "int8" else 4
    layers, weight_names = [], []
    width = d["input_length"]
    for number, (index, shape) in enumerate(zip((0, 4, 8), d["conv_shapes"])):
        outputs, inputs, kernel, convolved, pooled = shape
        original = [name for name in (f"model.{index}.weight", f"model.{index}.bias")
                    if name in state]
        weight_names.append(original[0])
        conv_payload = sum(payload.get(f"conv{number}_{suffix}", 0)
                           for suffix in ("weight", "bias", "bn_helpers"))
        layers.append(_layer(f"model.{index}", "Conv1d", [1, inputs, width],
                             [1, outputs, convolved], _parameter_count(state, original),
                             conv_payload, outputs * convolved * inputs * kernel,
                             activation_size, {"kernel_size": kernel, "stride": 1,
                                               "padding": 0, "groups": 1,
                                               "deployed_batchnorm_folded": True},
                             ["Deployment weights/bias include the following eval BatchNorm.",
                              "INT8 deployment fuses Conv/BN/ReLU/Pool; this is the logical pre-pool output."],
                             original))
        bn_names = [f"model.{index + 1}.{suffix}" for suffix in
                    ("weight", "bias", "running_mean", "running_var", "num_batches_tracked")
                    if f"model.{index + 1}.{suffix}" in state]
        eps = session.metadata.get("bn_eps", [1e-5] * 3)[number]
        layers.append(_layer(f"model.{index + 1}", "BatchNorm1d", [1, outputs, convolved],
                             [1, outputs, convolved], 2 * outputs, activation_size=activation_size,
                             attributes={"eps": float(eps), "folded_into": f"model.{index}"},
                             notes=["Eval BatchNorm is folded into Conv; no separate deployed parameter arrays.",
                                    "Running mean/variance and batch counter are buffers, not trainable parameters."],
                             tensors=bn_names))
        layers.append(_layer(f"model.{index + 2}", "ReLU", [1, outputs, convolved],
                             [1, outputs, convolved], activation_size=activation_size))
        layers.append(_layer(f"model.{index + 3}", "MaxPool1d", [1, outputs, convolved],
                             [1, outputs, pooled], activation_size=activation_size,
                             attributes={"kernel_size": 2, "stride": 2, "ceil_mode": False}))
        width = pooled
    layers.append(_layer("model.12", "Flatten", [1, d["conv_shapes"][-1][0], width],
                         [1, d["flatten_size"]], activation_size=activation_size,
                         attributes={"start_dim": 1, "end_dim": -1},
                         notes=["Shape-only view; activation_bytes is logical size, not a new allocation."]))
    for name, inputs, outputs in (("fc1", d["flatten_size"], d["hidden_size"]),
                                  ("fc2", d["hidden_size"], d["num_classes"])):
        original = [f"{name}.weight", f"{name}.bias"]
        weight_names.append(original[0])
        layers.append(_layer(name, "Linear", [1, inputs], [1, outputs],
                             _parameter_count(state, original),
                             payload[f"{name}_weight"] + payload[f"{name}_bias"],
                             inputs * outputs, 4,
                             {"activation": "none"},
                             (["FC1 has no ReLU; eval-mode Dropout is omitted."] if name == "fc1"
                              else ["Output logits; no softmax in the deployment graph."]), original))
    return layers, weight_names


def _recurrent_layers(session, precision, payload):
    spec, state = session.spec, session.state
    d = spec.dimensions
    i, h, t, classes = (d[key] for key in ("input_size", "hidden_size", "seq_length", "num_classes"))
    gates = 4 if spec.architecture == "lstm" else 1
    prefix = next(key[:-len("weight_ih_l0")] for key in state if key.endswith("weight_ih_l0"))
    head = next(p for p in ("head.", "fc.", "classifier.") if p + "weight" in state)
    original = [prefix + suffix for suffix in ("weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0")]
    weight_names = original[:2] + [head + "weight"]
    recurrent_payload = sum(value for name, value in payload.items() if not name.startswith("out_"))
    layers = [_layer(prefix.rstrip(".") or "recurrent", spec.architecture.upper(),
                     [1, t, i], [1, t, h], _parameter_count(state, original), recurrent_payload,
                     t * gates * h * (i + h), 1 if precision == "int8" else 4,
                     {"gates": gates, "hidden_size": h, "seq_length": t, "layers": 1,
                      "bidirectional": False, "initial_state": "zero", "gate_order": "ifgo" if gates == 4 else None,
                      "layout": "[time][feature], batch=1"},
                     ["Dense input and recurrent matrix MACs counted at every time step.",
                      "Bias additions, sigmoid/tanh and elementwise gate work are excluded from MAC FLOPs.",
                      "The head consumes only the final hidden state."], original)]
    head_tensors = [head + "weight", head + "bias"]
    layers.append(_layer(head.rstrip("."), "Linear", [1, h], [1, classes],
                         _parameter_count(state, head_tensors),
                         payload["out_weight"] + payload["out_bias"], h * classes, 4,
                         {"activation": "none", "input_time_step": t - 1},
                         ["Output logits from the final hidden state."], head_tensors))
    return layers, weight_names


def _json_safe(value):
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (str, bool, int)):
        return value
    return str(value)


def analyze_session(session, precision="int8", weight_scaling="maxabs", cache_policy="none"):
    """Return JSON-safe graph/inventory data without exporting or running inference.

    ``weight_bytes`` is deployed constant-array payload assigned to a logical
    layer, including bias/scales/row sums and any FP32 Conv helper constants.
    ``activation_bytes`` is a logical output size, even for folded/fused/views.
    The maximum is over logical layer outputs (including logits), omitting input.
    Neither value is a peak-liveness or exact ABI memory measurement.
    """
    if precision not in ("int8", "fp32", "both"):
        raise ConversionError("precision must be int8, fp32 or both")
    if weight_scaling not in ("maxabs", "mse"):
        raise ConversionError("weight_scaling must be maxabs or mse")
    if cache_policy not in ("none", "packed"):
        raise ConversionError("cache_policy must be none or packed")
    _validate_state(session.state)
    tensors = [_tensor_inventory(name, value) for name, value in session.state.items()]
    summary = {"total_parameters": None, "state_elements": sum(item["elements"] for item in tensors),
               "state_bytes": sum(item["bytes"] for item in tensors),
               "deployed_parameter_bytes": None, "workspace_bytes_lower_bound": None,
               "macs": None, "mac_flops": None, "max_activation_bytes": None,
               "weight_sparsity": None}
    warnings = [str(warning) for warning in session.warnings]
    layers, deployment = [], {"precision": precision, "weight_scaling": weight_scaling,
                              "cache_policy": cache_policy}
    if session.spec is not None:
        payload, memory = _deployment_memory(session.spec, precision, cache_policy)
        layers, weight_names = (_cnn_layers(session, precision, payload)
                                if session.spec.architecture == "cnn" else
                                _recurrent_layers(session, precision, payload))
        by_name = {item["name"]: item for item in tensors}
        weight_elements = sum(by_name[name]["elements"] for name in weight_names)
        weight_zeros = sum(int(torch.count_nonzero(session.state[name] == 0).item())
                           for name in weight_names)
        summary.update(total_parameters=sum(layer["parameters"] for layer in layers),
                       deployed_parameter_bytes=sum(payload.values()),
                       workspace_bytes_lower_bound=memory["workspace_bytes_lower_bound"],
                       macs=sum(layer["macs"] for layer in layers),
                       max_activation_bytes=max(layer["activation_bytes"] for layer in layers),
                       weight_sparsity=weight_zeros / weight_elements)
        summary["mac_flops"] = 2 * summary["macs"]
        deployment.update(memory)
        deployment["array_payload_by_parameter"] = payload
        warnings.extend(("MACs are theoretical dense counts for batch=1, not measured device latency. "
                         "MAC FLOPs = 2 * MACs; non-MAC operations are excluded.",
                         "Deployed bytes count constant array payload only, not full Flash/ELF size. "
                         "Workspace is the sum of emitter members assuming 1-byte uint8, 2-byte int16 and "
                         "4-byte float/int32/int; target ABI padding/alignment is excluded.",
                         "Activation bytes describe logical layer outputs; fused/folded/view outputs "
                         "do not imply separate allocation. The maximum is not peak live RAM.",
                         "Weight sparsity counts zeros in original Conv/Linear/recurrent matrices; "
                         "BatchNorm affine vectors and biases are excluded."))
    else:
        warnings.append("Tensor inventory only: graph structure, trainable parameters, MACs and deployment memory are unknown.")
    result = {"architecture": session.spec.architecture if session.spec else "unknown",
              "schema": session.spec.schema if session.spec else None,
              "supported": session.spec is not None, "summary": summary, "layers": layers,
              "tensors": tensors, "warnings": warnings,
              "dimensions": session.spec.dimensions if session.spec else {},
              "source": {"path": str(session.source) if session.source is not None else None,
                         "sha256": session.sha256}, "changes": session.changes,
              "deployment": deployment}
    return _json_safe(result)
