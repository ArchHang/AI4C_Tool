"""Validate only graphs whose semantics are explicitly supported by the C runtime."""

from dataclasses import dataclass

import numpy as np
import torch

from .quantization import fold_batch_norm


class ConversionError(ValueError):
    """An unsupported or inconsistent checkpoint, with an actionable diagnostic."""


@dataclass
class ModelSpec:
    architecture: str
    schema: str
    dimensions: dict
    arrays: dict
    metadata: dict


def validate_portable_storage(spec):
    """Reject graphs that cannot satisfy the runtime's 32-bit object-size guards.

    Every export has a float-input entry point, including int8 exports. Checking
    workspace members alone misses an oversized input whose hidden workspace is
    small. The recurrent runtime also bounds its matrices in float element units.
    """
    float_limit = 0x7fffffff // np.dtype(np.float32).itemsize
    dimensions = spec.dimensions
    counts = {"float input": dimensions["input_elements"],
              "logits": dimensions["num_classes"]}
    if spec.architecture == "cnn":
        for index, (outputs, inputs, kernel, width, pooled) in enumerate(dimensions["conv_shapes"]):
            counts[f"conv{index} output"] = outputs * width
            counts[f"conv{index} weights"] = outputs * inputs * kernel
    else:
        hidden, inputs = dimensions["hidden_size"], dimensions["input_size"]
        gates = 4 if spec.architecture == "lstm" else 1
        counts.update(hidden_sequence=hidden * dimensions["seq_length"],
                      input_weights=gates * hidden * inputs,
                      recurrent_weights=gates * hidden * hidden,
                      output_weights=hidden * dimensions["num_classes"])
        if spec.architecture == "lstm":
            counts["LSTM work"] = 6 * hidden
    counts.update((name, array.size) for name, array in spec.arrays.items())
    for name, count in counts.items():
        if count <= 0 or count > float_limit:
            raise ConversionError(f"{name} requires {count} elements; portable 32-bit runtime "
                                  f"supports at most {float_limit} float elements per object")


def tensor_array(state, key, shape=None):
    if key not in state:
        raise ConversionError(f"Missing tensor: {key}")
    value = state[key]
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise ConversionError(f"Tensor {key} must be floating point")
    if value.layout != torch.strided or value.device.type == "meta":
        raise ConversionError(f"Tensor {key} must contain dense, materialized strided values")
    try:
        array = value.detach().cpu().to(dtype=torch.float32).resolve_conj().resolve_neg().numpy().copy()
    except (RuntimeError, TypeError, ValueError) as error:
        raise ConversionError(f"Tensor {key} cannot be represented as a float32 array: {error}") from error
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ConversionError(f"Tensor {key} is empty or contains NaN/Inf")
    if shape is not None and array.shape != tuple(shape):
        raise ConversionError(f"Tensor {key}: expected shape {tuple(shape)}, got {array.shape}")
    return array


def positive_dimension(value, name, maximum=32896):
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or not 0 < value <= maximum:
        raise ConversionError(f"{name} must be an integer in [1, {maximum}]")
    return int(value)


def require_keys(state, required, optional=()):
    missing = set(required) - set(state)
    extra = set(state) - set(required) - set(optional)
    if missing or extra:
        raise ConversionError(f"Unsupported state_dict keys: missing={sorted(missing)}, "
                              f"extra={sorted(extra)}; stacked/bidirectional/custom graphs "
                              "require a separate supported schema")


def cnn_spec(state, metadata, input_length=None):
    if metadata.get("schema", "cnn_voice_v1") != "cnn_voice_v1":
        raise ConversionError("Only schema=cnn_voice_v1 is supported for CNN")
    required = []
    optional = []
    for index in (0, 4, 8):
        required.append(f"model.{index}.weight")
        optional.append(f"model.{index}.bias")
        bn = index + 1
        required.extend(f"model.{bn}.{suffix}" for suffix in
                        ("weight", "bias", "running_mean", "running_var"))
        optional.append(f"model.{bn}.num_batches_tracked")
    required.extend(("fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"))
    require_keys(state, required, optional)
    length = positive_dimension(metadata.get("input_length", 2048) if input_length is None else input_length,
                                "input_length", 0x7fffffff)
    first_weight = tensor_array(state, "model.0.weight")
    if first_weight.ndim != 3:
        raise ConversionError("CNN weights must have shape [output][input][kernel]")
    input_channels = positive_dimension(first_weight.shape[1], "input_channels")
    channels = input_channels
    arrays = {}
    widths = [length]
    conv_shapes = []
    epsilons = metadata.get("bn_eps", [1e-5] * 3)
    if not isinstance(epsilons, (list, tuple)) or len(epsilons) != 3:
        raise ConversionError("bn_eps metadata must contain three finite positive values")
    for layer, index in enumerate((0, 4, 8)):
        weight = tensor_array(state, f"model.{index}.weight")
        if weight.ndim != 3 or weight.shape[1] != channels or weight.shape[2] != 5:
            raise ConversionError("cnn_voice_v1 requires three valid stride-1 Conv1d kernels of width 5")
        output_channels = positive_dimension(weight.shape[0], "conv output channels")
        if channels * 5 > 32896:
            raise ConversionError("Conv dot product exceeds int32 MAC safety limit 32896")
        try:
            epsilon = float(epsilons[layer])
            with np.errstate(over="raise", invalid="raise"):
                stored_epsilon = np.float32(epsilon)
        except (TypeError, ValueError, OverflowError, FloatingPointError) as error:
            raise ConversionError("BatchNorm epsilon must be a finite positive float32 value") from error
        if (isinstance(epsilons[layer], bool) or not np.isfinite(stored_epsilon)
                or stored_epsilon <= 0):
            raise ConversionError("BatchNorm epsilon must be a finite positive float32 value")
        bias = (tensor_array(state, f"model.{index}.bias", (output_channels,))
                if f"model.{index}.bias" in state else np.zeros(output_channels, np.float32))
        bn = index + 1
        gamma, beta, mean, variance = [tensor_array(state, f"model.{bn}.{key}",
                                                  (output_channels,)) for key in
                                        ("weight", "bias", "running_mean", "running_var")]
        if np.any(variance < 0):
            raise ConversionError("BatchNorm running variance must be nonnegative")
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                arrays[f"conv{layer}_weight"], arrays[f"conv{layer}_bias"] = fold_batch_norm(
                    weight, bias, gamma, beta, mean, variance, stored_epsilon)
        except FloatingPointError as error:
            raise ConversionError("BatchNorm folding overflowed or produced invalid float32 arithmetic") from error
        if not all(np.all(np.isfinite(arrays[f"conv{layer}_{key}"])) for key in ("weight", "bias")):
            raise ConversionError("BatchNorm folding overflowed float32")
        width = widths[-1] - 4
        if width < 2:
            raise ConversionError("Input length is too short for three conv5 + pool2 blocks")
        conv_shapes.append([output_channels, channels, 5, width, width // 2])
        widths.append(width // 2)
        channels = output_channels
    flat_size = channels * widths[-1]
    hidden_weight = tensor_array(state, "fc1.weight")
    if hidden_weight.ndim != 2 or hidden_weight.shape[1] != flat_size:
        raise ConversionError(f"fc1 input must equal channels*pooled_width={flat_size}; "
                              "set --input-length to the actual graph input, not a flatten guess")
    hidden = positive_dimension(hidden_weight.shape[0], "FC hidden size")
    classes_weight = tensor_array(state, "fc2.weight")
    if classes_weight.ndim != 2 or classes_weight.shape[1] != hidden:
        raise ConversionError("fc2 weight must have shape [classes, fc1 outputs]")
    classes = positive_dimension(classes_weight.shape[0], "num_classes")
    if flat_size > 32896:
        raise ConversionError("FC dot product exceeds int32 MAC safety limit 32896")
    arrays.update(fc1_weight=hidden_weight, fc1_bias=tensor_array(state, "fc1.bias", (hidden,)),
                  fc2_weight=classes_weight, fc2_bias=tensor_array(state, "fc2.bias", (classes,)))
    return ModelSpec("cnn", "cnn_voice_v1", {"input_channels": input_channels,
                     "input_length": length, "input_elements": input_channels * length,
                     "num_classes": classes, "hidden_size": hidden, "flatten_size": flat_size,
                     "conv_shapes": conv_shapes}, arrays, dict(metadata))


def recurrent_spec(state, metadata, architecture, seq_length=None, explicit=False):
    supported_schema = "rnn_tanh_v1" if architecture == "rnn" else "lstm_ifgo_v1"
    if metadata.get("schema", supported_schema) != supported_schema:
        raise ConversionError(f"{architecture} requires schema={supported_schema}")
    # Plain state_dict cannot distinguish tanh from relu RNN, so auto requires schema metadata.
    if architecture == "rnn" and not explicit and metadata.get("schema") != "rnn_tanh_v1":
        raise ConversionError("RNN state_dict does not encode its activation: declare "
                              "--architecture rnn for the supported tanh graph or save schema=rnn_tanh_v1")
    if metadata.get("nonlinearity", "tanh") != "tanh":
        raise ConversionError("Only tanh RNN is supported")
    prefix = next((key[:-len("weight_ih_l0")] for key in state if key.endswith("weight_ih_l0")), None)
    if prefix is None:
        raise ConversionError("Missing recurrent weight_ih_l0")
    head_prefix = next((p for p in ("head.", "fc.", "classifier.")
                        if p + "weight" in state), None)
    if head_prefix is None:
        raise ConversionError("Supported recurrent classifiers require head.weight/head.bias (or fc/classifier)")
    recurrent = [prefix + name for name in ("weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0")]
    require_keys(state, recurrent + [head_prefix + "weight", head_prefix + "bias"])
    w_ih = tensor_array(state, recurrent[0])
    w_hh = tensor_array(state, recurrent[1])
    if w_ih.ndim != 2 or w_hh.ndim != 2:
        raise ConversionError("Recurrent weights must be matrices")
    hidden = positive_dimension(w_hh.shape[1], "hidden_size")
    gates = 4 if architecture == "lstm" else 1
    if w_ih.shape[0] != gates * hidden or w_hh.shape[0] != gates * hidden:
        raise ConversionError(f"{architecture} requires {gates}*hidden rows; stacked, projection "
                              "and bidirectional graphs are unsupported")
    input_size = positive_dimension(w_ih.shape[1], "input_size")
    seq = positive_dimension(metadata.get("seq_length") if seq_length is None else seq_length,
                             "seq_length", 0x7fffffff)
    out_w = tensor_array(state, head_prefix + "weight")
    if out_w.ndim != 2 or out_w.shape[1] != hidden:
        raise ConversionError("Classifier input must equal hidden_size")
    classes = positive_dimension(out_w.shape[0], "num_classes")
    arrays = {"w_ih": w_ih, "w_hh": w_hh, "out_weight": out_w,
              "out_bias": tensor_array(state, head_prefix + "bias", (classes,))}
    b_ih = tensor_array(state, recurrent[2], (gates * hidden,))
    b_hh = tensor_array(state, recurrent[3], (gates * hidden,))
    if architecture == "lstm":
        arrays["bias"] = (b_ih + b_hh).astype(np.float32)
        if not np.all(np.isfinite(arrays["bias"])):
            raise ConversionError("Merged LSTM bias overflowed float32")
    else:
        arrays.update(b_ih=b_ih, b_hh=b_hh)
    return ModelSpec(architecture, f"{architecture}_{'tanh' if gates == 1 else 'ifgo'}_v1",
                     {"input_size": input_size, "hidden_size": hidden, "seq_length": seq,
                      "input_elements": input_size * seq, "num_classes": classes}, arrays,
                     dict(metadata))


def parse_spec(state, metadata, architecture="auto", input_length=None, seq_length=None):
    if architecture not in ("auto", "cnn", "rnn", "lstm"):
        raise ConversionError(f"Unsupported architecture {architecture}")
    if architecture == "auto":
        if "model.0.weight" in state:
            architecture = "cnn"
        else:
            hh = next((value for key, value in state.items() if key.endswith("weight_hh_l0")), None)
            if hh is None or hh.ndim != 2 or hh.shape[1] == 0:
                raise ConversionError("Cannot identify a supported CNN/RNN/LSTM state_dict")
            ratio = hh.shape[0] // hh.shape[1]
            architecture = {1: "rnn", 4: "lstm"}.get(ratio)
            if architecture is None:
                raise ConversionError("Unsupported recurrent gate dimensions")
        explicit = False
    else:
        explicit = True
    if architecture == "cnn":
        spec = cnn_spec(state, metadata, input_length)
    else:
        spec = recurrent_spec(state, metadata, architecture, seq_length, explicit)
    validate_portable_storage(spec)
    return spec
