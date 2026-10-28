"""Bounded, schema-aware structured compression; no sparse runtime is required.

FPGM uses the sum-of-distances criterion on BN-folded filters. Taylor uses an
absolute first-order parameter sensitivity bound for known dependency groups,
with teacher pseudo-labels only. It is an adaptation, not a reproduction of a
paper's complete training schedule. A small ridge fit then reconstructs logits.
"""

import hashlib
import math
from pathlib import Path
import zipfile

import numpy as np
import torch
from torch.nn import functional as F

from model_converter.schema import ConversionError


MAX_CALIBRATION_SAMPLES = 128
MAX_DISTANCE_ELEMENTS = 4 * 1024 * 1024
MAX_DISTANCE_OPERATIONS = 200000000
MAX_FIT_ELEMENTS = 4 * 1024 * 1024
MAX_SVD_OPERATIONS = 200000000
MAX_FIT_OPERATIONS = 200000000
MAX_ADVANCED_WORKING_BYTES = 256 * 1024 * 1024


def _distance_dimensions(spec):
    dimensions = spec.dimensions
    if spec.architecture == "cnn":
        shapes = [(outputs, channels * kernel + 1)
                  for outputs, channels, kernel, _, _ in dimensions["conv_shapes"]]
        shapes.append((dimensions["hidden_size"], dimensions["flatten_size"] + 1))
        return shapes
    hidden = dimensions["hidden_size"]
    gates = 4 if spec.architecture == "lstm" else 1
    return [(hidden, gates * dimensions["input_size"] + 2 * gates * hidden
             + dimensions["num_classes"])]


def validate_advanced_budget(spec, method, amount):
    """Check matrix dimensions before cloning/casting or forming dense products."""
    if method == "svd" and spec.architecture != "cnn":
        raise ConversionError("Linear-head SVD requires cnn_voice_v1: FC1 and FC2 have no intervening activation")
    if amount == 0:
        return
    if method == "fpgm":
        for count, width in _distance_dimensions(spec):
            if (count * count > MAX_DISTANCE_ELEMENTS
                    or count * count * width > MAX_DISTANCE_OPERATIONS
                    or 8 * (3 * count * width + count * count) > MAX_ADVANCED_WORKING_BYTES):
                raise ConversionError("FPGM pairwise distance CPU/memory budget exceeded; use L1 or a smaller model")
    elif method == "svd":
        dimensions = spec.dimensions
        hidden, classes, width = (dimensions["hidden_size"], dimensions["num_classes"],
                                  dimensions["flatten_size"])
        small, large = min(classes, width), max(classes, width)
        operations = classes * hidden * width + small * small * large
        # Input FP64 casts, effective matrix, economy SVD factors/output and
        # additional solver scratch allowance. This is admission accounting,
        # not a guarantee about a vendor LAPACK allocator's peak RSS.
        working = 8 * (hidden * width + classes * hidden + 4 * classes * width
                       + 4 * small * small + hidden + classes)
        if (classes * width > MAX_FIT_ELEMENTS or operations > MAX_SVD_OPERATIONS
                or working > MAX_ADVANCED_WORKING_BYTES):
            raise ConversionError("Linear-head SVD CPU/memory budget exceeded")


def _prefixes(state):
    recurrent = next(key[:-len("weight_ih_l0")] for key in state if key.endswith("weight_ih_l0"))
    head = next(prefix for prefix in ("head.", "fc.", "classifier.") if prefix + "weight" in state)
    return recurrent, head


def _distance_score(vectors):
    count, width = vectors.shape
    if count * count > MAX_DISTANCE_ELEMENTS or count * count * width > MAX_DISTANCE_OPERATIONS:
        raise ConversionError("FPGM pairwise distance CPU budget exceeded; use L1 or a smaller model")
    distances = torch.cdist(vectors.double(), vectors.double(), p=2,
                            compute_mode="donot_use_mm_for_euclid_dist")
    scores = distances.sum(1)
    if not bool(torch.isfinite(scores).all()):
        raise ConversionError("FPGM scores overflowed; inspect the model weights")
    return scores


def geometric_scores(state, spec):
    validate_advanced_budget(spec, "fpgm", 1)
    scores = {}
    if spec.architecture == "cnn":
        for layer, index in enumerate((0, 4, 8)):
            # Compare deployed filters, so raw convolution scale hidden by BN
            # cannot reverse importance. Include the folded bias as a feature.
            weight = torch.from_numpy(spec.arrays[f"conv{layer}_weight"]).double().flatten(1)
            bias = torch.from_numpy(spec.arrays[f"conv{layer}_bias"]).double()[:, None]
            scores[f"conv{layer}"] = _distance_score(torch.cat((weight, bias), 1))
        scores["fc1"] = _distance_score(torch.cat((state["fc1.weight"].double(),
                                                   state["fc1.bias"].double()[:, None]), 1))
    else:
        prefix, head = _prefixes(state)
        hidden = spec.dimensions["hidden_size"]
        gates = 4 if spec.architecture == "lstm" else 1
        ih = state[prefix + "weight_ih_l0"].double().reshape(gates, hidden, -1)
        hh = state[prefix + "weight_hh_l0"].double().reshape(gates, hidden, hidden)
        # Dependency-group extension: keep IFGO units tied, and account for
        # incoming rows, recurrent columns and the classifier column together.
        vectors = torch.cat((ih.permute(1, 0, 2).reshape(hidden, -1),
                             hh.permute(1, 0, 2).reshape(hidden, -1),
                             hh.permute(2, 0, 1).reshape(hidden, -1),
                             state[head + "weight"].double().t()), 1)
        scores["hidden"] = _distance_score(vectors)
    return scores, {"criterion": "FPGM sum of pairwise Euclidean distances; remove central redundant groups",
                    "bn_folded": spec.architecture == "cnn",
                    "algorithm_scope": "One-shot criterion adaptation with known schema dependencies; no paper training schedule",
                    "reference": "https://openaccess.thecvf.com/content_CVPR_2019/html/He_Filter_Pruning_via_Geometric_Median_for_Deep_Convolutional_Neural_Networks_CVPR_2019_paper.html"}


def compress_linear_head(state, spec, amount):
    validate_advanced_budget(spec, "svd", amount)
    hidden = spec.dimensions["hidden_size"]
    classes, width = spec.dimensions["num_classes"], spec.dimensions["flatten_size"]
    if amount == 0:
        return {"fc1_neurons": {"before": hidden, "kept": hidden, "removed": 0}}, {
            "criterion": "Linear-head SVD; zero ratio is identity", "rank": hidden,
            "exact_linear_rank_retained": True}
    rank = min(max(1, hidden - math.floor(hidden * amount)), classes, width)
    small = min(classes, width)
    # FC1 -> dropout(eval identity) -> FC2 is affine. Use double while forming
    # and factorizing the effective map, then store normal FP32 model tensors.
    effective = state["fc2.weight"].double() @ state["fc1.weight"].double()
    bias = state["fc2.weight"].double() @ state["fc1.bias"].double() + state["fc2.bias"].double()
    u, singular, vh = torch.linalg.svd(effective, full_matrices=False)
    root = torch.sqrt(singular[:rank])
    state["fc1.weight"] = (root[:, None] * vh[:rank]).float().contiguous()
    state["fc1.bias"] = torch.zeros(rank, dtype=torch.float32)
    state["fc2.weight"] = (u[:, :rank] * root[None, :]).float().contiguous()
    state["fc2.bias"] = bias.float()
    total_energy = float(torch.square(singular).sum())
    discarded_energy = float(torch.square(singular[rank:]).sum())
    return {"fc1_neurons": {"before": hidden, "kept": rank, "removed": hidden - rank}}, {
        "criterion": "SVD of the effective affine FC2(FC1(x)); balanced square-root singular factors",
        "rank": rank, "effective_rank_limit": small,
        "rank_semantics": "Keep min(max(1,H-floor(H*amount)),classes,flatten_size); amount=0 is identity",
        "exact_linear_rank_retained": rank >= min(hidden, small),
        "discarded_singular_energy_fraction": discarded_energy / total_energy if total_energy else 0.0,
        "quality_note": "Full effective rank preserves the FP32 affine function up to rounding; truncated rank and INT8 still require validation.",
        "algorithm_scope": "Only this schema's consecutive affine heads; not valid across an activation or training dropout"}


def _calibration(session, data_path, max_samples, input_scale, input_zero_point):
    from .validation import (_validated_spec, _evaluation_plan, _manifest_data,
                             _numpy_data, _check_evaluation_count,
                             MAX_CPU_WORKING_BYTES, MAX_CPU_MODEL_BYTES)
    if data_path is None or not str(data_path).strip():
        raise ConversionError("Data-driven pruning requires a separate representative calibration dataset")
    if (isinstance(max_samples, bool) or not isinstance(max_samples, int)
            or not 1 <= max_samples <= MAX_CALIBRATION_SAMPLES):
        raise ConversionError(f"Calibration max_samples must be an integer in [1, {MAX_CALIBRATION_SAMPLES}]")
    try:
        scale = float(input_scale)
    except (ValueError, TypeError, OverflowError) as error:
        raise ConversionError("Calibration input_scale must be finite positive float32") from error
    if (isinstance(input_scale, bool) or not math.isfinite(scale) or scale <= 0
            or scale > np.finfo(np.float32).max or np.float32(scale) <= 0):
        raise ConversionError("Calibration input_scale must be finite positive float32")
    if (isinstance(input_zero_point, bool) or not isinstance(input_zero_point, int)
            or not 0 <= input_zero_point <= 255):
        raise ConversionError("Calibration input_zero_point must be an integer in [0,255]")
    spec = _validated_spec(session)
    # Six forward-equivalent graph costs cover one per-sample backward pass
    # and teacher/candidate feature reconstruction. Gradient working storage
    # is also explicitly admitted before dataset reading/model construction.
    plan = _evaluation_plan(*([spec] * 6))
    if plan["estimated_working_bytes_per_sample"] > MAX_CPU_WORKING_BYTES:
        raise ConversionError("Taylor gradient working-memory CPU budget exceeded")
    if plan["estimated_model_reconstruction_bytes"] > MAX_CPU_MODEL_BYTES:
        raise ConversionError("Taylor gradient model-memory CPU budget exceeded")
    if max_samples * max(spec.dimensions["hidden_size"], spec.dimensions["num_classes"]) > MAX_FIT_ELEMENTS:
        raise ConversionError("Taylor feature reconstruction memory budget exceeded")
    hidden = spec.dimensions["hidden_size"]
    if (hidden * hidden > MAX_DISTANCE_ELEMENTS
            or hidden ** 3 + max_samples * hidden * hidden > MAX_FIT_OPERATIONS):
        raise ConversionError("Taylor ridge reconstruction CPU/memory budget exceeded")
    path = Path(data_path).expanduser().resolve()
    if path.is_dir():
        path = path / "manifest.csv"
    if not path.is_file():
        raise ConversionError(f"Calibration dataset not found: {path}")
    try:
        if path.suffix.lower() == ".csv":
            inputs, _ = _manifest_data(path, spec, max_samples, plan)
        elif path.suffix.lower() in (".npy", ".npz"):
            inputs, _ = _numpy_data(path, spec, max_samples, plan)
        else:
            raise ConversionError("Calibration data must be .npy, .npz or manifest.csv")
    except ConversionError:
        raise
    except (OSError, ValueError, TypeError, EOFError, KeyError, RuntimeError, zipfile.BadZipFile) as error:
        raise ConversionError(f"Cannot read calibration dataset: {error}") from error
    _check_evaluation_count(len(inputs), max_samples, plan)
    with np.errstate(over="ignore", invalid="ignore"):
        floating = inputs.astype(np.float32)
        if inputs.dtype == np.uint8:
            floating = (floating - input_zero_point) * np.float32(scale)
    if not bool(np.isfinite(floating).all()):
        raise ConversionError("Calibration dequantization produced NaN/Inf")
    # Hash the exact selected input bytes plus shape/dtype and preprocessing,
    # rather than merely the manifest whose referenced files could change.
    digest = hashlib.sha256()
    digest.update(str((inputs.shape, str(inputs.dtype), scale, input_zero_point)).encode("utf-8"))
    digest.update(np.ascontiguousarray(inputs).tobytes())
    provenance = {"path": str(path), "selected_inputs_sha256": digest.hexdigest(),
                  "samples": len(inputs), "labels_used": False,
                  "input_scale": scale, "input_zero_point": input_zero_point,
                  "selection": "first max_samples entries in supplied calibration file",
                  "estimated_total_macs": len(inputs) * plan["estimated_macs_per_sample"],
                  "purpose": "channel selection and classifier reconstruction; use a separate validation set"}
    return torch.from_numpy(np.ascontiguousarray(floating)), provenance


def _parameter_sensitivity(session, model):
    values = {}
    prefix = head = None
    if session.spec.architecture != "cnn":
        prefix, head = _prefixes(session.state)
    for name, parameter in model.named_parameters():
        key = name
        if prefix is not None:
            key = (prefix + name[len("recurrent."):] if name.startswith("recurrent.")
                   else head + name[len("head."):])
        if parameter.grad is None:
            values[key] = torch.zeros_like(parameter, dtype=torch.float64)
        else:
            values[key] = (parameter.detach().double() * parameter.grad.detach().double()).abs()
            if not bool(torch.isfinite(values[key]).all()):
                raise ConversionError("Taylor sensitivities contain NaN/Inf")
    return values


def _sensitivity_groups(values, spec):
    if spec.architecture == "cnn":
        scores = {}
        for layer, index in enumerate((0, 4, 8)):
            score = values[f"model.{index}.weight"].sum((1, 2))
            if f"model.{index}.bias" in values:
                score += values[f"model.{index}.bias"]
            score += values[f"model.{index + 1}.weight"] + values[f"model.{index + 1}.bias"]
            if index < 8:
                score += values[f"model.{index + 4}.weight"].sum((0, 2))
            else:
                score += values["fc1.weight"].reshape(spec.dimensions["hidden_size"], len(score), -1).sum((0, 2))
            scores[f"conv{layer}"] = score
        scores["fc1"] = (values["fc1.weight"].sum(1) + values["fc1.bias"]
                          + values["fc2.weight"].sum(0))
        return scores
    prefix, head = _prefixes(values)
    hidden = spec.dimensions["hidden_size"]
    gates = 4 if spec.architecture == "lstm" else 1
    ih = values[prefix + "weight_ih_l0"].reshape(gates, hidden, -1)
    hh = values[prefix + "weight_hh_l0"].reshape(gates, hidden, hidden)
    score = ih.sum((0, 2)) + hh.sum((0, 2)) + hh.sum((0, 1)) + values[head + "weight"].sum(0)
    for name in ("bias_ih_l0", "bias_hh_l0"):
        score += values[prefix + name].reshape(gates, hidden).sum(0)
    return {"hidden": score}


@torch.enable_grad()
def taylor_scores(session, data_path, max_samples, input_scale, input_zero_point, skip=False):
    from .validation import build_model
    if skip:
        return None, None, {"criterion": "Taylor dependency-group sensitivity; zero ratio is identity",
                            "calibration": None}
    inputs, provenance = _calibration(session, data_path, max_samples, input_scale, input_zero_point)
    model = build_model(session)
    scores = None
    # Per-sample absolute products avoid cancelling opposing sample gradients.
    # Teacher pseudo-labels deliberately ignore any labels stored in the file.
    for sample in inputs:
        model.zero_grad(set_to_none=True)
        logits = model(sample[None])
        if not bool(torch.isfinite(logits).all()):
            raise ConversionError("Taylor teacher logits contain NaN/Inf")
        F.cross_entropy(logits, logits.detach().argmax(1)).backward()
        current = _sensitivity_groups(_parameter_sensitivity(session, model), session.spec)
        if scores is None:
            scores = current
        else:
            for key in scores:
                scores[key] += current[key]
    for key in scores:
        scores[key] /= len(inputs)
    return scores, (inputs, provenance), {
        "criterion": "Mean per-sample sum(abs(parameter * gradient)) over tied structural dependency groups",
        "calibration": provenance,
        "algorithm_scope": "First-order Taylor sensitivity-bound adaptation with teacher argmax pseudo-labels; no labels or iterative paper retraining",
        "reference": "https://openaccess.thecvf.com/content_CVPR_2019/html/Molchanov_Importance_Estimation_for_Neural_Network_Pruning_CVPR_2019_paper.html"}


def _features_and_logits(session, inputs):
    from .validation import build_model
    model = build_model(session)
    features, outputs = [], []
    head = model.fc2 if session.spec.architecture == "cnn" else model.head
    def capture(_, arguments):
        features.append(arguments[0].detach().double())
    handle = head.register_forward_pre_hook(capture)
    try:
        with torch.inference_mode():
            for sample in inputs:
                outputs.append(model(sample[None]).double())
    finally:
        handle.remove()
    x, y = torch.cat(features), torch.cat(outputs)
    if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(y).all()):
        raise ConversionError("Head reconstruction features/logits contain NaN/Inf")
    return x, y


def reconstruct_head(reference, candidate, calibration):
    inputs, _ = calibration
    _, target = _features_and_logits(reference, inputs)
    features, before = _features_and_logits(candidate, inputs)
    hidden = features.shape[1]
    if hidden * hidden > MAX_DISTANCE_ELEMENTS:
        raise ConversionError("Head ridge reconstruction CPU/memory budget exceeded")
    centered = features - features.mean(0)
    residual = target - before
    centered_residual = residual - residual.mean(0)
    gram = centered.t() @ centered
    ridge = max(float(gram.diag().mean()) * 1e-3, 1e-8)
    correction = torch.linalg.solve(gram + ridge * torch.eye(hidden, dtype=torch.float64),
                                    centered.t() @ centered_residual)
    key = "fc2." if candidate.spec.architecture == "cnn" else _prefixes(candidate.state)[1]
    weight = candidate.state[key + "weight"].double() + correction.t()
    bias = candidate.state[key + "bias"].double() + residual.mean(0) - features.mean(0) @ correction
    if not bool(torch.isfinite(weight).all()) or not bool(torch.isfinite(bias).all()):
        raise ConversionError("Head ridge reconstruction produced NaN/Inf")
    candidate.state[key + "weight"] = weight.float().contiguous()
    candidate.state[key + "bias"] = bias.float().contiguous()
    reconstructed = features @ candidate.state[key + "weight"].double().t() + candidate.state[key + "bias"].double()
    return {"method": "centered ridge correction to teacher FP32 logits", "ridge_lambda": ridge,
            "calibration_rmse_before": float(torch.square(before - target).mean().sqrt()),
            "calibration_rmse_after": float(torch.square(reconstructed - target).mean().sqrt()),
            "labels_used": False,
            "quality_note": "Calibration improvement is fitted, not independent validation or guaranteed task accuracy"}
