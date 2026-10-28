"""Bounded, label-free channel reconstruction for the supported affine-head CNN.

This is a schema-specific adaptation of next-layer reconstruction (ThiNet) and
group Optimal Brain Surgeon. It is not a reproduction of a paper's full method.
It creates smaller dense tensors, without adding a sparse execution dependency.
"""

import math

import torch
from torch.nn import functional as F

from model_converter.schema import ConversionError

from .advanced_pruning import (MAX_ADVANCED_WORKING_BYTES, MAX_CALIBRATION_SAMPLES,
                               MAX_FIT_ELEMENTS, MAX_FIT_OPERATIONS, _calibration,
                               validate_advanced_budget)


MAX_PATCH_POSITIONS = 32
CONV_DAMPING = 1e-3
HEAD_DAMPING_CANDIDATES = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)


def validate_reconstruction_budget(spec, amount):
    """Admit worst-case calibration dimensions before state cloning/allocation."""
    if spec.architecture != "cnn":
        raise ConversionError("Activation reconstruction currently requires cnn_voice_v1")
    if amount == 0:
        return
    dimensions = spec.dimensions
    count = MAX_CALIBRATION_SAMPLES
    activation_elements = count * sum(outputs * (width + pooled)
        for outputs, _, _, width, pooled in dimensions["conv_shapes"])
    for outputs, channels, kernel, width, _ in dimensions["conv_shapes"][1:]:
        rows = count * min(width, MAX_PATCH_POSITIONS)
        columns = channels * kernel
        # Covariance, inverse updates, greedy group solves and affine fitting.
        operations = (2 * rows * columns * columns + 3 * columns ** 3
                      + channels * channels * (kernel ** 3 + outputs * kernel * kernel)
                      + outputs * columns * columns)
        # Teacher activation storage coexists with selection/fit buffers.
        # Explicitly reserve sampled FP64 patches, centered patches, targets,
        # residuals, covariance/inverse copies, solve scratch and corrections.
        working = 16 * activation_elements + 8 * (
            3 * rows * columns + 6 * rows * outputs + 6 * columns * columns
            + 4 * outputs * columns)
        if (rows * max(columns, outputs) > MAX_FIT_ELEMENTS
                or columns * columns > MAX_FIT_ELEMENTS
                or operations > MAX_FIT_OPERATIONS or working > MAX_ADVANCED_WORKING_BYTES):
            raise ConversionError("Activation reconstruction convolution CPU/memory budget exceeded")
    width = dimensions["flatten_size"]
    classes = dimensions["num_classes"]
    last_channels = dimensions["conv_shapes"][-1][0]
    # Train/holdout and final dual Gram products; solve/evaluate six dampings.
    operations = (2 * count * count * width + 8 * count ** 3
                  + (2 * len(HEAD_DAMPING_CANDIDATES) + 6) * count * width * classes
                  + 2 * count * classes * last_channels ** 2 + 3 * last_channels ** 3)
    selection_working = 16 * activation_elements + 8 * (
        4 * count * last_channels * classes + 6 * last_channels ** 2
        + classes * width + count * classes)
    fit_working = 16 * activation_elements + 8 * (
        4 * count * width + 8 * count * classes + 4 * count ** 2
        + 6 * classes * width + 6 * min(classes, width) ** 2)
    working = max(selection_working, fit_working)
    if (count * width > MAX_FIT_ELEMENTS
            or count * last_channels * classes > MAX_FIT_ELEMENTS
            or operations > MAX_FIT_OPERATIONS
            or working > MAX_ADVANCED_WORKING_BYTES):
        raise ConversionError("Activation reconstruction classifier CPU/memory budget exceeded")
    validate_advanced_budget(spec, "svd", amount)


def _finite(value, name):
    if not bool(torch.isfinite(value).all()):
        raise ConversionError(f"Activation reconstruction {name} contains NaN/Inf")
    return value


def _block(state, metadata, ordinal, values):
    index = ordinal * 4
    weight = state[f"model.{index}.weight"].float()
    bias = state.get(f"model.{index}.bias")
    linear = F.conv1d(values, weight, None if bias is None else bias.float())
    prefix = f"model.{index + 1}."
    epsilon = float(metadata.get("bn_eps", [1e-5] * 3)[ordinal])
    normalized = F.batch_norm(linear, state[prefix + "running_mean"].float(),
        state[prefix + "running_var"].float(), state[prefix + "weight"].float(),
        state[prefix + "bias"].float(), training=False, eps=epsilon)
    return _finite(linear, "convolution"), _finite(F.max_pool1d(F.relu(normalized), 2), "pooling")


def _patches(values, kernel):
    length = values.shape[-1] - kernel + 1
    positions = torch.linspace(0, length - 1, min(length, MAX_PATCH_POSITIONS)).round().long()
    patches = values.unfold(-1, kernel, 1).index_select(2, positions)
    return patches.permute(0, 2, 1, 3).reshape(-1, values.shape[1] * kernel).double(), positions


def _obs_groups(features, weight, group_width, keep):
    """Greedily remove channel groups with the damped OBS quadratic loss.

    For deleted group G and inverse covariance P, loss = tr(W_G P_GG^-1
    W_G^T). Remaining W_R -= W_G P_GG^-1 P_GR, followed by a Schur
    complement update of P. Group order stays stable, including tied scores.
    """
    count = features.shape[1] // group_width
    if keep == count:
        return torch.arange(count), []
    centered = features - features.mean(0)
    gram = centered.t() @ centered / max(1, len(features))
    ridge = max(float(gram.diag().mean()) * CONV_DAMPING, 1e-12)
    inverse = torch.linalg.inv(gram + ridge * torch.eye(len(gram), dtype=torch.float64))
    current = weight.double().clone()
    remaining = list(range(count))
    trace = []
    while len(remaining) > keep:
        candidates = []
        for ordinal, channel in enumerate(remaining):
            indices = torch.arange(ordinal * group_width, (ordinal + 1) * group_width)
            block = inverse[indices[:, None], indices]
            local = current[:, indices]
            cost = float((local.t() * torch.linalg.solve(block, local.t())).sum())
            if not math.isfinite(cost):
                raise ConversionError("Activation reconstruction OBS score contains NaN/Inf")
            candidates.append((max(0.0, cost), channel, ordinal, indices))
        cost, channel, ordinal, removed = min(candidates, key=lambda item: item[:2])
        mask = torch.ones(len(inverse), dtype=torch.bool)
        mask[removed] = False
        retained = torch.where(mask)[0]
        block = inverse[removed[:, None], removed]
        cross = inverse[removed[:, None], retained]
        solved = torch.linalg.solve(block, cross)
        current = current[:, retained] - current[:, removed] @ solved
        inverse = inverse[retained[:, None], retained] - cross.t() @ solved
        inverse = (inverse + inverse.t()) * 0.5
        _finite(inverse, "inverse covariance")
        _finite(current, "OBS compensation")
        remaining.pop(ordinal)
        trace.append({"channel": channel, "quadratic_loss": cost})
    return torch.tensor(remaining, dtype=torch.long), trace


def _affine_context(features, target, weight, bias, dual):
    """Reusable statistics for a fixed training subset and affine prior."""
    features, target, weight, bias = (value.double() for value in (features, target, weight, bias))
    before = features @ weight.t() + bias
    residual = target - before
    centered = features - features.mean(0)
    centered_residual = residual - residual.mean(0)
    gram = centered @ centered.t() if dual else centered.t() @ centered
    return features, target, weight, bias, before, residual, centered, centered_residual, gram


def _affine_fit(features, target, weight, bias, damping, *, dual=False, prepared=None):
    """Centered ridge correction around existing weights, preserving the prior."""
    if prepared is None:
        prepared = _affine_context(features, target, weight, bias, dual)
    features, target, weight, bias, before, residual, centered, centered_residual, gram = prepared
    ridge = max(float(gram.diag().mean()) * damping, 1e-12)
    regularized = gram + ridge * torch.eye(len(gram), dtype=torch.float64)
    if dual:
        correction = torch.linalg.solve(regularized, centered_residual).t() @ centered
    else:
        correction = torch.linalg.solve(regularized, centered.t() @ centered_residual).t()
    fitted_weight = _finite(weight + correction, "ridge weights")
    fitted_bias = _finite(bias + residual.mean(0) - features.mean(0) @ correction.t(), "ridge bias")
    after = features @ fitted_weight.t() + fitted_bias
    return fitted_weight, fitted_bias, {"ridge_lambda": ridge, "damping": damping,
        "calibration_rmse_before": float((before - target).square().mean().sqrt()),
        "calibration_rmse_after": float((after - target).square().mean().sqrt()),
        "solver": "dual centered ridge" if dual else "primal centered ridge"}


def _slice_outputs(state, ordinal, kept):
    index = ordinal * 4
    for key in (f"model.{index}.weight", f"model.{index}.bias",
                *(f"model.{index + 1}.{name}" for name in
                  ("weight", "bias", "running_mean", "running_var"))):
        if key in state:
            state[key] = state[key].index_select(0, kept).contiguous()


def _reconstruct_next(state, metadata, ordinal, values, target, kept):
    index = (ordinal + 1) * 4
    weight = state[f"model.{index}.weight"].index_select(1, kept).contiguous()
    features, positions = _patches(values.index_select(1, kept), weight.shape[-1])
    targets = target.index_select(2, positions).permute(0, 2, 1).reshape(-1, weight.shape[0]).double()
    bias_key = f"model.{index}.bias"
    has_bias = bias_key in state
    bias = state[bias_key] if has_bias else torch.zeros(weight.shape[0])
    fitted_weight, fitted_bias, report = _affine_fit(
        features, targets, weight.flatten(1), bias, CONV_DAMPING)
    state[f"model.{index}.weight"] = fitted_weight.reshape_as(weight).float().contiguous()
    if has_bias:
        state[bias_key] = fitted_bias.float().contiguous()
    else:
        # Moving the fitted intercept into BN's running mean preserves the
        # original bias-free convolution schema and is exact in eval mode.
        mean_key = f"model.{index + 1}.running_mean"
        state[mean_key] = (state[mean_key].double() - fitted_bias).float().contiguous()
    report.update({"sampled_spatial_positions_per_input": len(positions), "patch_rows": len(features)})
    return report


def _head_fit(features, target, weight, bias):
    # The holdout comes from the supplied TRAIN calibration inputs. No labels
    # are read. The external evaluation set never chooses a channel or lambda.
    holdout = torch.arange(len(features)) % 4 == 0
    train = ~holdout
    selection = []
    if int(train.sum()) >= 2 and int(holdout.sum()) >= 2:
        training_features, training_target = features[train], target[train]
        prepared = _affine_context(training_features, training_target, weight, bias, True)
        for damping in HEAD_DAMPING_CANDIDATES:
            fitted, intercept, _ = _affine_fit(training_features, training_target, weight, bias,
                                              damping, dual=True, prepared=prepared)
            error = float((features[holdout] @ fitted.t() + intercept - target[holdout]).square().mean().sqrt())
            selection.append({"damping": damping, "holdout_teacher_rmse": error})
        damping = min(selection, key=lambda item: (item["holdout_teacher_rmse"], -item["damping"]))["damping"]
        del prepared, training_features, training_target
    else:
        damping = 1e-2
    fitted, intercept, report = _affine_fit(features, target, weight, bias, damping, dual=True)
    report.update({"damping_selection": "every fourth training-calibration input held out; teacher logits only",
                   "damping_candidates": selection, "final_fit_samples": len(features), "labels_used": False})
    return fitted, intercept, report


def _factor_head(state, fitted, bias, original_hidden):
    """Balanced SVD directly on the fitted map, without a classes² identity."""
    effective = fitted.float().double()
    classes, width = effective.shape
    rank = min(original_hidden, classes, width)
    u, singular, vh = torch.linalg.svd(effective, full_matrices=False)
    root = torch.sqrt(singular[:rank])
    state["fc1.weight"] = (root[:, None] * vh[:rank]).float().contiguous()
    state["fc1.bias"] = torch.zeros(rank, dtype=torch.float32)
    state["fc2.weight"] = (u[:, :rank] * root[None, :]).float().contiguous()
    state["fc2.bias"] = bias.float().contiguous()
    total = float(singular.square().sum())
    return {"method": "balanced SVD of reconstructed effective affine map",
            "rank": rank, "effective_rank_limit": min(classes, width),
            "exact_linear_rank_retained": rank >= min(classes, width),
            "discarded_singular_energy_fraction": float(singular[rank:].square().sum()) / total if total else 0.0,
            "quality_note": "The original hidden width caps rank. When it is below the fitted map rank, this is a lossy factorization; INT8 always needs independent validation."}


@torch.no_grad()
def reconstruct_channels(session, state, amount, data_path, max_samples, input_scale, input_zero_point):
    """Mutate only the supplied clone; return physical units and audit metadata."""
    spec = session.spec
    validate_reconstruction_budget(spec, amount)
    shapes = spec.dimensions["conv_shapes"]
    hidden = spec.dimensions["hidden_size"]
    units = {f"conv{ordinal}_channels": {"before": shape[0], "kept": shape[0], "removed": 0}
             for ordinal, shape in enumerate(shapes)}
    units["fc1_neurons"] = {"before": hidden, "kept": hidden, "removed": 0}
    if amount == 0:
        return units, {"criterion": "Activation reconstruction; zero ratio is identity", "calibration": None}
    inputs, provenance = _calibration(session, data_path, max_samples, input_scale, input_zero_point)
    teacher_linear, teacher_blocks = [], []
    values = inputs
    for ordinal in range(3):
        linear, values = _block(session.state, session.metadata, ordinal, values)
        teacher_linear.append(linear)
        teacher_blocks.append(values)
    effective = session.state["fc2.weight"].double() @ session.state["fc1.weight"].double()
    intercept = (session.state["fc2.weight"].double() @ session.state["fc1.bias"].double()
                 + session.state["fc2.bias"].double())
    teacher_logits = _finite(teacher_blocks[-1].flatten(1).double() @ effective.t() + intercept, "teacher logits")
    values = inputs
    layers = []
    for ordinal, (outputs, _, _, _, _) in enumerate(shapes):
        _, values = _block(state, session.metadata, ordinal, values)
        keep = max(1, outputs - math.floor(outputs * amount))
        if ordinal < 2:
            next_index = (ordinal + 1) * 4
            next_weight = state[f"model.{next_index}.weight"].double()
            features, _ = _patches(values, next_weight.shape[-1])
            bn_prefix = f"model.{next_index + 1}."
            epsilon = float(session.metadata.get("bn_eps", [1e-5] * 3)[ordinal + 1])
            bn_scale = state[bn_prefix + "weight"].double() / torch.sqrt(
                state[bn_prefix + "running_var"].double() + epsilon)
            kept, deletion = _obs_groups(features, next_weight.flatten(1) * bn_scale[:, None],
                                          next_weight.shape[-1], keep)
        else:
            # Each column is one channel's contribution to ALL class logits.
            # This small surrogate allows channel-scale compensation during
            # selection; the subsequent fit frees all retained FC coefficients.
            contribution = torch.einsum("nct,oct->nco", values.double(),
                effective.reshape(effective.shape[0], outputs, values.shape[-1]))
            centered = contribution - contribution.mean(0)
            features = centered.permute(0, 2, 1).reshape(-1, outputs)
            kept, deletion = _obs_groups(features, torch.ones(1, outputs, dtype=torch.float64), 1, keep)
            # Selection's class-contribution buffers need not coexist with the
            # full flattened-feature ridge contexts and SVD workspaces.
            del contribution, centered, features
        _slice_outputs(state, ordinal, kept)
        layer_report = {"layer": f"conv{ordinal}", "kept_channels": kept.tolist(), "deletion_order": deletion}
        units[f"conv{ordinal}_channels"].update(kept=keep, removed=outputs - keep)
        if ordinal < 2:
            if keep < outputs or ordinal > 0:
                layer_report["next_layer_reconstruction"] = _reconstruct_next(
                    state, session.metadata, ordinal, values, teacher_linear[ordinal + 1], kept)
            values = values.index_select(1, kept).contiguous()
        else:
            values = values.index_select(1, kept).contiguous()
            columns = (kept[:, None] * values.shape[-1] + torch.arange(values.shape[-1])).flatten()
            fitted, bias, head_report = _head_fit(values.flatten(1).double(), teacher_logits,
                                                   effective.index_select(1, columns), intercept)
            svd_report = _factor_head(state, fitted, bias, hidden)
            factorized = F.linear(F.linear(values.flatten(1), state["fc1.weight"],
                                          state["fc1.bias"]), state["fc2.weight"], state["fc2.bias"])
            head_report["calibration_rmse_after_factorization"] = float(
                (_finite(factorized, "factorized classifier").double() - teacher_logits).square().mean().sqrt())
            head_report["rmse_scope"] = "before/after refer to the affine ridge fit; after_factorization includes stored FP32 SVD factors and their FP32 execution"
            rank = state["fc1.weight"].shape[0]
            units["fc1_neurons"].update(kept=rank, removed=hidden - rank)
            layer_report["classifier_reconstruction"] = head_report
            layer_report["linear_head_factorization"] = svd_report
        layers.append(layer_report)
    return units, {"criterion": "Next-layer activation reconstruction with damped group OBS channel deletion",
        "calibration": provenance, "layers": layers,
        "algorithm_scope": "Known CNN channel dependencies; sampled next-convolution patch quadratic; final-channel class-contribution surrogate; ridge corrections; balanced affine-head SVD capped by original hidden width, classes and retained flatten width. Rank truncation can lose information. No task-label training or paper-level optimality claim.",
        "ratio_semantics": "Each convolution keeps max(1,C-floor(C*amount)); affine-head rank is min(original hidden width, classes, retained flatten width). Zero is identity.",
        "references": ["https://arxiv.org/abs/1707.06342", "https://arxiv.org/abs/2208.11580"],
        "quality_note": "Training-calibration fits and teacher holdout are not task accuracy. Validate INT8 on an independent representative labeled set; fine-tuning may still be needed."}
