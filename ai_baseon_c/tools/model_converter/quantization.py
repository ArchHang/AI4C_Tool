"""Deterministic float32, per-output-channel symmetric weight quantization."""

import numpy as np


def quantize_rows(weights, weight_scaling="maxabs", mse_min_ratio=0.5, mse_steps=101):
    if weight_scaling not in ("maxabs", "mse"):
        raise ValueError("weight_scaling must be maxabs or mse")
    if not np.isfinite(mse_min_ratio) or not 0.0 < mse_min_ratio <= 1.0:
        raise ValueError("mse_min_ratio must be finite in (0,1]")
    if not isinstance(mse_steps, int) or isinstance(mse_steps, bool) or not 2 <= mse_steps <= 10001:
        raise ValueError("mse_steps must be an integer in [2,10001]")
    if np.iscomplexobj(weights):
        raise ValueError("weights must be real-valued")
    with np.errstate(over="ignore", invalid="ignore"):
        weights = np.asarray(weights, dtype=np.float32)
    if weights.ndim < 2 or weights.size == 0 or not np.all(np.isfinite(weights)):
        raise ValueError("weights must have nonempty finite float32 rows")
    flat = weights.reshape(weights.shape[0], -1)
    if flat.shape[1] > np.iinfo(np.int32).max // 127:
        raise ValueError("weight row sum may exceed int32 capacity")
    scale = np.maximum(np.max(np.abs(flat), axis=1) / np.float32(127),
                       np.float32(1e-12)).astype(np.float32)
    quantized = np.clip(np.rint(flat / scale[:, None]), -127, 127).astype(np.int8)
    flat_double = flat.astype(np.float64)
    reconstructed = quantized.astype(np.float64) * scale.astype(np.float64)[:, None]
    error = reconstructed - flat_double
    maxabs_mse = np.mean(error ** 2, axis=1)
    best_mse = maxabs_mse.copy()
    base_scale = scale.copy()
    chosen_ratio = np.ones(scale.size, np.float32)
    if weight_scaling == "mse":
        # The baseline remains a candidate. Strictly smaller per-row MSE is required to replace it.
        # A subnormal ratio may round down; the positive scale floor still applies.
        with np.errstate(under="ignore"):
            ratios = np.linspace(1.0, mse_min_ratio, mse_steps, dtype=np.float32)[1:]
        for ratio in ratios:
            with np.errstate(under="ignore"):
                candidate_scale = np.maximum(base_scale * ratio, np.float32(1e-12)).astype(np.float32)
            # Tiny user-specified candidates can overflow the normalized float;
            # signed infinities intentionally saturate to the int8 endpoints.
            with np.errstate(over="ignore"):
                candidate_q = np.clip(np.rint(flat / candidate_scale[:, None]), -127, 127).astype(np.int8)
            candidate_float = candidate_q.astype(np.float64) * candidate_scale.astype(np.float64)[:, None]
            candidate_mse = np.mean((candidate_float - flat_double) ** 2, axis=1)
            accept = candidate_mse < best_mse
            quantized[accept] = candidate_q[accept]
            scale[accept] = candidate_scale[accept]
            best_mse[accept] = candidate_mse[accept]
            chosen_ratio[accept] = ratio
        reconstructed = quantized.astype(np.float64) * scale.astype(np.float64)[:, None]
        error = reconstructed - flat_double
    metrics = {"max_abs_error": float(np.max(np.abs(error))),
               "rmse": float(np.sqrt(np.mean(error ** 2))),
               "zero_rows": int(np.count_nonzero(np.max(np.abs(flat), axis=1) == 0)),
               "scheme": "symmetric_per_output_channel", "range": [-127, 127],
               "scale_selection": weight_scaling, "mse": float(np.mean(best_mse)),
               "maxabs_baseline_mse": float(np.mean(maxabs_mse)),
               "improved_rows": int(np.count_nonzero(best_mse < maxabs_mse)),
               "selected_ratio_min": float(np.min(chosen_ratio)),
               "selected_ratio_max": float(np.max(chosen_ratio)),
               "clipped_weights": int(np.count_nonzero(np.abs(flat_double)
                    > scale.astype(np.float64)[:, None] * 127.0))}
    rowsum = quantized.astype(np.int64).sum(axis=1)
    return quantized.reshape(weights.shape), scale, rowsum.astype(np.int32), metrics


def fold_batch_norm(weight, bias, gamma, beta, mean, variance, epsilon):
    """Fold eval-mode BN including an existing convolution bias."""
    factor = gamma / np.sqrt(variance + np.float32(epsilon))
    folded_weight = weight * factor.reshape((-1,) + (1,) * (weight.ndim - 1))
    folded_bias = (bias - mean) * factor + beta
    return folded_weight.astype(np.float32), folded_bias.astype(np.float32)
