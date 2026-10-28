/*
 * Copyright (c) 2026, VeriSilicon Holdings Co., Ltd. All rights reserved
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * @file rnn.c
 * @brief 简单循环网络（Elman RNN）推理实现（fp32 与 int8 动态量化）
 *
 * 数值约定：
 *   - 整数域累加在 int32 完成，最后反量化到 float32；
 *   - 隐藏状态动态量化保留 uint8 非对称的历史规则，零点向零截断；
 *     常规范围不强制包含零，与 CNN fc1 当前的零覆盖规则不同；
 *     当 (max-min)/255 < 1e-9 时采用幅值兜底：若 max|h| > 1e-9，
 *     则 scale=max|h|/128、zp=0/255；否则 scale=1e-9、zp=0。
 *     该规则保留非零常数的幅值，但不保证近常数向量逐元素无损；
 *   - int8 路径的零点不逐 MAC 减，而是通过预计算的权重行和折进
 *     累加器（acc = sum(w*h) - zp*rowsum），整数运算等价且更快；
 *   - 激活采用 deterministic_float32_v1：严格 binary32、无 FMA，避免 libm 差异。
 */

#include "../include/ai_baseon_c/rnn.h"

#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>

#include "../include/ai_baseon_c/cnn.h"
#include "kernels/nn_dot.h"
#include "quantization.h"
#include "nn_activations.h"

/* 覆盖 raw dot 与零点校正的最坏中间值，包括合法的 -128 权重。 */
enum { RNN_MAX_DOT_SIZE = NN_MAX_MACS,
       RNN_SIMD_MIN_SIZE = NN_HAS_AVX2 ? 64 : (NN_HAS_NEON ? 32 : 16) };

static bool rnn_shape_is_valid(uint32_t input_size, uint32_t hidden_size,
                               uint32_t seq_len, uint32_t num_classes)
{
    const uint64_t limit = (uint64_t)PTRDIFF_MAX / sizeof(float);

    if (input_size == 0 || hidden_size == 0 || seq_len == 0) {
        return false;
    }
    return (uint64_t)input_size <= limit / seq_len &&
           (uint64_t)hidden_size <= limit / seq_len &&
           (uint64_t)hidden_size <= limit / input_size &&
           (uint64_t)hidden_size <= limit / hidden_size &&
           (num_classes == 0 || (uint64_t)hidden_size <= limit / num_classes);
}

/* 固定模型元数据只在入口检查一次，不在每个时间步重复扫描。 */
static bool rnn_rows_are_valid(const float *scales, const float *bias,
                               const int32_t *rowsums, uint32_t rows, uint32_t width)
{
    const int32_t rowsum_limit = (int32_t)(128u * width);

    for (uint32_t row = 0; row < rows; row++) {
        if (!isfinite(scales[row]) || scales[row] <= 0.0f || !isfinite(bias[row]) ||
            (rowsums && (rowsums[row] < -rowsum_limit || rowsums[row] > rowsum_limit))) {
            return false;
        }
    }
    return true;
}

/**
 * @brief RNN fp32 前向推理。
 *
 * 隐藏单元按 4 行分块（4 个独立累加器，打破乘加依赖链、复用 h 向量读取，
 * 与 cnn 库 linear_layer 的优化手法一致）；每行内的累加顺序与朴素实现
 * 相同（b_ih+b_hh -> x 维度 -> h 维度），结果与 Python 严格复现一致。
 */
int rnn_forward_fp32(const rnn_fp32_param_t *p, const rnn_out_fp32_t *out, const float *restrict x,
                     uint32_t seq_len, float *restrict h_seq, float *restrict logits)
{
    uint32_t i, j, t;
    const uint32_t in = p ? p->input_size : 0;
    const uint32_t hid = p ? p->hidden_size : 0;

    if (!p || !p->w_ih || !p->b_ih || !p->w_hh || !p->b_hh || !x || !h_seq) {
        return CNN_POINTER_NULL;
    }
    if (!rnn_shape_is_valid(in, hid, seq_len, out ? out->num_classes : 0) ||
        (out && (!out->w || !out->b || out->hidden_size != hid || out->num_classes == 0)) ||
        (out && !logits)) {
        return CNN_DATA_EXCEPTION;
    }

    for (t = 0; t < seq_len; t++) {
        const float *restrict xt = x + (size_t)t * in;
        const float *restrict hp = (t == 0) ? NULL : h_seq + (size_t)(t - 1) * hid;
        float *restrict ht = h_seq + (size_t)t * hid;

        // 4 行分块：4 个独立累加器同时累加，编译器可向量化；保持无 FMA 的固定浮点求值顺序。
        for (j = 0; j + 4 <= hid; j += 4) {
            const float *w_ih_0 = p->w_ih + (size_t)(j + 0) * in;
            const float *w_ih_1 = p->w_ih + (size_t)(j + 1) * in;
            const float *w_ih_2 = p->w_ih + (size_t)(j + 2) * in;
            const float *w_ih_3 = p->w_ih + (size_t)(j + 3) * in;
            const float *w_hh_0 = p->w_hh + (size_t)(j + 0) * hid;
            const float *w_hh_1 = p->w_hh + (size_t)(j + 1) * hid;
            const float *w_hh_2 = p->w_hh + (size_t)(j + 2) * hid;
            const float *w_hh_3 = p->w_hh + (size_t)(j + 3) * hid;
            float a0 = p->b_ih[j + 0] + p->b_hh[j + 0];
            float a1 = p->b_ih[j + 1] + p->b_hh[j + 1];
            float a2 = p->b_ih[j + 2] + p->b_hh[j + 2];
            float a3 = p->b_ih[j + 3] + p->b_hh[j + 3];

            for (i = 0; i < in; i++) {
                const float xv = xt[i];
                a0 += w_ih_0[i] * xv;
                a1 += w_ih_1[i] * xv;
                a2 += w_ih_2[i] * xv;
                a3 += w_ih_3[i] * xv;
            }
            if (hp) {
                for (i = 0; i < hid; i++) {
                    const float hv = hp[i];
                    a0 += w_hh_0[i] * hv;
                    a1 += w_hh_1[i] * hv;
                    a2 += w_hh_2[i] * hv;
                    a3 += w_hh_3[i] * hv;
                }
            }
            ht[j + 0] = nn_tanh_f32(a0);
            ht[j + 1] = nn_tanh_f32(a1);
            ht[j + 2] = nn_tanh_f32(a2);
            ht[j + 3] = nn_tanh_f32(a3);
            /* 不把 NaN 隐藏状态作为成功结果传播到后续时间步。 */
            if (!isfinite(ht[j + 0]) || !isfinite(ht[j + 1]) ||
                !isfinite(ht[j + 2]) || !isfinite(ht[j + 3])) {
                return CNN_DATA_EXCEPTION;
            }
        }
        // 尾部（hidden 不是 4 的倍数时）。
        for (; j < hid; j++) {
            const float *w_ih_j = p->w_ih + (size_t)j * in;
            const float *w_hh_j = p->w_hh + (size_t)j * hid;
            float acc = p->b_ih[j] + p->b_hh[j];
            for (i = 0; i < in; i++) {
                acc += w_ih_j[i] * xt[i];
            }
            if (hp) {
                for (i = 0; i < hid; i++) {
                    acc += w_hh_j[i] * hp[i];
                }
            }
            ht[j] = nn_tanh_f32(acc);
            if (!isfinite(ht[j])) {
                return CNN_DATA_EXCEPTION;
            }
        }
    }

    if (out) {
        const float *h_last = h_seq + (size_t)(seq_len - 1) * hid;
        for (j = 0; j < out->num_classes; j++) {
            const float *wj = out->w + (size_t)j * hid;
            float acc = out->b[j];
            for (i = 0; i < hid; i++) {
                acc += wj[i] * h_last[i];
            }
            if (!isfinite(acc)) {
                return CNN_DATA_EXCEPTION;
            }
            logits[j] = acc;
        }
    }
    return CNN_NORMAL;
}

/**
 * @brief RNN int8 动态量化前向推理。
 *
 * 每个时间步：
 *   1) 整数域累加（int32）：acc_ih = sum(w_ih_q*x_q) - x_zp*rowsum_ih，
 *      acc_hh = sum(w_hh_q*h_q) - hprev_zp*rowsum_hh（零点折进行和，
 *      不逐 MAC 减零点；整数运算结果与直接减零点完全一致）；
 *   2) 反量化 y = acc_ih*(w_ih_scale*x_scale) + b_ih
 *              + acc_hh*(w_hh_scale*h_scale) + b_hh（float32）；
 *   3) h = tanh(y)（float32），同时跟踪 min/max；
 *   4) 对 h 做 uint8 非对称动态量化（min/max -> scale/zp）供下一时间步使用。
 */
int rnn_forward_int8(const rnn_int8_param_t *p, const rnn_out_int8_t *out, const uint8_t *restrict x_q,
                     uint32_t seq_len, float x_scale, int x_zp, uint8_t *restrict h_q_seq,
                     float *restrict h_scale_seq, int *restrict h_zp_seq, float *restrict h_work,
                     float *restrict logits)
{
    uint32_t i, j, t;
    const uint32_t in = p ? p->input_size : 0;
    const uint32_t hid = p ? p->hidden_size : 0;

    if (!p || !p->w_ih || !p->w_ih_scale || !p->w_ih_rowsum || !p->b_ih || !p->w_hh ||
        !p->w_hh_scale || !p->w_hh_rowsum || !p->b_hh || !x_q || !h_q_seq || !h_scale_seq ||
        !h_zp_seq || !h_work) {
        return CNN_POINTER_NULL;
    }
    if (!rnn_shape_is_valid(in, hid, seq_len, out ? out->num_classes : 0) ||
        in > RNN_MAX_DOT_SIZE || hid > RNN_MAX_DOT_SIZE ||
        !isfinite(x_scale) || x_scale <= 0.0f || x_zp < 0 || x_zp > 255 ||
        (out && (!out->w || !out->w_scale || !out->b || out->hidden_size != hid ||
                 out->num_classes == 0)) ||
        (out && !logits)) {
        return CNN_DATA_EXCEPTION;
    }
    if (!rnn_rows_are_valid(p->w_ih_scale, p->b_ih, p->w_ih_rowsum, hid, in) ||
        !rnn_rows_are_valid(p->w_hh_scale, p->b_hh, p->w_hh_rowsum, hid, hid) ||
        (out && !rnn_rows_are_valid(out->w_scale, out->b, NULL, out->num_classes, hid))) {
        return CNN_DATA_EXCEPTION;
    }

    for (t = 0; t < seq_len; t++) {
        const uint8_t *restrict xt = x_q + (size_t)t * in;
        const uint8_t *restrict hp = (t == 0) ? NULL : h_q_seq + (size_t)(t - 1) * hid;
        const float hprev_scale = (t == 0) ? 1.0f : h_scale_seq[t - 1];
        const int hprev_zp = (t == 0) ? 0 : h_zp_seq[t - 1];
        uint8_t *restrict hq = h_q_seq + (size_t)t * hid;
        float mn, mx, scale;
        int32_t zp;

        mn = 1e30f;
        mx = -1e30f;

        // 4 行分块：4 个独立 int32 累加器；零点折进行和，无逐 MAC 减法。
        for (j = 0; j + 4 <= hid; j += 4) {
            const int8_t *w_ih_0 = p->w_ih + (size_t)(j + 0) * in;
            const int8_t *w_ih_1 = p->w_ih + (size_t)(j + 1) * in;
            const int8_t *w_ih_2 = p->w_ih + (size_t)(j + 2) * in;
            const int8_t *w_ih_3 = p->w_ih + (size_t)(j + 3) * in;
            const int8_t *w_hh_0 = p->w_hh + (size_t)(j + 0) * hid;
            const int8_t *w_hh_1 = p->w_hh + (size_t)(j + 1) * hid;
            const int8_t *w_hh_2 = p->w_hh + (size_t)(j + 2) * hid;
            const int8_t *w_hh_3 = p->w_hh + (size_t)(j + 3) * hid;
            int32_t a0 = -x_zp * p->w_ih_rowsum[j + 0];
            int32_t a1 = -x_zp * p->w_ih_rowsum[j + 1];
            int32_t a2 = -x_zp * p->w_ih_rowsum[j + 2];
            int32_t a3 = -x_zp * p->w_ih_rowsum[j + 3];
            int32_t b0 = -hprev_zp * p->w_hh_rowsum[j + 0];
            int32_t b1 = -hprev_zp * p->w_hh_rowsum[j + 1];
            int32_t b2 = -hprev_zp * p->w_hh_rowsum[j + 2];
            int32_t b3 = -hprev_zp * p->w_hh_rowsum[j + 3];

#if NN_HAS_VECTOR_DOT
            if (in >= RNN_SIMD_MIN_SIZE) {
                const int8_t *const weights[4] = {w_ih_0, w_ih_1, w_ih_2, w_ih_3};
                int32_t sums[4];
                nn_dot4_u8_s8(xt, weights, in, sums);
                a0 += sums[0];
                a1 += sums[1];
                a2 += sums[2];
                a3 += sums[3];
            } else
#endif
            {
                for (i = 0; i < in; i++) {
                    const int32_t xv = (int32_t)xt[i];
                    a0 += (int32_t)w_ih_0[i] * xv;
                    a1 += (int32_t)w_ih_1[i] * xv;
                    a2 += (int32_t)w_ih_2[i] * xv;
                    a3 += (int32_t)w_ih_3[i] * xv;
                }
            }
            if (hp) {
#if NN_HAS_VECTOR_DOT
                if (hid >= RNN_SIMD_MIN_SIZE) {
                    const int8_t *const weights[4] = {w_hh_0, w_hh_1, w_hh_2, w_hh_3};
                    int32_t sums[4];
                    nn_dot4_u8_s8(hp, weights, hid, sums);
                    b0 += sums[0];
                    b1 += sums[1];
                    b2 += sums[2];
                    b3 += sums[3];
                } else
#endif
                {
                    for (i = 0; i < hid; i++) {
                        const int32_t hv = (int32_t)hp[i];
                        b0 += (int32_t)w_hh_0[i] * hv;
                        b1 += (int32_t)w_hh_1[i] * hv;
                        b2 += (int32_t)w_hh_2[i] * hv;
                        b3 += (int32_t)w_hh_3[i] * hv;
                    }
                }
            }
            {
                const float y0 = (float)a0 * (p->w_ih_scale[j + 0] * x_scale) + p->b_ih[j + 0] +
                                 (float)b0 * (p->w_hh_scale[j + 0] * hprev_scale) + p->b_hh[j + 0];
                const float y1 = (float)a1 * (p->w_ih_scale[j + 1] * x_scale) + p->b_ih[j + 1] +
                                 (float)b1 * (p->w_hh_scale[j + 1] * hprev_scale) + p->b_hh[j + 1];
                const float y2 = (float)a2 * (p->w_ih_scale[j + 2] * x_scale) + p->b_ih[j + 2] +
                                 (float)b2 * (p->w_hh_scale[j + 2] * hprev_scale) + p->b_hh[j + 2];
                const float y3 = (float)a3 * (p->w_ih_scale[j + 3] * x_scale) + p->b_ih[j + 3] +
                                 (float)b3 * (p->w_hh_scale[j + 3] * hprev_scale) + p->b_hh[j + 3];
                h_work[j + 0] = nn_tanh_f32(y0);
                h_work[j + 1] = nn_tanh_f32(y1);
                h_work[j + 2] = nn_tanh_f32(y2);
                h_work[j + 3] = nn_tanh_f32(y3);
                // 融合 min/max；无序比较同时拒绝 NaN，避免额外的逐单元检查。
                if (h_work[j + 0] < mn) {
                    mn = h_work[j + 0];
                } else if (!(h_work[j + 0] >= mn)) {
                    return CNN_DATA_EXCEPTION;
                }
                if (h_work[j + 0] > mx) {
                    mx = h_work[j + 0];
                }
                if (h_work[j + 1] < mn) {
                    mn = h_work[j + 1];
                } else if (!(h_work[j + 1] >= mn)) {
                    return CNN_DATA_EXCEPTION;
                }
                if (h_work[j + 1] > mx) {
                    mx = h_work[j + 1];
                }
                if (h_work[j + 2] < mn) {
                    mn = h_work[j + 2];
                } else if (!(h_work[j + 2] >= mn)) {
                    return CNN_DATA_EXCEPTION;
                }
                if (h_work[j + 2] > mx) {
                    mx = h_work[j + 2];
                }
                if (h_work[j + 3] < mn) {
                    mn = h_work[j + 3];
                } else if (!(h_work[j + 3] >= mn)) {
                    return CNN_DATA_EXCEPTION;
                }
                if (h_work[j + 3] > mx) {
                    mx = h_work[j + 3];
                }
            }
        }
        for (; j < hid; j++) {
            const int8_t *w_ih_j = p->w_ih + (size_t)j * in;
            const int8_t *w_hh_j = p->w_hh + (size_t)j * hid;
            int32_t acc_ih = -x_zp * p->w_ih_rowsum[j];
            int32_t acc_hh = -hprev_zp * p->w_hh_rowsum[j];
#if NN_HAS_VECTOR_DOT
            if (in >= RNN_SIMD_MIN_SIZE) {
                acc_ih += nn_dot_u8_s8(xt, w_ih_j, in);
            } else
#endif
            {
                for (i = 0; i < in; i++) {
                    acc_ih += (int32_t)w_ih_j[i] * (int32_t)xt[i];
                }
            }
            if (hp) {
#if NN_HAS_VECTOR_DOT
                if (hid >= RNN_SIMD_MIN_SIZE) {
                    acc_hh += nn_dot_u8_s8(hp, w_hh_j, hid);
                } else
#endif
                {
                    for (i = 0; i < hid; i++) {
                        acc_hh += (int32_t)w_hh_j[i] * (int32_t)hp[i];
                    }
                }
            }
            {
                const float y = (float)acc_ih * (p->w_ih_scale[j] * x_scale) + p->b_ih[j] +
                                (float)acc_hh * (p->w_hh_scale[j] * hprev_scale) + p->b_hh[j];
                h_work[j] = nn_tanh_f32(y);
                if (h_work[j] < mn) {
                    mn = h_work[j];
                } else if (!(h_work[j] >= mn)) {
                    return CNN_DATA_EXCEPTION;
                }
                if (h_work[j] > mx) {
                    mx = h_work[j];
                }
            }
        }

        // 隐藏状态动态量化（uint8 非对称）。
        // 当 (mx-mn)/255 < 1e-9 且幅值 > 1e-9 时，以 max|h|/128 作 scale，
        // mn<0 时 zp=255，否则 zp=0；近常数向量仍存在量化误差。
        nn_recurrent_quant_params(mn, mx, &scale, &zp);
        quantize_u8(h_work, hid, scale, zp, hq);
        h_scale_seq[t] = scale;
        h_zp_seq[t] = zp;
    }

    if (out) {
        return linear_int8(h_q_seq + (size_t)(seq_len - 1) * hid, hid, out->w, out->w_scale, out->b,
                           out->num_classes, h_scale_seq[seq_len - 1], h_zp_seq[seq_len - 1],
                           logits);
    }
    return CNN_NORMAL;
}
