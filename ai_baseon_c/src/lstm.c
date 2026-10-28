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
 * @file lstm.c
 * @brief LSTM 推理实现（fp32 与 int8 动态量化）
 *
 * 门顺序 i/f/g/o；权重行索引：
 *   i -> row j, f -> row H+j, g -> row 2H+j, o -> row 3H+j
 *
 * 数值约定：
 *   - 整数域累加在 int32 完成，反量化到 float32；
 *   - 隐藏状态逐时间步保留 uint8 非对称的历史规则，零点向零截断；
 *     常规范围不强制包含零。当 (max-min)/255 < 1e-9 时采用幅值兜底：
 *     若 max|h| > 1e-9，则 scale=max|h|/128、zp=0/255；
 *     否则 scale=1e-9、zp=0；近常数向量仍可能存在量化误差；
 *   - int8 路径的零点通过预计算的权重行和折进累加器，
 *     不逐 MAC 减零点；
 *   - 细胞状态 c 保持 float32。
 */

#include "../include/ai_baseon_c/lstm.h"

#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>

#include "../include/ai_baseon_c/cnn.h"
#include "kernels/nn_dot.h"
#include "quantization.h"
#include "nn_activations.h"

/* 覆盖 raw dot 与零点校正的最坏中间值，包括合法的 -128 权重。 */
enum { LSTM_MAX_DOT_SIZE = NN_MAX_MACS,
       LSTM_SIMD_MIN_SIZE = NN_HAS_AVX2 ? 64 : (NN_HAS_NEON ? 32 : 16) };

static bool lstm_shape_is_valid(uint32_t input_size, uint32_t hidden_size,
                                uint32_t seq_len, uint32_t num_classes)
{
    const uint64_t limit = (uint64_t)PTRDIFF_MAX / sizeof(float);
    const uint64_t rows = (uint64_t)hidden_size * 4;

    if (input_size == 0 || hidden_size == 0 || seq_len == 0 ||
        hidden_size > UINT32_MAX / 6) {
        return false;
    }
    return (uint64_t)input_size <= limit / seq_len &&
           (uint64_t)hidden_size <= limit / seq_len && rows <= limit / input_size &&
           rows <= limit / hidden_size && (uint64_t)hidden_size <= limit / 6 &&
           (num_classes == 0 || (uint64_t)hidden_size <= limit / num_classes);
}

/* 固定模型元数据只在入口检查一次，不在每个时间步重复扫描。 */
static bool lstm_rows_are_valid(const float *scales, const float *bias,
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
 * @brief LSTM fp32 前向推理。
 *
 * 隐藏单元按 4 行分块，每行同时累加 4 个门（共 16 个独立累加器），
 * 打破乘加依赖链并复用 x/h 向量读取；每行内的累加顺序与朴素实现
 * 相同（bias -> x 维度 -> h 维度）。
 */
int lstm_forward_fp32(const lstm_fp32_param_t *p, const lstm_out_fp32_t *out, const float *restrict x,
                      uint32_t seq_len, float *restrict c_state, float *restrict h_seq,
                      float *restrict logits)
{
    uint32_t i, j, t;
    const uint32_t in = p ? p->input_size : 0;
    const uint32_t hid = p ? p->hidden_size : 0;

    if (!p || !p->w_ih || !p->w_hh || !p->bias || !x || !c_state || !h_seq) {
        return CNN_POINTER_NULL;
    }
    if (!lstm_shape_is_valid(in, hid, seq_len, out ? out->num_classes : 0) ||
        (out && (!out->w || !out->b || out->hidden_size != hid || out->num_classes == 0)) ||
        (out && !logits)) {
        return CNN_DATA_EXCEPTION;
    }

    // 初始细胞状态为 0。
    for (j = 0; j < hid; j++) {
        c_state[j] = 0.0f;
    }

    for (t = 0; t < seq_len; t++) {
        const float *restrict xt = x + (size_t)t * in;
        const float *restrict hp = (t == 0) ? NULL : h_seq + (size_t)(t - 1) * hid;
        float *restrict ht = h_seq + (size_t)t * hid;

        // 4 行 x 4 门：16 个独立累加器。
        for (j = 0; j + 4 <= hid; j += 4) {
            float yi0 = p->bias[j + 0];
            float yi1 = p->bias[j + 1];
            float yi2 = p->bias[j + 2];
            float yi3 = p->bias[j + 3];
            float yf0 = p->bias[hid + j + 0];
            float yf1 = p->bias[hid + j + 1];
            float yf2 = p->bias[hid + j + 2];
            float yf3 = p->bias[hid + j + 3];
            float yg0 = p->bias[2 * hid + j + 0];
            float yg1 = p->bias[2 * hid + j + 1];
            float yg2 = p->bias[2 * hid + j + 2];
            float yg3 = p->bias[2 * hid + j + 3];
            float yo0 = p->bias[3 * hid + j + 0];
            float yo1 = p->bias[3 * hid + j + 1];
            float yo2 = p->bias[3 * hid + j + 2];
            float yo3 = p->bias[3 * hid + j + 3];
            const float *w_ih_i0 = p->w_ih + (size_t)(j + 0) * in;
            const float *w_ih_i1 = p->w_ih + (size_t)(j + 1) * in;
            const float *w_ih_i2 = p->w_ih + (size_t)(j + 2) * in;
            const float *w_ih_i3 = p->w_ih + (size_t)(j + 3) * in;
            const float *w_ih_f0 = p->w_ih + (size_t)(hid + j + 0) * in;
            const float *w_ih_f1 = p->w_ih + (size_t)(hid + j + 1) * in;
            const float *w_ih_f2 = p->w_ih + (size_t)(hid + j + 2) * in;
            const float *w_ih_f3 = p->w_ih + (size_t)(hid + j + 3) * in;
            const float *w_ih_g0 = p->w_ih + (size_t)(2 * hid + j + 0) * in;
            const float *w_ih_g1 = p->w_ih + (size_t)(2 * hid + j + 1) * in;
            const float *w_ih_g2 = p->w_ih + (size_t)(2 * hid + j + 2) * in;
            const float *w_ih_g3 = p->w_ih + (size_t)(2 * hid + j + 3) * in;
            const float *w_ih_o0 = p->w_ih + (size_t)(3 * hid + j + 0) * in;
            const float *w_ih_o1 = p->w_ih + (size_t)(3 * hid + j + 1) * in;
            const float *w_ih_o2 = p->w_ih + (size_t)(3 * hid + j + 2) * in;
            const float *w_ih_o3 = p->w_ih + (size_t)(3 * hid + j + 3) * in;
            const float *w_hh_i0 = p->w_hh + (size_t)(j + 0) * hid;
            const float *w_hh_i1 = p->w_hh + (size_t)(j + 1) * hid;
            const float *w_hh_i2 = p->w_hh + (size_t)(j + 2) * hid;
            const float *w_hh_i3 = p->w_hh + (size_t)(j + 3) * hid;
            const float *w_hh_f0 = p->w_hh + (size_t)(hid + j + 0) * hid;
            const float *w_hh_f1 = p->w_hh + (size_t)(hid + j + 1) * hid;
            const float *w_hh_f2 = p->w_hh + (size_t)(hid + j + 2) * hid;
            const float *w_hh_f3 = p->w_hh + (size_t)(hid + j + 3) * hid;
            const float *w_hh_g0 = p->w_hh + (size_t)(2 * hid + j + 0) * hid;
            const float *w_hh_g1 = p->w_hh + (size_t)(2 * hid + j + 1) * hid;
            const float *w_hh_g2 = p->w_hh + (size_t)(2 * hid + j + 2) * hid;
            const float *w_hh_g3 = p->w_hh + (size_t)(2 * hid + j + 3) * hid;
            const float *w_hh_o0 = p->w_hh + (size_t)(3 * hid + j + 0) * hid;
            const float *w_hh_o1 = p->w_hh + (size_t)(3 * hid + j + 1) * hid;
            const float *w_hh_o2 = p->w_hh + (size_t)(3 * hid + j + 2) * hid;
            const float *w_hh_o3 = p->w_hh + (size_t)(3 * hid + j + 3) * hid;

            for (i = 0; i < in; i++) {
                const float xv = xt[i];
                yi0 += w_ih_i0[i] * xv;
                yi1 += w_ih_i1[i] * xv;
                yi2 += w_ih_i2[i] * xv;
                yi3 += w_ih_i3[i] * xv;
                yf0 += w_ih_f0[i] * xv;
                yf1 += w_ih_f1[i] * xv;
                yf2 += w_ih_f2[i] * xv;
                yf3 += w_ih_f3[i] * xv;
                yg0 += w_ih_g0[i] * xv;
                yg1 += w_ih_g1[i] * xv;
                yg2 += w_ih_g2[i] * xv;
                yg3 += w_ih_g3[i] * xv;
                yo0 += w_ih_o0[i] * xv;
                yo1 += w_ih_o1[i] * xv;
                yo2 += w_ih_o2[i] * xv;
                yo3 += w_ih_o3[i] * xv;
            }
            if (hp) {
                for (i = 0; i < hid; i++) {
                    const float hv = hp[i];
                    yi0 += w_hh_i0[i] * hv;
                    yi1 += w_hh_i1[i] * hv;
                    yi2 += w_hh_i2[i] * hv;
                    yi3 += w_hh_i3[i] * hv;
                    yf0 += w_hh_f0[i] * hv;
                    yf1 += w_hh_f1[i] * hv;
                    yf2 += w_hh_f2[i] * hv;
                    yf3 += w_hh_f3[i] * hv;
                    yg0 += w_hh_g0[i] * hv;
                    yg1 += w_hh_g1[i] * hv;
                    yg2 += w_hh_g2[i] * hv;
                    yg3 += w_hh_g3[i] * hv;
                    yo0 += w_hh_o0[i] * hv;
                    yo1 += w_hh_o1[i] * hv;
                    yo2 += w_hh_o2[i] * hv;
                    yo3 += w_hh_o3[i] * hv;
                }
            }
            {
                const float gi0 = nn_sigmoid_f32(yi0);
                const float gi1 = nn_sigmoid_f32(yi1);
                const float gi2 = nn_sigmoid_f32(yi2);
                const float gi3 = nn_sigmoid_f32(yi3);
                const float gf0 = nn_sigmoid_f32(yf0);
                const float gf1 = nn_sigmoid_f32(yf1);
                const float gf2 = nn_sigmoid_f32(yf2);
                const float gf3 = nn_sigmoid_f32(yf3);
                const float gg0 = nn_tanh_f32(yg0);
                const float gg1 = nn_tanh_f32(yg1);
                const float gg2 = nn_tanh_f32(yg2);
                const float gg3 = nn_tanh_f32(yg3);
                const float go0 = nn_sigmoid_f32(yo0);
                const float go1 = nn_sigmoid_f32(yo1);
                const float go2 = nn_sigmoid_f32(yo2);
                const float go3 = nn_sigmoid_f32(yo3);
                const float cn0 = gf0 * c_state[j + 0] + gi0 * gg0;
                const float cn1 = gf1 * c_state[j + 1] + gi1 * gg1;
                const float cn2 = gf2 * c_state[j + 2] + gi2 * gg2;
                const float cn3 = gf3 * c_state[j + 3] + gi3 * gg3;
                c_state[j + 0] = cn0;
                c_state[j + 1] = cn1;
                c_state[j + 2] = cn2;
                c_state[j + 3] = cn3;
                ht[j + 0] = go0 * nn_tanh_f32(cn0);
                ht[j + 1] = go1 * nn_tanh_f32(cn1);
                ht[j + 2] = go2 * nn_tanh_f32(cn2);
                ht[j + 3] = go3 * nn_tanh_f32(cn3);
                /* 不把无效隐藏状态传播到后续时间步。 */
                if (!isfinite(ht[j + 0]) || !isfinite(ht[j + 1]) ||
                    !isfinite(ht[j + 2]) || !isfinite(ht[j + 3])) {
                    return CNN_DATA_EXCEPTION;
                }
            }
        }
        // 尾部（hidden 不是 4 的倍数时）。
        for (; j < hid; j++) {
            float yi = p->bias[j];
            float yf = p->bias[hid + j];
            float yg = p->bias[2 * hid + j];
            float yo = p->bias[3 * hid + j];
            const float *w_ih_i = p->w_ih + (size_t)j * in;
            const float *w_ih_f = p->w_ih + (size_t)(hid + j) * in;
            const float *w_ih_g = p->w_ih + (size_t)(2 * hid + j) * in;
            const float *w_ih_o = p->w_ih + (size_t)(3 * hid + j) * in;
            const float *w_hh_i = p->w_hh + (size_t)j * hid;
            const float *w_hh_f = p->w_hh + (size_t)(hid + j) * hid;
            const float *w_hh_g = p->w_hh + (size_t)(2 * hid + j) * hid;
            const float *w_hh_o = p->w_hh + (size_t)(3 * hid + j) * hid;

            for (i = 0; i < in; i++) {
                const float xv = xt[i];
                yi += w_ih_i[i] * xv;
                yf += w_ih_f[i] * xv;
                yg += w_ih_g[i] * xv;
                yo += w_ih_o[i] * xv;
            }
            if (hp) {
                for (i = 0; i < hid; i++) {
                    const float hv = hp[i];
                    yi += w_hh_i[i] * hv;
                    yf += w_hh_f[i] * hv;
                    yg += w_hh_g[i] * hv;
                    yo += w_hh_o[i] * hv;
                }
            }
            {
                const float gi = nn_sigmoid_f32(yi);
                const float gf = nn_sigmoid_f32(yf);
                const float gg = nn_tanh_f32(yg);
                const float go = nn_sigmoid_f32(yo);
                const float cnew = gf * c_state[j] + gi * gg;
                c_state[j] = cnew;
                ht[j] = go * nn_tanh_f32(cnew);
                if (!isfinite(ht[j])) {
                    return CNN_DATA_EXCEPTION;
                }
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
 * @brief LSTM int8 动态量化前向推理。
 *
 * work 布局（长度 6*hidden）：
 *   [0,  H)       -> y_i
 *   [H,  2H)      -> y_f
 *   [2H, 3H)      -> y_g
 *   [3H, 4H)      -> y_o
 *   [4H, 5H)      -> h_raw（tanh 后、量化前）
 *   [5H, 6H)      -> c_state
 *
 * 零点折进权重行和：acc = sum(w*q) - zp*rowsum，整数运算等价于逐 MAC
 * 减零点，但省去每次乘加前的减法。
 */
int lstm_forward_int8(const lstm_int8_param_t *p, const lstm_out_int8_t *out, const uint8_t *restrict x_q,
                      uint32_t seq_len, float x_scale, int x_zp, float *restrict work,
                      uint8_t *restrict h_q_seq, float *restrict h_scale_seq,
                      int *restrict h_zp_seq, float *restrict logits)
{
    uint32_t i, j, t;
    const uint32_t in = p ? p->input_size : 0;
    const uint32_t hid = p ? p->hidden_size : 0;

    if (!p || !p->w_ih || !p->w_ih_scale || !p->w_ih_rowsum || !p->w_hh || !p->w_hh_scale ||
        !p->w_hh_rowsum || !p->bias || !x_q || !work || !h_q_seq || !h_scale_seq || !h_zp_seq) {
        return CNN_POINTER_NULL;
    }
    if (!lstm_shape_is_valid(in, hid, seq_len, out ? out->num_classes : 0) ||
        in > LSTM_MAX_DOT_SIZE || hid > LSTM_MAX_DOT_SIZE ||
        !isfinite(x_scale) || x_scale <= 0.0f || x_zp < 0 || x_zp > 255 ||
        (out && (!out->w || !out->w_scale || !out->b || out->hidden_size != hid ||
                 out->num_classes == 0)) ||
        (out && !logits)) {
        return CNN_DATA_EXCEPTION;
    }
    if (!lstm_rows_are_valid(p->w_ih_scale, p->bias, p->w_ih_rowsum, 4 * hid, in) ||
        !lstm_rows_are_valid(p->w_hh_scale, p->bias, p->w_hh_rowsum, 4 * hid, hid) ||
        (out && !lstm_rows_are_valid(out->w_scale, out->b, NULL, out->num_classes, hid))) {
        return CNN_DATA_EXCEPTION;
    }

    /* 先校验空指针与尺寸，再形成 work 子区间指针。 */
    float *restrict yi = work;
    float *restrict yf = work + hid;
    float *restrict yg = work + (size_t)2 * hid;
    float *restrict yo = work + (size_t)3 * hid;
    float *restrict h_raw = work + (size_t)4 * hid;
    float *restrict c_state = work + (size_t)5 * hid;

    for (j = 0; j < hid; j++) {
        c_state[j] = 0.0f;
    }

    for (t = 0; t < seq_len; t++) {
        const uint8_t *restrict xt = x_q + (size_t)t * in;
        const uint8_t *restrict hp = (t == 0) ? NULL : h_q_seq + (size_t)(t - 1) * hid;
        const float hprev_scale = (t == 0) ? 1.0f : h_scale_seq[t - 1];
        const int hprev_zp = (t == 0) ? 0 : h_zp_seq[t - 1];
        uint8_t *restrict hq = h_q_seq + (size_t)t * hid;
        float mn, mx, scale;
        int32_t zp;

        for (j = 0; j < hid; j++) {
            const int8_t *w_ih_i = p->w_ih + (size_t)j * in;
            const int8_t *w_ih_f = p->w_ih + (size_t)(hid + j) * in;
            const int8_t *w_ih_g = p->w_ih + (size_t)(2 * hid + j) * in;
            const int8_t *w_ih_o = p->w_ih + (size_t)(3 * hid + j) * in;
            const int8_t *w_hh_i = p->w_hh + (size_t)j * hid;
            const int8_t *w_hh_f = p->w_hh + (size_t)(hid + j) * hid;
            const int8_t *w_hh_g = p->w_hh + (size_t)(2 * hid + j) * hid;
            const int8_t *w_hh_o = p->w_hh + (size_t)(3 * hid + j) * hid;
            // 零点折进权重行和（整数运算，结果与逐 MAC 减零点完全一致）。
            int32_t acc_ih_i = -x_zp * p->w_ih_rowsum[j];
            int32_t acc_ih_f = -x_zp * p->w_ih_rowsum[hid + j];
            int32_t acc_ih_g = -x_zp * p->w_ih_rowsum[2 * hid + j];
            int32_t acc_ih_o = -x_zp * p->w_ih_rowsum[3 * hid + j];
            int32_t acc_hh_i = -hprev_zp * p->w_hh_rowsum[j];
            int32_t acc_hh_f = -hprev_zp * p->w_hh_rowsum[hid + j];
            int32_t acc_hh_g = -hprev_zp * p->w_hh_rowsum[2 * hid + j];
            int32_t acc_hh_o = -hprev_zp * p->w_hh_rowsum[3 * hid + j];

#if NN_HAS_VECTOR_DOT
            if (in >= LSTM_SIMD_MIN_SIZE) {
                const int8_t *const weights[4] = {w_ih_i, w_ih_f, w_ih_g, w_ih_o};
                int32_t sums[4];
                nn_dot4_u8_s8(xt, weights, in, sums);
                acc_ih_i += sums[0];
                acc_ih_f += sums[1];
                acc_ih_g += sums[2];
                acc_ih_o += sums[3];
            } else
#endif
            {
                for (i = 0; i < in; i++) {
                    const int32_t xv = (int32_t)xt[i];
                    acc_ih_i += (int32_t)w_ih_i[i] * xv;
                    acc_ih_f += (int32_t)w_ih_f[i] * xv;
                    acc_ih_g += (int32_t)w_ih_g[i] * xv;
                    acc_ih_o += (int32_t)w_ih_o[i] * xv;
                }
            }
            if (hp) {
#if NN_HAS_VECTOR_DOT
                if (hid >= LSTM_SIMD_MIN_SIZE) {
                    const int8_t *const weights[4] = {w_hh_i, w_hh_f, w_hh_g, w_hh_o};
                    int32_t sums[4];
                    nn_dot4_u8_s8(hp, weights, hid, sums);
                    acc_hh_i += sums[0];
                    acc_hh_f += sums[1];
                    acc_hh_g += sums[2];
                    acc_hh_o += sums[3];
                } else
#endif
                {
                    for (i = 0; i < hid; i++) {
                        const int32_t hv = (int32_t)hp[i];
                        acc_hh_i += (int32_t)w_hh_i[i] * hv;
                        acc_hh_f += (int32_t)w_hh_f[i] * hv;
                        acc_hh_g += (int32_t)w_hh_g[i] * hv;
                        acc_hh_o += (int32_t)w_hh_o[i] * hv;
                    }
                }
            }
            // 偏置已合并（bias_ih + bias_hh），每个门单元只加一次。
            yi[j] = (float)acc_ih_i * (p->w_ih_scale[j] * x_scale) +
                    (float)acc_hh_i * (p->w_hh_scale[j] * hprev_scale) + p->bias[j];
            yf[j] = (float)acc_ih_f * (p->w_ih_scale[hid + j] * x_scale) +
                    (float)acc_hh_f * (p->w_hh_scale[hid + j] * hprev_scale) + p->bias[hid + j];
            yg[j] = (float)acc_ih_g * (p->w_ih_scale[2 * hid + j] * x_scale) +
                    (float)acc_hh_g * (p->w_hh_scale[2 * hid + j] * hprev_scale) +
                    p->bias[2 * hid + j];
            yo[j] = (float)acc_ih_o * (p->w_ih_scale[3 * hid + j] * x_scale) +
                    (float)acc_hh_o * (p->w_hh_scale[3 * hid + j] * hprev_scale) +
                    p->bias[3 * hid + j];
        }

        mn = 1e30f;
        mx = -1e30f;
        for (j = 0; j < hid; j++) {
            const float gi = nn_sigmoid_f32(yi[j]);
            const float gf = nn_sigmoid_f32(yf[j]);
            const float gg = nn_tanh_f32(yg[j]);
            const float go = nn_sigmoid_f32(yo[j]);
            const float cnew = gf * c_state[j] + gi * gg;
            c_state[j] = cnew;
            h_raw[j] = go * nn_tanh_f32(cnew);
            // 融合 min/max；无序比较同时拒绝 NaN，避免额外的逐单元检查。
            if (h_raw[j] < mn) {
                mn = h_raw[j];
            } else if (!(h_raw[j] >= mn)) {
                return CNN_DATA_EXCEPTION;
            }
            if (h_raw[j] > mx) {
                mx = h_raw[j];
            }
        }

        // 隐藏状态动态量化（uint8 非对称；退化情形对称兜底）。
        nn_recurrent_quant_params(mn, mx, &scale, &zp);
        quantize_u8(h_raw, hid, scale, zp, hq);
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
