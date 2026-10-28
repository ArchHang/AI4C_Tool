/*
 * Copyright (c) 2025, VeriSilicon Holdings Co., Ltd. All rights reserved
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
 * @file cnn.c
 * @brief 可移植 CNN 张量算子库实现（fp32 与 int8 量化推理）
 *
 * 文件组织：
 *   1. fp32 算子：卷积+BN、LeakyReLU、Softmax、argmax、全连接、池化、Sigmoid
 *   2. int8/uint8 量化算子：
 *      - 统一入口（结构体 API）：conv2d_int8_run / linear_int8_run /
 *        maxpool_u8_run / quantize_u8_run，与 fp32 的
 *        conv2d_bn_no_bias(in, cfg, out) 同构，量化参数（scale/zp）、
 *        量化模式（静态/动态）、权重指针全部封装在结构体中；
 *      - 共用整数内核：stride=1 界内窗口按 8 个输出位置分块，
 *        边界和其它步长逐点计算；动态模式可保存 acc 避免重复 MAC；
 *      - 兼容层（散参数旧 API）：conv1d_int8 / conv2d_int8 /
 *        conv1d_int8_dynamic / conv2d_int8_dynamic / linear_int8 /
 *        quantize_u8，内部构造结构体转调统一入口，行为逐位一致。
 *
 * 零点分派（统一入口自动完成，静态/动态同规则，零额外内存）：
 *   - 输入为对称量化或零点为 0（in_act.zp == 0）：走免减法快路径，
 *     内积不做“减零点”，每个 MAC 少一次减法；
 *   - 输入为非对称量化（零点非 0）：界内窗口把零点折进权重行和
 *     （acc = sum(w*x) - zp*rowsum，整数恒等变换，rowsum 运行时一次
 *     累加，不占额外内存），边界窗口逐 tap 减零点。
 *
 * 数值约定：
 *   - int8 量化链在整数域（int32）完成累加，最后才反量化为 float，
 *     保证与 Python 端 int32 精确复现一致；
 *   - 量化舍入保持 lrintf 的语义；M4/M7 DSP hard-float 可用有界硬件转换；
 *   - 卷积累加顺序固定为 channel -> kernel_row -> kernel_col，
 *     浮点路径与朴素参考实现保持逐位一致。
 *   - 动态量化（无需校准）的激活重量化在整数域完成：逐点只有
 *     int32 累加 + int64 乘加 + 移位，不产生逐点浮点转换；
 *     每通道计算量化系数（见 NN_QUANT_BITS）；极端系数采用安全浮点回退。
 */

#include "../include/ai_baseon_c/cnn.h"

#include <limits.h>
#include <stddef.h>

#include "kernels/nn_dot.h"
#include "quantization.h"

#define BN_EPS (1e-5f)

static int32_t nn_shape_product(uint32_t a, uint32_t b, uint32_t *result);

typedef struct nn_fir_param {
    uint32_t input_size;
    uint32_t output_size;
    uint32_t channels;
    uint32_t kernel;
    float scale;
    float shift;
} nn_fir_param_t;

// Isolate the contiguous FIR loop from validation so GCC preserves loop vectorization.
#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
static void nn_fir_accumulate(const float *restrict input, const float *restrict weights,
                              float *restrict output, const nn_fir_param_t *param)
{
    const uint32_t output_size = param->output_size;
    for (uint32_t index = 0; index < output_size; index++) {
        output[index] = 0.0f;
    }
    for (uint32_t channel = 0; channel < param->channels; channel++) {
        const float *values = input + (size_t)channel * param->input_size;
        const float *kernel = weights + (size_t)channel * param->kernel;
        for (uint32_t tap = 0; tap < param->kernel; tap++) {
            const float weight = kernel[tap];
            const float *source = values + tap;
            for (uint32_t index = 0; index < output_size; index++) {
                output[index] += weight * source[index];
            }
        }
    }
    for (uint32_t index = 0; index < output_size; index++) {
        output[index] = output[index] * param->scale + param->shift;
    }
}

/**
 * @brief 计算卷积/池化输出长度（valid 输出）。
 *
 * 非法几何、零步长或输出长度溢出时返回 0。
 */
uint32_t cal_conv_out_len(uint32_t raw_len, uint32_t pad_len, uint32_t filter_len,
                          uint32_t stride) {
    if (raw_len == 0 || filter_len == 0 || stride == 0
        || pad_len > (UINT32_MAX - raw_len) / 2
        || filter_len > raw_len + 2 * pad_len) {
        return 0;
    }
    const uint32_t extent = (raw_len + 2 * pad_len - filter_len) / stride;
    return extent == UINT32_MAX ? 0 : extent + 1;
}

/* =============================== fp32 算子 =============================== */

/**
 * @brief fp32 卷积 + BN（无卷积 bias），支持 1D/2D、zero-padding、任意 stride。
 *
 * 内部把 BN 折叠为每通道 scale/shift：
 *   scale = gamma / sqrt(var + BN_EPS)
 *   shift = beta - mean * scale
 * 推理公式：y = acc * scale + shift。
 *
 * 性能路径：
 *   - 1D（row=1、核 row=1）：
 *       * pad=0、stride=1 时使用 FIR 散射形式（内层对整行输出做连续累加，
 *         读写连续，编译器可向量化为 AVX2 FMA）；
 *       * pad=0、stride>1 时使用输出驻留循环；
 *       * pad>0 时使用虚拟 padding（窗口完全在界内走无分支快速路径，
 *         边界像素逐 tap 判断越界输入按 0 参与累加）。
 *   - 2D：指针游走 + 内层连续读取，累加顺序与朴素实现一致。
 *
 * 注意：输入输出 buffer 不允许重叠。
 *
 * @param input_feat  [in]  输入特征图；1D 时 row=1
 * @param param       [in]  卷积配置
 * @param output_feat [out] 输出特征图；data 由调用方提供，容量至少
 *                          filter_num * out_row * out_col
 * @return CNN_NORMAL / CNN_POINTER_NULL / CNN_DATA_EXCEPTION
 */
int conv2d_bn_no_bias(conv2d_data_t* input_feat, conv2d_config_t* param, conv2d_data_t* output_feat) {
    const batch_norm2d_t* bn = NULL;
    const conv2d_filter_t* filter = NULL;
    const float* restrict in = NULL;
    const float* restrict w = NULL;
    float* restrict out = NULL;
    uint32_t in_row = 0, in_col = 0;
    uint32_t f_row = 0, f_col = 0, f_chan = 0, f_num = 0;
    uint32_t out_row = 0, out_col = 0, out_chan = 0;
    uint32_t i = 0, j = 0, k = 0, ii = 0, jj = 0, kk = 0;

    // 指针与关键字段非空检查。
    if (!input_feat || !input_feat->data || !param || !param->bn || !param->bn->mean ||
        !param->bn->var || !param->bn->gamma || !param->bn->beta || !param->filter ||
        !param->filter->data || !output_feat || !output_feat->data) {
        return CNN_POINTER_NULL;
    }

    bn = param->bn;
    filter = param->filter;

    // 形状合法性检查：stride>=1、通道匹配、padding 不溢出、核不超界、BN 长度匹配。
    // 注意溢出保护：pad > (UINT32_MAX - row)/2 时 2*pad+row 可能溢出，需先判断。
    if (param->stride == 0 || input_feat->row == 0 || input_feat->col == 0
        || input_feat->channel == 0 || filter->row == 0 || filter->col == 0
        || filter->filter_num == 0 || input_feat->channel != filter->channel
        || param->stride > (uint32_t)INT32_MAX
        || input_feat->row > (uint32_t)INT32_MAX || input_feat->col > (uint32_t)INT32_MAX
        || param->pad > ((uint32_t)INT32_MAX - input_feat->row) / 2
        || param->pad > ((uint32_t)INT32_MAX - input_feat->col) / 2
        || filter->filter_num != bn->size ||
        filter->row > 2 * param->pad + input_feat->row ||
        filter->col > 2 * param->pad + input_feat->col) {
        return CNN_DATA_EXCEPTION;
    }

    in_row = input_feat->row;
    in_col = input_feat->col;
    f_row = filter->row;
    f_col = filter->col;
    f_chan = filter->channel;
    f_num = filter->filter_num;

    // 输出维度：1D 时行维恒为 1（与库内其它算子约定一致）。
    out_row = (in_row == 1 && f_row == 1) ? 1
        : cal_conv_out_len(in_row, param->pad, f_row, param->stride);
    out_col = cal_conv_out_len(in_col, param->pad, f_col, param->stride);
    out_chan = f_num;

    uint32_t input_size, weight_size, output_size;
    if (!nn_shape_product(in_row, in_col, &input_size)
        || !nn_shape_product(input_size, f_chan, &input_size)
        || !nn_shape_product(f_row, f_col, &weight_size)
        || !nn_shape_product(weight_size, f_chan, &weight_size)
        || !nn_shape_product(weight_size, f_num, &weight_size)
        || !nn_shape_product(out_row, out_col, &output_size)
        || !nn_shape_product(output_size, out_chan, &output_size)
        || input_size > (uint32_t)INT32_MAX / sizeof(float)
        || weight_size > (uint32_t)INT32_MAX / sizeof(float)
        || output_size > (uint32_t)INT32_MAX / sizeof(float)) {
        return CNN_DATA_EXCEPTION;
    }
    for (i = 0; i < out_chan; i++) {
        if (!isfinite(bn->mean[i]) || !isfinite(bn->var[i]) || bn->var[i] + BN_EPS <= 0.0f
            || !isfinite(bn->gamma[i]) || !isfinite(bn->beta[i])) {
            return CNN_DATA_EXCEPTION;
        }
    }

    in = input_feat->data;
    w = filter->data;
    out = output_feat->data;

    output_feat->row = out_row;
    output_feat->col = out_col;
    output_feat->channel = out_chan;

    // ------------------------- Fast path 1: 1D 卷积 -------------------------
    // 单行输入 + 单行核，任意 stride。
    // stride=1 时使用 FIR 散射：外层遍历 通道 x 卷积核抽头，内层对整行输出做
    // 连续累加。内层循环读写都连续、权重是标量，编译器可自动向量化（AVX2 FMA）。
    // 每个输出位置的累加顺序仍然是 channel(0..C-1) -> tap(0..K-1)，
    // 与参考实现完全一致（保证逐位一致）。
    if (in_row == 1 && f_row == 1) {
        for (i = 0; i < out_chan; i++) {
            const float* w_i = w + i * (f_chan * f_col);
            float* out_i = out + i * out_col;
            const float scale = bn->gamma[i] / sqrtf(bn->var[i] + BN_EPS);
            const float shift = bn->beta[i] - bn->mean[i] * scale;

            if (param->pad == 0) {
                // 无 padding：保持原有最优路径（FIR 散射 / 输出驻留），性能零损失。
                if (param->stride == 1) {
                    const nn_fir_param_t fir = {in_col, out_col, f_chan, f_col, scale, shift};
                    nn_fir_accumulate(in, w_i, out_i, &fir);
                } else {
                    // stride>1：输出驻留循环，每个输出点独立累加（无重叠写）。
                    for (j = 0; j < out_col; j++) {
                        float acc = 0.0f;
                        for (ii = 0; ii < f_chan; ii++) {
                            const float* in_ii = in + ii * in_col;
                            const float* w_ii = w_i + ii * f_col;
                            for (kk = 0; kk < f_col; kk++) {
                                acc += w_ii[kk] * in_ii[j * param->stride + kk];
                            }
                        }
                        out_i[j] = acc * scale + shift;
                    }
                }
            } else {
                // 虚拟 padding：输出驻留；窗口完全在界内走无分支快速路径，
                // 边界像素逐 tap 判断（越界输入按 0 参与累加）。
                for (j = 0; j < out_col; j++) {
                    const int s = (int)(j * param->stride) - (int)param->pad;
                    const int inside = (s >= 0 && (uint32_t)(s + f_col) <= in_col);
                    float acc = 0.0f;
                    for (ii = 0; ii < f_chan; ii++) {
                        const float* w_ii = w_i + ii * f_col;
                        if (inside) {
                            const float* in_ii = in + ii * in_col + (uint32_t)s;
                            for (kk = 0; kk < f_col; kk++) {
                                acc += w_ii[kk] * in_ii[kk];
                            }
                        } else {
                            const float* in_ii = in + ii * in_col;
                            for (kk = 0; kk < f_col; kk++) {
                                const int p = s + (int)kk;
                                if (p >= 0 && p < (int)in_col) {
                                    acc += w_ii[kk] * in_ii[p];
                                }
                            }
                        }
                    }
                    out_i[j] = acc * scale + shift;
                }
            }
        }

        return CNN_NORMAL;
    }

    // ------------------------- Fast path 2: 通用 2D 卷积 -------------------------
    // 循环顺序与累加顺序（channel -> kernel_row -> kernel_col）与参考实现一致，
    // 但全部改为指针游走：输出平面顺序写入，内层 kernel-col 循环读写连续 float，
    // 编译器可生成紧凑的 load-FMA 序列。
    for (i = 0; i < out_chan; i++) {
        const float* w_i = w + i * (f_chan * f_row * f_col);
        float* out_i = out + i * (out_row * out_col);
        const float scale = bn->gamma[i] / sqrtf(bn->var[i] + BN_EPS);
        const float shift = bn->beta[i] - bn->mean[i] * scale;

        if (param->pad == 0) {
            // 无 padding：指针游走，输出连续写入。
            for (j = 0; j < out_row; j++) {
                const float* row_base = in + j * param->stride * in_col;
                for (k = 0; k < out_col; k++) {
                    const float* col_base = row_base + k * param->stride;
                    float acc = 0.0f;
                    for (ii = 0; ii < f_chan; ii++) {
                        const float* in_c = col_base + ii * (in_row * in_col);
                        const float* w_c = w_i + ii * (f_row * f_col);
                        for (jj = 0; jj < f_row; jj++) {
                            const float* in_r = in_c + jj * in_col;
                            const float* w_r = w_c + jj * f_col;
                            for (kk = 0; kk < f_col; kk++) {
                                acc += w_r[kk] * in_r[kk];
                            }
                        }
                    }
                    *out_i++ = acc * scale + shift;
                }
            }
        } else {
            // 虚拟 padding：行/列窗口都完全在界内时走无分支快速路径，
            // 否则逐 tap 判断（越界输入按 0）。
            for (j = 0; j < out_row; j++) {
                const int r0 = (int)(j * param->stride) - (int)param->pad;
                const int r_in = (r0 >= 0 && (uint32_t)(r0 + f_row) <= in_row);
                for (k = 0; k < out_col; k++) {
                    const int c0 = (int)(k * param->stride) - (int)param->pad;
                    const int c_in = (c0 >= 0 && (uint32_t)(c0 + f_col) <= in_col);
                    float acc = 0.0f;
                    if (r_in && c_in) {
                        const float* col_base = in + (uint32_t)r0 * in_col + (uint32_t)c0;
                        for (ii = 0; ii < f_chan; ii++) {
                            const float* in_c = col_base + ii * (in_row * in_col);
                            const float* w_c = w_i + ii * (f_row * f_col);
                            for (jj = 0; jj < f_row; jj++) {
                                const float* in_r = in_c + jj * in_col;
                                const float* w_r = w_c + jj * f_col;
                                for (kk = 0; kk < f_col; kk++) {
                                    acc += w_r[kk] * in_r[kk];
                                }
                            }
                        }
                    } else {
                        // 边界窗口：逐元素检查越界（行越界整行跳过，列越界按 0 处理）。
                        for (ii = 0; ii < f_chan; ii++) {
                            const float* w_c = w_i + ii * (f_row * f_col);
                            for (jj = 0; jj < f_row; jj++) {
                                const int rr = r0 + (int)jj;
                                if (rr < 0 || rr >= (int)in_row) {
                                    continue;
                                }
                                const float* w_r = w_c + jj * f_col;
                                for (kk = 0; kk < f_col; kk++) {
                                    const int cc = c0 + (int)kk;
                                    if (cc >= 0 && cc < (int)in_col) {
                                        acc += w_r[kk] * in[ii * (in_row * in_col) +
                                                            (uint32_t)rr * in_col + (uint32_t)cc];
                                    }
                                }
                            }
                        }
                    }
                    *out_i++ = acc * scale + shift;
                }
            }
        }
    }

    return CNN_NORMAL;
}

/**
 * @brief LeakyReLU 激活。
 *
 * y = x >= 0 ? x : x * neg_slope。
 * 支持 in-place（out == inp）与非 in-place 两种形式。
 *
 * @param neg_slope 负半轴斜率；传 0 即 ReLU
 * @param inp       [in]  输入数据
 * @param inp_size  [in]  输入长度
 * @param out       [out] 输出数据
 * @return CNN_NORMAL / CNN_POINTER_NULL
 */
int leaky_relu(float neg_slope, float* inp, uint32_t inp_size, float* out) {
    uint32_t i = 0;

    if (!inp || !out) {
        return CNN_POINTER_NULL;
    }

    if (inp == out) {
        // 原地版本：直接读写同一 buffer。
        for (; i < inp_size; i++) {
            const float v = inp[i];
            out[i] = (v < 0.0f) ? v * neg_slope : v;
        }
    } else {
        // 非原地版本：用 restrict 指针帮助编译器向量化。
        const float* restrict x = inp;
        float* restrict o = out;
        for (; i < inp_size; i++) {
            const float v = x[i];
            o[i] = (v < 0.0f) ? v * neg_slope : v;
        }
    }

    return CNN_NORMAL;
}

/**
 * @brief Softmax 激活（数值稳定）。
 *
 * 先减最大值再 exp，避免 exp 溢出为 inf/nan，最后归一化使和为 1。
 * 支持 in-place（out == inp）。
 *
 * @param inp  [in]  输入数据
 * @param size [in]  输入长度（>0）
 * @param out  [out] 输出概率
 * @return CNN_NORMAL / CNN_POINTER_NULL
 */
int softmax(const float* inp, uint32_t size, float* out) {
    float maxv, sum;
    uint32_t i;

    if (!inp || !out || size == 0) {
        return CNN_POINTER_NULL;
    }

    // 第一遍找最大值，用于数值稳定。
    maxv = inp[0];
    if (!isfinite(maxv)) {
        return CNN_DATA_EXCEPTION;
    }
    for (i = 1; i < size; i++) {
        if (!isfinite(inp[i])) {
            return CNN_DATA_EXCEPTION;
        }
        if (inp[i] > maxv) {
            maxv = inp[i];
        }
    }

    // 第二遍计算 exp(x - max) 并求和。
    sum = 0.0f;
    for (i = 0; i < size; i++) {
        out[i] = expf(inp[i] - maxv);
        sum += out[i];
    }

    // 归一化。
    for (i = 0; i < size; i++) {
        out[i] /= sum;
    }

    return CNN_NORMAL;
}

/**
 * @brief argmax：返回最大值下标（并列时返回第一个）。
 *
 * @param inp     [in]  输入数据
 * @param size    [in]  输入长度
 * @param max_val [out] 可选；返回最大值，传 NULL 忽略
 * @return 最大值下标；参数非法时返回 0
 */
uint32_t argmax(const float* inp, uint32_t size, float* max_val) {
    uint32_t i, idx = 0;
    float m;

    if (!inp || size == 0) {
        if (max_val) {
            *max_val = 0.0f;
        }
        return 0;
    }

    m = inp[0];
    for (i = 1; i < size; i++) {
        if (inp[i] > m) {
            m = inp[i];
            idx = i;
        }
    }
    if (max_val) {
        *max_val = m;
    }
    return idx;
}

/* ============================= 量化算子 ============================= */

/* Integer convolution implementation. Public buffers never allocate on the heap. */

typedef struct nn_conv_shape {
    uint32_t output_row;
    uint32_t output_col;
    uint32_t input_plane;
    uint32_t output_plane;
    uint32_t kernel_plane;
    uint32_t kernel_size;
    uint32_t output_size;
} nn_conv_shape_t;

typedef struct nn_requant {
    int64_t multiplier;
    int64_t bias;
    float real_multiplier;
    float real_bias;
    float scale;
    int32_t has_fixed;
} nn_requant_t;

typedef struct nn_conv_context {
    const conv2d_data_i8_t *input;
    const conv2d_config_i8_t *config;
    const int8_t *weights;
    const nn_conv_shape_t *shape;
    int32_t correction;
} nn_conv_context_t;

static int32_t nn_shape_product(uint32_t a, uint32_t b, uint32_t *result)
{
    if (a == 0 || b == 0 || a > (uint32_t)INT32_MAX / b) {
        return 0;
    }
    *result = a * b;
    return 1;
}

static int32_t nn_quant_valid(const quant_param_t *quant)
{
    return nn_quant_metadata_is_valid(quant->scale, quant->zp);
}

static int32_t nn_conv_validate(const conv2d_data_i8_t *input,
                                const conv2d_config_i8_t *config, nn_conv_shape_t *shape)
{
    uint32_t total, weight_size;
    const conv2d_filter_i8_t *filter;
    if (input == NULL || input->data == NULL || config == NULL || config->filter == NULL
        || config->out_act == NULL) {
        return CNN_POINTER_NULL;
    }
    filter = config->filter;
    if (filter->data == NULL || filter->scale == NULL || filter->bias == NULL) {
        return CNN_POINTER_NULL;
    }
    if (!nn_quant_valid(&config->in_act)
        || (config->dynamic != 0 && config->dynamic != 1)
        || (!config->dynamic && !nn_quant_valid(config->out_act))
        || input->row == 0 || input->col == 0 || input->channel == 0
        || filter->row == 0 || filter->col == 0 || filter->filter_num == 0
        || filter->channel != input->channel || config->stride == 0
        || config->stride > (uint32_t)INT32_MAX
        || input->row > (uint32_t)INT32_MAX || input->col > (uint32_t)INT32_MAX
        || config->pad > ((uint32_t)INT32_MAX - input->row) / 2
        || config->pad > ((uint32_t)INT32_MAX - input->col) / 2
        || filter->row > input->row + 2 * config->pad
        || filter->col > input->col + 2 * config->pad) {
        return CNN_DATA_EXCEPTION;
    }
    shape->output_row = input->row == 1 && filter->row == 1 ? 1
        : cal_conv_out_len(input->row, config->pad, filter->row, config->stride);
    shape->output_col = cal_conv_out_len(input->col, config->pad,
                                        filter->col, config->stride);
    if (!nn_shape_product(input->row, input->col, &shape->input_plane)
        || !nn_shape_product(shape->input_plane, input->channel, &total)
        || !nn_shape_product(filter->row, filter->col, &shape->kernel_plane)
        || !nn_shape_product(shape->kernel_plane, input->channel, &shape->kernel_size)
        || shape->kernel_size > NN_MAX_MACS
        || !nn_shape_product(shape->kernel_size, filter->filter_num, &weight_size)
        || !nn_shape_product(shape->output_row, shape->output_col, &shape->output_plane)
        || !nn_shape_product(shape->output_plane, filter->filter_num, &shape->output_size)
        || shape->output_size > (uint32_t)INT32_MAX / sizeof(int32_t)) {
        return CNN_DATA_EXCEPTION;
    }
    for (uint32_t channel = 0; channel < filter->filter_num; channel++) {
        const float multiplier = filter->scale[channel] * config->in_act.scale;
        if (!isfinite(filter->scale[channel]) || filter->scale[channel] <= 0.0f
            || !isfinite(multiplier) || multiplier <= 0.0f
            || !isfinite(filter->bias[channel])) {
            return CNN_DATA_EXCEPTION;
        }
    }
    return CNN_NORMAL;
}

static int32_t nn_weight_sum(const int8_t *weights, uint32_t size)
{
    int32_t sum = 0;
    for (uint32_t index = 0; index < size; index++) {
        sum += (int32_t)weights[index];
    }
    return sum;
}

static int32_t nn_conv_point(const nn_conv_context_t *context, int32_t row, int32_t col)
{
    const conv2d_data_i8_t *input = context->input;
    const conv2d_config_i8_t *config = context->config;
    const int8_t *weights = context->weights;
    const nn_conv_shape_t *shape = context->shape;
    const conv2d_filter_i8_t *filter = config->filter;
    const int32_t inside = row >= 0 && col >= 0
        && (uint32_t)row <= input->row - (filter->row <= input->row ? filter->row : 0)
        && filter->row <= input->row
        && (uint32_t)col <= input->col - (filter->col <= input->col ? filter->col : 0)
        && filter->col <= input->col;
    int32_t sum = 0;
    for (uint32_t channel = 0; channel < input->channel; channel++) {
        const uint8_t *plane = input->data + (size_t)channel * shape->input_plane;
        const int8_t *kernel = weights + (size_t)channel * shape->kernel_plane;
        for (uint32_t kernel_row = 0; kernel_row < filter->row; kernel_row++) {
            const int32_t source_row = row + (int32_t)kernel_row;
            const int8_t *kernel_values = kernel + (size_t)kernel_row * filter->col;
            if (inside) {
                sum += nn_dot_u8_s8(plane + (size_t)source_row * input->col + (uint32_t)col,
                                    kernel_values, filter->col);
            } else if (source_row >= 0 && (uint32_t)source_row < input->row) {
                for (uint32_t kernel_col = 0; kernel_col < filter->col; kernel_col++) {
                    const int32_t source_col = col + (int32_t)kernel_col;
                    if (source_col >= 0 && (uint32_t)source_col < input->col) {
                        sum += (int32_t)kernel_values[kernel_col]
                            * ((int32_t)plane[(size_t)source_row * input->col
                                              + (uint32_t)source_col] - config->in_act.zp);
                    }
                }
            }
        }
    }
    return inside ? sum - context->correction : sum;
}

static void nn_conv_tile(const nn_conv_context_t *context, int32_t row, int32_t col,
                         int32_t sums[8])
{
    const conv2d_data_i8_t *input = context->input;
    const conv2d_config_i8_t *config = context->config;
    const int8_t *weights = context->weights;
    const nn_conv_shape_t *shape = context->shape;
    const conv2d_filter_i8_t *filter = config->filter;
#if NN_HAS_AVX2
    __m256i accum = _mm256_setzero_si256();
#elif NN_HAS_NEON
    int32x4_t accum_first = vdupq_n_s32(0);
    int32x4_t accum_second = vdupq_n_s32(0);
#else
    for (uint32_t lane = 0; lane < 8; lane++) {
        sums[lane] = 0;
    }
#endif
    for (uint32_t channel = 0; channel < input->channel; channel++) {
        const uint8_t *plane = input->data + (size_t)channel * shape->input_plane
            + (size_t)(uint32_t)row * input->col + (uint32_t)col;
        const int8_t *kernel = weights + (size_t)channel * shape->kernel_plane;
        for (uint32_t kr = 0; kr < filter->row; kr++) {
            const uint8_t *source = plane + (size_t)kr * input->col;
#if NN_HAS_ARM_DSP
            const int8_t *kernel_row = kernel + (size_t)kr * filter->col;
            uint32_t kc = 0;
            for (; filter->col - kc >= 4; kc += 4) {
                const uint32_t packed_weights = nn_read_u32(kernel_row + kc);
                const uint32_t even_weights = nn_arm_sxtb16(packed_weights);
                const uint32_t odd_weights = nn_arm_sxtb16(nn_arm_ror8(packed_weights));
                /* Share expanded weights across spatial windows; each SMLAD is exact. */
                for (uint32_t lane = 0; lane < 8; lane++) {
                    const uint32_t packed = nn_read_u32(source + kc + lane);
                    const uint32_t even = nn_arm_uxtb16(packed);
                    const uint32_t odd = nn_arm_uxtb16(nn_arm_ror8(packed));
                    sums[lane] = (int32_t)nn_arm_smlad(even, even_weights,
                                                 sums[lane]);
                    sums[lane] = (int32_t)nn_arm_smlad(odd, odd_weights,
                                                 sums[lane]);
                }
            }
            for (; kc < filter->col; kc++) {
                const int32_t weight = (int32_t)kernel_row[kc];
                for (uint32_t lane = 0; lane < 8; lane++) {
                    sums[lane] += weight * (int32_t)source[kc + lane];
                }
            }
#else
            for (uint32_t kc = 0; kc < filter->col; kc++) {
                const int32_t weight = (int32_t)kernel[(size_t)kr * filter->col + kc];
#if NN_HAS_AVX2
                const __m256i values = _mm256_cvtepu8_epi32(
                    _mm_loadl_epi64((const __m128i *)(const void *)(source + kc)));
                accum = _mm256_add_epi32(accum,
                    _mm256_mullo_epi32(values, _mm256_set1_epi32(weight)));
#elif NN_HAS_NEON
                const int16x8_t values = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(source + kc)));
                const int16x4_t coefficient = vdup_n_s16((int16_t)weight);
                accum_first = vmlal_s16(accum_first, vget_low_s16(values), coefficient);
                accum_second = vmlal_s16(accum_second, vget_high_s16(values), coefficient);
#else
                for (uint32_t lane = 0; lane < 8; lane++) {
                    sums[lane] += weight * (int32_t)source[kc + lane];
                }
#endif
            }
#endif
        }
    }
#if NN_HAS_AVX2
    accum = _mm256_sub_epi32(accum, _mm256_set1_epi32(context->correction));
    _mm256_storeu_si256((__m256i *)(void *)sums, accum);
#elif NN_HAS_NEON
    const int32x4_t correction = vdupq_n_s32(context->correction);
    vst1q_s32(sums, vsubq_s32(accum_first, correction));
    vst1q_s32(sums + 4, vsubq_s32(accum_second, correction));
#else
    for (uint32_t lane = 0; lane < 8; lane++) {
        sums[lane] -= context->correction;
    }
#endif
}

static uint32_t nn_conv_sums(const nn_conv_context_t *context, uint32_t row, uint32_t col,
                             int32_t sums[8])
{
    const conv2d_data_i8_t *input = context->input;
    const conv2d_config_i8_t *config = context->config;
    const nn_conv_shape_t *shape = context->shape;
    const conv2d_filter_i8_t *filter = config->filter;
    const int32_t source_row = input->row == 1 && filter->row == 1 ? 0
        : (int32_t)(row * config->stride) - (int32_t)config->pad;
    const int32_t source_col = (int32_t)(col * config->stride) - (int32_t)config->pad;
    if (config->stride == 1 && shape->output_col - col >= 8
        && source_row >= 0 && source_col >= 0
        && (uint32_t)source_row + filter->row <= input->row
        && (uint64_t)(uint32_t)source_col + filter->col + 7 <= input->col) {
        nn_conv_tile(context, source_row, source_col, sums);
        return 8;
    }
    sums[0] = nn_conv_point(context, source_row, source_col);
    return 1;
}

static nn_requant_t nn_prepare_requant(float multiplier, float bias, float scale,
                                       uint32_t kernel_size)
{
    nn_requant_t quant = {0, 0, multiplier, bias, scale, 0};
    const double unit = (double)(UINT64_C(1) << NN_QUANT_BITS);
    const double fixed_multiplier = (double)multiplier / (double)scale * unit;
    const double fixed_bias = (double)bias / (double)scale * unit;
    // Check a conservative bound before conversion AND multiplication, not after overflow.
    const double bound = fabs(fixed_multiplier) * (double)(kernel_size * 32640u)
        + fabs(fixed_bias) + (double)(kernel_size * 32640u) + 1.0;
    if (isfinite(bound) && bound < 9.0e18) {
        quant.multiplier = (int64_t)llrint(fixed_multiplier);
        quant.bias = (int64_t)llrint(fixed_bias);
        quant.has_fixed = 1;
    }
    return quant;
}

static uint8_t nn_requantize(int32_t sum, const nn_requant_t *quant)
{
    if (quant->has_fixed) {
        const int64_t value = (int64_t)sum * quant->multiplier + quant->bias;
        const uint64_t unit = UINT64_C(1) << NN_QUANT_BITS;
        if (value <= 0) {
            return 0;
        }
        if ((uint64_t)value >= 255u * unit) {
            return 255;
        }
        const uint64_t remainder = (uint64_t)value & (unit - 1u);
        uint32_t result = (uint32_t)((uint64_t)value >> NN_QUANT_BITS);
        if (remainder > unit / 2u || (remainder == unit / 2u && (result & 1u))) {
            result++;
        }
        return (uint8_t)result;
    }
    const float value = (float)sum * quant->real_multiplier + quant->real_bias;
    return nn_quantize_u8_scalar(value > 0.0f ? value : 0.0f, quant->scale, 0);
}

#ifndef NN_M4_PATCH_MAX
#define NN_M4_PATCH_MAX 160u
#endif
#if NN_M4_PATCH_MAX < 4u || NN_M4_PATCH_MAX > 256u || (NN_M4_PATCH_MAX % 4u) != 0u
#error "NN_M4_PATCH_MAX must be a multiple of 4 between 4 and 256"
#endif

#if NN_HAS_ARM_DSP
typedef struct nn_conv_patch_context {
    const conv2d_data_i8_t *input;
    const conv2d_config_i8_t *config;
    const nn_conv_shape_t *shape;
} nn_conv_patch_context_t;

static int32_t nn_conv_use_patches(const conv2d_data_i8_t *input,
                                  const conv2d_config_i8_t *config,
                                  const nn_conv_shape_t *shape, const int32_t *scratch)
{
    return scratch != NULL && input->row == 1 && config->filter->row == 1
        && config->stride == 1 && config->pad == 0 && shape->kernel_size >= 4
        && shape->kernel_size <= NN_M4_PATCH_MAX && config->filter->filter_num >= 4
        && config->filter->filter_num <= 32;
}

/* A small pair of lowered windows reuses input expansion across output channels.
 * Four-byte groups are stored as x0,x2,x1,x3 to match SXTB16 on int8 weights.
 * Patch arrays use 4*NN_M4_PATCH_MAX bytes plus a 128-byte channel maximum array.
 * This is stack storage, independent of feature-map length; packed weights keep
 * their separate 64-element API limit. The cap must cover whole shuffled groups. */
static int32_t nn_conv_patch_collect(const nn_conv_patch_context_t *context,
                                     int32_t *scratch, int32_t pooled, float *maximum)
{
    const conv2d_data_i8_t *input = context->input;
    const conv2d_config_i8_t *config = context->config;
    const nn_conv_shape_t *shape = context->shape;
    int16_t patches[2][NN_M4_PATCH_MAX];
    int32_t channel_maximum[32];
    const conv2d_filter_i8_t *filter = config->filter;
    const uint32_t output_col = shape->output_col;
    const uint32_t stored_col = pooled ? output_col / 2 : output_col;
    const uint32_t packed_size = shape->kernel_size & ~3u;
    for (uint32_t channel = 0; channel < filter->filter_num; channel++) {
        channel_maximum[channel] = INT32_MIN;
    }
    for (uint32_t col = 0; col < output_col; col += 2) {
        const uint32_t second_col = col + 1 < output_col ? col + 1 : col;
        for (uint32_t channel = 0; channel < input->channel; channel++) {
            const uint8_t *source = input->data + (size_t)channel * input->col;
            for (uint32_t kc = 0; kc < filter->col; kc++) {
                const uint32_t index = channel * filter->col + kc;
                const uint32_t shuffled = (index & ~3u)
                    | ((index & 1u) << 1) | ((index & 2u) >> 1);
                patches[0][shuffled] = (int16_t)((int32_t)source[col + kc]
                                               - config->in_act.zp);
                patches[1][shuffled] = (int16_t)((int32_t)source[second_col + kc]
                                               - config->in_act.zp);
            }
        }
        uint32_t channel = 0;
        /* Two output rows share all four expanded patch loads in each MAC group. */
        for (; filter->filter_num - channel >= 2; channel += 2) {
            const int8_t *first_weights = filter->data + (size_t)channel * shape->kernel_size;
            const int8_t *second_weights = first_weights + shape->kernel_size;
            int32_t sum00 = 0, sum01 = 0, sum10 = 0, sum11 = 0;
            uint32_t index = 0;
            for (; index < packed_size; index += 4) {
                const uint32_t input00 = nn_read_u32(patches[0] + index);
                const uint32_t input01 = nn_read_u32(patches[0] + index + 2);
                const uint32_t input10 = nn_read_u32(patches[1] + index);
                const uint32_t input11 = nn_read_u32(patches[1] + index + 2);
                const uint32_t packed_first = nn_read_u32(first_weights + index);
                const uint32_t first_even = nn_arm_sxtb16(packed_first);
                const uint32_t first_odd = nn_arm_sxtb16(nn_arm_ror8(packed_first));
                const uint32_t packed_second = nn_read_u32(second_weights + index);
                const uint32_t second_even = nn_arm_sxtb16(packed_second);
                const uint32_t second_odd = nn_arm_sxtb16(nn_arm_ror8(packed_second));
                sum00 = nn_arm_smlad(input00, first_even, sum00);
                sum00 = nn_arm_smlad(input01, first_odd, sum00);
                sum01 = nn_arm_smlad(input10, first_even, sum01);
                sum01 = nn_arm_smlad(input11, first_odd, sum01);
                sum10 = nn_arm_smlad(input00, second_even, sum10);
                sum10 = nn_arm_smlad(input01, second_odd, sum10);
                sum11 = nn_arm_smlad(input10, second_even, sum11);
                sum11 = nn_arm_smlad(input11, second_odd, sum11);
            }
            for (; index < shape->kernel_size; index++) {
                const uint32_t shuffled = (index & ~3u)
                    | ((index & 1u) << 1) | ((index & 2u) >> 1);
                const int32_t first_input = (int32_t)patches[0][shuffled];
                const int32_t second_input = (int32_t)patches[1][shuffled];
                sum00 += first_input * (int32_t)first_weights[index];
                sum01 += second_input * (int32_t)first_weights[index];
                sum10 += first_input * (int32_t)second_weights[index];
                sum11 += second_input * (int32_t)second_weights[index];
            }
            const int32_t first_maximum = sum00 > sum01 ? sum00 : sum01;
            const int32_t second_maximum = sum10 > sum11 ? sum10 : sum11;
            if (first_maximum > channel_maximum[channel]) {
                channel_maximum[channel] = first_maximum;
            }
            if (second_maximum > channel_maximum[channel + 1]) {
                channel_maximum[channel + 1] = second_maximum;
            }
            const size_t offset = (size_t)channel * stored_col;
            if (pooled) {
                if (col + 1 < output_col) {
                    scratch[offset + col / 2] = first_maximum;
                    scratch[offset + stored_col + col / 2] = second_maximum;
                }
            } else {
                scratch[offset + col] = sum00;
                scratch[offset + stored_col + col] = sum10;
                if (col + 1 < output_col) {
                    scratch[offset + col + 1] = sum01;
                    scratch[offset + stored_col + col + 1] = sum11;
                }
            }
        }
        for (; channel < filter->filter_num; channel++) {
            const int8_t *weights = filter->data + (size_t)channel * shape->kernel_size;
            int32_t sum0 = 0, sum1 = 0;
            uint32_t index = 0;
            for (; index < packed_size; index += 4) {
                const uint32_t packed = nn_read_u32(weights + index);
                const uint32_t even = nn_arm_sxtb16(packed);
                const uint32_t odd = nn_arm_sxtb16(nn_arm_ror8(packed));
                sum0 = (int32_t)nn_arm_smlad(nn_read_u32(patches[0] + index), even,
                                        sum0);
                sum0 = (int32_t)nn_arm_smlad(nn_read_u32(patches[0] + index + 2), odd,
                                        sum0);
                sum1 = (int32_t)nn_arm_smlad(nn_read_u32(patches[1] + index), even,
                                        sum1);
                sum1 = (int32_t)nn_arm_smlad(nn_read_u32(patches[1] + index + 2), odd,
                                        sum1);
            }
            for (; index < shape->kernel_size; index++) {
                const uint32_t shuffled = (index & ~3u)
                    | ((index & 1u) << 1) | ((index & 2u) >> 1);
                sum0 += (int32_t)patches[0][shuffled] * (int32_t)weights[index];
                sum1 += (int32_t)patches[1][shuffled] * (int32_t)weights[index];
            }
            const int32_t pair_maximum = sum0 > sum1 ? sum0 : sum1;
            if (pair_maximum > channel_maximum[channel]) {
                channel_maximum[channel] = pair_maximum;
            }
            const size_t offset = (size_t)channel * stored_col;
            if (pooled) {
                if (col + 1 < output_col) {
                    scratch[offset + col / 2] = pair_maximum;
                }
            } else {
                scratch[offset + col] = sum0;
                if (col + 1 < output_col) {
                    scratch[offset + col + 1] = sum1;
                }
            }
        }
    }
    *maximum = 0.0f;
    for (uint32_t channel = 0; channel < filter->filter_num; channel++) {
        const float value = (float)channel_maximum[channel]
            * (filter->scale[channel] * config->in_act.scale) + filter->bias[channel];
        if (!isfinite(value)) {
            return CNN_DATA_EXCEPTION;
        }
        if (value > *maximum) {
            *maximum = value;
        }
    }
    return CNN_NORMAL;
}

static void nn_requantize_buffer(const int32_t *sums, uint32_t size,
                                 const nn_requant_t *quant, uint8_t *output)
{
    if (quant->has_fixed && quant->multiplier <= INT32_MAX
        && quant->multiplier >= INT32_MIN) {
        /* Narrow the checked coefficient once: Cortex-M4 emits a single SMULL. */
        const int32_t multiplier = (int32_t)quant->multiplier;
        const int64_t bias = quant->bias;
        const uint64_t unit = UINT64_C(1) << NN_QUANT_BITS;
        for (uint32_t index = 0; index < size; index++) {
            const int64_t value = (int64_t)sums[index] * multiplier + bias;
            uint32_t result;
            if (value <= 0) {
                result = 0;
            } else if ((uint64_t)value >= 255u * unit) {
                result = 255;
            } else {
                const uint32_t remainder = (uint32_t)value & (uint32_t)(unit - 1u);
                result = (uint32_t)((uint64_t)value >> NN_QUANT_BITS);
                /* RNE increment without a data-dependent half-way branch.
                 * remainder < 2^28, so this addition fits uint32_t exactly. */
                result += (remainder + (uint32_t)(unit / 2u - 1u) + (result & 1u))
                    >> NN_QUANT_BITS;
            }
            output[index] = (uint8_t)result;
        }
    } else {
        for (uint32_t index = 0; index < size; index++) {
            output[index] = nn_requantize(sums[index], quant);
        }
    }
}
#endif

static int32_t nn_conv_collect(const conv2d_data_i8_t *input,
                               const conv2d_config_i8_t *config, const nn_conv_shape_t *shape,
                               int32_t *scratch, float *maximum)
{
#if NN_HAS_ARM_DSP
    if (nn_conv_use_patches(input, config, shape, scratch)) {
        const nn_conv_patch_context_t context = {input, config, shape};
        return nn_conv_patch_collect(&context, scratch, 0, maximum);
    }
#endif
    const conv2d_filter_i8_t *filter = config->filter;
    *maximum = 0.0f;
    for (uint32_t channel = 0; channel < filter->filter_num; channel++) {
        const int8_t *weights = filter->data + (size_t)channel * shape->kernel_size;
        const int32_t correction = config->in_act.zp == 0 ? 0
            : config->in_act.zp * nn_weight_sum(weights, shape->kernel_size);
        const nn_conv_context_t context = {input, config, weights, shape, correction};
        int32_t channel_maximum = INT32_MIN;
        for (uint32_t row = 0; row < shape->output_row; row++) {
            for (uint32_t col = 0; col < shape->output_col;) {
                int32_t sums[8];
                const uint32_t count = nn_conv_sums(&context, row, col, sums);
                for (uint32_t lane = 0; lane < count; lane++) {
                    if (sums[lane] > channel_maximum) {
                        channel_maximum = sums[lane];
                    }
                    if (scratch != NULL) {
                        scratch[(size_t)channel * shape->output_plane
                            + (size_t)row * shape->output_col + col + lane] = sums[lane];
                    }
                }
                col += count;
            }
        }
        const float value = (float)channel_maximum
            * (filter->scale[channel] * config->in_act.scale) + filter->bias[channel];
        if (!isfinite(value)) {
            return CNN_DATA_EXCEPTION;
        }
        if (value > *maximum) {
            *maximum = value;
        }
    }
    return CNN_NORMAL;
}

static void nn_conv_output(const conv2d_data_i8_t *input, const conv2d_config_i8_t *config,
                           const nn_conv_shape_t *shape, const int32_t *scratch, uint8_t *output)
{
    const conv2d_filter_i8_t *filter = config->filter;
    for (uint32_t channel = 0; channel < filter->filter_num; channel++) {
        const int8_t *weights = filter->data + (size_t)channel * shape->kernel_size;
        const float multiplier = filter->scale[channel] * config->in_act.scale;
        const float bias = filter->bias[channel];
        const nn_requant_t quant = config->dynamic
            ? nn_prepare_requant(multiplier, bias, config->out_act->scale, shape->kernel_size)
            : (nn_requant_t){0, 0, 0.0f, 0.0f, 0.0f, 0};
#if NN_HAS_ARM_DSP
        if (config->dynamic && scratch != NULL) {
            const size_t offset = (size_t)channel * shape->output_plane;
            nn_requantize_buffer(scratch + offset, shape->output_plane, &quant, output + offset);
            continue;
        }
#endif
        const int32_t correction = config->in_act.zp == 0 || scratch != NULL ? 0
            : config->in_act.zp * nn_weight_sum(weights, shape->kernel_size);
        const nn_conv_context_t context = {input, config, weights, shape, correction};
        for (uint32_t row = 0; row < shape->output_row; row++) {
            for (uint32_t col = 0; col < shape->output_col;) {
                int32_t sums[8];
                const size_t offset = (size_t)channel * shape->output_plane
                    + (size_t)row * shape->output_col + col;
                const uint32_t count = scratch != NULL
                    ? (shape->output_col - col >= 8 ? 8 : 1)
                    : nn_conv_sums(&context, row, col, sums);
                for (uint32_t lane = 0; lane < count; lane++) {
                    const int32_t sum = scratch != NULL ? scratch[offset + lane] : sums[lane];
                    if (config->dynamic) {
                        output[offset + lane] = nn_requantize(sum, &quant);
                    } else {
                        const float value = (float)sum * multiplier + bias;
                        output[offset + lane] = nn_quantize_u8_scalar(value > 0.0f ? value : 0.0f,
                            config->out_act->scale, config->out_act->zp);
                    }
                }
                col += count;
            }
        }
    }
}

/** @brief Return dynamic-convolution scratch bytes, or zero for invalid/static geometry. */
size_t conv2d_int8_get_scratch_size(const conv2d_data_i8_t *input,
                                  const conv2d_config_i8_t *config)
{
    nn_conv_shape_t shape;
    if (nn_conv_validate(input, config, &shape) != CNN_NORMAL || !config->dynamic) {
        return 0;
    }
    return (size_t)shape.output_size * sizeof(int32_t);
}

/** @brief Run convolution with an optional caller-owned int32 accumulator buffer. */
int32_t conv2d_int8_run_with_scratch(const conv2d_data_i8_t *input,
                                    const conv2d_config_i8_t *config, conv2d_data_i8_t *output,
                                    int32_t *scratch, size_t scratch_size)
{
    nn_conv_shape_t shape;
    int32_t status;
    if (output == NULL || output->data == NULL) {
        return CNN_POINTER_NULL;
    }
    status = nn_conv_validate(input, config, &shape);
    if (status != CNN_NORMAL) {
        return status;
    }
    // out_act may refer to the caller's in_act; preserve the input quantization by value.
    const conv2d_config_i8_t config_snapshot = *config;
    config = &config_snapshot;
    if (scratch == NULL && scratch_size != 0) {
        return CNN_POINTER_NULL;
    }
    if (scratch != NULL && (scratch_size < (size_t)shape.output_size * sizeof(int32_t)
        || (uintptr_t)scratch % _Alignof(int32_t) != 0)) {
        return CNN_DATA_EXCEPTION;
    }
    if (config->dynamic) {
        float maximum;
        status = nn_conv_collect(input, config, &shape, scratch, &maximum);
        if (status != CNN_NORMAL) {
            return status;
        }
        config->out_act->scale = fmaxf(maximum / 255.0f, 1e-9f);
        config->out_act->zp = 0;
        if (maximum <= 0.0f) {
            memset(output->data, 0, shape.output_size);
        } else {
            nn_conv_output(input, config, &shape, scratch, output->data);
        }
    } else {
        nn_conv_output(input, config, &shape, NULL, output->data);
    }
    output->row = shape.output_row;
    output->col = shape.output_col;
    output->channel = config->filter->filter_num;
    return CNN_NORMAL;
}

static int32_t nn_conv_pool2_validate(const conv2d_data_i8_t *input,
                                     const conv2d_config_i8_t *config, nn_conv_shape_t *shape)
{
    const int32_t status = nn_conv_validate(input, config, shape);
    if (status != CNN_NORMAL) {
        return status;
    }
    if (input->row != 1 || config->filter->row != 1 || config->stride != 1
        || config->pad != 0 || config->dynamic != 1 || shape->output_col < 2) {
        return CNN_DATA_EXCEPTION;
    }
    return CNN_NORMAL;
}

static int32_t nn_conv_pool2_collect(const conv2d_data_i8_t *input,
                                    const conv2d_config_i8_t *config,
                                    const nn_conv_shape_t *shape, int32_t *scratch,
                                    float *maximum)
{
#if NN_HAS_ARM_DSP
    if (nn_conv_use_patches(input, config, shape, scratch)) {
        const nn_conv_patch_context_t context = {input, config, shape};
        return nn_conv_patch_collect(&context, scratch, 1, maximum);
    }
#endif
    const conv2d_filter_i8_t *filter = config->filter;
    const uint32_t pooled_col = shape->output_col / 2;
    *maximum = 0.0f;
    for (uint32_t channel = 0; channel < filter->filter_num; channel++) {
        const int8_t *weights = filter->data + (size_t)channel * shape->kernel_size;
        const int32_t correction = config->in_act.zp == 0 ? 0
            : config->in_act.zp * nn_weight_sum(weights, shape->kernel_size);
        const nn_conv_context_t context = {input, config, weights, shape, correction};
        int32_t channel_maximum = INT32_MIN;
        int32_t first = 0;
        for (uint32_t col = 0; col < shape->output_col;) {
            int32_t sums[8];
            const uint32_t count = nn_conv_sums(&context, 0, col, sums);
            for (uint32_t lane = 0; lane < count; lane++) {
                const uint32_t position = col + lane;
                const int32_t sum = sums[lane];
                if (sum > channel_maximum) {
                    channel_maximum = sum;
                }
                if ((position & 1u) == 0) {
                    first = sum;
                } else if (scratch != NULL) {
                    scratch[(size_t)channel * pooled_col + position / 2]
                        = sum > first ? sum : first;
                }
            }
            col += count;
        }
        // An odd discarded tail still determines the unfused convolution's dynamic scale.
        const float value = (float)channel_maximum
            * (filter->scale[channel] * config->in_act.scale) + filter->bias[channel];
        if (!isfinite(value)) {
            return CNN_DATA_EXCEPTION;
        }
        if (value > *maximum) {
            *maximum = value;
        }
    }
    return CNN_NORMAL;
}

static void nn_conv_pool2_output(const conv2d_data_i8_t *input,
                                const conv2d_config_i8_t *config,
                                const nn_conv_shape_t *shape, const int32_t *scratch,
                                uint8_t *output)
{
    const conv2d_filter_i8_t *filter = config->filter;
    const uint32_t pooled_col = shape->output_col / 2;
    for (uint32_t channel = 0; channel < filter->filter_num; channel++) {
        const int8_t *weights = filter->data + (size_t)channel * shape->kernel_size;
        const float multiplier = filter->scale[channel] * config->in_act.scale;
        const nn_requant_t quant = nn_prepare_requant(multiplier, filter->bias[channel],
            config->out_act->scale, shape->kernel_size);
        uint8_t *destination = output + (size_t)channel * pooled_col;
        if (scratch != NULL) {
            const int32_t *source = scratch + (size_t)channel * pooled_col;
#if NN_HAS_ARM_DSP
            nn_requantize_buffer(source, pooled_col, &quant, destination);
#else
            for (uint32_t col = 0; col < pooled_col; col++) {
                destination[col] = nn_requantize(source[col], &quant);
            }
#endif
            continue;
        }
        const int32_t correction = config->in_act.zp == 0 ? 0
            : config->in_act.zp * nn_weight_sum(weights, shape->kernel_size);
        const nn_conv_context_t context = {input, config, weights, shape, correction};
        int32_t first = 0;
        for (uint32_t col = 0; col < pooled_col * 2;) {
            int32_t sums[8];
            const uint32_t count = nn_conv_sums(&context, 0, col, sums);
            for (uint32_t lane = 0; lane < count; lane++) {
                const uint32_t position = col + lane;
                if ((position & 1u) == 0) {
                    first = sums[lane];
                } else {
                    const int32_t maximum = sums[lane] > first ? sums[lane] : first;
                    destination[position / 2] = nn_requantize(maximum, &quant);
                }
            }
            col += count;
        }
    }
}

size_t conv2d_int8_maxpool2_get_scratch_size(const conv2d_data_i8_t *input,
                                           const conv2d_config_i8_t *config)
{
    nn_conv_shape_t shape;
    if (nn_conv_pool2_validate(input, config, &shape) != CNN_NORMAL) {
        return 0;
    }
    return (size_t)(shape.output_col / 2) * config->filter->filter_num * sizeof(int32_t);
}

int32_t conv2d_int8_maxpool2_run_with_scratch(const conv2d_data_i8_t *input,
                                            const conv2d_config_i8_t *config,
                                            conv2d_data_i8_t *output, int32_t *scratch,
                                            size_t scratch_size)
{
    nn_conv_shape_t shape;
    if (output == NULL || output->data == NULL) {
        return CNN_POINTER_NULL;
    }
    int32_t status = nn_conv_pool2_validate(input, config, &shape);
    if (status != CNN_NORMAL) {
        return status;
    }
    // Preserve in_act when the caller uses the same quantization object for both layers.
    const conv2d_config_i8_t config_snapshot = *config;
    config = &config_snapshot;
    const uint32_t output_col = shape.output_col / 2;
    const size_t output_size = (size_t)output_col * config->filter->filter_num;
    if (scratch == NULL && scratch_size != 0) {
        return CNN_POINTER_NULL;
    }
    if (scratch != NULL && (scratch_size < output_size * sizeof(int32_t)
        || (uintptr_t)scratch % _Alignof(int32_t) != 0)) {
        return CNN_DATA_EXCEPTION;
    }
    float maximum;
    status = nn_conv_pool2_collect(input, config, &shape, scratch, &maximum);
    if (status != CNN_NORMAL) {
        return status;
    }
    config->out_act->scale = fmaxf(maximum / 255.0f, 1e-9f);
    config->out_act->zp = 0;
    if (maximum <= 0.0f) {
        memset(output->data, 0, output_size);
    } else {
        nn_conv_pool2_output(input, config, &shape, scratch, output->data);
    }
    output->row = 1;
    output->col = output_col;
    output->channel = config->filter->filter_num;
    return CNN_NORMAL;
}

/* The optional packed path is separate so the original Flash/int8 hot loop is unchanged. */
typedef struct nn_packed_memory {
    const void *data;
    size_t size;
} nn_packed_memory_t;

typedef struct nn_packed_conv {
    const conv2d_data_i8_t *input;
    const conv2d_config_i8_t *config;
    const nn_conv_shape_t *shape;
    const conv2d_packed_context_t *context;
} nn_packed_conv_t;

static int32_t nn_packed_memory_valid(nn_packed_memory_t memory)
{
    return memory.data != NULL && memory.size != 0
        && memory.size - 1 <= UINTPTR_MAX - (uintptr_t)memory.data;
}

static int32_t nn_packed_memory_overlap(nn_packed_memory_t first,
                                        nn_packed_memory_t second)
{
    const uintptr_t first_address = (uintptr_t)first.data;
    const uintptr_t second_address = (uintptr_t)second.data;
    return first_address <= second_address
        ? second_address - first_address < first.size
        : first_address - second_address < second.size;
}

size_t conv2d_int8_packed_weights_size(const conv2d_filter_i8_t *filter)
{
    if (filter == NULL || (uintptr_t)filter % _Alignof(conv2d_filter_i8_t) != 0) {
        return 0;
    }
    if (filter->data == NULL || filter->row != 1 || filter->col == 0
        || filter->channel == 0 || filter->col > 64 || filter->channel > 64 / filter->col
        || filter->filter_num == 0 || filter->filter_num > 32) {
        return 0;
    }
    const uint32_t kernel_size = filter->col * filter->channel;
    return (size_t)((kernel_size + 3u) & ~3u) * filter->filter_num * sizeof(int16_t);
}

int32_t conv2d_int8_pack_weights(const conv2d_filter_i8_t *filter,
                                int16_t *output, size_t output_size)
{
    if (filter == NULL || output == NULL) {
        return CNN_POINTER_NULL;
    }
    if ((uintptr_t)filter % _Alignof(conv2d_filter_i8_t) != 0) {
        return CNN_DATA_EXCEPTION;
    }
    if (filter->data == NULL) {
        return CNN_POINTER_NULL;
    }
    const size_t required = conv2d_int8_packed_weights_size(filter);
    if (required == 0 || output_size < required
        || (uintptr_t)output % _Alignof(int32_t) != 0) {
        return CNN_DATA_EXCEPTION;
    }
    const uint32_t kernel_size = filter->col * filter->channel;
    const uint32_t packed_size = (kernel_size + 3u) & ~3u;
    const nn_packed_memory_t destination = {output, required};
    const nn_packed_memory_t sources[] = {
        {filter, sizeof(*filter)},
        {filter->data, (size_t)kernel_size * filter->filter_num},
        {filter->scale, (size_t)filter->filter_num * sizeof(float)},
        {filter->bias, (size_t)filter->filter_num * sizeof(float)}
    };
    if (!nn_packed_memory_valid(destination)) {
        return CNN_DATA_EXCEPTION;
    }
    for (size_t index = 0; index < sizeof(sources) / sizeof(sources[0]); index++) {
        if (sources[index].data != NULL
            && (!nn_packed_memory_valid(sources[index])
                || nn_packed_memory_overlap(destination, sources[index]))) {
            return CNN_DATA_EXCEPTION;
        }
    }
    memset(output, 0, required);
    for (uint32_t channel = 0; channel < filter->filter_num; channel++) {
        for (uint32_t index = 0; index < kernel_size; index++) {
            const uint32_t shuffled = (index & ~3u)
                | ((index & 1u) << 1) | ((index & 2u) >> 1);
            output[(size_t)channel * packed_size + shuffled]
                = (int16_t)filter->data[(size_t)channel * kernel_size + index];
        }
    }
    return CNN_NORMAL;
}

static void nn_packed_patch_fill(const nn_packed_conv_t *conv, uint32_t col,
                                  int16_t patches[2][64])
{
    const conv2d_data_i8_t *input = conv->input;
    const conv2d_config_i8_t *config = conv->config;
    const uint32_t second_col = col + 1 < conv->shape->output_col ? col + 1 : col;
    for (uint32_t channel = 0; channel < input->channel; channel++) {
        const uint8_t *source = input->data + (size_t)channel * input->col;
        for (uint32_t kc = 0; kc < config->filter->col; kc++) {
            const uint32_t index = channel * config->filter->col + kc;
            const uint32_t shuffled = (index & ~3u)
                | ((index & 1u) << 1) | ((index & 2u) >> 1);
            patches[0][shuffled] = (int16_t)((int32_t)source[col + kc] - config->in_act.zp);
            patches[1][shuffled] = (int16_t)((int32_t)source[second_col + kc]
                - config->in_act.zp);
        }
    }
}

static void nn_packed_patch_pair(const int16_t patches[2][64],
                                  const int16_t *const weights[2], uint32_t packed_size,
                                  int32_t sums[4])
{
    int32_t sum00 = 0, sum01 = 0, sum10 = 0, sum11 = 0;
    for (uint32_t index = 0; index < packed_size; index += 4) {
#if NN_HAS_ARM_DSP
        const uint32_t input00 = nn_read_u32(patches[0] + index);
        const uint32_t input01 = nn_read_u32(patches[0] + index + 2);
        const uint32_t input10 = nn_read_u32(patches[1] + index);
        const uint32_t input11 = nn_read_u32(patches[1] + index + 2);
        const uint32_t first_even = nn_read_u32(weights[0] + index);
        const uint32_t first_odd = nn_read_u32(weights[0] + index + 2);
        const uint32_t second_even = nn_read_u32(weights[1] + index);
        const uint32_t second_odd = nn_read_u32(weights[1] + index + 2);
        sum00 = nn_arm_smlad(input00, first_even, sum00);
        sum00 = nn_arm_smlad(input01, first_odd, sum00);
        sum01 = nn_arm_smlad(input10, first_even, sum01);
        sum01 = nn_arm_smlad(input11, first_odd, sum01);
        sum10 = nn_arm_smlad(input00, second_even, sum10);
        sum10 = nn_arm_smlad(input01, second_odd, sum10);
        sum11 = nn_arm_smlad(input10, second_even, sum11);
        sum11 = nn_arm_smlad(input11, second_odd, sum11);
#else
        /* Same exact centered integer dot; shuffled inputs and weights cancel. */
        for (uint32_t lane = 0; lane < 4; lane++) {
            const int32_t first_input = (int32_t)patches[0][index + lane];
            const int32_t second_input = (int32_t)patches[1][index + lane];
            sum00 += first_input * (int32_t)weights[0][index + lane];
            sum01 += second_input * (int32_t)weights[0][index + lane];
            sum10 += first_input * (int32_t)weights[1][index + lane];
            sum11 += second_input * (int32_t)weights[1][index + lane];
        }
#endif
    }
    sums[0] = sum00;
    sums[1] = sum01;
    sums[2] = sum10;
    sums[3] = sum11;
}

static int32_t nn_packed_patch_collect(const nn_packed_conv_t *conv, float *maximum)
{
    const conv2d_filter_i8_t *filter = conv->config->filter;
    const uint32_t packed_size = (conv->shape->kernel_size + 3u) & ~3u;
    const uint32_t pooled_col = conv->shape->output_col / 2;
    _Alignas(int32_t) int16_t patches[2][64] = {{0}};
    int32_t channel_maximum[32];
    for (uint32_t channel = 0; channel < filter->filter_num; channel++) {
        channel_maximum[channel] = INT32_MIN;
    }
    for (uint32_t col = 0; col < conv->shape->output_col; col += 2) {
        nn_packed_patch_fill(conv, col, patches);
        for (uint32_t channel = 0; channel < filter->filter_num; channel += 2) {
            const int32_t has_second = channel + 1 < filter->filter_num;
            const int16_t *first = conv->context->weights + (size_t)channel * packed_size;
            const int16_t *weights[2] = {first, has_second ? first + packed_size : first};
            int32_t sums[4];
            nn_packed_patch_pair((const int16_t (*)[64])patches, weights, packed_size, sums);
            const int32_t first_maximum = sums[0] > sums[1] ? sums[0] : sums[1];
            const int32_t second_maximum = sums[2] > sums[3] ? sums[2] : sums[3];
            if (first_maximum > channel_maximum[channel]) {
                channel_maximum[channel] = first_maximum;
            }
            if (has_second && second_maximum > channel_maximum[channel + 1]) {
                channel_maximum[channel + 1] = second_maximum;
            }
            if (col + 1 < conv->shape->output_col) {
                const size_t offset = (size_t)channel * pooled_col + col / 2;
                conv->context->scratch[offset] = first_maximum;
                if (has_second) {
                    conv->context->scratch[offset + pooled_col] = second_maximum;
                }
            }
        }
    }
    *maximum = 0.0f;
    for (uint32_t channel = 0; channel < filter->filter_num; channel++) {
        const float value = (float)channel_maximum[channel]
            * (filter->scale[channel] * conv->config->in_act.scale) + filter->bias[channel];
        if (!isfinite(value)) {
            return CNN_DATA_EXCEPTION;
        }
        if (value > *maximum) {
            *maximum = value;
        }
    }
    return CNN_NORMAL;
}

static int32_t nn_packed_metadata_valid(const conv2d_data_i8_t *input,
                                        const conv2d_config_i8_t *config,
                                        const conv2d_data_i8_t *output,
                                        const conv2d_packed_context_t *context)
{
    if (input == NULL || config == NULL || output == NULL || context == NULL) {
        return CNN_POINTER_NULL;
    }
    if ((uintptr_t)input % _Alignof(conv2d_data_i8_t) != 0
        || (uintptr_t)config % _Alignof(conv2d_config_i8_t) != 0
        || (uintptr_t)output % _Alignof(conv2d_data_i8_t) != 0
        || (uintptr_t)context % _Alignof(conv2d_packed_context_t) != 0) {
        return CNN_DATA_EXCEPTION;
    }
    if (input->data == NULL || output->data == NULL || config->filter == NULL
        || config->out_act == NULL || context->weights == NULL || context->scratch == NULL) {
        return CNN_POINTER_NULL;
    }
    if ((uintptr_t)config->filter % _Alignof(conv2d_filter_i8_t) != 0
        || (uintptr_t)config->out_act % _Alignof(quant_param_t) != 0
        || (uintptr_t)context->weights % _Alignof(int32_t) != 0
        || (uintptr_t)context->scratch % _Alignof(int32_t) != 0) {
        return CNN_DATA_EXCEPTION;
    }
    if (config->filter->scale == NULL || config->filter->bias == NULL) {
        return CNN_POINTER_NULL;
    }
    if ((uintptr_t)config->filter->scale % _Alignof(float) != 0
        || (uintptr_t)config->filter->bias % _Alignof(float) != 0) {
        return CNN_DATA_EXCEPTION;
    }
    return CNN_NORMAL;
}

static int32_t nn_packed_buffers_valid(const nn_packed_conv_t *conv,
                                       const conv2d_data_i8_t *output)
{
    const conv2d_filter_i8_t *filter = conv->config->filter;
    const size_t output_size = (size_t)(conv->shape->output_col / 2) * filter->filter_num;
    const size_t weights_size = conv2d_int8_packed_weights_size(filter);
    if (weights_size == 0 || conv->context->weights_size < weights_size
        || conv->context->scratch_size < output_size * sizeof(int32_t)) {
        return CNN_DATA_EXCEPTION;
    }
    const nn_packed_memory_t buffers[] = {
        {conv->input->data, (size_t)conv->input->col * conv->input->channel},
        {output->data, output_size},
        {conv->context->weights, weights_size},
        {conv->context->scratch, output_size * sizeof(int32_t)}
    };
    const nn_packed_memory_t metadata[] = {
        {conv->input, sizeof(*conv->input)},
        {conv->config, sizeof(*conv->config)},
        {output, sizeof(*output)},
        {conv->context, sizeof(*conv->context)},
        {filter, sizeof(*filter)},
        {filter->data, (size_t)conv->shape->kernel_size * filter->filter_num},
        {filter->scale, (size_t)filter->filter_num * sizeof(float)},
        {filter->bias, (size_t)filter->filter_num * sizeof(float)},
        {conv->config->out_act, sizeof(*conv->config->out_act)}
    };
    for (size_t index = 0; index < sizeof(buffers) / sizeof(buffers[0]); index++) {
        if (!nn_packed_memory_valid(buffers[index])) {
            return CNN_DATA_EXCEPTION;
        }
        for (size_t other = 0; other < index; other++) {
            if (nn_packed_memory_overlap(buffers[index], buffers[other])) {
                return CNN_DATA_EXCEPTION;
            }
        }
        for (size_t other = 0; other < sizeof(metadata) / sizeof(metadata[0]); other++) {
            if (!nn_packed_memory_valid(metadata[other])
                || nn_packed_memory_overlap(buffers[index], metadata[other])) {
                return CNN_DATA_EXCEPTION;
            }
        }
    }
    const nn_packed_memory_t quant = metadata[8];
    if (nn_packed_memory_overlap(quant, metadata[0])
        || (nn_packed_memory_overlap(quant, metadata[1])
            && conv->config->out_act != &conv->config->in_act)) {
        return CNN_DATA_EXCEPTION;
    }
    for (size_t index = 2; index < 8; index++) {
        if (nn_packed_memory_overlap(quant, metadata[index])) {
            return CNN_DATA_EXCEPTION;
        }
    }
    return CNN_NORMAL;
}

int32_t conv2d_int8_maxpool2_run_packed(const conv2d_data_i8_t *input,
                                      const conv2d_config_i8_t *config,
                                      conv2d_data_i8_t *output,
                                      const conv2d_packed_context_t *context)
{
    int32_t status = nn_packed_metadata_valid(input, config, output, context);
    if (status != CNN_NORMAL) {
        return status;
    }
    nn_conv_shape_t shape;
    status = nn_conv_pool2_validate(input, config, &shape);
    if (status != CNN_NORMAL) {
        return status;
    }
    const nn_packed_conv_t validation = {input, config, &shape, context};
    status = nn_packed_buffers_valid(&validation, output);
    if (status != CNN_NORMAL) {
        return status;
    }
    /* Retain the native AVX2/NEON/scalar backend outside Cortex-M DSP targets.
     * Cache storage remains validated, so the contract is identical on all platforms. */
    if (!NN_HAS_ARM_DSP) {
        return conv2d_int8_maxpool2_run_with_scratch(input, config, output,
            context->scratch, context->scratch_size);
    }
    /* Capture before out_act is written, including the supported in_act alias. */
    const conv2d_config_i8_t config_snapshot = *config;
    const conv2d_packed_context_t context_snapshot = *context;
    const nn_packed_conv_t conv = {input, &config_snapshot, &shape, &context_snapshot};
    float maximum;
    status = nn_packed_patch_collect(&conv, &maximum);
    if (status != CNN_NORMAL) {
        return status;
    }
    config->out_act->scale = fmaxf(maximum / 255.0f, 1e-9f);
    config->out_act->zp = 0;
    const size_t output_size = (size_t)(shape.output_col / 2) * config->filter->filter_num;
    if (maximum <= 0.0f) {
        memset(output->data, 0, output_size);
    } else {
        nn_conv_pool2_output(input, &config_snapshot, &shape, context_snapshot.scratch,
            output->data);
    }
    output->row = 1;
    output->col = shape.output_col / 2;
    output->channel = config->filter->filter_num;
    return CNN_NORMAL;
}

int conv2d_int8_run(const conv2d_data_i8_t *input, const conv2d_config_i8_t *config,
                   conv2d_data_i8_t *output)
{
    return (int)conv2d_int8_run_with_scratch(input, config, output, NULL, 0);
}

void quantize_u8_run(const float *input, uint32_t size, const quant_param_t *quant,
                     uint8_t *output)
{
    if (input == NULL || output == NULL || quant == NULL || !nn_quant_valid(quant)) {
        return;
    }
    for (uint32_t index = 0; index < size; index++) {
        output[index] = nn_quantize_u8_scalar(input[index], quant->scale, quant->zp);
    }
}

void quantize_u8(const float *input, uint32_t size, float scale, int zero_point, uint8_t *output)
{
    const quant_param_t quant = {scale, zero_point};
    quantize_u8_run(input, size, &quant, output);
}


int conv1d_int8(const uint8_t* input, uint32_t in_col, uint32_t in_chan, const int8_t* weights, uint32_t kernel,
                uint32_t stride, uint32_t pad, uint32_t out_chan, const float* weights_scale,
                const float* bias, float in_act_scale, int in_act_zp, float out_act_scale,
                int out_act_zp, uint8_t* output) {
    conv2d_data_i8_t in_feat = {1, in_col, in_chan, (uint8_t*)input};
    conv2d_filter_i8_t filter = {1, kernel, in_chan, out_chan, weights, weights_scale, bias};
    quant_param_t out_quant = {out_act_scale, out_act_zp};
    conv2d_config_i8_t config = {stride, pad, &filter, {in_act_scale, in_act_zp}, 0, &out_quant};
    conv2d_data_i8_t out_feat = {1, 0, 0, output};
    return conv2d_int8_run(&in_feat, &config, &out_feat);
}

int conv2d_int8(const uint8_t* input, uint32_t in_row, uint32_t in_col, uint32_t in_chan,
                const int8_t* weights, uint32_t k_row, uint32_t k_col, uint32_t stride, uint32_t pad,
                uint32_t out_chan, const float* weights_scale, const float* bias, float in_act_scale,
                int in_act_zp, float out_act_scale, int out_act_zp, uint8_t* output) {
    conv2d_data_i8_t in_feat = {in_row, in_col, in_chan, (uint8_t*)input};
    conv2d_filter_i8_t filter = {k_row, k_col, in_chan, out_chan, weights, weights_scale, bias};
    quant_param_t out_quant = {out_act_scale, out_act_zp};
    conv2d_config_i8_t config = {stride, pad, &filter, {in_act_scale, in_act_zp}, 0, &out_quant};
    conv2d_data_i8_t out_feat = {0, 0, 0, output};
    return conv2d_int8_run(&in_feat, &config, &out_feat);
}

int conv1d_int8_dynamic(const uint8_t* input, uint32_t in_col, uint32_t in_chan, const int8_t* weights,
                        uint32_t kernel, uint32_t stride, uint32_t pad, uint32_t out_chan,
                        const float* weights_scale, const float* bias, float in_act_scale, int in_act_zp,
                        float* out_act_scale, uint8_t* output) {
    if (out_act_scale == NULL) {
        return CNN_POINTER_NULL;
    }

    conv2d_data_i8_t in_feat = {1, in_col, in_chan, (uint8_t*)input};
    conv2d_filter_i8_t filter = {1, kernel, in_chan, out_chan, weights, weights_scale, bias};
    quant_param_t quant = {0.0f, 0};
    conv2d_config_i8_t config = {stride, pad, &filter, {in_act_scale, in_act_zp}, 1, &quant};
    conv2d_data_i8_t out_feat = {1, 0, 0, output};
    const int ret = conv2d_int8_run(&in_feat, &config, &out_feat);
    if (ret == CNN_NORMAL) {
        *out_act_scale = quant.scale;
    }
    return ret;
}

int conv2d_int8_dynamic(const uint8_t* input, uint32_t in_row, uint32_t in_col, uint32_t in_chan,
                        const int8_t* weights, uint32_t k_row, uint32_t k_col, uint32_t stride,
                        uint32_t pad, uint32_t out_chan, const float* weights_scale, const float* bias,
                        float in_act_scale, int in_act_zp, float* out_act_scale, uint8_t* output) {
    if (out_act_scale == NULL) {
        return CNN_POINTER_NULL;
    }

    conv2d_data_i8_t in_feat = {in_row, in_col, in_chan, (uint8_t*)input};
    conv2d_filter_i8_t filter = {k_row, k_col, in_chan, out_chan, weights, weights_scale, bias};
    quant_param_t quant = {0.0f, 0};
    conv2d_config_i8_t config = {stride, pad, &filter, {in_act_scale, in_act_zp}, 1, &quant};
    conv2d_data_i8_t out_feat = {0, 0, 0, output};
    const int ret = conv2d_int8_run(&in_feat, &config, &out_feat);
    if (ret == CNN_NORMAL) {
        *out_act_scale = quant.scale;
    }
    return ret;
}

/**
 * @brief uint8 1D 最大池化（量化域直接取 max）。
 *
 * 输出长度 = (n - k) / stride + 1（valid 池化，与 fp32 maxpool 一致）。
 * 量化函数单调，uint8 直接取 max 与浮点池化等价。
 *
 * @param in     [in]  输入 uint8
 * @param n      [in]  输入长度
 * @param out    [out] 输出 uint8
 * @param k      [in]  核宽
 * @param stride [in]  步长
 */
void maxpool1d_u8(const uint8_t* input, uint32_t size, uint8_t* output, uint32_t kernel, uint32_t stride) {
    uint32_t out_n;
    uint32_t in_feat, j;
    if (!input || !output || kernel == 0 || stride == 0 || kernel > size) {
        return;
    }
    out_n = (size - kernel) / stride + 1;
    for (in_feat = 0; in_feat < out_n; in_feat++) {
        uint8_t m = input[in_feat * stride];
        for (j = 1; j < kernel; j++) {
            if (input[in_feat * stride + j] > m) {
                m = input[in_feat * stride + j];
            }
        }
        output[in_feat] = m;
    }
}

/**
 * @brief uint8 2D 最大池化（量化域直接取 max）。
 *
 * @param in     [in]  输入 uint8，布局 [in_row][in_col]
 * @param in_row [in]  输入行数
 * @param in_col [in]  输入列数
 * @param out    [out] 输出 uint8
 * @param k_row  [in]  核高
 * @param k_col  [in]  核宽
 * @param stride [in]  步长
 */
void maxpool2d_u8(const uint8_t* input, uint32_t in_row, uint32_t in_col, uint8_t* output, uint32_t k_row,
                  uint32_t k_col, uint32_t stride) {
    uint32_t out_row, out_col;
    uint32_t j, kernel, jj, kk;

    // 参数校验：核大于输入时 (input - kernel) 会 uint32 下溢导致越界写，必须先检查。
    if (!input || !output || k_row == 0 || k_col == 0 || stride == 0 || k_row > in_row ||
        k_col > in_col) {
        return;
    }
    out_row = (in_row - k_row) / stride + 1;
    out_col = (in_col - k_col) / stride + 1;

    for (j = 0; j < out_row; j++) {
        for (kernel = 0; kernel < out_col; kernel++) {
            uint8_t max_num = input[(j * stride) * in_col + kernel * stride];
            for (jj = 0; jj < k_row; jj++) {
                for (kk = 0; kk < k_col; kk++) {
                    uint8_t v = input[(j * stride + jj) * in_col + kernel * stride + kk];
                    if (v > max_num) {
                        max_num = v;
                    }
                }
            }
            output[j * out_col + kernel] = max_num;
        }
    }
}

/**
 * @brief uint8 最大池化统一入口（1D/2D 自动分派）。
 *
 * @param in     [in]  输入 uint8，单通道平面
 * @param in_row [in]  输入行数（1D 时为 1）
 * @param in_col [in]  输入列数
 * @param out    [out] 输出 uint8，单通道平面
 * @param k_row  [in]  核高（1D 时为 1）
 * @param k_col  [in]  核宽
 * @param stride [in]  步长
 */
void maxpool_u8(const uint8_t* input, uint32_t in_row, uint32_t in_col, uint8_t* output, uint32_t k_row,
                uint32_t k_col, uint32_t stride) {
    // 统一入口：纯 1D（单行输入 + 单行核）走 1D 优化算法，否则 2D。
    if (in_row == 1 && k_row == 1) {
        maxpool1d_u8(input, in_col, output, k_col, stride);
    } else {
        maxpool2d_u8(input, in_row, in_col, output, k_row, k_col, stride);
    }
}

/**
 * @brief uint8 最大池化统一入口（1D/2D 自动分派，带校验与维度写回）。
 *
 * 与 fp32 的 maxpool 同构：输入/配置/输出三个结构体。量化函数单调，
 * uint8 直接取 max 与浮点池化等价；支持 in-place（输出升序写入安全）。
 * 输出几何形状写回 out 的 row/col/channel 字段。
 */
int maxpool_u8_run(const conv2d_data_i8_t* input, const max_pool_config_t* param, conv2d_data_i8_t* output) {
    uint32_t in_r, in_c, n_ch, out_row, out_col, in_plane, out_plane;
    uint32_t config;

    if (!input || !input->data || !param || !output || !output->data) {
        return CNN_POINTER_NULL;
    }
    if (param->row == 0 || param->col == 0 || param->stride == 0 || param->pad != 0 ||
        param->row > input->row || param->col > input->col) {
        return CNN_DATA_EXCEPTION;
    }
    uint32_t input_size;
    if (!nn_shape_product(input->row, input->col, &input_size)
        || !nn_shape_product(input_size, input->channel, &input_size)) {
        return CNN_DATA_EXCEPTION;
    }

    // 快照输入维度（input-place 时写回输出维度前必须先取原值）。
    in_r = input->row;
    in_c = input->col;
    n_ch = input->channel;
    out_row = (in_r == 1) ? 1 : (in_r - param->row) / param->stride + 1;
    out_col = (in_c - param->col) / param->stride + 1;
    in_plane = in_r * in_c;
    out_plane = out_row * out_col;

    // 写回输出几何形状。
    output->row = out_row;
    output->col = out_col;
    output->channel = n_ch;

    // 逐通道平面分派到 1D/2D 池化（升序写入，input-place 安全）。
    for (config = 0; config < n_ch; config++) {
        const uint8_t* src = input->data + config * in_plane;
        uint8_t* dst = output->data + config * out_plane;
        if (in_r == 1 && param->row == 1) {
            maxpool1d_u8(src, in_c, dst, param->col, param->stride);
        } else {
            maxpool2d_u8(src, in_r, in_c, dst, param->row, param->col, param->stride);
        }
    }
    return CNN_NORMAL;
}

/**
 * @brief int8 全连接统一入口：uint8 输入 -> fp32 输出（不做输出量化）。
 *
 * 整数域累加（int32）-> 反量化。最长 4032 项时上界约 1.31e8 < INT32_MAX，安全。
 * 输出行做 4 行分块（4 个独立累加器，打破乘加依赖链、便于向量化）；
 * 每行内的累加顺序仍为 j 从小到大，与 Python int32 复现一致。
 * 零点分派：in_act->zp == 0 时走免减法快路径；非 0 时零点折进权重
 * 行和（整数恒等）。
 */
int linear_int8_run(const uint8_t *input, const linear_param_i8_t *config,
                    const quant_param_t *in_act, float *output)
{
    uint32_t total;
    if (input == NULL || config == NULL || config->weight == NULL || config->scale == NULL
        || config->bias == NULL || in_act == NULL || output == NULL) {
        return CNN_POINTER_NULL;
    }
    if (!nn_quant_valid(in_act) || config->inp_size > NN_MAX_MACS
        || config->fea_size > (uint32_t)INT32_MAX / sizeof(float)
        || !nn_shape_product(config->inp_size, config->fea_size, &total)) {
        return CNN_DATA_EXCEPTION;
    }
    for (uint32_t row = 0; row < config->fea_size; row++) {
        if (!isfinite(config->scale[row]) || config->scale[row] <= 0.0f
            || !isfinite(config->scale[row] * in_act->scale)
            || !isfinite(config->bias[row])) {
            return CNN_DATA_EXCEPTION;
        }
    }
    uint32_t row = 0;
    for (; config->fea_size - row >= 4; row += 4) {
        const int8_t *weights = config->weight + (size_t)row * config->inp_size;
        const int8_t *const weight_rows[4] = {weights, weights + config->inp_size,
            weights + (size_t)2 * config->inp_size, weights + (size_t)3 * config->inp_size};
        int32_t sums[4];
#if NN_HAS_VECTOR_DOT
        if (config->inp_size >= 64) {
            nn_dot4_u8_s8_centered(input, weight_rows, config->inp_size, in_act->zp, sums);
        } else
#endif
        {
            sums[0] = 0;
            sums[1] = 0;
            sums[2] = 0;
            sums[3] = 0;
            if (in_act->zp == 0) {
                nn_dot4_u8_s8(input, weight_rows, config->inp_size, sums);
            } else {
                for (uint32_t index = 0; index < config->inp_size; index++) {
                    const int32_t value = (int32_t)input[index] - in_act->zp;
                    sums[0] += value * (int32_t)weights[index];
                    sums[1] += value * (int32_t)weights[config->inp_size + index];
                    sums[2] += value * (int32_t)weights[(size_t)2 * config->inp_size + index];
                    sums[3] += value * (int32_t)weights[(size_t)3 * config->inp_size + index];
                }
            }
        }
        for (uint32_t lane = 0; lane < 4; lane++) {
            output[row + lane] = (float)sums[lane] * (config->scale[row + lane] * in_act->scale)
                + config->bias[row + lane];
            if (!isfinite(output[row + lane])) {
                return CNN_DATA_EXCEPTION;
            }
        }
    }
    for (; row < config->fea_size; row++) {
        const int8_t *weights = config->weight + (size_t)row * config->inp_size;
        int32_t sum = 0;
#if NN_HAS_VECTOR_DOT
        if (config->inp_size >= 64) {
            sum = nn_dot_u8_s8(input, weights, config->inp_size);
            if (in_act->zp != 0) {
                sum -= in_act->zp * nn_weight_sum(weights, config->inp_size);
            }
        } else
#endif
        {
            for (uint32_t index = 0; index < config->inp_size; index++) {
                sum += (int32_t)weights[index] * ((int32_t)input[index] - in_act->zp);
            }
        }
        output[row] = (float)sum * (config->scale[row] * in_act->scale) + config->bias[row];
        if (!isfinite(output[row])) {
            return CNN_DATA_EXCEPTION;
        }
    }
    return CNN_NORMAL;
}


/**
 * @brief int8 全连接层（兼容封装，等价于 linear_int8_run）。
 *
 * 构造结构体后转调 linear_int8_run，行为与结果与直接调用统一入口
 * 逐位一致。
 */
int linear_int8(const uint8_t* input, uint32_t in_size, const int8_t* weights, const float* weights_scale,
                const float* bias, uint32_t out_size, float in_act_scale, int in_act_zp,
                float* output) {
    const linear_param_i8_t config = {in_size, out_size, weights, weights_scale, bias};
    const quant_param_t ia = {in_act_scale, in_act_zp};
    return linear_int8_run(input, &config, &ia, output);
}

/**
 * @brief fp32 全连接层（线性层）。
 *
 * 对输出行做 4 行分块：输入向量 x 一次加载被 4 行权重复用（缓存友好），
 * 4 个独立累加器消除乘加链相关性，编译器可向量化为 FMA。
 * 累加顺序（每行内 j 从小到大）与朴素实现一致。
 *
 * @param inp           [in]  输入 float，长度 inp_size
 * @param linear_config [in]  全连接配置
 * @param out           [out] 输出 float，长度 fea_size
 * @return CNN_NORMAL / CNN_POINTER_NULL / CNN_DATA_EXCEPTION。
 *         非有限计算结果被拒绝；失败时输出缓冲可能已经部分更新。
 */
int linear_layer(float* inp, linear_param_t* linear_config, float* out) {
    uint32_t i, j, fea_size, inp_size;
    const float* restrict w;
    const float* restrict x;
    float* restrict o;

    if (!inp || !linear_config || !linear_config->weight || !linear_config->bias || !out) {
        return CNN_POINTER_NULL;
    }

    fea_size = linear_config->fea_size;
    inp_size = linear_config->inp_size;
    uint32_t weight_size;
    if (!nn_shape_product(inp_size, fea_size, &weight_size)
        || weight_size > (uint32_t)INT32_MAX / sizeof(float)) {
        return CNN_DATA_EXCEPTION;
    }
    w = linear_config->weight;
    x = inp;
    o = out;

    // 4 行同时累加：a0..a3 为 4 个独立累加器。
    for (i = 0; i + 4 <= fea_size; i += 4) {
        float a0 = linear_config->bias[i];
        float a1 = linear_config->bias[i + 1];
        float a2 = linear_config->bias[i + 2];
        float a3 = linear_config->bias[i + 3];
        const float* w0 = w + i * inp_size;
        const float* w1 = w0 + inp_size;
        const float* w2 = w1 + inp_size;
        const float* w3 = w2 + inp_size;

        for (j = 0; j < inp_size; j++) {
            const float xv = x[j];
            a0 += w0[j] * xv;
            a1 += w1[j] * xv;
            a2 += w2[j] * xv;
            a3 += w3[j] * xv;
        }

        if (!isfinite(a0) || !isfinite(a1) || !isfinite(a2) || !isfinite(a3)) {
            return CNN_DATA_EXCEPTION;
        }
        o[i] = a0;
        o[i + 1] = a1;
        o[i + 2] = a2;
        o[i + 3] = a3;
    }

    // 尾部（fea_size 不是 4 的倍数时）。
    for (; i < fea_size; i++) {
        float acc = linear_config->bias[i];
        const float* w_i = w + i * inp_size;
        for (j = 0; j < inp_size; j++) {
            acc += x[j] * w_i[j];
        }
        if (!isfinite(acc)) {
            return CNN_DATA_EXCEPTION;
        }
        o[i] = acc;
    }

    return CNN_NORMAL;
}

/**
 * @brief fp32 最大池化（通用 1D/2D，任意核大小与步长）。
 *
 * 当前仅支持 pad=0。支持 in-place（input_feat->data == output_feat->data）：
 * 输出按升序写入，且每个输出窗口先完全读入再写，任意 stride>=1 均安全。
 *
 * @param input_feat  [in]  输入特征图
 * @param param       [in]  池化配置
 * @param output_feat [out] 输出特征图
 * @return CNN_NORMAL / CNN_POINTER_NULL / CNN_DATA_EXCEPTION
 */
int maxpool(conv2d_data_t* input_feat, max_pool_config_t* param, conv2d_data_t* output_feat) {
    uint32_t out_row = 0, out_col = 0, out_chan = 0;
    uint32_t in_row = 0, in_col = 0, in_chan = 0;
    uint32_t i = 0, j = 0, k = 0, jj = 0, kk = 0;

    if (!input_feat || !input_feat->data || !param || !output_feat || !output_feat->data) {
        return CNN_POINTER_NULL;
    }
    if (param->row == 0 || param->col == 0 || param->stride == 0 || param->pad != 0) {
        return CNN_DATA_EXCEPTION;
    }

    in_row = input_feat->row;
    in_col = input_feat->col;
    in_chan = input_feat->channel;

    uint32_t input_size;
    if (param->row > in_row || param->col > in_col
        || !nn_shape_product(in_row, in_col, &input_size)
        || !nn_shape_product(input_size, in_chan, &input_size)
        || input_size > (uint32_t)INT32_MAX / sizeof(float)) {
        return CNN_DATA_EXCEPTION;
    }

    out_row = (in_row == 1) ? 1 : cal_conv_out_len(in_row, param->pad, param->row, param->stride);
    out_col = cal_conv_out_len(input_feat->col, param->pad, param->col, param->stride);
    out_chan = in_chan;

    // 通用 1D 路径：input row=1、核 row=1，任意核宽与步长（如 stride=2/3）。
    if (in_row == 1 && param->row == 1) {
        for (i = 0; i < out_chan; i++) {
            const float* s = input_feat->data + i * in_col;
            float* d = output_feat->data + i * out_col;
            for (k = 0; k < out_col; k++) {
                const float* w = s + k * param->stride;
                float max_num = w[0];
                for (kk = 1; kk < param->col; kk++) {
                    if (w[kk] > max_num) {
                        max_num = w[kk];
                    }
                }
                d[k] = max_num;
            }
        }
    } else {
        // 通用 2D 路径：任意核大小与步长。
        for (i = 0; i < out_chan; i++) {
            const float* in_c = input_feat->data + i * (in_row * in_col);
            float* out_c = output_feat->data + i * (out_row * out_col);

            for (j = 0; j < out_row; j++) {
                const float* row_base = in_c + j * param->stride * in_col;
                for (k = 0; k < out_col; k++) {
                    const float* col_base = row_base + k * param->stride;
                    float max_num = col_base[0];

                    // 第一行从列 1 开始（max_num 已初始化为列 0），其余行从列 0 开始。
                    for (jj = 0; jj < param->row; jj++) {
                        const float* prow = col_base + jj * in_col;
                        for (kk = (jj == 0) ? 1 : 0; kk < param->col; kk++) {
                            if (prow[kk] > max_num) {
                                max_num = prow[kk];
                            }
                        }
                    }

                    *out_c++ = max_num;
                }
            }
        }
    }

    output_feat->row = out_row;
    output_feat->col = out_col;
    output_feat->channel = out_chan;

    return CNN_NORMAL;
}

/**
 * @brief 线性层 DSP 加速入口。
 *
 * PC 版不再依赖 RISC-V DSP 库，统一走优化后的普通线性层实现。
 */
int linear_layer_dsp(float* inp, linear_param_t* linear_config, float* out) {
    return linear_layer(inp, linear_config, out);
}

/**
 * @brief Sigmoid 激活（原地计算）。
 *
 * y = 1 / (1 + exp(-x))
 *
 * @param x    [in,out] 输入/输出数据
 * @param size [in]     长度
 */
void sigmoid(float* x, uint32_t size) {
    for (uint32_t i = 0; i < size; i++) {
        *(x + i) = (1 / (1 + expf(-*(x + i))));
    }
}
