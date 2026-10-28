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
 * @file lstm.h
 * @brief LSTM 推理算子库（fp32 与 int8 动态量化）
 *
 * 单元计算公式（与 PyTorch nn.LSTM 一致，门顺序 i/f/g/o）：
 *   i_t = sigmoid(W_ii*x_t + b_ii + W_hi*h_{t-1} + b_hi)
 *   f_t = sigmoid(W_if*x_t + b_if + W_hf*h_{t-1} + b_hf)
 *   g_t = tanh   (W_ig*x_t + b_ig + W_hg*h_{t-1} + b_hg)
 *   o_t = sigmoid(W_io*x_t + b_io + W_ho*h_{t-1} + b_ho)
 *   c_t = f_t*c_{t-1} + i_t*g_t
 *   h_t = o_t*tanh(c_t)
 *
 * 权重布局（行主序，门顺序 i/f/g/o，每个门 hidden 行）：
 *   - w_ih: [4*hidden][input]
 *   - w_hh: [4*hidden][hidden]
 *   - bias: [4*hidden]（导出时已合并 bias_ih + bias_hh）
 *   - 输出层 w: [num_classes][hidden]
 *
 * int8 动态量化方案：
 *   - 权重：离线 per-row 对称 int8，scale = max|w|/127；
 *   - 输入与隐藏状态：uint8 动态量化（隐藏状态每个时间步在线求
 *     min/max -> scale/zp，供下一时间步整数域累加）；
 *   - 细胞状态 c 保持 float32（不量化），门激活在浮点域计算；
 *   - 整数域只做累加（int32），零点通过预计算的权重行和折进累加器，
 *     不逐 MAC 减零点。
 *
 * 本库只依赖 C 标准库与 cnn 库的 quantize_u8 / linear_int8，无动态内存分配。
 */

#ifndef AI_BASEON_C_LSTM_H_
#define AI_BASEON_C_LSTM_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief LSTM fp32 权重参数。
 */
typedef struct lstm_fp32_param {
    uint32_t input_size;   ///< 输入维度
    uint32_t hidden_size;  ///< 隐藏单元数
    const float *w_ih;     ///< [4*hidden][input]，门顺序 i/f/g/o
    const float *w_hh;     ///< [4*hidden][hidden]，门顺序 i/f/g/o
    const float *bias;     ///< [4*hidden]，bias_ih + bias_hh 合并
} lstm_fp32_param_t;

/**
 * @brief LSTM int8 动态量化权重参数。
 */
typedef struct lstm_int8_param {
    uint32_t input_size;         ///< 输入维度
    uint32_t hidden_size;        ///< 隐藏单元数
    const int8_t *w_ih;          ///< int8 权重 [4*hidden][input]
    const float *w_ih_scale;     ///< per-row 权重步长 [4*hidden]
    const int32_t *w_ih_rowsum;  ///< int8 权重行和 [4*hidden]（零点折叠用）
    const int8_t *w_hh;          ///< int8 权重 [4*hidden][hidden]
    const float *w_hh_scale;     ///< per-row 权重步长 [4*hidden]
    const int32_t *w_hh_rowsum;  ///< int8 权重行和 [4*hidden]（零点折叠用）
    const float *bias;           ///< fp32 偏置 [4*hidden]
} lstm_int8_param_t;

/**
 * @brief 输出（分类）层 fp32 参数（与 rnn.h 相同结构）。
 */
typedef struct lstm_out_fp32 {
    uint32_t hidden_size;
    uint32_t num_classes;
    const float *w;
    const float *b;
} lstm_out_fp32_t;

/**
 * @brief 输出（分类）层 int8 参数（与 rnn.h 相同结构）。
 */
typedef struct lstm_out_int8 {
    uint32_t hidden_size;
    uint32_t num_classes;
    const int8_t *w;
    const float *w_scale;
    const float *b;
} lstm_out_int8_t;

/**
 * @brief LSTM int8 前向所需的临时浮点缓冲大小（float 个数）。
 *
 * 布局：4 个门预激活（4*hidden）+ 原始 h（hidden）+ 细胞状态 c（hidden）。
 */
#define LSTM_INT8_WORK_SIZE(hidden) ((size_t)6 * (size_t)(hidden))

/**
 * @brief LSTM fp32 前向推理。
 *
 * 初始 c_0 = h_0 = 0。若 out 与 logits 均非 NULL，则用最后一个 h 计算 logits。
 * 输入、输出与状态缓冲不得重叠；调用方必须提供各布局的完整容量。
 * 无法在目标平台上表示的数组尺寸返回 CNN_DATA_EXCEPTION。
 * 非有限细胞状态、隐藏状态或 logits 返回 CNN_DATA_EXCEPTION；失败时缓冲可能已部分更新。
 *
 * @param p        [in]  LSTM fp32 权重
 * @param out      [in]  输出层参数；传 NULL 表示不计算 logits
 * @param x        [in]  输入序列，布局 [seq_len][input_size]
 * @param seq_len  [in]  序列长度，>0
 * @param c_state  [in]  细胞状态临时缓冲，长度 hidden_size（函数内初始化为 0）
 * @param h_seq    [out] 每步隐藏状态，布局 [seq_len][hidden_size]
 * @param logits   [out] 可选；长度 num_classes
 * @return CNN_NORMAL / CNN_POINTER_NULL / CNN_DATA_EXCEPTION
 */
int lstm_forward_fp32(const lstm_fp32_param_t *p, const lstm_out_fp32_t *out, const float * x,
                      uint32_t seq_len, float * c_state, float * h_seq,
                      float * logits);

/**
 * @brief LSTM int8 动态量化前向推理。
 *
 * 输入为 uint8 量化域；隐藏状态每个时间步动态量化（uint8 非对称）；
 * 细胞状态保持 float32。
 * input_size / hidden_size 最大 32896，确保含零点校正的 int32 中间值安全。
 * 权重 scale 必须有限且 >0，偏置必须有限，rowsum 必须等于对应权重行和。
 * 输入、输出与临时缓冲不得重叠；调用方必须提供完整容量。
 * 常量隐藏态使用 |v|/128、zp=0（正数）或 255（负数）兜底；常规 zp 向零截断。
 *
 * @param p           [in]  LSTM int8 权重
 * @param out         [in]  输出层 int8 参数；传 NULL 表示不计算 logits
 * @param x_q         [in]  量化输入序列，布局 [seq_len][input_size]
 * @param seq_len     [in]  序列长度，>0
 * @param x_scale     [in]  输入量化步长，必须有限且 >0
 * @param x_zp        [in]  输入量化零点，范围 [0,255]
 * @param work        [in]  临时缓冲，长度 LSTM_INT8_WORK_SIZE(hidden_size)
 * @param h_q_seq     [out] 每步量化隐藏状态，布局 [seq_len][hidden_size]
 * @param h_scale_seq [out] 每步隐藏状态动态 scale，长度 seq_len
 * @param h_zp_seq    [out] 每步隐藏状态动态 zp，长度 seq_len
 * @param logits      [out] 可选；长度 num_classes
 * @return CNN_NORMAL / CNN_POINTER_NULL / CNN_DATA_EXCEPTION
 */
int lstm_forward_int8(const lstm_int8_param_t *p, const lstm_out_int8_t *out, const uint8_t * x_q,
                      uint32_t seq_len, float x_scale, int x_zp, float * work,
                      uint8_t * h_q_seq, float * h_scale_seq,
                      int * h_zp_seq, float * logits);

#ifdef __cplusplus
}
#endif

#endif  // AI_BASEON_C_LSTM_H_
