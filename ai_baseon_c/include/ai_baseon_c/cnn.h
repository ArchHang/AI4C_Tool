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
 * @file cnn.h
 * @brief 可移植 CNN 张量算子库（fp32 与 int8 量化推理）
 *
 * 本库只实现张量算子：卷积、批归一化、激活、全连接、池化、量化等，
 * 不关心数据来源（音频、图像或任意张量），也不包含任何预处理。
 * 全部代码仅依赖 C 标准库，可直接交叉编译到嵌入式平台（如 RISC-V/DSP），
 * 无第三方依赖、无动态内存分配（调用方提供所有 buffer）。
 *
 * 内存布局约定（全库统一，NCHW 连续存储）：
 *   - 特征图:  [channel][row][col]，data 容量为 channel * row * col
 *   - 卷积核:  [filter_num][channel][k_row][k_col]
 *   - 全连接:  weight 为 [fea_size][inp_size]
 *   - uint8 特征图布局与 fp32 相同（[channel][row][col]）
 *
 * 命名规范（全库统一，新增/修改代码必须遵循）：
 *   - 类型：snake_case，typedef 以 _t 结尾（conv2d_config_i8_t）
 *   - 函数/变量/结构体字段：snake_case
 *   - 宏与常量：SCREAMING_SNAKE_CASE（DYNAMIC_QUANT_FRAC_BITS）
 *   - 公开 API 形参：完整单词（input/config/output/weights/kernel/
 *     in_row/in_col/in_chan/out_chan/...），不使用单字母缩写
 *   - static 内核与内部实现：允许紧凑别名（input/weights/output），与 fp32
 *     conv2d_bn_no_bias 的内部 restrict 别名（input/weights/output）风格一致
 *   - 卷积族循环变量约定（与 fp32 实现一致）：
 *     i=输出通道, j/kernel=输出行列, ii=输入通道, jj/kk=核行列, t=抽头
 *   - 量化公式符号保留数学记号（scale/zp/acc/y/quant）
 */

#ifndef AI_BASEON_C_CNN_H_
#define AI_BASEON_C_CNN_H_

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief CNN 推理库错误码（所有 int 返回值函数使用）
 *
 * 返回 CNN_NORMAL(0) 表示成功；其余为非零错误码（保留既有枚举值）。
 */
typedef enum cnn_error {
    CNN_ERR_GENERIC = -1,   ///< 通用错误
    CNN_NORMAL = 0,         ///< 成功
    CNN_MALLOC_FAIL,        ///< 内存分配失败（本库不主动分配，保留兼容）
    CNN_DATA_NOT_ENOUGH,    ///< 数据长度不足
    CNN_DATA_EXCEPTION,     ///< 数据/参数异常（形状、步长、核大小等不合法）
    CNN_DATA_NULL,          ///< 数据指针为空
    CNN_DATA_INVALID,       ///< 数据内容非法
    CNN_DATA_TOO_MANY,      ///< 数据量过大
    CNN_ERR_LOCAL_FREE,     ///< 本地释放失败（保留兼容）
    CNN_POINTER_NULL,       ///< 必要指针为空（NULL 参数）
    CNN_DATA_QUALITY_POOR,  ///< 数据质量差
    CNN_IO_EXCEPTION        ///< IO 异常
} cnn_error_t;

/**
 * @brief 卷积层输入/输出特征图
 *
 * data 指向连续内存，布局为 [channel][row][col]，
 * 容量至少 channel * row * col 个 float。
 * 1D 特征（如 2048 点 FFT 幅度谱）表示为 row=1、col=长度。
 */
typedef struct conv2d_data {
    uint32_t row;      ///< 高；1D 特征恒为 1
    uint32_t col;      ///< 宽（1D 特征的长度）
    uint32_t channel;  ///< 通道数
    float* data;       ///< 特征图数据，NCHW 连续存储
} conv2d_data_t;

/**
 * @brief 卷积层权重（卷积核）
 *
 * data 布局为 [filter_num][channel][row][col]，
 * 容量 filter_num * channel * row * col 个 float。
 * 1D 卷积表示为 row=1、col=kernel 宽度。
 */
typedef struct conv2d_filter {
    uint32_t row;         ///< 核高（1D 为 1）
    uint32_t col;         ///< 核宽
    uint32_t channel;     ///< 输入通道数（必须等于特征图通道数）
    uint32_t filter_num;  ///< 输出通道数
    float* data;          ///< 权重数据
} conv2d_filter_t;

/**
 * @brief 批归一化（BatchNorm）参数
 *
 * 推理公式（eps=1e-5）：
 *   y = gamma * (x - mean) / sqrt(var + eps) + beta
 *
 * 本库内部会将其折叠为每通道 scale/shift 以加速，且不改动调用方数据。
 * size 必须等于卷积输出通道数 filter_num。
 */
typedef struct batch_norm2d {
    uint32_t size;  ///< 参数长度，必须等于 filter_num
    float* mean;    ///< running_mean，长度 size
    float* var;     ///< running_var，长度 size
    float* gamma;   ///< weight，长度 size
    float* beta;    ///< bias，长度 size
} batch_norm2d_t;

/**
 * @brief 卷积层配置（卷积核 + BN + 步长 + padding）
 *
 * padding 为“虚拟 padding”：不分配临时缓冲、不搬运数据，
 * 卷积内部对越界输入按 0 处理，仅边界输出像素有额外判断，
 * 对嵌入式性能影响很小。
 */
typedef struct conv2d_config {
    uint32_t stride;       ///< 步长，>=1
    uint32_t pad;          ///< 每侧 zero-padding 宽度（行列相同，>=0）
    conv2d_filter_t* filter;  ///< 卷积核
    batch_norm2d_t* bn;       ///< BN 参数
} conv2d_config_t;

/**
 * @brief 全连接层（线性层）配置
 *
 * weight 布局为 [fea_size][inp_size]，bias 长度为 fea_size。
 */
typedef struct linear_param {
    uint32_t inp_size;  ///< 输入维度
    uint32_t fea_size;  ///< 输出维度
    float* weight;      ///< 权重，[fea_size][inp_size]
    float* bias;        ///< 偏置，长度 fea_size
} linear_param_t;

/**
 * @brief 最大池化配置
 *
 * 当前仅支持 pad=0（valid 池化）。
 * 1D 池化表示为 row=1、col=核宽。
 */
typedef struct max_pool_config {
    uint32_t row;     ///< 池化核高
    uint32_t col;     ///< 池化核宽
    uint32_t stride;  ///< 步长，>=1
    uint32_t pad;     ///< padding，当前仅支持 0
} max_pool_config_t;

/**
 * @brief 计算卷积/池化输出长度（valid 输出）
 *
 * 公式：output = (raw_len + 2*pad_len - filter_len) / stride + 1
 *
 * @param raw_len    输入长度
 * @param pad_len    每侧 padding 宽度
 * @param filter_len 核大小
 * @param stride     步长
 * @return 输出长度；非法几何、零步长或溢出时返回 0。
 */
uint32_t cal_conv_out_len(uint32_t raw_len, uint32_t pad_len, uint32_t filter_len, uint32_t stride);

/**
 * @brief fp32 卷积 + BN（无卷积 bias，BN 的 beta 充当 bias）
 *
 * 支持 1D/2D、zero-padding、任意 stride>=1。
 * 内部把 BN 折叠为 per-channel scale/shift，纯 C 实现。
 *
 * 累加顺序（保证与朴素参考实现一致）：
 *   channel(0..C-1) -> kernel_row -> kernel_col
 *
 * @param input_feat  [input]  输入特征图；1D 时 row=1
 * @param param       [input]  卷积配置（stride/pad/filter/bn）
 * @param output_feat [output] 输出特征图；data 由调用方提供，
 *                           容量至少 filter_num * out_row * out_col
 * @return CNN_NORMAL / CNN_POINTER_NULL / CNN_DATA_EXCEPTION
 */
int conv2d_bn_no_bias(conv2d_data_t* input_feat, conv2d_config_t* param, conv2d_data_t* output_feat);

/**
 * @brief LeakyReLU 激活函数
 *
 * y = x >= 0 ? x : x * neg_slope
 *
 * @param neg_slope 负半轴斜率；传 0 即 ReLU
 * @param inp       [input]  输入数据
 * @param inp_size  [input]  输入长度
 * @param output       [output] 输出数据；支持 input-place（output == inp）
 * @return CNN_NORMAL / CNN_POINTER_NULL
 */
int leaky_relu(float neg_slope, float* inp, uint32_t inp_size, float* output);

/**
 * @brief Softmax 激活（数值稳定：先减最大值再 exp，避免溢出）
 *
 * @param inp  [input]  输入数据，长度 size
 * @param size [input]  输入长度，必须 >0
 * @param output  [output] 输出概率，和为 1；支持 input-place（output == inp）
 * @return CNN_NORMAL / CNN_POINTER_NULL
 */
int softmax(const float* inp, uint32_t size, float* output);

/**
 * @brief 返回最大值所在下标（argmax）
 *
 * 并列时返回第一个最大值下标。
 *
 * @param inp     [input]  输入数据
 * @param size    [input]  输入长度，必须 >0
 * @param max_val [output] 可选；返回最大值，传 NULL 可忽略
 * @return 最大值下标；参数非法时返回 0
 */
uint32_t argmax(const float* inp, uint32_t size, float* max_val);

/* ============================ int8 量化算子 ============================ */

/**
 * @brief uint8 激活量化参数（非对称量化域）
 *
 * scale 为量化步长（>0），zp 为零点（量化域中对应浮点 0 的值，[0,255]）。
 * zp == 0 表示对称量化或零点为 0 的非对称量化，量化算子据此自动走
 * 免减法快路径。
 */
typedef struct quant_param {
    float scale;  ///< 量化步长，>0
    int zp;       ///< 零点，[0,255]；==0 时自动走免减法快路径
} quant_param_t;

/**
 * @brief int8 链路量化特征图（uint8 数据）
 *
 * 布局与容量约定同 conv2d_data_t（[channel][row][col] 连续存储），
 * 仅数据类型为 uint8（int8 链路的激活以 uint8 编码）。1D 特征表示为
 * row=1。结构体后缀 I8 表示“int8 链路”，与 conv2d_filter_i8_t /
 * conv2d_config_i8_t / linear_param_i8_t 同族。
 */
typedef struct conv2d_data_i8 {
    uint32_t row;      ///< 高；1D 特征恒为 1
    uint32_t col;      ///< 宽（1D 特征的长度）
    uint32_t channel;  ///< 通道数
    uint8_t* data;     ///< 特征图数据，NCHW 连续存储
} conv2d_data_i8_t;

/**
 * @brief int8 量化卷积核（权重 + per-channel scale + 折叠 bias）
 *
 * data 布局 [filter_num][channel][row][col]（与 conv2d_filter_t 相同，
 * 元素为 int8）；scale 为 per-output-channel 权重步长（长度 filter_num）；
 * bias 为 BN 折叠后的 float 偏置（长度 filter_num）。
 */
typedef struct conv2d_filter_i8 {
    uint32_t row;         ///< 核高（1D 为 1）
    uint32_t col;         ///< 核宽
    uint32_t channel;     ///< 输入通道数（必须等于特征图通道数）
    uint32_t filter_num;  ///< 输出通道数
    const int8_t* data;   ///< int8 权重
    const float* scale;   ///< per-channel 权重步长，长度 filter_num
    const float* bias;    ///< BN 折叠后的偏置，长度 filter_num
} conv2d_filter_i8_t;

/**
 * @brief int8 卷积配置（量化参数内嵌，自动分派内核）
 *
 * 统一入口 conv2d_int8_run 根据本结构体自动分派：
 *   - 形状：input->row==1 且 filter->row==1 时走 1D 优化算法；
 *   - 零点：in_act.zp == 0（对称量化/零点为 0）走免减法快路径，
 *     否则走通用路径（零点折进权重行和）；
 *   - 模式：dynamic==0 静态量化（out_act 为校准参数，只读语义）；
 *     dynamic==1 动态量化（在线计算 scale 写回 out_act，zp 恒 0，
 *     可直接作为下一层的 in_act）。
 */
typedef struct conv2d_config_i8 {
    uint32_t stride;               ///< 步长，>=1
    uint32_t pad;                  ///< 每侧 zero-padding 宽度（虚拟 padding）
    const conv2d_filter_i8_t* filter;  ///< 卷积核
    quant_param_t in_act;             ///< 输入激活量化参数
    int dynamic;                   ///< 0=静态量化；1=动态量化（无需校准）
    quant_param_t* out_act;           ///< 静态：输出量化参数（只读语义）；
                                   ///< 动态：接收在线计算的 scale/zp（出参）
} conv2d_config_i8_t;

typedef struct conv2d_packed_context {
    const int16_t *weights;  ///< 只读预展开权重；生命周期覆盖推理调用。
    size_t weights_size;    ///< weights 容量，字节。
    int32_t *scratch;       ///< 可在层间复用的池化后 int32 累加值。
    size_t scratch_size;    ///< scratch 容量，字节。
} conv2d_packed_context_t;

/**
 * @brief 查询受支持卷积核预展开后的字节数，不分配内存。
 *
 * 仅支持 row=1、每输出通道 1..64 个权重、1..32 个输出通道。
 * @param filter [in] 原始卷积核；data 必须非空，scale/bias 不参与此查询。
 * @return 每通道补齐后的 int16_t 权重总字节数；非法或不支持时为 0。
 */
size_t conv2d_int8_packed_weights_size(const conv2d_filter_i8_t *filter);

/**
 * @brief 将 int8 权重符号扩展、重排并补零至调用方提供的缓冲。
 *
 * output 必须按 int32_t 对齐，不得覆盖 filter、原始权重、scale 或 bias。
 * 失败时不写 output；不修改 filter 的数据指针，不分配内存。
 * @param filter [in] 原始 int8 卷积核，支持范围与大小查询一致。
 * @param output [out] 预展开权重缓冲。
 * @param output_size [in] output 容量，单位字节。
 * @return CNN_NORMAL、CNN_POINTER_NULL 或 CNN_DATA_EXCEPTION。
 */
int32_t conv2d_int8_pack_weights(const conv2d_filter_i8_t *filter,
                                int16_t *output, size_t output_size);

/**
 * @brief 使用预展开权重执行动态 1D 卷积、ReLU、量化与 maxpool2。
 *
 * 只支持 stride=1、pad=0、dynamic=1、input/filter row=1、卷积输出宽至少 2，
 * 权重形状限制见大小查询。scratch 必须非空且不小于原融合算子的大小查询值。
 * input/output、预展开权重和 scratch 的有效字节范围必须彼此分离；输出和 scratch
 * 也不得覆盖卷积核元数据、原权重、scale 或 bias。所有容量均由调用方保证。
 * out_act 可指向 config.in_act，奇数输出宽被池化丢弃的末位置仍参与动态 scale。
 * ARM DSP 目标使用双位置、双通道 SMLAD；其它后端自动转调原融合入口。
 * 不分配内存；缓存须先由 pack_weights 初始化且匹配原始权重，结果与量化舍入规则相同。
 * @param input [in] CHW 连续 uint8 输入，零点范围 0..255。
 * @param config [in] 动态卷积配置；filter.data 保留原始 int8 权重。
 * @param output [out] 池化后特征与成功时写回的形状。
 * @param context [in] 预展开权重和调用方所有的 int32 累加缓冲。
 * @return CNN_NORMAL、CNN_POINTER_NULL 或 CNN_DATA_EXCEPTION。
 */
int32_t conv2d_int8_maxpool2_run_packed(const conv2d_data_i8_t *input,
                                      const conv2d_config_i8_t *config,
                                      conv2d_data_i8_t *output,
                                      const conv2d_packed_context_t *context);

/**
 * @brief int8 全连接配置
 *
 * weight 布局 [fea_size][inp_size]（int8），scale 为 per-output 权重步长
 * （长度 fea_size），bias 为 float 偏置（长度 fea_size）。
 */
typedef struct linear_param_i8 {
    uint32_t inp_size;    ///< 输入维度
    uint32_t fea_size;    ///< 输出维度
    const int8_t* weight; ///< int8 权重，[fea_size][inp_size]
    const float* scale;   ///< per-output 权重步长，长度 fea_size
    const float* bias;    ///< float 偏置，长度 fea_size
} linear_param_i8_t;

/**
 * @brief int8 卷积统一入口（静态/动态量化、1D/2D、零点分派全自动）
 *
 * 与 fp32 的 conv2d_bn_no_bias 同构：输入/配置/输出三个结构体。
 * 分派规则（全部自动，无需调用方选择函数）：
 *   1) input->row==1 且 filter->row==1 -> 1D 优化算法，否则 2D 通用算法；
 *   2) in_act.zp == 0 -> 免减法快路径（内积不做“减零点”），
 *      否则通用路径（界内窗口把零点折进权重行和，整数恒等）；
 *   3) dynamic==0 -> 静态量化（out_act 为校准得到的输出量化参数）；
 *      dynamic==1 -> 动态量化（整数域两遍法在线计算 scale，写回
 *      out_act->scale 并置 out_act->zp=0）。
 *
 * ReLU 内置于卷积（量化链路约定）；输出几何形状由本函数写回 output 的
 * row/col/channel 字段。
 *
 * @param input  [input]  输入量化特征图；data 为 uint8，维度字段必须填好
 * @param config [input]  卷积配置（含输入/输出量化参数与模式）
 * @param output [output] 输出量化特征图；data 由调用方提供，容量至少
 *                  filter_num * out_row * out_col；维度字段由本函数写回
 * @return CNN_NORMAL / CNN_POINTER_NULL / CNN_DATA_EXCEPTION
 */
int conv2d_int8_run(const conv2d_data_i8_t* input, const conv2d_config_i8_t* config, conv2d_data_i8_t* output);

/**
 * @brief 查询动态卷积的可选累加缓冲字节数。
 *
 * 静态模式或非法配置返回 0；动态模式返回输出元素数乘 sizeof(int32_t)。
 * @param input [in] 输入特征，布局 NCHW。
 * @param config [in] 卷积配置，包含非空权重和量化参数。
 * @return 可复用的工作缓冲字节数，库内部不分配内存。
 */
size_t conv2d_int8_get_scratch_size(const conv2d_data_i8_t *input,
                                  const conv2d_config_i8_t *config);

/**
 * @brief 使用可选调用方缓冲执行整数卷积。
 *
 * 动态模式保存完整 int32 输出，避免第二遍 MAC；scratch=NULL、size=0 使用两遍法。
 * 输入、输出和 scratch 不允许重叠。scratch 需按 int32_t 对齐，可在层间复用。
 * 静态模式忽略合法 scratch。所有张量容量由调用方保证。
 * 每个输出的 MAC 项数限制为 32896，防止含 -128 权重的 int32 中间量溢出。
 * @param input [in] uint8 输入 NCHW。
 * @param config [in] 卷积配置；动态 out_act 写回 scale/zp。
 * @param output [out] 输出特征和几何形状。
 * @param scratch [in,out] 可选 int32 累加缓冲，长度至少查询结果除以 sizeof(int32_t)。
 * @param scratch_size [in] scratch 容量，单位字节。
 * @return CNN_NORMAL、CNN_POINTER_NULL 或 CNN_DATA_EXCEPTION。
 */
int32_t conv2d_int8_run_with_scratch(const conv2d_data_i8_t *input,
                                    const conv2d_config_i8_t *config, conv2d_data_i8_t *output,
                                    int32_t *scratch, size_t scratch_size);

/**
 * @brief 查询动态 1D 卷积与横向 maxpool2 融合的累加缓冲字节数。
 *
 * 仅支持 input/filter 的 row=1、stride=1、pad=0、dynamic=1，卷积输出宽至少 2。
 * 池化固定为 kernel=2、stride=2，输出宽为卷积输出宽除以 2 向下取整。
 * @param input [in] CHW 连续 uint8 输入。
 * @param config [in] 动态卷积配置。
 * @return 池化输出元素数乘 sizeof(int32_t)；非法或不支持的配置返回 0。
 */
size_t conv2d_int8_maxpool2_get_scratch_size(const conv2d_data_i8_t *input,
                                           const conv2d_config_i8_t *config);

/**
 * @brief 融合动态 1D 卷积、ReLU、量化和非重叠 maxpool2。
 *
 * 形状限制与 conv2d_int8_maxpool2_get_scratch_size 相同。动态 scale 来自全部未池化
 * 卷积位置，包括奇数宽时池化丢弃的最后位置，结果与卷积后 maxpool_u8_run 逐位一致。
 * scratch 只保存每对位置的 int32 最大值；NULL 且 size=0 使用两遍计算。
 * 输入、输出和 scratch 不允许重叠，scratch 按 int32_t 对齐，容量由调用方保证。
 * config.out_act 可指向 config.in_act。库内部不分配内存。
 * @param input [in] uint8 输入，布局 CHW。
 * @param config [in] 卷积配置；out_act 写回动态 scale/zp。
 * @param output [out] 池化后 uint8 特征，至少 filter_num * floor(conv_out_col/2) 字节。
 * @param scratch [in,out] 可选调用方 int32 缓冲，可在层间复用。
 * @param scratch_size [in] scratch 容量，单位字节。
 * @return CNN_NORMAL、CNN_POINTER_NULL 或 CNN_DATA_EXCEPTION。
 */
int32_t conv2d_int8_maxpool2_run_with_scratch(const conv2d_data_i8_t *input,
                                            const conv2d_config_i8_t *config,
                                            conv2d_data_i8_t *output, int32_t *scratch,
                                            size_t scratch_size);

/**
 * @brief int8 全连接统一入口：uint8 输入 -> fp32 输出（不做输出量化）
 *
 * 计算流程：
 *   1) 整数域累加：acc += w_q * (in_q - zp)，int32 累加
 *      （zp==0 时自动走免减法快路径；zp!=0 时零点折进权重行和）
 *   2) 反量化：output = acc * (scale * in_act->scale) + bias
 *
 * 无法表示的输出字节数或非有限结果返回 CNN_DATA_EXCEPTION；
 * 失败时输出缓冲可能已经部分更新。
 *
 * @param input      [input]  输入 uint8，长度 config->inp_size
 * @param config       [input]  全连接配置（权重/步长/偏置）
 * @param in_act  [input]  输入激活量化参数
 * @param output     [output] 输出 fp32，长度 config->fea_size
 * @return CNN_NORMAL / CNN_POINTER_NULL / CNN_DATA_EXCEPTION
 */
int linear_int8_run(const uint8_t* input, const linear_param_i8_t* config, const quant_param_t* in_act,
                    float* output);

/**
 * @brief uint8 最大池化统一入口（1D/2D 自动分派，带校验与维度写回）
 *
 * 量化函数单调，uint8 直接取 max 与浮点池化等价；支持 input-place
 * （input->data == output->data，输出升序写入安全）。
 *
 * @param input    [input]  输入量化特征图
 * @param param [input]  池化配置（row/col/stride，pad 仅支持 0）
 * @param output   [output] 输出量化特征图；维度字段由本函数写回
 * @return CNN_NORMAL / CNN_POINTER_NULL / CNN_DATA_EXCEPTION
 */
int maxpool_u8_run(const conv2d_data_i8_t* input, const max_pool_config_t* param, conv2d_data_i8_t* output);

/**
 * @brief float -> uint8 量化统一入口
 *
 * quant = clamp(round(x / quant->scale) + quant->zp, 0, 255)，遵循当前浮点舍入模式
 * （部署使用最近偶数）；有界M4/M7硬件路径与lrintf回退语义相同。参数非法时不写output。
 *
 * @param input  [input]  输入 float 数组，长度 size
 * @param size   [input]  元素个数，必须 >0
 * @param quant   [input]  量化参数（scale 必须大于 0）
 * @param output [output] 输出 uint8 数组，长度 size
 */
void quantize_u8_run(const float* input, uint32_t size, const quant_param_t* quant, uint8_t* output);

/**
 * @brief float -> uint8 非对称量化（兼容封装，等价于 quantize_u8_run）
 *
 * 公式：
 *   quant = clamp(round(x / scale) + zp, 0, 255)
 *
 * 舍入规则与 C 端其余量化算子一致：部署使用最近偶数，有界硬件或lrintf回退。
 * 注意：本函数为 void，参数非法时直接返回（不写 output）。
 *
 * @param input    [input]  输入 float 数组，长度 size
 * @param size     [input]  元素个数，必须 >0
 * @param scale [input]  量化步长，必须 >0
 * @param zp    [input]  零点（量化域中对应浮点 0 的值）
 * @param output   [output] 输出 uint8 数组，长度 size
 */
void quantize_u8(const float* input, uint32_t size, float scale, int zp, uint8_t* output);

/* -------------------- int8 兼容层（散参数旧 API） --------------------
 * 以下函数为兼容旧调用方保留：内部构造结构体后转调上面的统一入口
 * （conv2d_int8_run / linear_int8_run），行为与结果逐位一致。
 * 新代码建议直接使用统一入口。
 */

/**
 * @brief int8 1D 卷积 + ReLU + 输出量化（静态量化，兼容封装）
 *
 * 等价于 conv2d_int8_run：in_act_zp == 0 时自动走免减法快路径，
 * 非对称输入在界内窗口把零点折进权重行和（整数恒等）。
 *
 * 输入 uint8，布局 [in_chan][in_col]；权重 int8 per-channel，
 * 布局 [out_chan][in_chan][kernel]；bias 为 BN 折叠后的 float 偏置。
 *
 * @param input             [input]  输入 uint8 特征，长度 in_chan * in_col
 * @param in_col         [input]  输入宽度
 * @param in_chan        [input]  输入通道数，>0
 * @param weights              [input]  int8 权重
 * @param kernel              [input]  核宽，>0
 * @param stride         [input]  步长，>=1（支持任意值，如 2/3）
 * @param pad            [input]  每侧 zero-padding 宽度（虚拟 padding）
 * @param out_chan       [input]  输出通道数，>0
 * @param weights_scale        [input]  per-channel 权重步长，长度 out_chan
 * @param bias           [input]  折叠后的偏置，长度 out_chan
 * @param in_act_scale   [input]  输入激活步长
 * @param in_act_zp      [input]  输入激活零点
 * @param out_act_scale  [input]  输出激活步长
 * @param out_act_zp     [input]  输出激活零点
 * @param output            [output] 输出 uint8，容量 out_chan *
 *                             ((in_col + 2*pad - kernel) / stride + 1)
 * @return CNN_NORMAL / CNN_POINTER_NULL / CNN_DATA_EXCEPTION
 */
int conv1d_int8(const uint8_t* input, uint32_t in_col, uint32_t in_chan, const int8_t* weights, uint32_t kernel,
                uint32_t stride, uint32_t pad, uint32_t out_chan, const float* weights_scale,
                const float* bias, float in_act_scale, int in_act_zp, float out_act_scale,
                int out_act_zp, uint8_t* output);

/**
 * @brief int8 卷积统一入口 + ReLU + 输出量化（兼容封装，1D/2D 自动分派）
 *
 * 等价于 conv2d_int8_run：当 in_row==1 且 k_row==1 时自动分派到 1D
 * 优化算法，否则走 2D 通用算法；零点分派规则同 conv1d_int8。
 *
 * 输入 uint8，布局 [in_chan][in_row][in_col]；
 * 权重 int8，布局 [out_chan][in_chan][k_row][k_col]。
 *
 * @param input             [input]  输入 uint8 特征
 * @param in_row         [input]  输入行数（1D 时为 1）
 * @param in_col         [input]  输入列数
 * @param in_chan        [input]  输入通道数
 * @param weights              [input]  int8 权重
 * @param k_row          [input]  核高（1D 时为 1）
 * @param k_col          [input]  核宽
 * @param stride         [input]  行列共用步长，>=1
 * @param pad            [input]  每侧 zero-padding 宽度（行列相同，虚拟 padding）
 * @param out_chan       [input]  输出通道数
 * @param weights_scale        [input]  per-channel 权重步长，长度 out_chan
 * @param bias           [input]  折叠后的偏置，长度 out_chan
 * @param in_act_scale   [input]  输入激活步长
 * @param in_act_zp      [input]  输入激活零点
 * @param out_act_scale  [input]  输出激活步长
 * @param out_act_zp     [input]  输出激活零点
 * @param output            [output] 输出 uint8，容量
 *   out_chan * ((in_row + 2*pad - k_row) / stride + 1) *
 *              ((in_col + 2*pad - k_col) / stride + 1)
 * @return CNN_NORMAL / CNN_POINTER_NULL / CNN_DATA_EXCEPTION
 */
int conv2d_int8(const uint8_t* input, uint32_t in_row, uint32_t in_col, uint32_t in_chan,
                const int8_t* weights, uint32_t k_row, uint32_t k_col, uint32_t stride, uint32_t pad,
                uint32_t out_chan, const float* weights_scale, const float* bias, float in_act_scale,
                int in_act_zp, float out_act_scale, int out_act_zp, uint8_t* output);

/**
 * @brief int8 1D 卷积 + 动态量化（兼容封装，无需校准数据）
 *
 * 等价于 conv2d_int8_run 的 dynamic=1 模式：整数域两遍法在线计算
 * 激活 scale（第一遍找每通道最大 acc，第二遍定点重量化，零额外内存），
 * 本层动态 scale 通过 out_act_scale 返回，供下一层反量化使用。
 *
 * @param input             [input]  输入 uint8 特征
 * @param in_col         [input]  输入宽度
 * @param in_chan        [input]  输入通道数
 * @param weights              [input]  int8 权重
 * @param kernel              [input]  核宽
 * @param stride         [input]  步长
 * @param pad            [input]  每侧 zero-padding 宽度（虚拟 padding）
 * @param out_chan       [input]  输出通道数
 * @param weights_scale        [input]  per-channel 权重步长
 * @param bias           [input]  折叠后的偏置
 * @param in_act_scale   [input]  输入激活步长（上一层返回的动态 scale）
 * @param in_act_zp      [input]  输入激活零点（动态量化链通常为 0）
 * @param out_act_scale  [output] 本层动态 scale（= max/255）
 * @param output            [output] 输出 uint8，容量 out_chan * out_col
 * @return CNN_NORMAL / CNN_POINTER_NULL / CNN_DATA_EXCEPTION
 */
int conv1d_int8_dynamic(const uint8_t* input, uint32_t in_col, uint32_t in_chan, const int8_t* weights,
                        uint32_t kernel, uint32_t stride, uint32_t pad, uint32_t out_chan,
                        const float* weights_scale, const float* bias, float in_act_scale, int in_act_zp,
                        float* out_act_scale, uint8_t* output);

/**
 * @brief int8 卷积统一入口 + 动态量化（兼容封装，1D/2D 自动分派）
 *
 * 等价于 conv2d_int8_run 的 dynamic=1 模式；1D/2D 自动分派规则同
 * conv2d_int8，返回 out_act_scale 供下一层使用。
 *
 * @param input             [input]  输入 uint8 特征
 * @param in_row         [input]  输入行数（1D 时为 1）
 * @param in_col         [input]  输入列数
 * @param in_chan        [input]  输入通道数
 * @param weights              [input]  int8 权重
 * @param k_row          [input]  核高（1D 时为 1）
 * @param k_col          [input]  核宽
 * @param stride         [input]  步长
 * @param pad            [input]  每侧 zero-padding 宽度（虚拟 padding）
 * @param out_chan       [input]  输出通道数
 * @param weights_scale        [input]  per-channel 权重步长
 * @param bias           [input]  折叠后的偏置
 * @param in_act_scale   [input]  输入激活步长
 * @param in_act_zp      [input]  输入激活零点
 * @param out_act_scale  [output] 本层动态 scale
 * @param output            [output] 输出 uint8
 * @return CNN_NORMAL / CNN_POINTER_NULL / CNN_DATA_EXCEPTION
 */
int conv2d_int8_dynamic(const uint8_t* input, uint32_t in_row, uint32_t in_col, uint32_t in_chan,
                        const int8_t* weights, uint32_t k_row, uint32_t k_col, uint32_t stride,
                        uint32_t pad, uint32_t out_chan, const float* weights_scale, const float* bias,
                        float in_act_scale, int in_act_zp, float* out_act_scale, uint8_t* output);

/**
 * @brief uint8 1D 最大池化（量化域直接取 max，与浮点池化等价）
 *
 * 输出长度：out_n = (size - kernel) / stride + 1（valid 池化）。
 *
 * @param input     [input]  输入 uint8，长度 size
 * @param size      [input]  输入长度
 * @param output    [output] 输出 uint8，长度 out_n
 * @param kernel      [input]  核宽
 * @param stride [input]  步长
 */
void maxpool1d_u8(const uint8_t* input, uint32_t size, uint8_t* output, uint32_t kernel, uint32_t stride);

/**
 * @brief uint8 2D 最大池化（量化域直接取 max）
 *
 * 输入布局 [in_row][in_col]（单通道），输出
 * [((in_row-k_row)/stride+1)] x [((in_col-k_col)/stride+1)]。
 * 多通道由调用方按通道平面循环调用。
 *
 * @param input     [input]  输入 uint8，长度 in_row * in_col
 * @param in_row [input]  输入行数
 * @param in_col [input]  输入列数
 * @param output    [output] 输出 uint8
 * @param k_row  [input]  核高
 * @param k_col  [input]  核宽
 * @param stride [input]  步长
 */
void maxpool2d_u8(const uint8_t* input, uint32_t in_row, uint32_t in_col, uint8_t* output, uint32_t k_row,
                  uint32_t k_col, uint32_t stride);

/**
 * @brief uint8 最大池化统一入口（1D/2D 自动分派）
 *
 * 当 in_row==1 且 k_row==1 时走 1D 优化算法，否则走 2D 通用算法。
 * 量化函数单调，max(quantize(a), quantize(b)) == quantize(max(a,b))，
 * 因此 uint8 直接取 max 与浮点池化等价。
 *
 * @param input     [input]  输入 uint8，单通道平面
 * @param in_row [input]  输入行数（1D 时为 1）
 * @param in_col [input]  输入列数
 * @param output    [output] 输出 uint8，单通道平面
 * @param k_row  [input]  核高（1D 时为 1）
 * @param k_col  [input]  核宽
 * @param stride [input]  步长
 */
void maxpool_u8(const uint8_t* input, uint32_t in_row, uint32_t in_col, uint8_t* output, uint32_t k_row,
                uint32_t k_col, uint32_t stride);

/**
 * @brief int8 全连接层：uint8 输入 -> fp32 输出（不做输出量化）
 *
 * 计算流程：
 *   1) 整数域累加：acc += w_q * (in_q - zp)，int32 累加
 *   2) 反量化：output = acc * (weights_scale * in_act_scale) + bias
 *
 * @param input           [input]  输入 uint8，长度 in_size
 * @param in_size      [input]  输入维度，>0
 * @param weights            [input]  int8 权重，布局 [out_size][in_size]
 * @param weights_scale      [input]  per-output 权重步长，长度 out_size
 * @param bias         [input]  float 偏置，长度 out_size
 * @param out_size     [input]  输出维度，>0
 * @param in_act_scale [input]  输入激活步长
 * @param in_act_zp    [input]  输入激活零点
 * @param output          [output] 输出 fp32 logits，长度 out_size
 * @return CNN_NORMAL / CNN_POINTER_NULL / CNN_DATA_EXCEPTION
 */
int linear_int8(const uint8_t* input, uint32_t in_size, const int8_t* weights, const float* weights_scale,
                const float* bias, uint32_t out_size, float in_act_scale, int in_act_zp,
                float* output);

/**
 * @brief fp32 全连接层（线性层）
 *
 * output = weights * input + bias。内部对输出行做 4 行分块累加（多独立累加器），
 * 以打破乘加链依赖、便于编译器向量化；累加顺序与朴素实现一致。
 *
 * @param inp           [input]  输入 float，长度 inp_size
 * @param linear_config [input]  全连接配置
 * @param output           [output] 输出 float，长度 fea_size
 * @return CNN_NORMAL / CNN_POINTER_NULL
 */
int linear_layer(float* inp, linear_param_t* linear_config, float* output);

/**
 * @brief fp32 最大池化（通用 1D/2D，任意核大小与步长）
 *
 * 1D：input row=1 且 param->row=1，按列滑动；
 * 2D：按行列滑动。当前仅支持 pad=0。
 *
 * 支持 input-place（input_feat->data == output_feat->data）：
 * 实现按输出升序写入，且先读完整个输入窗口再写输出位置，
 * 任意 stride>=1 均安全。
 *
 * @param input_feat  [input]  输入特征图
 * @param param       [input]  池化配置（row/col/stride/pad=0）
 * @param output_feat [output] 输出特征图，data 由调用方提供
 * @return CNN_NORMAL / CNN_POINTER_NULL / CNN_DATA_EXCEPTION
 */
int maxpool(conv2d_data_t* input_feat, max_pool_config_t* param, conv2d_data_t* output_feat);

/**
 * @brief 线性层 DSP 加速入口
 *
 * PC 版不再依赖 RISC-V DSP 库，统一走优化后的普通线性层实现。
 *
 * @param inp           [input]  输入 float
 * @param linear_config [input]  全连接配置
 * @param output           [output] 输出 float
 * @return error code（与 linear_layer 相同）
 */
int linear_layer_dsp(float* inp, linear_param_t* linear_config, float* output);

/**
 * @brief Sigmoid 激活（原地计算）
 *
 * y = 1 / (1 + exp(-x))
 *
 * @param x    [input,output] 输入/输出数据
 * @param size [input]     长度
 */
void sigmoid(float* x, uint32_t size);

#ifdef __cplusplus
}
#endif

#endif  // AI_BASEON_C_CNN_H_
