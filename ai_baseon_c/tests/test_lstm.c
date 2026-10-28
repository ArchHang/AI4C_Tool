/*
 * LSTM C 推理验证工具：读取输入 bin（uint8 量化 + float32 两份），
 * 分别运行 fp32 与 int8 动态量化前向，导出每步 h 与 logits 供 Python 对拍。
 *
 * 输入布局：[N][SEQ_LEN][INPUT_SIZE]
 * 输出布局：
 *   fp32.bin 每样本：h_seq float32[T*H] + logits float32[C]
 *   int8.bin 每样本：h_q uint8[T*H] + scale float32[T] + zp int32[T] +
 *                    logits float32[C]
 *
 * 用法：test_lstm.exe <x_u8.bin> <x_f32.bin> <N> <out_fp32.bin> <out_int8.bin>
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#endif

#include "ai_baseon_c/cnn.h"
#include "ai_baseon_c/lstm.h"
#include "lstm_model.h"

#define SEQ_LEN 16

static double now_ms(void)
{
#ifdef _WIN32
    static LARGE_INTEGER frequency = {{0}};
    LARGE_INTEGER counter;
    if (frequency.QuadPart == 0) {
        QueryPerformanceFrequency(&frequency);
    }
    QueryPerformanceCounter(&counter);
    return 1000.0 * (double)counter.QuadPart / (double)frequency.QuadPart;
#else
    struct timespec timestamp;
    if (timespec_get(&timestamp, TIME_UTC) != TIME_UTC) {
        return 0.0;
    }
    return 1000.0 * (double)timestamp.tv_sec + (double)timestamp.tv_nsec / 1000000.0;
#endif
}

int main(int argc, char** argv) {
    FILE *fu8, *ff32, *fo_fp32, *fo_int8;
    uint8_t* x_q = NULL;
    float* x_f = NULL;
    uint8_t* h_q = NULL;
    float *h_seq = NULL, *h_scale = NULL, *work = NULL, *c_state = NULL, *logits = NULL;
    int32_t* h_zp = NULL;
    const uint32_t in = LSTM_INPUT_SIZE;
    const uint32_t hid = LSTM_HIDDEN_SIZE;
    const uint32_t cls = LSTM_NUM_CLASSES;
    uint32_t n, s;
    double t0, t1;
    size_t total_u8, total_f32;

    if (argc < 6) {
        printf("用法: test_lstm.exe <x_u8.bin> <x_f32.bin> <N> <out_fp32.bin> <out_int8.bin>\n");
        return 1;
    }
    n = (uint32_t)atoi(argv[3]);

    fu8 = fopen(argv[1], "rb");
    ff32 = fopen(argv[2], "rb");
    fo_fp32 = fopen(argv[4], "wb");
    fo_int8 = fopen(argv[5], "wb");
    if (!fu8 || !ff32 || !fo_fp32 || !fo_int8) {
        printf("无法打开输入/输出文件\n");
        return 1;
    }

    total_u8 = (size_t)n * SEQ_LEN * in;
    total_f32 = total_u8 * sizeof(float);
    x_q = (uint8_t*)malloc(total_u8);
    x_f = (float*)malloc(total_f32);
    h_q = (uint8_t*)malloc((size_t)n * SEQ_LEN * hid);
    h_seq = (float*)malloc((size_t)n * SEQ_LEN * hid * sizeof(float));
    h_scale = (float*)malloc((size_t)n * SEQ_LEN * sizeof(float));
    h_zp = (int32_t*)malloc((size_t)n * SEQ_LEN * sizeof(int32_t));
    c_state = (float*)malloc(hid * sizeof(float));
    work = (float*)malloc(LSTM_INT8_WORK_SIZE(hid) * sizeof(float));
    logits = (float*)malloc(cls * sizeof(float));
    if (!x_q || !x_f || !h_q || !h_seq || !h_scale || !h_zp || !c_state || !work || !logits) {
        printf("内存不足\n");
        return 1;
    }
    if (fread(x_q, 1, total_u8, fu8) != total_u8 ||
        fread(x_f, sizeof(float), total_u8, ff32) != total_u8) {
        printf("输入长度不足\n");
        return 1;
    }

    {
        static const lstm_fp32_param_t p_fp32 = {LSTM_INPUT_SIZE, LSTM_HIDDEN_SIZE, lstm_w_ih,
                                             lstm_w_hh, lstm_bias};
        static const lstm_out_fp32_t out_fp32 = {LSTM_HIDDEN_SIZE, LSTM_NUM_CLASSES, lstm_out_w,
                                             lstm_out_b};
        static const lstm_int8_param_t p_int8 = {LSTM_INPUT_SIZE, LSTM_HIDDEN_SIZE, lstm_w_ih_q,
                                             lstm_w_ih_scale, lstm_w_ih_rowsum, lstm_w_hh_q,
                                             lstm_w_hh_scale, lstm_w_hh_rowsum, lstm_bias};
        static const lstm_out_int8_t out_int8 = {LSTM_HIDDEN_SIZE, LSTM_NUM_CLASSES, lstm_out_w_q,
                                             lstm_out_w_scale, lstm_out_b};

        t0 = now_ms();
        for (s = 0; s < n; s++) {
            const float* xs = x_f + (size_t)s * SEQ_LEN * in;
            float* hs = h_seq + (size_t)s * SEQ_LEN * hid;
            lstm_forward_fp32(&p_fp32, &out_fp32, xs, SEQ_LEN, c_state, hs, logits);
            fwrite(hs, sizeof(float), SEQ_LEN * hid, fo_fp32);
            fwrite(logits, sizeof(float), cls, fo_fp32);
        }
        t1 = now_ms();
        printf("LSTM fp32: %u samples, %.3f ms total, %.4f ms/sample\n", n, t1 - t0, (t1 - t0) / n);

        t0 = now_ms();
        for (s = 0; s < n; s++) {
            const uint8_t* xs = x_q + (size_t)s * SEQ_LEN * in;
            uint8_t* hs = h_q + (size_t)s * SEQ_LEN * hid;
            float* ss = h_scale + (size_t)s * SEQ_LEN;
            int32_t* zs = h_zp + (size_t)s * SEQ_LEN;
            lstm_forward_int8(&p_int8, &out_int8, xs, SEQ_LEN, 1.0f / 255.0f, 0, work, hs, ss, zs,
                              logits);
            fwrite(hs, 1, SEQ_LEN * hid, fo_int8);
            fwrite(ss, sizeof(float), SEQ_LEN, fo_int8);
            fwrite(zs, sizeof(int32_t), SEQ_LEN, fo_int8);
            fwrite(logits, sizeof(float), cls, fo_int8);
        }
        t1 = now_ms();
        printf("LSTM int8: %u samples, %.3f ms total, %.4f ms/sample\n", n, t1 - t0, (t1 - t0) / n);
    }

    free(x_q);
    free(x_f);
    free(h_q);
    free(h_seq);
    free(h_scale);
    free(h_zp);
    free(c_state);
    free(work);
    free(logits);
    fclose(fu8);
    fclose(ff32);
    fclose(fo_fp32);
    fclose(fo_int8);
    return 0;
}
