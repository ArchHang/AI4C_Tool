/**
 * @file test_fused.c
 * @brief Independent scalar tests for dynamic CHW convolution plus maxpool2 fusion.
 *
 * References quantize both convolution positions before pooling, include discarded odd tails
 * in dynamic scale, and use division/remainder for the Q28 half-even rounding reference.
 */
#include "../include/ai_baseon_c/cnn.h"

#include <float.h>
#include <inttypes.h>
#include <stdio.h>

#define TEST_CAPACITY 4096u
#define TEST_CHANNELS 12u
#define TEST_GUARD UINT8_C(173)
#define TEST_ACC_GUARD INT32_C(73911921)
#define TEST_FIXED_UNIT INT64_C(268435456)

typedef struct test_storage {
    uint8_t input[TEST_CAPACITY];
    int8_t weights[TEST_CAPACITY];
    float scales[TEST_CHANNELS];
    float biases[TEST_CHANNELS];
    int32_t accumulators[TEST_CAPACITY];
    int32_t scratch[TEST_CAPACITY + 2];
    uint8_t reference[TEST_CAPACITY];
    uint8_t unfused[TEST_CAPACITY];
    uint8_t pooled[TEST_CAPACITY];
    uint8_t actual[TEST_CAPACITY + 2];
} test_storage_t;

static test_storage_t storage;
static uint32_t rng_state = UINT32_C(0x361547a1);
static uint32_t checks;
static uint32_t failures;

static uint32_t next_random(void)
{
    rng_state = rng_state * UINT32_C(1664525) + UINT32_C(1013904223);
    return rng_state >> 8;
}

static void check_condition(int32_t condition, const char *name)
{
    checks++;
    if (!condition) {
        failures++;
        if (failures <= 20) {
            printf("FAIL %s\n", name);
        }
    }
}

static uint8_t reference_quantize(int32_t sum, float multiplier, float bias, float scale,
                                 uint32_t kernel_size)
{
    const double coefficient = (double)multiplier / (double)scale * (double)TEST_FIXED_UNIT;
    const double offset = (double)bias / (double)scale * (double)TEST_FIXED_UNIT;
    const double bound = fabs(coefficient) * (double)(kernel_size * 32640u)
        + fabs(offset) + (double)(kernel_size * 32640u) + 1.0;
    if (isfinite(bound) && bound < 9.0e18) {
        const int64_t value = (int64_t)sum * (int64_t)nearbyint(coefficient)
            + (int64_t)nearbyint(offset);
        if (value <= 0) {
            return 0;
        }
        if (value >= INT64_C(255) * TEST_FIXED_UNIT) {
            return 255;
        }
        int64_t quotient = value / TEST_FIXED_UNIT;
        const int64_t remainder = value % TEST_FIXED_UNIT;
        if (remainder > TEST_FIXED_UNIT / 2
            || (remainder == TEST_FIXED_UNIT / 2 && (quotient & INT64_C(1)) != 0)) {
            quotient++;
        }
        return (uint8_t)quotient;
    }
    const float value = (float)sum * multiplier + bias;
    const float normalized = (value > 0.0f ? value : 0.0f) / scale;
    if (normalized <= 0.0f || isnan(normalized)) {
        return 0;
    }
    if (normalized >= 255.0f) {
        return 255;
    }
    return (uint8_t)nearbyint((double)normalized);
}

static float reference_run(const conv2d_data_i8_t *input, const conv2d_config_i8_t *config)
{
    const conv2d_filter_i8_t *filter = config->filter;
    const uint32_t conv_col = input->col - filter->col + 1;
    const uint32_t pool_col = conv_col / 2;
    const uint32_t kernel_size = filter->channel * filter->col;
    float maximum = 0.0f;
    for (uint32_t channel = 0; channel < filter->filter_num; channel++) {
        int32_t channel_maximum = INT32_MIN;
        for (uint32_t col = 0; col < conv_col; col++) {
            int32_t sum = 0;
            for (uint32_t source_channel = 0; source_channel < input->channel; source_channel++) {
                for (uint32_t tap = 0; tap < filter->col; tap++) {
                    const int32_t value = (int32_t)input->data[source_channel * input->col
                        + col + tap] - config->in_act.zp;
                    sum += value * (int32_t)filter->data[channel * kernel_size
                        + source_channel * filter->col + tap];
                }
            }
            storage.accumulators[channel * conv_col + col] = sum;
            if (sum > channel_maximum) {
                channel_maximum = sum;
            }
        }
        const float value = (float)channel_maximum
            * (filter->scale[channel] * config->in_act.scale) + filter->bias[channel];
        if (value > maximum) {
            maximum = value;
        }
    }
    const float scale = fmaxf(maximum / 255.0f, 1e-9f);
    for (uint32_t channel = 0; channel < filter->filter_num; channel++) {
        for (uint32_t col = 0; col < pool_col; col++) {
            const float multiplier = filter->scale[channel] * config->in_act.scale;
            const uint32_t index = channel * conv_col + col * 2;
            const uint8_t left = reference_quantize(storage.accumulators[index], multiplier,
                filter->bias[channel], scale, kernel_size);
            const uint8_t right = reference_quantize(storage.accumulators[index + 1], multiplier,
                filter->bias[channel], scale, kernel_size);
            storage.reference[channel * pool_col + col] = left > right ? left : right;
        }
    }
    return scale;
}

static void check_case(conv2d_data_i8_t *input, conv2d_config_i8_t *config)
{
    quant_param_t *const saved_output_quant = config->out_act;
    const uint32_t conv_col = input->col - config->filter->col + 1;
    const uint32_t pool_col = conv_col / 2;
    const uint32_t output_size = pool_col * config->filter->filter_num;
    const float reference_scale = reference_run(input, config);
    const quant_param_t input_quant = config->in_act;
    quant_param_t output_quant = {0.0f, 0};
    conv2d_data_i8_t output = {0, 0, 0, storage.unfused};
    const max_pool_config_t pool = {1, 2, 2, 0};
    config->out_act = &output_quant;
    check_condition(conv2d_int8_run(input, config, &output) == CNN_NORMAL, "unfused return");
    conv2d_data_i8_t pooled = {0, 0, 0, storage.pooled};
    check_condition(maxpool_u8_run(&output, &pool, &pooled) == CNN_NORMAL, "unfused pool return");
    check_condition(memcmp(storage.pooled, storage.reference, output_size) == 0,
                    "independent reference equals unfused");
    check_condition(memcmp(&output_quant.scale, &reference_scale, sizeof(float)) == 0,
                    "unfused scale reference bits");
    const size_t scratch_size = (size_t)output_size * sizeof(int32_t);
    check_condition(conv2d_int8_maxpool2_get_scratch_size(input, config) == scratch_size,
                    "pooled scratch size");
    for (uint32_t mode = 0; mode < 4; mode++) {
        config->in_act = input_quant;
        config->out_act = mode >= 2 ? &config->in_act : &output_quant;
        output = (conv2d_data_i8_t){0, 0, 0, storage.actual + 1};
        memset(storage.actual, TEST_GUARD, (size_t)output_size + 2);
        storage.scratch[0] = TEST_ACC_GUARD;
        storage.scratch[output_size + 1] = TEST_ACC_GUARD;
        int32_t *scratch = (mode & 1u) != 0 ? storage.scratch + 1 : NULL;
        const int32_t status = conv2d_int8_maxpool2_run_with_scratch(input, config, &output,
            scratch, scratch != NULL ? scratch_size : 0);
        check_condition(status == CNN_NORMAL, "fused return");
        check_condition(memcmp(storage.actual + 1, storage.reference, output_size) == 0,
                        "fused reference bytes");
        check_condition(output.row == 1 && output.col == pool_col
            && output.channel == config->filter->filter_num, "fused geometry");
        check_condition(memcmp(&config->out_act->scale, &reference_scale, sizeof(float)) == 0
            && config->out_act->zp == 0, "fused scale reference bits");
        check_condition(storage.actual[0] == TEST_GUARD
            && storage.actual[output_size + 1] == TEST_GUARD, "fused byte guards");
        check_condition(storage.scratch[0] == TEST_ACC_GUARD
            && storage.scratch[output_size + 1] == TEST_ACC_GUARD, "fused scratch guards");
    }
    config->in_act = input_quant;
    *saved_output_quant = output_quant;
    config->out_act = saved_output_quant;
}

static void test_random_cases(void)
{
    for (uint32_t test = 0; test < 512; test++) {
        const uint32_t kernel = 1 + next_random() % 13;
        const uint32_t conv_col = 2 + next_random() % 254;
        const uint32_t channels = 1 + next_random() % 8;
        const uint32_t outputs = 1 + next_random() % 9;
        const uint32_t input_col = conv_col + kernel - 1;
        const uint32_t kernel_size = channels * kernel;
        for (uint32_t index = 0; index < input_col * channels; index++) {
            storage.input[index] = (uint8_t)next_random();
        }
        for (uint32_t index = 0; index < kernel_size * outputs; index++) {
            storage.weights[index] = (int8_t)((int32_t)(next_random() % 256) - 128);
        }
        for (uint32_t index = 0; index < outputs; index++) {
            storage.scales[index] = (float)(1 + next_random() % 1024) / 65536.0f;
            storage.biases[index] = (float)((int32_t)(next_random() % 8193) - 4096) / 1024.0f;
        }
        conv2d_data_i8_t input = {1, input_col, channels, storage.input};
        const conv2d_filter_i8_t filter = {1, kernel, channels, outputs, storage.weights,
            storage.scales, storage.biases};
        quant_param_t output_quant = {0.0f, 0};
        const int32_t zero_points[] = {0, 1, 127, 128, 254, 255};
        const int32_t zero_point = zero_points[test % 6];
        conv2d_config_i8_t config = {1, 0, &filter,
            {(float)(1 + next_random() % 1024) / 65536.0f, zero_point}, 1, &output_quant};
        check_case(&input, &config);
    }
}

static void test_boundaries(void)
{
    uint8_t input_data[] = {1, 2, 3, 4, 255};
    int8_t weights[] = {1, 1};
    float scales[] = {1.0f, 1.0f};
    float biases[] = {0.0f, 0.0f};
    conv2d_data_i8_t input = {1, 5, 1, input_data};
    conv2d_filter_i8_t filter = {1, 1, 1, 1, weights, scales, biases};
    quant_param_t output_quant = {0.0f, 0};
    conv2d_config_i8_t config = {1, 0, &filter, {1.0f, 0}, 1, &output_quant};
    check_case(&input, &config);
    check_condition(storage.reference[0] == 2 && storage.reference[1] == 4,
                    "odd discarded maximum preserves scale");
    check_condition(output_quant.scale == 1.0f, "odd discarded maximum exact scale");
    input_data[0] = 0;
    input_data[1] = 0;
    input_data[2] = 4;
    input_data[3] = 4;
    scales[0] = 0.5f;
    biases[0] = 127.5f;
    check_case(&input, &config);
    check_condition(storage.reference[0] == 128 && storage.reference[1] == 130
        && output_quant.scale == 1.0f, "pooled exact half-even ties");
    input_data[0] = 0;
    input_data[1] = 1;
    input_data[2] = 127;
    input_data[3] = 254;
    scales[1] = 1e12f;
    scales[0] = 1.0f;
    biases[0] = 1.0f;
    biases[1] = -(255.0f * 1e12f);
    weights[0] = 0;
    filter.filter_num = 2;
    check_case(&input, &config);
    check_condition(storage.reference[0] == 255 && storage.reference[1] == 255
        && storage.reference[2] == 0 && storage.reference[3] == 0,
        "large coefficient and bias float fallback");
    check_condition(output_quant.scale == 1.0f / 255.0f, "float fallback exact scale");
    filter.filter_num = 1;
    scales[0] = 1.0f;
    biases[0] = -300.0f;
    weights[0] = 1;
    check_case(&input, &config);
    check_condition(storage.reference[0] == 0 && storage.reference[1] == 0
        && output_quant.scale == 1e-9f, "all negative scale floor");
    biases[0] = 0.5f;
    check_case(&input, &config);
    biases[0] = 1.0f;
    scales[0] = FLT_MAX / 512.0f;
    check_case(&input, &config);
}

static void test_invalid_cases(void)
{
    uint8_t input_data[] = {0, 1, 2, 3};
    const int8_t weights[] = {1};
    float scale = 1.0f;
    float bias = 0.0f;
    conv2d_data_i8_t input = {1, 4, 1, input_data};
    conv2d_data_i8_t output = {71, 72, 73, storage.actual};
    conv2d_filter_i8_t filter = {1, 1, 1, 1, weights, &scale, &bias};
    quant_param_t output_quant = {0.0f, 0};
    conv2d_config_i8_t config = {1, 0, &filter, {1.0f, 0}, 1, &output_quant};
    check_condition(conv2d_int8_maxpool2_run_with_scratch(NULL, &config, &output, NULL, 0)
        == CNN_POINTER_NULL, "null input");
    check_condition(conv2d_int8_maxpool2_run_with_scratch(&input, NULL, &output, NULL, 0)
        == CNN_POINTER_NULL, "null config");
    check_condition(conv2d_int8_maxpool2_run_with_scratch(&input, &config, NULL, NULL, 0)
        == CNN_POINTER_NULL, "null output");
    check_condition(conv2d_int8_maxpool2_run_with_scratch(&input, &config, &output, NULL, 8)
        == CNN_POINTER_NULL, "null scratch with bytes");
    check_condition(conv2d_int8_maxpool2_run_with_scratch(&input, &config, &output,
        storage.scratch, 7) == CNN_DATA_EXCEPTION, "short scratch");
    check_condition(conv2d_int8_maxpool2_run_with_scratch(&input, &config, &output,
        (int32_t *)(void *)((uint8_t *)storage.scratch + 1), 8) == CNN_DATA_EXCEPTION,
        "unaligned scratch");
    for (uint32_t test = 0; test < 12; test++) {
        const conv2d_data_i8_t saved_input = input;
        const conv2d_filter_i8_t saved_filter = filter;
        const conv2d_config_i8_t saved_config = config;
        const float saved_scale = scale;
        const float saved_bias = bias;
        switch (test) {
            case 0:
                config.stride = 2;
                break;
            case 1:
                config.pad = 1;
                break;
            case 2:
                config.dynamic = 0;
                output_quant.scale = 1.0f;
                break;
            case 3:
                input.row = 2;
                break;
            case 4:
                filter.row = 2;
                input.row = 2;
                break;
            case 5:
                input.col = 1;
                break;
            case 6:
                filter.channel = 2;
                break;
            case 7:
                config.in_act.zp = 256;
                break;
            case 8:
                scale = NAN;
                break;
            case 9:
                bias = INFINITY;
                break;
            case 10:
                input.channel = UINT32_MAX;
                filter.channel = UINT32_MAX;
                break;
            default:
                input.col = UINT32_MAX;
                break;
        }
        check_condition(conv2d_int8_maxpool2_get_scratch_size(&input, &config) == 0,
                        "invalid geometry scratch query");
        check_condition(conv2d_int8_maxpool2_run_with_scratch(&input, &config, &output,
            NULL, 0) == CNN_DATA_EXCEPTION, "invalid geometry return");
        check_condition(output.row == 71 && output.col == 72 && output.channel == 73,
                        "invalid output geometry unchanged");
        input = saved_input;
        filter = saved_filter;
        config = saved_config;
        scale = saved_scale;
        bias = saved_bias;
    }
}

int main(void)
{
    test_random_cases();
    test_boundaries();
    test_invalid_cases();
    printf("fused checks=%" PRIu32 " failures=%" PRIu32 " seed=0x361547a1\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
