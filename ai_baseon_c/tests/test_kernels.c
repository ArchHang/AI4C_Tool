/**
 * @file test_kernels.c
 * @brief Deterministic CNN reference, boundary and timing tests, without model training.
 *
 * NCHW input and output; independent scalar references use channel -> row -> column MACs.
 * Build against either the snapshot or current cnn.c using identical compiler options.
 * Default: functional tests. --bench: median timing CSV. --robust-case NAME: one invalid case.
 * Invalid cases should be isolated by tools/test_robustness.py because old versions may crash.
 */
#include "../include/ai_baseon_c/cnn.h"

#include <float.h>
#include <inttypes.h>
#include <stdio.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#endif

#define TEST_CAPACITY 262144u
#define TEST_CHANNELS 256u
#define TEST_GUARD 173u
#define TEST_FIXED_BITS 28
#define TEST_FIXED_UNIT (INT64_C(1) << TEST_FIXED_BITS)

typedef struct test_shape {
    uint32_t row;
    uint32_t col;
    uint32_t channels;
    uint32_t kernel_row;
    uint32_t kernel_col;
    uint32_t stride;
    uint32_t pad;
    uint32_t outputs;
} test_shape_t;

typedef struct test_buffers {
    float input[TEST_CAPACITY];
    float weights[TEST_CAPACITY];
    float output[TEST_CAPACITY + 2];
    float reference[TEST_CAPACITY];
    uint8_t input_u8[TEST_CAPACITY];
    int8_t weights_i8[TEST_CAPACITY];
    uint8_t output_u8[TEST_CAPACITY + 2];
    uint8_t reference_u8[TEST_CAPACITY];
    uint8_t float_reference_u8[TEST_CAPACITY];
    int32_t accumulators[TEST_CAPACITY];
    float mean[TEST_CHANNELS];
    float variance[TEST_CHANNELS];
    float gamma[TEST_CHANNELS];
    float bias[TEST_CHANNELS];
    float scale[TEST_CHANNELS];
} test_buffers_t;

static test_buffers_t test_buffers;
static uint32_t test_rng_state = UINT32_C(0x761c02ab);
static uint32_t test_checks;
static uint32_t test_failures;
static uint32_t test_dynamic_float_differences;
static volatile double test_benchmark_checksum;

static uint32_t next_random(void)
{
    test_rng_state = test_rng_state * UINT32_C(1664525) + UINT32_C(1013904223);
    return test_rng_state >> 8;
}

static float next_float(void)
{
    return (float)((int32_t)(next_random() % 2049u) - 1024) / 1024.0f;
}

static void check_condition(int condition, const char *name)
{
    test_checks++;
    if (!condition) {
        test_failures++;
        if (test_failures <= 24u) {
            printf("FAIL %s\n", name);
        }
    }
}

static void check_float_array(const float *actual, const float *expected, uint32_t size,
                              const char *name)
{
    uint32_t index;
    for (index = 0; index < size; index++) {
        const float error = fabsf(actual[index] - expected[index]);
        const float tolerance = 1e-6f + 2e-6f * fabsf(expected[index]);
        if (!isfinite(actual[index]) || error > tolerance) {
            printf("DIFF %s index=%" PRIu32 " actual=%.9g expected=%.9g\n",
                   name, index, (double)actual[index], (double)expected[index]);
            break;
        }
    }
    check_condition(index == size, name);
}

static void check_u8_array(const uint8_t *actual, const uint8_t *expected, uint32_t size,
                          const char *name)
{
    uint32_t index;
    for (index = 0; index < size; index++) {
        if (actual[index] != expected[index]) {
            printf("DIFF %s index=%" PRIu32 " actual=%u expected=%u\n",
                   name, index, (unsigned)actual[index], (unsigned)expected[index]);
            break;
        }
    }
    check_condition(index == size, name);
}

static uint32_t output_rows(const test_shape_t *shape)
{
    if (shape->row == 1u && shape->kernel_row == 1u) {
        return 1u;
    }
    return (shape->row + 2u * shape->pad - shape->kernel_row) / shape->stride + 1u;
}

static uint32_t output_columns(const test_shape_t *shape)
{
    return (shape->col + 2u * shape->pad - shape->kernel_col) / shape->stride + 1u;
}

static uint8_t quantize_reference(float value, float scale, int zero_point)
{
    const float normalized = value / scale;
    double rounded;
    if (normalized >= (float)(255 - zero_point)) {
        return 255u;
    }
    if (normalized <= (float)(-zero_point)) {
        return 0u;
    }
    rounded = nearbyint((double)normalized) + (double)zero_point;
    return (uint8_t)rounded;
}

/* Division and remainder provide a reference independent of the library's bit-shift helper. */
static uint8_t requantize_fixed_reference(int32_t accumulator, float multiplier, float bias,
                                        float scale)
{
    const int64_t coefficient = (int64_t)nearbyint((double)multiplier / (double)scale *
                                                  (double)TEST_FIXED_UNIT);
    const int64_t offset = (int64_t)nearbyint((double)bias / (double)scale *
                                             (double)TEST_FIXED_UNIT);
    const int64_t value = (int64_t)accumulator * coefficient + offset;
    int64_t quotient;
    int64_t remainder;
    if (value <= 0) {
        return 0u;
    }
    if (value >= INT64_C(255) * TEST_FIXED_UNIT) {
        return 255u;
    }
    quotient = value / TEST_FIXED_UNIT;
    remainder = value % TEST_FIXED_UNIT;
    if (remainder > TEST_FIXED_UNIT / 2
        || (remainder == TEST_FIXED_UNIT / 2 && (quotient % 2) != 0)) {
        quotient++;
    }
    return (uint8_t)quotient;
}

static int input_index_for_tap(const test_shape_t *shape, uint32_t output_index,
                               uint32_t tap, uint32_t *input_index)
{
    const uint32_t out_col = output_columns(shape);
    const uint32_t kernel_size = shape->kernel_row * shape->kernel_col;
    const uint32_t channel = tap / kernel_size;
    const uint32_t kernel_row = (tap % kernel_size) / shape->kernel_col;
    const uint32_t kernel_col = tap % shape->kernel_col;
    const int64_t row = shape->row == 1u && shape->kernel_row == 1u ? 0 :
        (int64_t)(output_index / out_col) * shape->stride - shape->pad + kernel_row;
    const int64_t col = (int64_t)(output_index % out_col) * shape->stride - shape->pad +
                        kernel_col;
    if (row < 0 || col < 0 || row >= shape->row || col >= shape->col) {
        return 0;
    }
    *input_index = (channel * shape->row + (uint32_t)row) * shape->col + (uint32_t)col;
    return 1;
}

static void reference_conv_fp32(const test_shape_t *shape)
{
    const uint32_t out_plane = output_rows(shape) * output_columns(shape);
    const uint32_t taps = shape->channels * shape->kernel_row * shape->kernel_col;
    for (uint32_t channel = 0; channel < shape->outputs; channel++) {
        const float scale = test_buffers.gamma[channel] /
                            sqrtf(test_buffers.variance[channel] + 1e-5f);
        const float shift = test_buffers.bias[channel] - test_buffers.mean[channel] * scale;
        for (uint32_t index = 0; index < out_plane; index++) {
            float accumulator = 0.0f;
            for (uint32_t tap = 0; tap < taps; tap++) {
                uint32_t input_index;
                if (input_index_for_tap(shape, index, tap, &input_index)) {
                    accumulator += test_buffers.input[input_index] *
                                   test_buffers.weights[channel * taps + tap];
                }
            }
            test_buffers.reference[channel * out_plane + index] = accumulator * scale + shift;
        }
    }
}

static void reference_conv_accumulators(const test_shape_t *shape, int zero_point)
{
    const uint32_t out_plane = output_rows(shape) * output_columns(shape);
    const uint32_t taps = shape->channels * shape->kernel_row * shape->kernel_col;
    for (uint32_t channel = 0; channel < shape->outputs; channel++) {
        for (uint32_t index = 0; index < out_plane; index++) {
            int64_t accumulator = 0;
            for (uint32_t tap = 0; tap < taps; tap++) {
                uint32_t input_index;
                if (input_index_for_tap(shape, index, tap, &input_index)) {
                    accumulator += (int64_t)test_buffers.weights_i8[channel * taps + tap] *
                                   ((int64_t)test_buffers.input_u8[input_index] - zero_point);
                }
            }
            test_buffers.accumulators[channel * out_plane + index] = (int32_t)accumulator;
        }
    }
}

static float reference_conv_u8(const test_shape_t *shape, int zero_point, int dynamic,
                               const quant_param_t *out_quant)
{
    const uint32_t out_plane = output_rows(shape) * output_columns(shape);
    const uint32_t size = shape->outputs * out_plane;
    float maximum = 0.0f;
    float scale = out_quant->scale;
    reference_conv_accumulators(shape, zero_point);
    for (uint32_t index = 0; index < size; index++) {
        const uint32_t channel = index / out_plane;
        const float multiplier = test_buffers.scale[channel] * 0.01f;
        float value = (float)test_buffers.accumulators[index] * multiplier +
                      test_buffers.bias[channel];
        if (value < 0.0f) {
            value = 0.0f;
        }
        test_buffers.reference[index] = value;
        if (value > maximum) {
            maximum = value;
        }
    }
    if (dynamic) {
        scale = fmaxf(maximum / 255.0f, 1e-9f);
    }
    for (uint32_t index = 0; index < size; index++) {
        const uint32_t channel = index / out_plane;
        const float multiplier = test_buffers.scale[channel] * 0.01f;
        test_buffers.float_reference_u8[index] = quantize_reference(test_buffers.reference[index],
                                                                    scale, dynamic ? 0 :
                                                                    out_quant->zp);
        test_buffers.reference_u8[index] = dynamic ?
            requantize_fixed_reference(test_buffers.accumulators[index], multiplier,
                                       test_buffers.bias[channel], scale) :
            test_buffers.float_reference_u8[index];
    }
    return scale;
}

static void fill_shape(const test_shape_t *shape)
{
    const uint32_t input_size = shape->row * shape->col * shape->channels;
    const uint32_t weight_size = shape->kernel_row * shape->kernel_col * shape->channels *
                                 shape->outputs;
    if (input_size > TEST_CAPACITY || weight_size > TEST_CAPACITY
        || shape->outputs > TEST_CHANNELS) {
        fprintf(stderr, "Test fixture exceeds capacity\n");
        exit(2);
    }
    for (uint32_t index = 0; index < input_size; index++) {
        test_buffers.input[index] = next_float();
        test_buffers.input_u8[index] = (uint8_t)(next_random() % 256u);
    }
    for (uint32_t index = 0; index < weight_size; index++) {
        test_buffers.weights[index] = next_float();
        test_buffers.weights_i8[index] = (int8_t)((int32_t)(next_random() % 256u) - 128);
    }
    for (uint32_t index = 0; index < shape->outputs; index++) {
        test_buffers.mean[index] = next_float();
        test_buffers.variance[index] = fabsf(next_float()) + 0.1f;
        test_buffers.gamma[index] = next_float();
        test_buffers.bias[index] = next_float();
        test_buffers.scale[index] = (float)(1u + next_random() % 31u) * 0.001f;
    }
}

static void test_conv_shape(const test_shape_t *shape, int zero_point)
{
    const uint32_t size = output_rows(shape) * output_columns(shape) * shape->outputs;
    conv2d_data_t input = {shape->row, shape->col, shape->channels, test_buffers.input};
    conv2d_data_t output = {0, 0, 0, test_buffers.output + 1};
    conv2d_filter_t filter = {shape->kernel_row, shape->kernel_col, shape->channels,
                             shape->outputs, test_buffers.weights};
    batch_norm2d_t bn = {shape->outputs, test_buffers.mean, test_buffers.variance,
                         test_buffers.gamma, test_buffers.bias};
    conv2d_config_t config = {shape->stride, shape->pad, &filter, &bn};
    conv2d_data_i8_t input_u8 = {shape->row, shape->col, shape->channels, test_buffers.input_u8};
    conv2d_data_i8_t output_u8 = {0, 0, 0, test_buffers.output_u8 + 1};
    conv2d_filter_i8_t filter_i8 = {shape->kernel_row, shape->kernel_col, shape->channels,
                                   shape->outputs, test_buffers.weights_i8, test_buffers.scale,
                                   test_buffers.bias};
    quant_param_t out_quant = {0.03f, zero_point};
    conv2d_config_i8_t config_i8 = {shape->stride, shape->pad, &filter_i8, {0.01f, zero_point},
                                   0, &out_quant};
    fill_shape(shape);
    reference_conv_fp32(shape);
    test_buffers.output[0] = 9876.0f;
    test_buffers.output[size + 1u] = 9876.0f;
    check_condition(conv2d_bn_no_bias(&input, &config, &output) == CNN_NORMAL, "conv fp32 return");
    check_float_array(output.data, test_buffers.reference, size, "conv fp32 reference");
    check_condition(output.row == output_rows(shape) && output.col == output_columns(shape)
                    && output.channel == shape->outputs, "conv fp32 shape");
    check_condition(test_buffers.output[0] == 9876.0f &&
                    test_buffers.output[size + 1u] == 9876.0f, "conv fp32 guards");
    for (int dynamic = 0; dynamic <= 1; dynamic++) {
        const float reference_scale = reference_conv_u8(shape, zero_point, dynamic, &out_quant);
        config_i8.dynamic = dynamic;
        test_buffers.output_u8[0] = TEST_GUARD;
        test_buffers.output_u8[size + 1u] = TEST_GUARD;
        check_condition(conv2d_int8_run(&input_u8, &config_i8, &output_u8) == CNN_NORMAL,
                        "conv int8 return");
        check_u8_array(output_u8.data, test_buffers.reference_u8, size,
                       dynamic ? "conv dynamic fixed reference" : "conv static reference");
        check_condition(output_u8.row == output_rows(shape) && output_u8.col ==
                        output_columns(shape) && output_u8.channel == shape->outputs,
                        "conv int8 shape");
        check_condition(test_buffers.output_u8[0] == TEST_GUARD &&
                        test_buffers.output_u8[size + 1u] == TEST_GUARD, "conv int8 guards");
        check_condition(out_quant.scale == reference_scale && out_quant.zp ==
                        (dynamic ? 0 : zero_point), "conv output quantization");
#ifdef TEST_HAS_SCRATCH
        {
            const size_t scratch_size = conv2d_int8_get_scratch_size(&input_u8, &config_i8);
            const float saved_scale = out_quant.scale;
            memset(output_u8.data, TEST_GUARD, size);
            check_condition(scratch_size == (dynamic ? (size_t)size * sizeof(int32_t) : 0u),
                            "conv scratch size");
            check_condition(conv2d_int8_run_with_scratch(&input_u8, &config_i8, &output_u8,
                                                        dynamic ? test_buffers.accumulators : NULL,
                                                        scratch_size)
                            == CNN_NORMAL, "conv scratch return");
            check_u8_array(output_u8.data, test_buffers.reference_u8, size,
                           "conv scratch independent reference");
            check_condition(saved_scale == out_quant.scale, "conv scratch scale");
        }
#endif
        for (uint32_t index = 0; index < size && dynamic; index++) {
            const int difference = abs((int)output_u8.data[index] -
                                       (int)test_buffers.float_reference_u8[index]);
            test_dynamic_float_differences += difference != 0;
            if (difference > 1) {
                check_condition(0, "dynamic float reference exceeds one code");
                break;
            }
        }
    }
}

static void test_dynamic_ties(void)
{
    uint8_t input_data[] = {1, 5, 9, 255};
    int8_t weights[] = {0, 1};
    float scales[] = {1.0f, 1.0f};
    float bias[] = {255.0f, -0.5f};
    uint8_t output_data[8];
    const uint8_t expected[] = {255, 255, 255, 255, 0, 4, 8, 254};
    conv2d_data_i8_t input = {1, 4, 1, input_data};
    conv2d_data_i8_t output = {0, 0, 0, output_data};
    conv2d_filter_i8_t filter = {1, 1, 1, 2, weights, scales, bias};
    quant_param_t out_quant = {0.0f, 0};
    conv2d_config_i8_t config = {1, 0, &filter, {1.0f, 0}, 1, &out_quant};
    check_condition(conv2d_int8_run(&input, &config, &output) == CNN_NORMAL,
                    "dynamic half-even return");
    check_condition(out_quant.scale == 1.0f, "dynamic half-even scale");
    check_u8_array(output_data, expected, 8, "dynamic exact ties to even");
    weights[1] = 0;
    bias[0] = -1.0f;
    bias[1] = -5.0f;
    memset(output_data, TEST_GUARD, sizeof(output_data));
    check_condition(conv2d_int8_run(&input, &config, &output) == CNN_NORMAL,
                    "dynamic all-negative return");
    check_condition(out_quant.scale == 1e-9f && out_quant.zp == 0,
                    "dynamic zero-range scale floor");
    for (uint32_t index = 0; index < 8u; index++) {
        check_condition(output_data[index] == 0u, "dynamic all-negative output");
    }
}

static void test_quantization(void)
{
    float input[] = {-10000.0f, -128.5f, -2.5f, -1.5f, -0.5f, 0.0f, 0.5f,
                     1.5f, 2.5f, 127.5f, 254.5f, 255.5f, 10000.0f};
    const uint32_t size = (uint32_t)(sizeof(input) / sizeof(input[0]));
    const int zero_points[] = {0, 128, 255};
    for (uint32_t index = 0; index < 3u; index++) {
        quant_param_t quant = {1.0f, zero_points[index]};
        for (uint32_t point = 0; point < size; point++) {
            test_buffers.reference_u8[point] = quantize_reference(input[point], quant.scale,
                                                                 quant.zp);
        }
        quantize_u8_run(input, size, &quant, test_buffers.output_u8 + 1);
        check_u8_array(test_buffers.output_u8 + 1, test_buffers.reference_u8, size,
                       "quantize half-even/saturation");
    }
}

static void test_quantization_alias(void)
{
    uint8_t data[] = {0, 1, 17, 127, 128, 191, 254, 255};
    int8_t weights[] = {-128, 127, 64};
    float weight_scale = 0.02f;
    float bias = 0.37f;
    uint8_t reference[8];
    uint8_t actual[8];
    conv2d_data_i8_t input = {1, 8, 1, data};
    conv2d_data_i8_t output = {0, 0, 0, reference};
    conv2d_filter_i8_t filter = {1, 3, 1, 1, weights, &weight_scale, &bias};
    quant_param_t out_quant = {0.0f, 0};
    conv2d_config_i8_t config = {1, 1, &filter, {0.01f, 128}, 1, &out_quant};
    const int zero_points[] = {0, 128, 255};
    for (uint32_t zero = 0; zero < 3u; zero++) {
        config.in_act = (quant_param_t){0.01f, zero_points[zero]};
        config.out_act = &out_quant;
        output.data = reference;
        check_condition(conv2d_int8_run(&input, &config, &output) == CNN_NORMAL,
                        "dynamic non-alias reference return");
        config.out_act = &config.in_act;
        output.data = actual;
        check_condition(conv2d_int8_run(&input, &config, &output) == CNN_NORMAL,
                        "dynamic quantization alias return");
        check_u8_array(actual, reference, 8, "dynamic quantization alias bytes");
        check_condition(config.in_act.scale == out_quant.scale && config.in_act.zp == 0,
                        "dynamic quantization alias scale");
#ifdef TEST_HAS_SCRATCH
        {
            int32_t scratch[8];
            config.in_act = (quant_param_t){0.01f, zero_points[zero]};
            check_condition(conv2d_int8_run_with_scratch(&input, &config, &output, scratch,
                                                        sizeof(scratch)) == CNN_NORMAL,
                            "dynamic scratch alias return");
            check_u8_array(actual, reference, 8, "dynamic scratch alias bytes");
            check_condition(config.in_act.scale == out_quant.scale && config.in_act.zp == 0,
                            "dynamic scratch alias scale");
        }
#endif
    }
}

static void test_dynamic_coefficient_fallback(void)
{
    uint8_t data[] = {0, 1, 127, 254, 255};
    int8_t weights[] = {0, 1};
    float weight_scale[] = {1.0f, 1e12f};
    float bias[] = {1.0f, -(255.0f * 1e12f)};
    uint8_t actual[10];
    const uint8_t reference[] = {255, 255, 255, 255, 255, 0, 0, 0, 0, 0};
    conv2d_data_i8_t input = {1, 5, 1, data};
    conv2d_data_i8_t output = {0, 0, 0, actual};
    conv2d_filter_i8_t filter = {1, 1, 1, 2, weights, weight_scale, bias};
    quant_param_t out_quant = {0.0f, 0};
    conv2d_config_i8_t config = {1, 0, &filter, {1.0f, 0}, 1, &out_quant};
    // Channel 1 cancels at its maximum; its fixed bias and products exceed int64 range.
    check_condition(conv2d_int8_run(&input, &config, &output) == CNN_NORMAL,
                    "dynamic coefficient fallback return");
    check_u8_array(actual, reference, 10, "dynamic coefficient fallback output");
    check_condition(out_quant.scale == 1.0f / 255.0f, "dynamic coefficient fallback scale");
#ifdef TEST_HAS_SCRATCH
    {
        int32_t scratch[10];
        check_condition(conv2d_int8_run_with_scratch(&input, &config, &output, scratch,
                                                    sizeof(scratch)) == CNN_NORMAL,
                        "dynamic coefficient fallback scratch return");
        check_u8_array(actual, reference, 10, "dynamic coefficient fallback scratch output");
        check_condition(out_quant.scale == 1.0f / 255.0f,
                        "dynamic coefficient fallback scratch scale");
    }
#endif
}

static void test_linear_shape(uint32_t inputs, uint32_t outputs, int zero_point)
{
    linear_param_t config = {inputs, outputs, test_buffers.weights, test_buffers.bias};
    linear_param_i8_t config_i8 = {inputs, outputs, test_buffers.weights_i8,
                                   test_buffers.scale, test_buffers.bias};
    quant_param_t quant = {0.01f, zero_point};
    const test_shape_t shape = {1, inputs, 1, 1, inputs, 1, 0, outputs};
    fill_shape(&shape);
    for (uint32_t output = 0; output < outputs; output++) {
        float accumulator = test_buffers.bias[output];
        for (uint32_t input = 0; input < inputs; input++) {
            accumulator += test_buffers.input[input] *
                           test_buffers.weights[output * inputs + input];
        }
        test_buffers.reference[output] = accumulator;
    }
    test_buffers.output[0] = 9876.0f;
    test_buffers.output[outputs + 1u] = 9876.0f;
    check_condition(linear_layer(test_buffers.input, &config, test_buffers.output + 1) == CNN_NORMAL,
                    "linear fp32 return");
    check_float_array(test_buffers.output + 1, test_buffers.reference, outputs,
                       "linear fp32 reference");
    for (uint32_t output = 0; output < outputs; output++) {
        int64_t accumulator = 0;
        for (uint32_t input = 0; input < inputs; input++) {
            accumulator += (int64_t)test_buffers.weights_i8[output * inputs + input] *
                           ((int64_t)test_buffers.input_u8[input] - zero_point);
        }
        test_buffers.reference[output] = (float)accumulator *
                                         (test_buffers.scale[output] * quant.scale) +
                                         test_buffers.bias[output];
    }
    check_condition(linear_int8_run(test_buffers.input_u8, &config_i8, &quant,
                                    test_buffers.output + 1) == CNN_NORMAL, "linear int8 return");
    check_float_array(test_buffers.output + 1, test_buffers.reference, outputs,
                       "linear int8 reference");
    check_condition(test_buffers.output[0] == 9876.0f &&
                    test_buffers.output[outputs + 1u] == 9876.0f, "linear guards");
}

static void test_pool_shape(const test_shape_t *shape)
{
    const uint32_t out_plane = output_rows(shape) * output_columns(shape);
    const uint32_t input_size = shape->row * shape->col * shape->channels;
    const uint32_t output_size = out_plane * shape->channels;
    conv2d_data_t input = {shape->row, shape->col, shape->channels, test_buffers.input};
    conv2d_data_t output = {0, 0, 0, test_buffers.output + 1};
    conv2d_data_i8_t input_u8 = {shape->row, shape->col, shape->channels, test_buffers.input_u8};
    conv2d_data_i8_t output_u8 = {0, 0, 0, test_buffers.output_u8 + 1};
    max_pool_config_t config = {shape->kernel_row, shape->kernel_col, shape->stride, 0};
    fill_shape(shape);
    for (uint32_t index = 0; index < output_size; index++) {
        float maximum = -FLT_MAX;
        uint8_t maximum_u8 = 0;
        const uint32_t channel = index / out_plane;
        for (uint32_t tap = 0; tap < shape->kernel_row * shape->kernel_col; tap++) {
            uint32_t input_index;
            input_index = 0;
            (void)input_index_for_tap(shape, index % out_plane, tap, &input_index);
            input_index += channel * shape->row * shape->col;
            maximum = fmaxf(maximum, test_buffers.input[input_index]);
            if (test_buffers.input_u8[input_index] > maximum_u8) {
                maximum_u8 = test_buffers.input_u8[input_index];
            }
        }
        test_buffers.reference[index] = maximum;
        test_buffers.reference_u8[index] = maximum_u8;
    }
    check_condition(maxpool(&input, &config, &output) == CNN_NORMAL, "pool fp32 return");
    check_float_array(output.data, test_buffers.reference, output_size, "pool fp32 reference");
    check_condition(maxpool_u8_run(&input_u8, &config, &output_u8) == CNN_NORMAL, "pool u8 return");
    check_u8_array(output_u8.data, test_buffers.reference_u8, output_size, "pool u8 reference");
    memcpy(test_buffers.output + 1, test_buffers.input, (size_t)input_size * sizeof(float));
    memcpy(test_buffers.output_u8 + 1, test_buffers.input_u8, input_size);
    output = (conv2d_data_t){shape->row, shape->col, shape->channels, test_buffers.output + 1};
    output_u8 = (conv2d_data_i8_t){shape->row, shape->col, shape->channels,
                                  test_buffers.output_u8 + 1};
    check_condition(maxpool(&output, &config, &output) == CNN_NORMAL, "pool fp32 in-place return");
    check_float_array(output.data, test_buffers.reference, output_size, "pool fp32 in-place");
    check_condition(maxpool_u8_run(&output_u8, &config, &output_u8) == CNN_NORMAL,
                    "pool u8 in-place return");
    check_u8_array(output_u8.data, test_buffers.reference_u8, output_size, "pool u8 in-place");
}

static void test_maximum_dot(void)
{
    const uint32_t width = 32896u;
    const uint32_t outputs = 4u;
    linear_param_i8_t linear = {width, outputs, test_buffers.weights_i8,
                                test_buffers.scale, test_buffers.bias};
    conv2d_data_i8_t input = {1, width, 1, test_buffers.input_u8};
    conv2d_data_i8_t output = {0, 0, 0, test_buffers.output_u8};
    conv2d_filter_i8_t filter = {1, width, 1, outputs, test_buffers.weights_i8,
                                test_buffers.scale, test_buffers.bias};
    quant_param_t quant = {0.0001f, 0};
    quant_param_t out_quant = {0.1f, 0};
    conv2d_config_i8_t config = {1, 0, &filter, quant, 0, &out_quant};
    for (uint32_t row = 0; row < outputs; row++) {
        memset(test_buffers.weights_i8 + (size_t)row * width, row & 1u ? 127 : -128, width);
        test_buffers.scale[row] = 0.0001f;
        test_buffers.bias[row] = 0.25f;
    }
    for (uint32_t zero_point = 0; zero_point <= 255; zero_point += 255) {
        quant.zp = (int)zero_point;
        config.in_act = quant;
        for (uint32_t value = 0; value <= 255; value += 255) {
            memset(test_buffers.input_u8, (int)value, width);
            for (uint32_t row = 0; row < outputs; row++) {
                const int64_t weight = row & 1u ? 127 : -128;
                const int64_t sum = (int64_t)width * weight
                    * ((int64_t)value - (int64_t)zero_point);
                const float result = (float)sum * (quant.scale * test_buffers.scale[row])
                    + test_buffers.bias[row];
                test_buffers.reference[row] = result;
                test_buffers.reference_u8[row] = quantize_reference(
                    result > 0.0f ? result : 0.0f, out_quant.scale, 0);
            }
            check_condition(linear_int8_run(input.data, &linear, &quant, test_buffers.output)
                            == CNN_NORMAL, "maximum legal dot return");
            check_float_array(test_buffers.output, test_buffers.reference, outputs,
                               "maximum legal dot int64 oracle");
            check_condition(conv2d_int8_run(&input, &config, &output) == CNN_NORMAL,
                            "maximum legal convolution return");
            check_u8_array(output.data, test_buffers.reference_u8, outputs,
                            "maximum legal convolution int64 oracle");
        }
    }
}

static int run_functional_tests(void)
{
    const test_shape_t shapes[] = {
        {1, 33, 3, 1, 3, 1, 0, 5}, {1, 35, 3, 1, 5, 2, 2, 7},
        {1, 19, 5, 1, 1, 3, 1, 3}, {1, 1, 1, 1, 1, 1, 0, 1},
        {7, 11, 3, 3, 3, 1, 1, 5}, {9, 13, 5, 2, 3, 2, 0, 7},
        {5, 7, 2, 3, 2, 3, 2, 3}, {1, 9, 3, 3, 3, 1, 1, 3},
        {3, 1, 3, 1, 1, 1, 0, 5}, {2, 3, 1, 5, 5, 2, 3, 1},
        {1, 9, 3, 3, 3, 1, 2, 3}
    };
    const test_shape_t pools[] = {
        {1, 35, 3, 1, 3, 1, 0, 3}, {1, 35, 5, 1, 3, 3, 0, 5},
        {9, 13, 3, 3, 2, 2, 0, 3}, {7, 11, 5, 2, 3, 3, 0, 5}
    };
    const uint32_t linear_inputs[] = {1, 3, 7, 15, 16, 17, 31, 32, 33, 129, 4032};
    const int zero_points[] = {0, 128, 255};
    test_quantization();
    test_dynamic_ties();
    test_quantization_alias();
    test_dynamic_coefficient_fallback();
    test_maximum_dot();
    for (uint32_t zero = 0; zero < 3u; zero++) {
        for (uint32_t index = 0; index < sizeof(shapes) / sizeof(shapes[0]); index++) {
            test_conv_shape(&shapes[index], zero_points[zero]);
        }
        for (uint32_t index = 0; index < sizeof(linear_inputs) / sizeof(linear_inputs[0]); index++) {
            test_linear_shape(linear_inputs[index], 1u + 2u * (index % 4u), zero_points[zero]);
        }
    }
    for (uint32_t index = 0; index < sizeof(pools) / sizeof(pools[0]); index++) {
        test_pool_shape(&pools[index]);
    }
    for (uint32_t repeat = 0; repeat < 64u; repeat++) {
        test_shape_t shape = {1, 3u + next_random() % 40u, 1u + next_random() % 5u,
                              1, 1u + next_random() % 3u, 1u + next_random() % 3u,
                              next_random() % 3u, 1u + next_random() % 7u};
        test_conv_shape(&shape, zero_points[repeat % 3u]);
    }
    printf("RESULT checks=%" PRIu32 " failures=%" PRIu32
           " dynamic_float_one_code_differences=%" PRIu32 " seed=0x761c02ab\n",
           test_checks, test_failures, test_dynamic_float_differences);
    return test_failures == 0u ? 0 : 1;
}

static double monotonic_seconds(void)
{
#ifdef _WIN32
    LARGE_INTEGER frequency;
    LARGE_INTEGER counter;
    (void)QueryPerformanceFrequency(&frequency);
    (void)QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)frequency.QuadPart;
#else
    return (double)clock() / (double)CLOCKS_PER_SEC;
#endif
}

static int compare_double(const void *left, const void *right)
{
    const double a = *(const double *)left;
    const double b = *(const double *)right;
    return (a > b) - (a < b);
}

typedef struct benchmark_context {
    int operation;
    conv2d_data_t input;
    conv2d_data_t output;
    conv2d_filter_t filter;
    batch_norm2d_t bn;
    conv2d_config_t conv;
    conv2d_data_i8_t input_u8;
    conv2d_data_i8_t output_u8;
    conv2d_filter_i8_t filter_i8;
    quant_param_t out_quant;
    conv2d_config_i8_t conv_i8;
    linear_param_t linear;
    linear_param_i8_t linear_i8;
    max_pool_config_t pool;
} benchmark_context_t;

static int benchmark_call(benchmark_context_t *context)
{
    switch (context->operation) {
        case 0:
            return conv2d_bn_no_bias(&context->input, &context->conv, &context->output);
        case 1:
        case 2:
            return conv2d_int8_run(&context->input_u8, &context->conv_i8, &context->output_u8);
        case 3:
            return linear_layer(context->input.data, &context->linear, context->output.data);
        case 4:
            return linear_int8_run(context->input_u8.data, &context->linear_i8,
                                    &context->conv_i8.in_act, context->output.data);
        case 5:
            return maxpool(&context->input, &context->pool, &context->output);
        case 6:
            return maxpool_u8_run(&context->input_u8, &context->pool, &context->output_u8);
#ifdef TEST_HAS_SCRATCH
        case 7:
            return conv2d_int8_run_with_scratch(&context->input_u8, &context->conv_i8,
                                               &context->output_u8, test_buffers.accumulators,
                                               sizeof(test_buffers.accumulators));
#endif
        default:
            return CNN_DATA_EXCEPTION;
    }
}

static void benchmark_one(const char *name, const test_shape_t *shape, int operation, int zero_point)
{
    benchmark_context_t context;
    double samples[9];
    uint32_t iterations = 1;
    double elapsed;
    fill_shape(shape);
    context.operation = operation;
    context.input = (conv2d_data_t){shape->row, shape->col, shape->channels, test_buffers.input};
    context.output = (conv2d_data_t){0, 0, 0, test_buffers.output + 1};
    context.filter = (conv2d_filter_t){shape->kernel_row, shape->kernel_col, shape->channels,
                                      shape->outputs, test_buffers.weights};
    context.bn = (batch_norm2d_t){shape->outputs, test_buffers.mean, test_buffers.variance,
                                 test_buffers.gamma, test_buffers.bias};
    context.conv = (conv2d_config_t){shape->stride, shape->pad, &context.filter, &context.bn};
    context.input_u8 = (conv2d_data_i8_t){shape->row, shape->col, shape->channels,
                                         test_buffers.input_u8};
    context.output_u8 = (conv2d_data_i8_t){0, 0, 0, test_buffers.output_u8 + 1};
    context.filter_i8 = (conv2d_filter_i8_t){shape->kernel_row, shape->kernel_col, shape->channels,
                                           shape->outputs, test_buffers.weights_i8,
                                           test_buffers.scale, test_buffers.bias};
    context.out_quant = (quant_param_t){0.03f, 0};
    context.conv_i8 = (conv2d_config_i8_t){shape->stride, shape->pad, &context.filter_i8,
                                          {0.01f, zero_point}, operation == 2 || operation == 7,
                                          &context.out_quant};
    context.linear = (linear_param_t){shape->col, shape->outputs, test_buffers.weights,
                                      test_buffers.bias};
    context.linear_i8 = (linear_param_i8_t){shape->col, shape->outputs, test_buffers.weights_i8,
                                           test_buffers.scale, test_buffers.bias};
    context.pool = (max_pool_config_t){shape->kernel_row, shape->kernel_col, shape->stride, 0};
    if (benchmark_call(&context) != CNN_NORMAL) {
        fprintf(stderr, "Benchmark case failed: %s\n", name);
        exit(2);
    }
    do {
        const double start = monotonic_seconds();
        for (uint32_t iteration = 0; iteration < iterations; iteration++) {
            (void)benchmark_call(&context);
        }
        elapsed = monotonic_seconds() - start;
        if (elapsed < 0.008) {
            iterations *= 2u;
        }
    } while (elapsed < 0.008);
    for (uint32_t sample = 0; sample < 9u; sample++) {
        const double start = monotonic_seconds();
        for (uint32_t iteration = 0; iteration < iterations; iteration++) {
            (void)benchmark_call(&context);
        }
        samples[sample] = (monotonic_seconds() - start) * 1e6 / (double)iterations;
    }
    qsort(samples, 9u, sizeof(samples[0]), compare_double);
    test_benchmark_checksum += operation == 1 || operation == 2 || operation == 6 || operation == 7 ?
        (double)context.output_u8.data[0] : (double)context.output.data[0];
    printf("%s,%.6f,%.6f,%.6f,%" PRIu32 ",9\n", name, samples[4], samples[0], samples[8],
           iterations);
}

static int run_benchmarks(void)
{
    const test_shape_t conv1 = {1, 2048, 1, 1, 3, 1, 0, 16};
    const test_shape_t conv2 = {1, 1023, 16, 1, 3, 1, 0, 32};
    const test_shape_t conv_2d = {32, 32, 8, 3, 3, 1, 1, 16};
    const test_shape_t linear = {1, 4032, 1, 1, 4032, 1, 0, 32};
    const test_shape_t pool = {32, 32, 16, 2, 2, 2, 0, 16};
    puts("case,median_us,min_us,max_us,iterations,samples");
    benchmark_one("conv1d_fp32_2048x1x16", &conv1, 0, 0);
    benchmark_one("conv1d_static_zp0_2048x1x16", &conv1, 1, 0);
    benchmark_one("conv1d_static_zp128_1023x16x32", &conv2, 1, 128);
    benchmark_one("conv1d_dynamic_zp0_1023x16x32", &conv2, 2, 0);
    benchmark_one("conv2d_fp32_32x32x8x16", &conv_2d, 0, 0);
    benchmark_one("conv2d_static_zp128_32x32x8x16", &conv_2d, 1, 128);
    benchmark_one("conv2d_dynamic_zp0_32x32x8x16", &conv_2d, 2, 0);
#ifdef TEST_HAS_SCRATCH
    benchmark_one("conv1d_dynamic_scratch_zp0_1023x16x32", &conv2, 7, 0);
    benchmark_one("conv2d_dynamic_scratch_zp0_32x32x8x16", &conv_2d, 7, 0);
#endif
    benchmark_one("linear_fp32_4032x32", &linear, 3, 0);
    benchmark_one("linear_int8_zp0_4032x32", &linear, 4, 0);
    benchmark_one("linear_int8_zp128_4032x32", &linear, 4, 128);
    benchmark_one("maxpool_fp32_32x32x16", &pool, 5, 0);
    benchmark_one("maxpool_u8_32x32x16", &pool, 6, 0);
    fprintf(stderr, "checksum=%.9g seed=0x761c02ab\n", test_benchmark_checksum);
    return 0;
}

static int run_robust_case(const char *name)
{
    float floats[16] = {1.0f};
    uint8_t bytes[16] = {1};
    int8_t weights[16] = {1};
    float scales[4] = {1.0f};
    float bias[4] = {0.0f};
    float output[16] = {9876.0f};
    uint8_t output_u8[16] = {TEST_GUARD};
    conv2d_data_t input = {1, 1, 1, floats};
    conv2d_data_t result = {0, 0, 0, output};
    conv2d_filter_t filter = {1, 1, 1, 1, floats};
    batch_norm2d_t bn = {1, bias, scales, scales, bias};
    conv2d_config_t config = {1, 0, &filter, &bn};
    conv2d_data_i8_t input_u8 = {1, 1, 1, bytes};
    conv2d_data_i8_t result_u8 = {0, 0, 0, output_u8};
    conv2d_filter_i8_t filter_i8 = {1, 1, 1, 1, weights, scales, bias};
    quant_param_t quant = {1.0f, 0};
    conv2d_config_i8_t config_i8 = {1, 0, &filter_i8, {1.0f, 0}, 0, &quant};
    linear_param_t linear = {1, 1, floats, bias};
    linear_param_i8_t linear_i8 = {1, 1, weights, scales, bias};
    max_pool_config_t pool = {1, 1, 1, 0};
    int status = CNN_NORMAL;
    if (strcmp(name, "conv_int8_null_input") == 0) {
        status = conv2d_int8_run(NULL, &config_i8, &result_u8);
    } else if (strcmp(name, "conv_int8_zero_stride") == 0) {
        config_i8.stride = 0;
        status = conv2d_int8_run(&input_u8, &config_i8, &result_u8);
    } else if (strcmp(name, "conv_fp32_zero_stride") == 0) {
        config.stride = 0;
        status = conv2d_bn_no_bias(&input, &config, &result);
    } else if (strcmp(name, "pool_u8_large_kernel") == 0) {
        pool.col = 2;
        status = maxpool_u8_run(&input_u8, &pool, &result_u8);
    } else if (strcmp(name, "conv_int8_overflow_padding") == 0) {
        config_i8.pad = UINT32_MAX;
        status = conv2d_int8_run(&input_u8, &config_i8, &result_u8);
    } else if (strcmp(name, "conv_int8_mac_limit") == 0) {
        input_u8.channel = 32897u;
        filter_i8.channel = 32897u;
        status = conv2d_int8_run(&input_u8, &config_i8, &result_u8);
    } else if (strcmp(name, "cal_conv_zero_stride") == 0) {
        return cal_conv_out_len(1, 0, 1, 0) == 0u ? 0 : 1;
    } else if (strcmp(name, "cal_conv_overflow_padding") == 0) {
        return cal_conv_out_len(1, UINT32_MAX, 1, 1) == 0u ? 0 : 1;
    } else if (strcmp(name, "conv_fp32_zero_channels") == 0) {
        input.channel = 0;
        filter.channel = 0;
        status = conv2d_bn_no_bias(&input, &config, &result);
    } else if (strcmp(name, "conv_fp32_zero_kernel") == 0) {
        filter.col = 0;
        status = conv2d_bn_no_bias(&input, &config, &result);
    } else if (strcmp(name, "conv_fp32_bn_nan") == 0) {
        scales[0] = NAN;
        status = conv2d_bn_no_bias(&input, &config, &result);
    } else if (strcmp(name, "conv_fp32_overflow_shape") == 0) {
        input.row = 65536u;
        input.col = 65536u;
        status = conv2d_bn_no_bias(&input, &config, &result);
    } else if (strcmp(name, "conv_int8_overflow_shape") == 0) {
        input_u8.row = 65536u;
        input_u8.col = 65536u;
        status = conv2d_int8_run(&input_u8, &config_i8, &result_u8);
    } else if (strcmp(name, "conv_int8_nan_scale") == 0) {
        config_i8.in_act.scale = NAN;
        status = conv2d_int8_run(&input_u8, &config_i8, &result_u8);
    } else if (strcmp(name, "conv_int8_inf_scale") == 0) {
        config_i8.in_act.scale = INFINITY;
        status = conv2d_int8_run(&input_u8, &config_i8, &result_u8);
    } else if (strcmp(name, "conv_int8_negative_zp") == 0) {
        config_i8.in_act.zp = -1;
        status = conv2d_int8_run(&input_u8, &config_i8, &result_u8);
    } else if (strcmp(name, "conv_int8_large_output_zp") == 0) {
        quant.zp = 256;
        status = conv2d_int8_run(&input_u8, &config_i8, &result_u8);
    } else if (strcmp(name, "conv_int8_negative_weight_scale") == 0) {
        scales[0] = -1.0f;
        config_i8.dynamic = 1;
        status = conv2d_int8_run(&input_u8, &config_i8, &result_u8);
    } else if (strcmp(name, "conv_int8_inf_bias") == 0) {
        bias[0] = INFINITY;
        status = conv2d_int8_run(&input_u8, &config_i8, &result_u8);
    } else if (strcmp(name, "conv_int8_bad_dynamic") == 0) {
        config_i8.dynamic = 2;
        status = conv2d_int8_run(&input_u8, &config_i8, &result_u8);
    } else if (strcmp(name, "conv_dynamic_null_scale") == 0) {
        status = conv1d_int8_dynamic(bytes, 1, 1, weights, 1, 1, 0, 1, scales, bias,
                                     1.0f, 0, NULL, output_u8);
    } else if (strcmp(name, "linear_fp32_zero_inputs") == 0) {
        linear.inp_size = 0;
        status = linear_layer(floats, &linear, output);
    } else if (strcmp(name, "linear_int8_nan_scale") == 0) {
        quant.scale = NAN;
        status = linear_int8_run(bytes, &linear_i8, &quant, output);
    } else if (strcmp(name, "linear_int8_large_zp") == 0) {
        quant.zp = 256;
        status = linear_int8_run(bytes, &linear_i8, &quant, output);
    } else if (strcmp(name, "linear_int8_overflow_shape") == 0) {
        linear_i8.inp_size = UINT32_MAX;
        linear_i8.fea_size = 2;
        status = linear_int8_run(bytes, &linear_i8, &quant, output);
    } else if (strcmp(name, "linear_int8_overflow_output_bytes") == 0) {
        linear_i8.fea_size = (uint32_t)INT32_MAX / sizeof(float) + 1u;
        status = linear_int8_run(bytes, &linear_i8, &quant, output);
    } else if (strcmp(name, "linear_int8_nonfinite_output_tail") == 0
               || strcmp(name, "linear_int8_nonfinite_output_tile") == 0) {
        const uint32_t rows = strcmp(name, "linear_int8_nonfinite_output_tile") == 0 ? 4u : 1u;
        linear_i8.fea_size = rows;
        for (uint32_t row = 0; row < rows; row++) {
            scales[row] = FLT_MAX;
            weights[row] = 2;
        }
        status = linear_int8_run(bytes, &linear_i8, &quant, output);
    } else if (strcmp(name, "linear_fp32_overflow_shape") == 0) {
        linear.inp_size = UINT32_MAX;
        linear.fea_size = 2;
        status = linear_layer(floats, &linear, output);
    } else if (strcmp(name, "linear_fp32_nonfinite_output_tail") == 0
               || strcmp(name, "linear_fp32_nonfinite_output_tile") == 0) {
        linear.fea_size = strcmp(name, "linear_fp32_nonfinite_output_tile") == 0 ? 4u : 1u;
        floats[0] = FLT_MAX;
        status = linear_layer(floats, &linear, output);
    } else if (strcmp(name, "linear_fp32_nan_input") == 0) {
        floats[0] = NAN;
        status = linear_layer(floats, &linear, output);
    } else if (strcmp(name, "pool_fp32_zero_channels") == 0) {
        input.channel = 0;
        status = maxpool(&input, &pool, &result);
    } else if (strcmp(name, "pool_u8_zero_channels") == 0) {
        input_u8.channel = 0;
        status = maxpool_u8_run(&input_u8, &pool, &result_u8);
    } else if (strcmp(name, "pool_u8_overflow_shape") == 0) {
        input_u8.row = 65536u;
        input_u8.col = 65536u;
        status = maxpool_u8_run(&input_u8, &pool, &result_u8);
    } else if (strcmp(name, "quant_nan_scale") == 0 || strcmp(name, "quant_inf_scale") == 0
               || strcmp(name, "quant_large_zp") == 0) {
        quant.scale = strcmp(name, "quant_nan_scale") == 0 ? NAN :
                      (strcmp(name, "quant_inf_scale") == 0 ? INFINITY : 1.0f);
        quant.zp = strcmp(name, "quant_large_zp") == 0 ? 256 : 0;
        quantize_u8_run(floats, 1, &quant, output_u8);
        return output_u8[0] == TEST_GUARD ? 0 : 1;
    } else if (strcmp(name, "quant_extreme_finite") == 0) {
        floats[0] = FLT_MAX;
        floats[1] = -FLT_MAX;
        quantize_u8_run(floats, 2, &quant, output_u8);
        return output_u8[0] == 255u && output_u8[1] == 0u ? 0 : 1;
    } else if (strcmp(name, "softmax_nan_input") == 0) {
        floats[0] = NAN;
        status = softmax(floats, 1, output);
#ifdef TEST_HAS_SCRATCH
    } else if (strcmp(name, "scratch_small") == 0) {
        int32_t scratch[1];
        config_i8.dynamic = 1;
        status = conv2d_int8_run_with_scratch(&input_u8, &config_i8, &result_u8,
                                             scratch, sizeof(scratch) - 1u);
    } else if (strcmp(name, "scratch_unaligned") == 0) {
        _Alignas(int32_t) uint8_t scratch[8];
        config_i8.dynamic = 1;
        status = conv2d_int8_run_with_scratch(&input_u8, &config_i8, &result_u8,
                                             (int32_t *)(scratch + 1), 4u);
    } else if (strcmp(name, "scratch_null_nonzero") == 0) {
        config_i8.dynamic = 1;
        status = conv2d_int8_run_with_scratch(&input_u8, &config_i8, &result_u8, NULL, 4u);
    } else if (strcmp(name, "scratch_invalid_query") == 0) {
        config_i8.in_act.scale = NAN;
        config_i8.dynamic = 1;
        return conv2d_int8_get_scratch_size(&input_u8, &config_i8) == 0u ? 0 : 1;
#endif
    } else {
        fprintf(stderr, "Unknown robustness case: %s\n", name);
        return 2;
    }
    printf("%s status=%d\n", name, status);
    return status == CNN_NORMAL ? 1 : 0;
}

/** @brief Run deterministic validation or an explicitly selected timing/invalid-input case. */
int main(int argc, char **argv)
{
    if (argc == 2 && strcmp(argv[1], "--bench") == 0) {
        return run_benchmarks();
    }
    if (argc == 3 && strcmp(argv[1], "--robust-case") == 0) {
        return run_robust_case(argv[2]);
    }
    if (argc != 1) {
        fprintf(stderr, "Usage: %s [--bench | --robust-case NAME]\n", argv[0]);
        return 2;
    }
    return run_functional_tests();
}
