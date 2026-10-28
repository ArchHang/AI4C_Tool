/**
 * @file test_weight_cache.c
 * @brief Independent exactness and rejection tests for the three weight locations.
 *
 * The reference uses int64 scalar MACs and division/remainder Q28 rounding. It does
 * not call production MAC, packing or requantization helpers. All storage is static.
 */
#include "ai_baseon_c/cnn.h"

#include <float.h>
#include <inttypes.h>
#include <limits.h>
#include <stdio.h>

#define TEST_CAPACITY 16384u
#define TEST_CHANNELS 32u
#define TEST_PACK_CAPACITY 2048u
#define TEST_BYTE_GUARD UINT8_C(173)
#define TEST_HALF_GUARD INT16_C(26983)
#define TEST_WORD_GUARD INT32_C(73911921)
#define TEST_FIXED_UNIT INT64_C(268435456)

typedef struct test_storage {
    uint8_t input[TEST_CAPACITY + 8u];
    uint8_t input_snapshot[TEST_CAPACITY];
    int8_t weights[TEST_PACK_CAPACITY];
    int8_t ram_weights[TEST_PACK_CAPACITY + 2u];
    float scales[TEST_CHANNELS];
    float biases[TEST_CHANNELS];
    int64_t accumulators[TEST_CAPACITY];
    int32_t pool_reference[TEST_CAPACITY];
    int32_t scratch[TEST_CAPACITY + 2u];
    _Alignas(int32_t) int16_t packed_halves[TEST_PACK_CAPACITY + 8u];
    int16_t packed_snapshot[TEST_PACK_CAPACITY];
    uint8_t reference[TEST_CAPACITY];
    uint8_t actual[TEST_CAPACITY + 2u];
} test_storage_t;

static test_storage_t storage;
static test_storage_t rejection_snapshot;
static uint32_t rng_state = UINT32_C(0x9c17e48b);
static uint32_t checks;
static uint32_t failures;
static uint32_t cases;
static uint32_t rejected_cases;
static uint32_t packed_values;

static uint32_t next_random(void)
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

static void check_condition(int32_t condition, const char *name)
{
    (void)name;
    checks++;
    if (!condition) {
        failures++;
        if (failures <= 30u) {
#ifndef TEST_NO_STDIO
            printf("FAIL case=%" PRIu32 " %s\n", cases, name);
#endif
        }
    }
}

static uint8_t reference_quantize(int64_t sum, float multiplier, float bias,
                                  float scale, uint32_t kernel_size)
{
    const double coefficient = (double)multiplier / (double)scale * TEST_FIXED_UNIT;
    const double offset = (double)bias / (double)scale * TEST_FIXED_UNIT;
    const double bound = fabs(coefficient) * (double)(kernel_size * 32640u)
        + fabs(offset) + (double)(kernel_size * 32640u) + 1.0;
    if (isfinite(bound) && bound < 9.0e18) {
        const int64_t value = sum * (int64_t)nearbyint(coefficient)
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
            || (remainder == TEST_FIXED_UNIT / 2 && (quotient % 2) != 0)) {
            quotient++;
        }
        return (uint8_t)quotient;
    }
    const float real_value = (float)sum * multiplier + bias;
    const float normalized = (real_value > 0.0f ? real_value : 0.0f) / scale;
    if (normalized <= 0.0f || isnan(normalized)) {
        return 0;
    }
    if (normalized >= 255.0f) {
        return 255;
    }
    return (uint8_t)nearbyint((double)normalized);
}

static float reference_run(const conv2d_data_i8_t *input,
                           const conv2d_config_i8_t *config)
{
    const conv2d_filter_i8_t *filter = config->filter;
    const uint32_t conv_col = input->col - filter->col + 1u;
    const uint32_t pool_col = conv_col / 2u;
    const uint32_t kernel_size = filter->channel * filter->col;
    float maximum = 0.0f;
    for (uint32_t channel = 0; channel < filter->filter_num; channel++) {
        int64_t channel_maximum = INT64_MIN;
        for (uint32_t col = 0; col < conv_col; col++) {
            int64_t sum = 0;
            for (uint32_t source = 0; source < input->channel; source++) {
                for (uint32_t tap = 0; tap < filter->col; tap++) {
                    const int64_t value = (int64_t)input->data[source * input->col + col + tap]
                        - (int64_t)config->in_act.zp;
                    sum += value * (int64_t)filter->data[channel * kernel_size
                        + source * filter->col + tap];
                }
            }
            storage.accumulators[channel * conv_col + col] = sum;
            if (sum > channel_maximum) {
                channel_maximum = sum;
            }
        }
        const float real_maximum = (float)channel_maximum
            * (filter->scale[channel] * config->in_act.scale) + filter->bias[channel];
        if (real_maximum > maximum) {
            maximum = real_maximum;
        }
    }
    const float scale = fmaxf(maximum / 255.0f, 1e-9f);
    for (uint32_t channel = 0; channel < filter->filter_num; channel++) {
        for (uint32_t col = 0; col < pool_col; col++) {
            const uint32_t index = channel * conv_col + col * 2u;
            const int64_t left = storage.accumulators[index];
            const int64_t right = storage.accumulators[index + 1u];
            const int64_t maximum_acc = left > right ? left : right;
            storage.pool_reference[channel * pool_col + col] = (int32_t)maximum_acc;
            storage.reference[channel * pool_col + col] = reference_quantize(maximum_acc,
                filter->scale[channel] * config->in_act.scale, filter->bias[channel],
                scale, kernel_size);
        }
    }
    return scale;
}

static int16_t *get_packed(void)
{
    return storage.packed_halves + 2;
}

static void check_packing(const conv2d_filter_i8_t *filter)
{
    const uint32_t kernel_size = filter->channel * filter->col;
    const uint32_t padded_size = (kernel_size + 3u) / 4u * 4u;
    const size_t expected = (size_t)padded_size * filter->filter_num * sizeof(int16_t);
    int16_t *const packed = get_packed();
    storage.packed_halves[0] = TEST_HALF_GUARD;
    storage.packed_halves[expected / sizeof(int16_t) + 2u] = TEST_HALF_GUARD;
    check_condition(conv2d_int8_packed_weights_size(filter) == expected, "packed byte size");
    check_condition(conv2d_int8_pack_weights(filter, packed, expected) == CNN_NORMAL,
                    "packing status");
    int32_t correct = 1;
    for (uint32_t channel = 0; channel < filter->filter_num; channel++) {
        for (uint32_t offset = 0; offset < padded_size; offset++) {
            const uint32_t permutation[4] = {0u, 2u, 1u, 3u};
            const uint32_t tap = offset / 4u * 4u + permutation[offset % 4u];
            const int16_t expected_value = tap < kernel_size
                ? (int16_t)filter->data[channel * kernel_size + tap] : 0;
            if (packed[channel * padded_size + offset] != expected_value) {
                correct = 0;
            }
            packed_values++;
        }
    }
    check_condition(correct, "packed layout and sign extension");
    check_condition(storage.packed_halves[0] == TEST_HALF_GUARD
        && storage.packed_halves[expected / sizeof(int16_t) + 2u] == TEST_HALF_GUARD,
        "packing guards");
}

static void check_outputs(const conv2d_data_i8_t *input, conv2d_config_i8_t *config,
                          const conv2d_data_i8_t *output, float scale, size_t output_size)
{
    check_condition(memcmp(output->data, storage.reference, output_size) == 0,
                    "oracle uint8 exact");
    check_condition(memcmp(&config->out_act->scale, &scale, sizeof(float)) == 0
        && config->out_act->zp == 0, "oracle scale IEEE bits");
    check_condition(output->row == 1u && output->channel == config->filter->filter_num
        && output->col == (input->col - config->filter->col + 1u) / 2u,
        "output geometry");
    check_condition(storage.actual[0] == TEST_BYTE_GUARD
        && storage.actual[output_size + 1u] == TEST_BYTE_GUARD, "output guards");
    check_condition(storage.scratch[0] == TEST_WORD_GUARD
        && storage.scratch[output_size + 1u] == TEST_WORD_GUARD, "scratch guards");
    check_condition(memcmp(storage.scratch + 1, storage.pool_reference,
        output_size * sizeof(int32_t)) == 0, "int64 oracle pooled accumulators");
    check_condition(memcmp(input->data, storage.input_snapshot,
        (size_t)input->col * input->channel) == 0, "input unchanged");
}

static void check_case(conv2d_data_i8_t *input, conv2d_config_i8_t *config)
{
    cases++;
    conv2d_filter_i8_t filter = *config->filter;
    const quant_param_t in_act = config->in_act;
    const float reference_scale = reference_run(input, config);
    const uint32_t output_size = filter.filter_num * ((input->col - filter.col + 1u) / 2u);
    const size_t scratch_size = (size_t)output_size * sizeof(int32_t);
    const size_t weight_size = (size_t)filter.channel * filter.col * filter.filter_num;
    const size_t packed_size = (size_t)((filter.channel * filter.col + 3u) / 4u * 4u)
        * filter.filter_num * sizeof(int16_t);
    memcpy(storage.input_snapshot, input->data, (size_t)input->col * input->channel);
    memcpy(storage.ram_weights + 1, filter.data, weight_size);
    storage.ram_weights[0] = (int8_t)77;
    storage.ram_weights[weight_size + 1u] = (int8_t)77;
    check_packing(&filter);
    memcpy(storage.packed_snapshot, get_packed(), packed_size);
    quant_param_t output_quant = {0.0f, 0};
    for (uint32_t mode = 0; mode < 6u; mode++) {
        conv2d_config_i8_t current = *config;
        current.in_act = in_act;
        current.out_act = mode >= 3u ? &current.in_act : &output_quant;
        current.filter = &filter;
        filter.data = mode % 3u == 1u ? storage.ram_weights + 1 : config->filter->data;
        conv2d_data_i8_t output = {0u, 0u, 0u, storage.actual + 1};
        memset(storage.actual, TEST_BYTE_GUARD, (size_t)output_size + 2u);
        storage.scratch[0] = TEST_WORD_GUARD;
        storage.scratch[output_size + 1u] = TEST_WORD_GUARD;
        int32_t status;
        if (mode % 3u == 2u) {
            const conv2d_packed_context_t context = {get_packed(), packed_size,
                storage.scratch + 1, scratch_size};
            status = conv2d_int8_maxpool2_run_packed(input, &current, &output, &context);
        } else {
            status = conv2d_int8_maxpool2_run_with_scratch(input, &current, &output,
                storage.scratch + 1, scratch_size);
        }
        check_condition(status == CNN_NORMAL, "inference status");
        check_outputs(input, &current, &output, reference_scale, output_size);
    }
    check_condition(storage.ram_weights[0] == (int8_t)77
        && storage.ram_weights[weight_size + 1u] == (int8_t)77, "RAM weights guards");
    check_condition(memcmp(storage.ram_weights + 1, config->filter->data, weight_size) == 0,
                    "int8 RAM copy unchanged");
    check_condition(memcmp(storage.packed_snapshot, get_packed(), packed_size) == 0,
                    "packed weights unchanged by inference");
    check_condition(storage.input[0] == TEST_BYTE_GUARD
        && storage.input[1] == TEST_BYTE_GUARD
        && storage.input[2] == TEST_BYTE_GUARD
        && storage.input[3] == TEST_BYTE_GUARD
        && storage.input[(size_t)input->col * input->channel + 4u] == TEST_BYTE_GUARD,
        "input guards");
}

static void populate_case(uint32_t input_size, uint32_t weight_size, uint32_t outputs,
                           uint32_t pattern)
{
    memset(storage.input, TEST_BYTE_GUARD, 4u);
    memset(storage.input + input_size + 4u, TEST_BYTE_GUARD, 4u);
    for (uint32_t index = 0; index < input_size; index++) {
        const uint8_t value = pattern == 1u ? 255u : (pattern == 2u ? 0u
            : (pattern == 3u ? (uint8_t)index : (uint8_t)next_random()));
        storage.input[index + 4u] = value;
    }
    for (uint32_t index = 0; index < weight_size; index++) {
        const int32_t value = pattern == 1u ? -128 : (pattern == 2u ? 127
            : (pattern == 3u ? (index % 2u == 0u ? -128 : 127)
            : (int32_t)(next_random() & 255u) - 128));
        storage.weights[index] = (int8_t)value;
    }
    for (uint32_t channel = 0; channel < outputs; channel++) {
        storage.scales[channel] = (float)(1u + next_random() % 1024u) / 65536.0f;
        storage.biases[channel] = (float)((int32_t)(next_random() % 8193u) - 4096) / 1024.0f;
    }
}

static void test_matrix(void)
{
    const uint32_t lengths[] = {1u, 2u, 3u, 4u, 5u, 7u, 20u, 40u, 63u, 64u};
    const uint32_t outputs[] = {1u, 2u, 3u, 4u, 5u, 16u, 31u, 32u};
    const uint32_t widths[] = {2u, 3u, 4u, 5u, 7u, 8u, 17u, 32u, 33u};
    const int32_t zero_points[] = {0, 1, 128, 255};
    for (uint32_t li = 0; li < sizeof(lengths) / sizeof(lengths[0]); li++) {
        for (uint32_t oi = 0; oi < sizeof(outputs) / sizeof(outputs[0]); oi++) {
            for (uint32_t wi = 0; wi < 9u; wi++) {
                for (uint32_t zi = 0; zi < 4u; zi++) {
                    for (uint32_t shape = 0; shape < 2u; shape++) {
                        const uint32_t kernel = shape == 0u ? lengths[li] : 1u;
                        const uint32_t channels = lengths[li] / kernel;
                        const uint32_t input_width = widths[wi] + kernel - 1u;
                        populate_case(input_width * channels, lengths[li] * outputs[oi],
                            outputs[oi], (li + oi + wi + zi + shape) % 4u);
                        conv2d_data_i8_t input = {1u, input_width, channels, storage.input + 4};
                        conv2d_filter_i8_t filter = {1u, kernel, channels, outputs[oi],
                            storage.weights, storage.scales, storage.biases};
                        quant_param_t output_quant = {0.0f, 0};
                        conv2d_config_i8_t config = {1u, 0u, &filter,
                            {0.0078125f, zero_points[zi]}, 1, &output_quant};
                        check_case(&input, &config);
                    }
                }
            }
        }
    }
}

static void test_special_cases(void)
{
    const uint32_t shapes[][3] = {{1u, 5u, 4u}, {4u, 5u, 8u}, {8u, 5u, 16u}};
    for (uint32_t shape = 0; shape < 3u; shape++) {
        const uint32_t channels = shapes[shape][0];
        const uint32_t kernel = shapes[shape][1];
        const uint32_t outputs = shapes[shape][2];
        const uint32_t width = 35u + kernel - 1u;
        populate_case(width * channels, kernel * channels * outputs, outputs, 0u);
        conv2d_data_i8_t input = {1u, width, channels, storage.input + 4};
        conv2d_filter_i8_t filter = {1u, kernel, channels, outputs,
            storage.weights, storage.scales, storage.biases};
        quant_param_t output_quant = {0.0f, 0};
        conv2d_config_i8_t config = {1u, 0u, &filter, {1.0f, 0}, 1, &output_quant};
        check_case(&input, &config);
    }
    populate_case(20u, 16u, 4u, 0u);
    memset(storage.input + 4, 1, 20u);
    memset(storage.weights, 0, 16u);
    for (uint32_t channel = 0; channel < 4u; channel++) {
        storage.weights[channel * 4u + 3u] = 1;
        storage.scales[channel] = 1.0f;
        storage.biases[channel] = 0.0f;
        storage.input[4u + channel * 5u + 4u] = 255u;
    }
    conv2d_data_i8_t input = {1u, 5u, 4u, storage.input + 4};
    conv2d_filter_i8_t filter = {1u, 1u, 4u, 4u,
        storage.weights, storage.scales, storage.biases};
    quant_param_t output_quant = {0.0f, 0};
    conv2d_config_i8_t config = {1u, 0u, &filter, {1.0f, 0}, 1, &output_quant};
    check_case(&input, &config);
    check_condition(storage.reference[0] == 1u && storage.reference[1] == 1u,
                    "discarded odd tail controls scale");
    for (uint32_t variant = 0; variant < 4u; variant++) {
        if (variant == 0u) {
            for (uint32_t channel = 0; channel < 4u; channel++) {
                storage.input[4u + channel * 5u] = 0u;
                storage.input[4u + channel * 5u + 1u] = 0u;
                storage.input[4u + channel * 5u + 2u] = 4u;
                storage.input[4u + channel * 5u + 3u] = 4u;
            }
        }
        for (uint32_t channel = 0; channel < 4u; channel++) {
            storage.scales[channel] = variant == 3u ? FLT_MAX / 512.0f
                : (variant == 2u && channel == 1u ? 1e12f : 0.5f);
            storage.biases[channel] = variant == 0u ? 127.5f
                : (variant == 1u ? -300.0f
                : (variant == 2u && channel == 1u ? -(255.0f * 1e12f) : 1.0f));
        }
        check_case(&input, &config);
        if (variant == 0u) {
            check_condition(storage.reference[0] == 128u && storage.reference[1] == 130u,
                            "exact 127.5 and 129.5 half-even ties");
        } else if (variant == 1u) {
            check_condition(storage.reference[0] == 0u && storage.reference[1] == 0u,
                            "all negative ReLU output");
        }
    }
}

static void test_rejections(void)
{
    populate_case(20u, 16u, 4u, 0u);
    conv2d_data_i8_t input_base = {1u, 5u, 4u, storage.input + 4};
    conv2d_filter_i8_t filter_base = {1u, 1u, 4u, 4u,
        storage.weights, storage.scales, storage.biases};
    quant_param_t output_quant = {7.0f, 17};
    conv2d_config_i8_t config_base = {1u, 0u, &filter_base, {1.0f, 0}, 1, &output_quant};
    check_packing(&filter_base);
    for (uint32_t variant = 0; variant < 57u; variant++) {
        conv2d_data_i8_t input = input_base;
        conv2d_data_i8_t output = {71u, 72u, 73u, storage.actual + 1};
        conv2d_filter_i8_t filter = filter_base;
        conv2d_config_i8_t config = config_base;
        config.filter = &filter;
        conv2d_packed_context_t context = {get_packed(), 32u, storage.scratch + 1, 32u};
        const float scale = storage.scales[0];
        const float bias = storage.biases[0];
        int32_t expected = CNN_DATA_EXCEPTION;
        const conv2d_data_i8_t *input_pointer = &input;
        const conv2d_config_i8_t *config_pointer = &config;
        conv2d_data_i8_t *output_pointer = &output;
        const conv2d_packed_context_t *context_pointer = &context;
        switch (variant) {
            case 0: input_pointer = NULL; expected = CNN_POINTER_NULL; break;
            case 1: config_pointer = NULL; expected = CNN_POINTER_NULL; break;
            case 2: output_pointer = NULL; expected = CNN_POINTER_NULL; break;
            case 3: context_pointer = NULL; expected = CNN_POINTER_NULL; break;
            case 4: input.data = NULL; expected = CNN_POINTER_NULL; break;
            case 5: output.data = NULL; expected = CNN_POINTER_NULL; break;
            case 6: config.filter = NULL; expected = CNN_POINTER_NULL; break;
            case 7: config.out_act = NULL; expected = CNN_POINTER_NULL; break;
            case 8: filter.data = NULL; expected = CNN_POINTER_NULL; break;
            case 9: filter.scale = NULL; expected = CNN_POINTER_NULL; break;
            case 10: filter.bias = NULL; expected = CNN_POINTER_NULL; break;
            case 11: context.weights = NULL; expected = CNN_POINTER_NULL; break;
            case 12: context.scratch = NULL; expected = CNN_POINTER_NULL; break;
            case 13: context.weights_size = 31u; break;
            case 14: context.scratch_size = 31u; break;
            case 15: context.weights = get_packed() + 1; break;
            case 16:
                context.scratch = (int32_t *)(void *)((uint8_t *)storage.scratch + 1);
                break;
            case 17: config.stride = 2u; break;
            case 18: config.pad = 1u; break;
            case 19: config.dynamic = 0; break;
            case 20: input.row = 2u; break;
            case 21: filter.row = 2u; break;
            case 22: input.col = 1u; break;
            case 23: filter.channel = 3u; break;
            case 24: filter.filter_num = 33u; break;
            case 25: config.in_act.zp = -1; break;
            case 26: config.in_act.zp = 256; break;
            case 27: config.in_act.scale = NAN; break;
            case 28: config.in_act.scale = INFINITY; break;
            case 29: storage.scales[0] = NAN; break;
            case 30: storage.scales[0] = INFINITY; break;
            case 31: storage.biases[0] = NAN; break;
            case 32: storage.biases[0] = -INFINITY; break;
            case 33: output.data = input.data; break;
            case 34: output.data = (uint8_t *)(void *)context.weights; break;
            case 35: context.scratch = (int32_t *)(void *)context.weights; break;
            case 36: context.scratch = (int32_t *)(void *)input.data; break;
            case 37: output.data = (uint8_t *)(void *)storage.weights; break;
            case 38: config.in_act.scale = 0.0f; break;
            case 39: config.in_act.scale = -1.0f; break;
            case 40: storage.scales[0] = 0.0f; break;
            case 41: storage.scales[0] = -1.0f; break;
            case 42: storage.biases[0] = INFINITY; break;
            case 43: filter.filter_num = 0u; break;
            case 44: context.weights_size = 0u; break;
            case 45: context.scratch_size = 0u; break;
            case 46: output.data = (uint8_t *)(void *)storage.scales; break;
            case 47: output.data = (uint8_t *)(void *)storage.biases; break;
            case 48: context.scratch = (int32_t *)(void *)storage.scales; break;
            case 49: context.scratch = (int32_t *)(void *)storage.biases; break;
            case 50: config.out_act = (quant_param_t *)(void *)input.data; break;
            case 51: config.out_act = (quant_param_t *)(void *)context.weights; break;
            case 52: config.out_act = (quant_param_t *)(void *)context.scratch; break;
            case 53: config.out_act = (quant_param_t *)(void *)storage.weights; break;
            case 54: config.out_act = (quant_param_t *)(void *)&config; break;
            case 55: config.out_act = (quant_param_t *)(void *)&input; break;
            default:
                input.col = UINT32_MAX;
                break;
        }
        memcpy(&rejection_snapshot, &storage, sizeof(storage));
        check_condition(conv2d_int8_maxpool2_run_packed(input_pointer, config_pointer,
            output_pointer, context_pointer) == expected, "packed rejection status");
        check_condition(memcmp(&rejection_snapshot, &storage, sizeof(storage)) == 0,
                        "rejected inference storage unchanged");
        check_condition(output.row == 71u && output.col == 72u && output.channel == 73u,
                        "rejected output geometry unchanged");
        storage.scales[0] = scale;
        storage.biases[0] = bias;
        rejected_cases++;
    }
    check_condition(conv2d_int8_packed_weights_size(NULL) == 0u, "null size query");
    check_condition(conv2d_int8_pack_weights(NULL, get_packed(), 32u) == CNN_POINTER_NULL,
                    "pack null filter");
    check_condition(conv2d_int8_pack_weights(&filter_base, NULL, 32u) == CNN_POINTER_NULL,
                    "pack null output");
    check_condition(conv2d_int8_pack_weights(&filter_base, get_packed(), 31u)
        == CNN_DATA_EXCEPTION, "pack short capacity");
    check_condition(conv2d_int8_pack_weights(&filter_base, get_packed() + 1, 32u)
        == CNN_DATA_EXCEPTION, "pack unaligned output");
    check_condition(conv2d_int8_pack_weights(&filter_base,
        (int16_t *)(void *)storage.weights, 32u) == CNN_DATA_EXCEPTION,
        "pack input output overlap");
    rejected_cases += 6u;
    for (uint32_t variant = 0; variant < 10u; variant++) {
        conv2d_filter_i8_t filter = filter_base;
        int16_t *destination = get_packed();
        int32_t expected = CNN_DATA_EXCEPTION;
        switch (variant) {
            case 0: filter.data = NULL; expected = CNN_POINTER_NULL; break;
            case 1: filter.row = 2u; break;
            case 2: filter.channel = 0u; break;
            case 3: filter.channel = 65u; break;
            case 4: filter.filter_num = 0u; break;
            case 5: filter.filter_num = 33u; break;
            case 6: filter.col = 0u; break;
            case 7: filter.channel = UINT32_MAX; filter.col = UINT32_MAX; break;
            case 8: destination = (int16_t *)(void *)storage.scales; break;
            default: destination = (int16_t *)(void *)storage.biases; break;
        }
        check_condition(conv2d_int8_pack_weights(&filter, destination, 32u) == expected,
                        "packer rejection status");
        if (variant < 8u) {
            check_condition(conv2d_int8_packed_weights_size(&filter) == 0u,
                            "unsupported filter size query");
        }
        rejected_cases++;
    }
}

int main(void)
{
    test_matrix();
    test_special_cases();
    test_rejections();
#ifndef TEST_NO_STDIO
    printf("weight_cache cases=%" PRIu32 " checks=%" PRIu32 " failures=%" PRIu32
        " rejected=%" PRIu32 " packed_values=%" PRIu32 " seed=0x9c17e48b\n",
        cases, checks, failures, rejected_cases, packed_values);
#endif
    return failures == 0u ? 0 : 1;
}
