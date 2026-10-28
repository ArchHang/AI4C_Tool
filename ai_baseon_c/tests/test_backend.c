#include "ai_baseon_c/nn_platform.h"
#include "kernels/nn_dot.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BACKEND_TEST_MAX_SIZE 65280U
#define BACKEND_TEST_OFFSET_SIZE 32U

static uint32_t random_state = UINT32_C(0x71ac52e9);
static uint32_t test_count;

static uint32_t random_next(void)
{
    random_state ^= random_state << 13;
    random_state ^= random_state >> 17;
    random_state ^= random_state << 5;
    return random_state;
}

static int32_t check_dot(const uint8_t *input, const int8_t *const weights[4], uint32_t size,
                         int32_t zero_point)
{
    int32_t actual[4];
    int32_t raw[4];
    nn_dot4_u8_s8_centered(input, weights, size, zero_point, actual);
    nn_dot4_u8_s8(input, weights, size, raw);
    for (uint32_t row = 0; row < 4; row++) {
        int64_t expected = 0;
        int64_t expected_raw = 0;
        for (uint32_t index = 0; index < size; index++) {
            expected += ((int64_t)input[index] - (int64_t)zero_point) * weights[row][index];
            expected_raw += (int64_t)input[index] * weights[row][index];
        }
        if ((expected < INT32_MIN) || (expected > INT32_MAX) ||
            (expected_raw < INT32_MIN) || (expected_raw > INT32_MAX)) {
            (void)fprintf(stderr, "Test oracle overflow: size=%u row=%u\n", size, row);
            return 1;
        }
        const int32_t actual_raw = nn_dot_u8_s8(input, weights[row], size);
        if ((actual[row] != (int32_t)expected) || (raw[row] != (int32_t)expected_raw) ||
            (actual_raw != (int32_t)expected_raw)) {
            (void)fprintf(stderr,
                          "Dot mismatch: backend=%s size=%u zp=%d row=%u\n",
                          nn_get_backend_name(), size, zero_point, row);
            return 1;
        }
    }
    test_count++;
    return 0;
}

int main(void)
{
    static const int32_t zero_points[] = {0, 1, 127, 128, 254, 255};
    static const uint32_t large_sizes[] = {127U, 128U, 129U, 255U, 256U, 257U,
                                          1023U, 1024U, 1025U, 4032U, 65279U, 65280U};
    const uint32_t buffer_size = BACKEND_TEST_MAX_SIZE + BACKEND_TEST_OFFSET_SIZE;
    uint8_t *input_buffer = (uint8_t *)malloc(buffer_size);
    int8_t *weight_buffers[4] = {NULL, NULL, NULL, NULL};
    int32_t status = 0;
    for (uint32_t row = 0; row < 4; row++) {
        weight_buffers[row] = (int8_t *)malloc(buffer_size);
    }
    if ((input_buffer == NULL) || (weight_buffers[0] == NULL) ||
        (weight_buffers[1] == NULL) || (weight_buffers[2] == NULL) ||
        (weight_buffers[3] == NULL)) {
        (void)fprintf(stderr, "Unable to allocate backend test buffers\n");
        status = 1;
        goto cleanup;
    }
    for (uint32_t index = 0; index < buffer_size; index++) {
        input_buffer[index] = (uint8_t)random_next();
        for (uint32_t row = 0; row < 4; row++) {
            weight_buffers[row][index] = (int8_t)((int32_t)(random_next() & 255U) - 128);
        }
    }
    for (uint32_t offset = 0; offset < BACKEND_TEST_OFFSET_SIZE; offset++) {
        const int8_t *weights[4];
        for (uint32_t row = 0; row < 4; row++) {
            weights[row] = weight_buffers[row] + ((offset + row * 7U) % 32U);
        }
        for (uint32_t size = 0; size <= 65U; size++) {
            for (uint32_t point = 0; point < sizeof(zero_points) / sizeof(zero_points[0]);
                 point++) {
                if (check_dot(input_buffer + offset, weights, size, zero_points[point]) != 0) {
                    status = 1;
                    goto cleanup;
                }
            }
        }
        for (uint32_t index = 0; index < sizeof(large_sizes) / sizeof(large_sizes[0]); index++) {
            if (check_dot(input_buffer + offset, weights, large_sizes[index], 255) != 0) {
                status = 1;
                goto cleanup;
            }
        }
    }
    /* Repeated maximum products exercise lane and horizontal int32 accumulations. */
    (void)memset(input_buffer, 255, buffer_size);
    (void)memset(weight_buffers[0], -128, buffer_size);
    (void)memset(weight_buffers[1], 127, buffer_size);
    (void)memset(weight_buffers[2], -1, buffer_size);
    (void)memset(weight_buffers[3], 1, buffer_size);
    const int8_t *extreme_weights[4] = {weight_buffers[0] + 1, weight_buffers[1] + 3,
                                       weight_buffers[2] + 7, weight_buffers[3] + 15};
    for (uint32_t point = 0; point < sizeof(zero_points) / sizeof(zero_points[0]); point++) {
        if (check_dot(input_buffer + 1, extreme_weights, BACKEND_TEST_MAX_SIZE,
                      zero_points[point]) != 0) {
            status = 1;
            goto cleanup;
        }
    }
    (void)memset(input_buffer, 0, buffer_size);
    if (check_dot(input_buffer + 1, extreme_weights, BACKEND_TEST_MAX_SIZE, 255) != 0) {
        status = 1;
        goto cleanup;
    }
    const int8_t *empty_weights[4] = {NULL, NULL, NULL, NULL};
    if (check_dot(NULL, empty_weights, 0, 255) != 0) {
        status = 1;
        goto cleanup;
    }
#if NN_HAS_NEON
    for (uint32_t offset = 0; offset < 32U; offset++) {
        for (int32_t weight = -128; weight <= 127; weight++) {
            int32_t sums[8] = {1, -2, 3, -4, 5, -6, 7, -8};
            int32_t expected[8];
            for (uint32_t lane = 0; lane < 8U; lane++) {
                input_buffer[offset + lane] = (uint8_t)random_next();
                expected[lane] = sums[lane] + (int32_t)input_buffer[offset + lane] * weight;
            }
            nn_neon_mac8_u8_s8(input_buffer + offset, (int8_t)weight, sums);
            if (memcmp(sums, expected, sizeof(sums)) != 0) {
                (void)fprintf(stderr, "NEON spatial tile mismatch\n");
                status = 1;
                goto cleanup;
            }
            test_count++;
        }
    }
#endif
    if ((nn_get_backend() != (nn_backend_t)NN_SELECTED_BACKEND) ||
        (nn_get_platform_info()->required_buffer_alignment != 1U)) {
        (void)fprintf(stderr, "Platform information does not match selected backend\n");
        status = 1;
        goto cleanup;
    }
    (void)printf("backend=%s architecture=%s cases=%u all exact\n", nn_get_backend_name(),
                 nn_get_platform_info()->architecture_name, test_count);

cleanup:
    free(input_buffer);
    for (uint32_t row = 0; row < 4; row++) {
        free(weight_buffers[row]);
    }
    return status;
}
