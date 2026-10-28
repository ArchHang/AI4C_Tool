#ifndef AI_BASEON_C_SRC_KERNELS_BACKENDS_PORTABLE_H_
#define AI_BASEON_C_SRC_KERNELS_BACKENDS_PORTABLE_H_

#include <stdint.h>

static inline int32_t nn_dot_u8_s8(const uint8_t *input, const int8_t *weights,
                                 uint32_t size)
{
    int32_t sum = 0;
    for (uint32_t index = 0; index < size; index++) {
        sum += (int32_t)input[index] * (int32_t)weights[index];
    }
    return sum;
}

static inline void nn_dot4_u8_s8_centered(const uint8_t *input,
                                        const int8_t *const weights[4], uint32_t size,
                                        int32_t zero_point, int32_t sums[4])
{
    int32_t sum0 = 0;
    int32_t sum1 = 0;
    int32_t sum2 = 0;
    int32_t sum3 = 0;
    for (uint32_t index = 0; index < size; index++) {
        const int32_t value = (int32_t)input[index] - zero_point;
        sum0 += value * (int32_t)weights[0][index];
        sum1 += value * (int32_t)weights[1][index];
        sum2 += value * (int32_t)weights[2][index];
        sum3 += value * (int32_t)weights[3][index];
    }
    sums[0] = sum0;
    sums[1] = sum1;
    sums[2] = sum2;
    sums[3] = sum3;
}

#endif
