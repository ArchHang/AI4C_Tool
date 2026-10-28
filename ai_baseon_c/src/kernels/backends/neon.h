/**
 * @file neon.h
 * @brief Exact baseline NEON dots for Armv7-A and Armv8-A.
 *
 * Operands widen to int16 and products accumulate in int32. No v8.2 dot-product
 * extension, saturated pair addition, extra padding, or alignment is required.
 */
#ifndef AI_BASEON_C_SRC_KERNELS_BACKENDS_NEON_H_
#define AI_BASEON_C_SRC_KERNELS_BACKENDS_NEON_H_

#include <stdint.h>

#include <arm_neon.h>

static inline int32_t nn_neon_sum_i32x4(int32x4_t value)
{
    int32x2_t sum = vadd_s32(vget_low_s32(value), vget_high_s32(value));
    sum = vpadd_s32(sum, sum);
    return vget_lane_s32(sum, 0);
}

static inline int32x4_t nn_neon_mac8_i16(int32x4_t sum, int16x8_t input,
                                       const int8_t *weights)
{
    const int16x8_t values = vmovl_s8(vld1_s8(weights));
    sum = vmlal_s16(sum, vget_low_s16(input), vget_low_s16(values));
    return vmlal_s16(sum, vget_high_s16(input), vget_high_s16(values));
}

static inline int32_t nn_dot_u8_s8(const uint8_t *input, const int8_t *weights,
                                 uint32_t size)
{
    uint32_t index = 0;
    int32x4_t accum = vdupq_n_s32(0);
    for (; size - index >= 8; index += 8) {
        const int16x8_t values = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(input + index)));
        accum = nn_neon_mac8_i16(accum, values, weights + index);
    }
    int32_t sum = nn_neon_sum_i32x4(accum);
    for (; index < size; index++) {
        sum += (int32_t)input[index] * (int32_t)weights[index];
    }
    return sum;
}

static inline void nn_dot4_u8_s8_centered(const uint8_t *input,
                                        const int8_t *const weights[4], uint32_t size,
                                        int32_t zero_point, int32_t sums[4])
{
    const int16x8_t offset = vdupq_n_s16((int16_t)zero_point);
    uint32_t index = 0;
    int32x4_t accum0 = vdupq_n_s32(0);
    int32x4_t accum1 = accum0;
    int32x4_t accum2 = accum0;
    int32x4_t accum3 = accum0;
    for (; size - index >= 8; index += 8) {
        const int16x8_t values = vsubq_s16(
            vreinterpretq_s16_u16(vmovl_u8(vld1_u8(input + index))), offset);
        accum0 = nn_neon_mac8_i16(accum0, values, weights[0] + index);
        accum1 = nn_neon_mac8_i16(accum1, values, weights[1] + index);
        accum2 = nn_neon_mac8_i16(accum2, values, weights[2] + index);
        accum3 = nn_neon_mac8_i16(accum3, values, weights[3] + index);
    }
    int32_t sum0 = nn_neon_sum_i32x4(accum0);
    int32_t sum1 = nn_neon_sum_i32x4(accum1);
    int32_t sum2 = nn_neon_sum_i32x4(accum2);
    int32_t sum3 = nn_neon_sum_i32x4(accum3);
    for (; index < size; index++) {
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

/* One kernel tap across eight adjacent spatial positions; input has exactly eight bytes. */
static inline void nn_neon_mac8_u8_s8(const uint8_t *input, int8_t weight, int32_t sums[8])
{
    const int16x8_t values = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(input)));
    const int16x4_t coefficient = vdup_n_s16((int16_t)weight);
    int32x4_t first = vld1q_s32(sums);
    int32x4_t second = vld1q_s32(sums + 4);
    first = vmlal_s16(first, vget_low_s16(values), coefficient);
    second = vmlal_s16(second, vget_high_s16(values), coefficient);
    vst1q_s32(sums, first);
    vst1q_s32(sums + 4, second);
}

#endif
