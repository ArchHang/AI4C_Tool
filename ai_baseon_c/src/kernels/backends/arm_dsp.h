#ifndef AI_BASEON_C_SRC_KERNELS_BACKENDS_ARM_DSP_H_
#define AI_BASEON_C_SRC_KERNELS_BACKENDS_ARM_DSP_H_

#include <stdint.h>
#include <string.h>

#include <arm_acle.h>

static inline uint32_t nn_read_u32(const void *address)
{
    uint32_t value;
    /* Byte buffers may be unaligned and must not violate strict aliasing. */
    memcpy(&value, address, sizeof(value));
    return value;
}

static inline uint32_t nn_arm_ror8(uint32_t value)
{
    return (value >> 8) | (value << 24);
}

static inline uint32_t nn_arm_sxtb16(uint32_t value)
{
    return (uint32_t)__sxtb16((int8x4_t)value);
}

static inline uint32_t nn_arm_uxtb16(uint32_t value)
{
    return (uint32_t)__uxtb16((uint8x4_t)value);
}

static inline uint32_t nn_arm_ssub16(uint32_t first, uint32_t second)
{
    return (uint32_t)__ssub16((int16x2_t)first, (int16x2_t)second);
}

static inline int32_t nn_arm_smlad(uint32_t first, uint32_t second, int32_t sum)
{
    return __smlad((int16x2_t)first, (int16x2_t)second, sum);
}

static inline int32_t nn_arm_mac4(uint32_t input_even, uint32_t input_odd,
                                const int8_t *weights, int32_t sum)
{
    const uint32_t packed = nn_read_u32(weights);
    const uint32_t even = nn_arm_sxtb16(packed);
    const uint32_t odd = nn_arm_sxtb16(nn_arm_ror8(packed));
    sum = nn_arm_smlad(input_even, even, sum);
    return nn_arm_smlad(input_odd, odd, sum);
}

static inline int32_t nn_dot_u8_s8(const uint8_t *input, const int8_t *weights,
                                 uint32_t size)
{
    uint32_t index = 0;
    int32_t sum = 0;
    for (; size - index >= 4; index += 4) {
        const uint32_t packed = nn_read_u32(input + index);
        sum = nn_arm_mac4(nn_arm_uxtb16(packed), nn_arm_uxtb16(nn_arm_ror8(packed)),
                         weights + index, sum);
    }
    for (; index < size; index++) {
        sum += (int32_t)input[index] * (int32_t)weights[index];
    }
    return sum;
}

static inline void nn_dot4_u8_s8_centered(const uint8_t *input,
                                        const int8_t *const weights[4], uint32_t size,
                                        int32_t zero_point, int32_t sums[4])
{
    uint32_t index = 0;
    int32_t sum0 = 0;
    int32_t sum1 = 0;
    int32_t sum2 = 0;
    int32_t sum3 = 0;
    /* Common ReLU/FC inputs need no halfword zero-point subtraction. */
    if (zero_point == 0) {
        for (; size - index >= 4; index += 4) {
            const uint32_t packed = nn_read_u32(input + index);
            const uint32_t even = nn_arm_uxtb16(packed);
            const uint32_t odd = nn_arm_uxtb16(nn_arm_ror8(packed));
            sum0 = nn_arm_mac4(even, odd, weights[0] + index, sum0);
            sum1 = nn_arm_mac4(even, odd, weights[1] + index, sum1);
            sum2 = nn_arm_mac4(even, odd, weights[2] + index, sum2);
            sum3 = nn_arm_mac4(even, odd, weights[3] + index, sum3);
        }
    } else {
        const uint32_t offset = (uint32_t)zero_point * UINT32_C(0x00010001);
        for (; size - index >= 4; index += 4) {
            const uint32_t packed = nn_read_u32(input + index);
            const uint32_t even = nn_arm_ssub16(nn_arm_uxtb16(packed), offset);
            const uint32_t odd = nn_arm_ssub16(nn_arm_uxtb16(nn_arm_ror8(packed)), offset);
            sum0 = nn_arm_mac4(even, odd, weights[0] + index, sum0);
            sum1 = nn_arm_mac4(even, odd, weights[1] + index, sum1);
            sum2 = nn_arm_mac4(even, odd, weights[2] + index, sum2);
            sum3 = nn_arm_mac4(even, odd, weights[3] + index, sum3);
        }
    }
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

#endif
