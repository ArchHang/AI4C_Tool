#ifndef AI_BASEON_C_SRC_KERNELS_BACKENDS_AVX2_H_
#define AI_BASEON_C_SRC_KERNELS_BACKENDS_AVX2_H_

#include <stdint.h>

#include <immintrin.h>

static inline int32_t nn_sum_i32x8(__m256i value)
{
    __m128i sum = _mm_add_epi32(_mm256_castsi256_si128(value),
                              _mm256_extracti128_si256(value, 1));
    sum = _mm_hadd_epi32(sum, sum);
    sum = _mm_hadd_epi32(sum, sum);
    return _mm_cvtsi128_si32(sum);
}

static inline int32_t nn_dot_u8_s8(const uint8_t *input, const int8_t *weights,
                                 uint32_t size)
{
    uint32_t index = 0;
    __m256i accum = _mm256_setzero_si256();
    for (; size - index >= 16; index += 16) {
        const __m256i inputs = _mm256_cvtepu8_epi16(
            _mm_loadu_si128((const __m128i *)(const void *)(input + index)));
        const __m256i values = _mm256_cvtepi8_epi16(
            _mm_loadu_si128((const __m128i *)(const void *)(weights + index)));
        /* maddubs saturates adjacent int16 sums; widening preserves all products. */
        accum = _mm256_add_epi32(accum, _mm256_madd_epi16(inputs, values));
    }
    int32_t sum = nn_sum_i32x8(accum);
    for (; index < size; index++) {
        sum += (int32_t)input[index] * (int32_t)weights[index];
    }
    return sum;
}

static inline void nn_dot4_u8_s8_centered(const uint8_t *input,
                                        const int8_t *const weights[4], uint32_t size,
                                        int32_t zero_point, int32_t sums[4])
{
    const __m256i offset = _mm256_set1_epi16((int16_t)zero_point);
    uint32_t index = 0;
    __m256i accum0 = _mm256_setzero_si256();
    __m256i accum1 = accum0;
    __m256i accum2 = accum0;
    __m256i accum3 = accum0;
    for (; size - index >= 16; index += 16) {
        const __m256i inputs = _mm256_sub_epi16(_mm256_cvtepu8_epi16(
            _mm_loadu_si128((const __m128i *)(const void *)(input + index))), offset);
        const __m256i w0 = _mm256_cvtepi8_epi16(
            _mm_loadu_si128((const __m128i *)(const void *)(weights[0] + index)));
        const __m256i w1 = _mm256_cvtepi8_epi16(
            _mm_loadu_si128((const __m128i *)(const void *)(weights[1] + index)));
        const __m256i w2 = _mm256_cvtepi8_epi16(
            _mm_loadu_si128((const __m128i *)(const void *)(weights[2] + index)));
        const __m256i w3 = _mm256_cvtepi8_epi16(
            _mm_loadu_si128((const __m128i *)(const void *)(weights[3] + index)));
        accum0 = _mm256_add_epi32(accum0, _mm256_madd_epi16(inputs, w0));
        accum1 = _mm256_add_epi32(accum1, _mm256_madd_epi16(inputs, w1));
        accum2 = _mm256_add_epi32(accum2, _mm256_madd_epi16(inputs, w2));
        accum3 = _mm256_add_epi32(accum3, _mm256_madd_epi16(inputs, w3));
    }
    int32_t sum0 = nn_sum_i32x8(accum0);
    int32_t sum1 = nn_sum_i32x8(accum1);
    int32_t sum2 = nn_sum_i32x8(accum2);
    int32_t sum3 = nn_sum_i32x8(accum3);
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
