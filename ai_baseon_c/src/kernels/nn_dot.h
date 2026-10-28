/**
 * @file nn_dot.h
 * @brief Internal exact uint8 x int8 dots selected by compiler target features.
 *
 * Buffers need no extra alignment or padding. The caller must validate size and
 * bound the absolute int32 sum; centered dots require a uint8 zero point (0..255).
 */
#ifndef AI_BASEON_C_NN_DOT_H_
#define AI_BASEON_C_NN_DOT_H_

#include "../../include/ai_baseon_c/nn_platform.h"

#if NN_HAS_NEON
#include "backends/neon.h"
#elif NN_HAS_ARM_DSP
#include "backends/arm_dsp.h"
#elif NN_HAS_AVX2
#include "backends/avx2.h"
#else
#include "backends/portable.h"
#endif

static inline void nn_dot4_u8_s8(const uint8_t *input, const int8_t *const weights[4],
                               uint32_t size, int32_t sums[4])
{
    nn_dot4_u8_s8_centered(input, weights, size, 0, sums);
}

#endif
