/** @file quantization.h
 *  @brief Shared scalar quantization rules with a compile-time target fallback.
 */
#ifndef AI_BASEON_C_SRC_QUANTIZATION_H_
#define AI_BASEON_C_SRC_QUANTIZATION_H_

#include <math.h>
#include <stdint.h>

#include "../include/ai_baseon_c/nn_platform.h"

/* Bound both raw dot and zero-point correction, including int8 -128 weights. */
#define NN_MAX_MACS 32896u
#define NN_QUANT_BITS 28u

static inline int32_t nn_quant_metadata_is_valid(float scale, int32_t zero_point)
{
    return isfinite(scale) && scale > 0.0f && zero_point >= 0 && zero_point <= 255;
}

/* User-requested automatic target optimization: M4/M7 hard-float DSP only.
 * The GNU/Clang instruction is unavailable on soft/softfp ABI, non-7EM cores,
 * or a forced-portable backend. Other targets retain the standard C fallback.
 */
#if defined(__ARM_ARCH_7EM__) && defined(__ARM_FP) && ((__ARM_FP & 4) != 0) && \
    defined(__ARM_PCS_VFP) && (defined(__GNUC__) || defined(__clang__)) && NN_HAS_ARM_DSP
#define NN_ARM_HW_RNE_AVAILABLE 1
#else
#define NN_ARM_HW_RNE_AVAILABLE 0
#endif

#ifndef NN_USE_ARM_HW_RNE
#define NN_USE_ARM_HW_RNE NN_ARM_HW_RNE_AVAILABLE
#endif
#if NN_USE_ARM_HW_RNE != 0 && NN_USE_ARM_HW_RNE != 1
#error "NN_USE_ARM_HW_RNE must be 0 or 1"
#endif

/* The caller already rejected NaN and saturated values outside (-255,255).
 * VCVTR uses FPSCR rounding (deployment requires nearest-even), unlike VCVT's
 * truncation. The S32 result is in a VFP register; =t makes GCC/Clang move its
 * integer bits to the core register without a second numerical conversion.
 */
static inline int32_t nn_round_bounded_i32(float value)
{
#if NN_USE_ARM_HW_RNE && NN_ARM_HW_RNE_AVAILABLE
    int32_t result;
    __asm__ __volatile__("vcvtr.s32.f32 %0, %1" : "=t"(result) : "t"(value));
    return result;
#else
    return (int32_t)lrintf(value);
#endif
}

static inline uint8_t nn_quantize_u8_scalar(float value, float scale, int32_t zero_point)
{
    const float normalized = value / scale;
    /* Clamp before bounded rounding, including infinities. NaN maps to zero point. */
    if (isnan(normalized)) {
        return (uint8_t)zero_point;
    }
    if (normalized <= (float)-zero_point) {
        return 0;
    }
    if (normalized >= (float)(255 - zero_point)) {
        return 255;
    }
    return (uint8_t)(nn_round_bounded_i32(normalized) + zero_point);
}

/* Preserve the established recurrent rule: zero-point truncates toward zero.
 * Constant hidden states use magnitude/128; ordinary min/max quantization would
 * erase them when its range is zero. The CNN dynamic ReLU rule is separate.
 */
static inline void nn_recurrent_quant_params(float minimum, float maximum,
                                             float *scale, int32_t *zero_point)
{
    *scale = (maximum - minimum) / 255.0f;
    if (*scale < 1e-9f) {
        const float magnitude = fabsf(minimum) > fabsf(maximum)
                                    ? fabsf(minimum) : fabsf(maximum);
        if (magnitude > 1e-9f) {
            *scale = magnitude / 128.0f;
            *zero_point = minimum < 0.0f ? 255 : 0;
        } else {
            *scale = 1e-9f;
            *zero_point = 0;
        }
    } else {
        const float offset = -minimum / *scale;
        *zero_point = (int32_t)(offset < 0.0f ? 0.0f : (offset > 255.0f ? 255.0f : offset));
    }
}

#endif
