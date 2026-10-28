/* Deterministic binary32 activation math: deterministic_float32_v1.
 *
 * Mathematical construction: exp(x) = 2^n exp(r), |r| <= log(2)/2;
 * exp(r) uses a degree-7 Taylor polynomial. Small tanh uses its degree-13
 * odd Taylor polynomial, avoiding cancellation in (1-exp(-2*x))/(1+exp(-2*x)).
 * No third-party implementation or fitted coefficient table was copied.
 *
 * Contract: IEEE binary32, round-to-nearest-even, gradual underflow, no FMA,
 * no fast-math. Compile with -ffp-contract=off. Extended evaluation is rejected.
 * NaNs become canonical quiet NaN, tanh preserves signed zero, sigmoid(+-0)=.5.
 * Infinities saturate naturally. This is not a correctly-rounded libm claim.
 */
#ifndef NN_ACTIVATIONS_H
#define NN_ACTIVATIONS_H

#include <float.h>
#include <stdint.h>

#if defined(__FAST_MATH__)
#error "Deterministic activations require strict floating-point evaluation"
#endif
#if FLT_RADIX != 2 || FLT_MANT_DIG != 24 || FLT_MAX_EXP != 128 || FLT_EVAL_METHOD != 0
#error "Deterministic activations require binary32 without excess precision"
#endif

typedef union nn_float_bits {
    float value;
    uint32_t bits;
} nn_float_bits_t;

static inline float nn_from_bits(uint32_t bits)
{
    nn_float_bits_t item;
    item.bits = bits;
    return item.value;
}

static inline uint32_t nn_to_bits(float value)
{
    nn_float_bits_t item;
    item.value = value;
    return item.bits;
}

static inline float nn_exp_f32(float value)
{
    const uint32_t bits = nn_to_bits(value);
    const uint32_t magnitude = bits & UINT32_C(0x7fffffff);
    if (magnitude > UINT32_C(0x7f800000)) {
        return nn_from_bits(UINT32_C(0x7fc00000));
    }
    if (value >= 88.72283935546875f) {
        return nn_from_bits(UINT32_C(0x7f800000));
    }
    if (value <= -104.0f) {
        return 0.0f;
    }

    /* Split log(2): the leading part has few enough significant bits that
     * n*hi is exact for every n in the supported exponent range. */
    const float scaled = value * 1.4426950408889634f;
    const int32_t exponent = (int32_t)(scaled + (scaled < 0.0f ? -0.5f : 0.5f));
    const float reduced_hi = value - (float)exponent * 0.693145751953125f;
    const float reduced = reduced_hi - (float)exponent * 0.000001428606765330187f;
    float polynomial = 0.0001984126984126984f;
    polynomial = 0.001388888888888889f + reduced * polynomial;
    polynomial = 0.008333333333333333f + reduced * polynomial;
    polynomial = 0.041666666666666664f + reduced * polynomial;
    polynomial = 0.16666666666666666f + reduced * polynomial;
    polynomial = 0.5f + reduced * polynomial;
    polynomial = 1.0f + reduced * polynomial;
    polynomial = 1.0f + reduced * polynomial;
    if (exponent > 127) {
        return (polynomial * nn_from_bits(UINT32_C(0x7f000000))) * 2.0f;
    }
    if (exponent < -126) {
        const uint32_t power = (uint32_t)(exponent + 24 + 127) << 23;
        return (polynomial * nn_from_bits(power)) * 0x1p-24f;
    }
    const uint32_t power = (uint32_t)(exponent + 127) << 23;
    return polynomial * nn_from_bits(power);
}

static inline float nn_tanh_f32(float value)
{
    const uint32_t bits = nn_to_bits(value);
    const uint32_t magnitude = bits & UINT32_C(0x7fffffff);
    if (magnitude > UINT32_C(0x7f800000)) {
        return nn_from_bits(UINT32_C(0x7fc00000));
    }
    if (magnitude == 0) {
        return value;
    }
    const float absolute = nn_from_bits(magnitude);
    float result;
    if (absolute <= 0.5f) {
        const float square = absolute * absolute;
        float polynomial = 0.003592128036572481f;
        polynomial = -0.008863235529902197f + square * polynomial;
        polynomial = 0.021869488536155203f + square * polynomial;
        polynomial = -0.05396825396825397f + square * polynomial;
        polynomial = 0.13333333333333333f + square * polynomial;
        polynomial = -0.33333333333333333f + square * polynomial;
        result = absolute + (absolute * square) * polynomial;
    } else if (absolute >= 10.0f) {
        result = 1.0f;
    } else {
        const float exponential = nn_exp_f32(-2.0f * absolute);
        const float numerator = 1.0f - exponential;
        const float denominator = 1.0f + exponential;
        result = numerator / denominator;
    }
    return (bits & UINT32_C(0x80000000)) != 0 ? -result : result;
}

static inline float nn_sigmoid_f32(float value)
{
    const uint32_t bits = nn_to_bits(value);
    const uint32_t magnitude = bits & UINT32_C(0x7fffffff);
    if (magnitude > UINT32_C(0x7f800000)) {
        return nn_from_bits(UINT32_C(0x7fc00000));
    }
    /* Preserve the LSTM's original reciprocal formulation and evaluation order.
     * nn_exp_f32 explicitly handles overflow, so large negative inputs yield 0. */
    const float exponential = nn_exp_f32(-value);
    const float denominator = 1.0f + exponential;
    return 1.0f / denominator;
}

#endif
