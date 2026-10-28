/**
 * @file nn_platform.h
 * @brief Compile-time backend selection and immutable platform information.
 *
 * Selection uses compiler target features, never the host running the build.
 * Define NN_BACKEND or NN_FORCE_PORTABLE before including any library header.
 * All library translation units must use the same target and selection flags.
 */
#ifndef AI_BASEON_C_INCLUDE_NN_PLATFORM_H_
#define AI_BASEON_C_INCLUDE_NN_PLATFORM_H_

#include <stdint.h>

#define NN_BACKEND_AUTO       0
#define NN_BACKEND_PORTABLE   1
#define NN_BACKEND_ARM_DSP    2
#define NN_BACKEND_NEON       3
#define NN_BACKEND_AVX2       4

#ifndef NN_BACKEND
#define NN_BACKEND NN_BACKEND_AUTO
#endif

#ifndef NN_FORCE_PORTABLE
#define NN_FORCE_PORTABLE 0
#endif

#if (NN_BACKEND < NN_BACKEND_AUTO) || (NN_BACKEND > NN_BACKEND_AVX2)
#error "NN_BACKEND must be AUTO, PORTABLE, ARM_DSP, NEON, or AVX2"
#endif
#if (NN_FORCE_PORTABLE != 0) && (NN_FORCE_PORTABLE != 1)
#error "NN_FORCE_PORTABLE must be 0 or 1"
#endif

/* Legacy switches remain numeric: 1 requests a backend; 0 disables its auto selection. */
#if defined(NN_USE_ARM_DSP) && (NN_USE_ARM_DSP != 0)
#define NN_REQUEST_ARM_DSP 1
#else
#define NN_REQUEST_ARM_DSP 0
#endif
#if defined(NN_USE_NEON) && (NN_USE_NEON != 0)
#define NN_REQUEST_NEON 1
#else
#define NN_REQUEST_NEON 0
#endif
#if defined(NN_USE_AVX2) && (NN_USE_AVX2 != 0)
#define NN_REQUEST_AVX2 1
#else
#define NN_REQUEST_AVX2 0
#endif

#if (NN_REQUEST_ARM_DSP + NN_REQUEST_NEON + NN_REQUEST_AVX2) > 1
#error "Select only one integer dot backend"
#endif
#if NN_FORCE_PORTABLE && (NN_REQUEST_ARM_DSP || NN_REQUEST_NEON || NN_REQUEST_AVX2)
#error "NN_FORCE_PORTABLE conflicts with an enabled legacy backend switch"
#endif
#if NN_FORCE_PORTABLE && (NN_BACKEND != NN_BACKEND_AUTO) && \
    (NN_BACKEND != NN_BACKEND_PORTABLE)
#error "NN_FORCE_PORTABLE conflicts with NN_BACKEND"
#endif
#if NN_REQUEST_ARM_DSP && (NN_BACKEND != NN_BACKEND_AUTO) && \
    (NN_BACKEND != NN_BACKEND_ARM_DSP)
#error "NN_USE_ARM_DSP conflicts with NN_BACKEND"
#endif
#if NN_REQUEST_NEON && (NN_BACKEND != NN_BACKEND_AUTO) && (NN_BACKEND != NN_BACKEND_NEON)
#error "NN_USE_NEON conflicts with NN_BACKEND"
#endif
#if NN_REQUEST_AVX2 && (NN_BACKEND != NN_BACKEND_AUTO) && (NN_BACKEND != NN_BACKEND_AVX2)
#error "NN_USE_AVX2 conflicts with NN_BACKEND"
#endif

#if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__)
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#define NN_PLATFORM_LITTLE_ENDIAN 1
#else
#define NN_PLATFORM_LITTLE_ENDIAN 0
#endif
#elif defined(__arm__) || defined(__aarch64__)
#if defined(__ARM_BIG_ENDIAN) && (__ARM_BIG_ENDIAN != 0)
#define NN_PLATFORM_LITTLE_ENDIAN 0
#else
#define NN_PLATFORM_LITTLE_ENDIAN 1
#endif
#elif defined(_WIN32)
#define NN_PLATFORM_LITTLE_ENDIAN 1
#else
#define NN_PLATFORM_LITTLE_ENDIAN 0
#endif

#if defined(__ARM_NEON) && (__ARM_NEON != 0)
#define NN_TARGET_HAS_NEON 1
#else
#define NN_TARGET_HAS_NEON 0
#endif
#if defined(__ARM_FEATURE_DSP) && (__ARM_FEATURE_DSP != 0) && \
    defined(__ARM_FEATURE_SIMD32) && (__ARM_FEATURE_SIMD32 != 0) && \
    NN_PLATFORM_LITTLE_ENDIAN
#define NN_TARGET_HAS_ARM_DSP 1
#else
#define NN_TARGET_HAS_ARM_DSP 0
#endif
#if defined(__AVX2__) && (__AVX2__ != 0)
#define NN_TARGET_HAS_AVX2 1
#else
#define NN_TARGET_HAS_AVX2 0
#endif

#if NN_FORCE_PORTABLE
#define NN_SELECTED_BACKEND NN_BACKEND_PORTABLE
#elif NN_REQUEST_ARM_DSP
#define NN_SELECTED_BACKEND NN_BACKEND_ARM_DSP
#elif NN_REQUEST_NEON
#define NN_SELECTED_BACKEND NN_BACKEND_NEON
#elif NN_REQUEST_AVX2
#define NN_SELECTED_BACKEND NN_BACKEND_AVX2
#elif NN_BACKEND != NN_BACKEND_AUTO
#define NN_SELECTED_BACKEND NN_BACKEND
#elif NN_TARGET_HAS_NEON && (!defined(NN_USE_NEON) || (NN_USE_NEON != 0))
#define NN_SELECTED_BACKEND NN_BACKEND_NEON
#elif NN_TARGET_HAS_ARM_DSP && defined(__ARM_ARCH_PROFILE) && \
    (__ARM_ARCH_PROFILE == 'M') && (!defined(NN_USE_ARM_DSP) || (NN_USE_ARM_DSP != 0))
#define NN_SELECTED_BACKEND NN_BACKEND_ARM_DSP
#elif NN_TARGET_HAS_AVX2 && (!defined(NN_USE_AVX2) || (NN_USE_AVX2 != 0))
#define NN_SELECTED_BACKEND NN_BACKEND_AVX2
#else
#define NN_SELECTED_BACKEND NN_BACKEND_PORTABLE
#endif

#if (NN_SELECTED_BACKEND == NN_BACKEND_ARM_DSP) && !NN_TARGET_HAS_ARM_DSP
#error "ARM_DSP requires little-endian ARM DSP and SIMD32 target instructions"
#endif
#if (NN_SELECTED_BACKEND == NN_BACKEND_NEON) && !NN_TARGET_HAS_NEON
#error "NEON requires compiling with ARM Advanced SIMD enabled"
#endif
#if (NN_SELECTED_BACKEND == NN_BACKEND_AVX2) && !NN_TARGET_HAS_AVX2
#error "AVX2 requires compiling with AVX2 enabled"
#endif
#if (NN_SELECTED_BACKEND == NN_BACKEND_ARM_DSP) && defined(NN_USE_ARM_DSP) && \
    (NN_USE_ARM_DSP == 0)
#error "NN_USE_ARM_DSP=0 conflicts with NN_BACKEND=ARM_DSP"
#endif
#if (NN_SELECTED_BACKEND == NN_BACKEND_NEON) && defined(NN_USE_NEON) && (NN_USE_NEON == 0)
#error "NN_USE_NEON=0 conflicts with NN_BACKEND=NEON"
#endif
#if (NN_SELECTED_BACKEND == NN_BACKEND_AVX2) && defined(NN_USE_AVX2) && (NN_USE_AVX2 == 0)
#error "NN_USE_AVX2=0 conflicts with NN_BACKEND=AVX2"
#endif

#define NN_HAS_ARM_DSP (NN_SELECTED_BACKEND == NN_BACKEND_ARM_DSP)
#define NN_HAS_NEON (NN_SELECTED_BACKEND == NN_BACKEND_NEON)
#define NN_HAS_AVX2 (NN_SELECTED_BACKEND == NN_BACKEND_AVX2)
#define NN_HAS_VECTOR_DOT (NN_HAS_ARM_DSP || NN_HAS_NEON || NN_HAS_AVX2)

typedef enum nn_backend {
    NN_BACKEND_ID_PORTABLE = NN_BACKEND_PORTABLE,
    NN_BACKEND_ID_ARM_DSP = NN_BACKEND_ARM_DSP,
    NN_BACKEND_ID_NEON = NN_BACKEND_NEON,
    NN_BACKEND_ID_AVX2 = NN_BACKEND_AVX2
} nn_backend_t;

typedef struct nn_platform_info {
    const char *backend_name;
    const char *architecture_name;
    nn_backend_t backend;
    uint32_t integer_vector_bits;
    uint32_t required_buffer_alignment;
} nn_platform_info_t;

#ifdef __cplusplus
extern "C" {
#endif

/** @return Backend compiled into the library; no runtime probing is performed. */
nn_backend_t nn_get_backend(void);

/** @return Static backend name: portable, arm_dsp, neon, or avx2. */
const char *nn_get_backend_name(void);

/** @return Immutable information owned by the library; never free the pointer. */
const nn_platform_info_t *nn_get_platform_info(void);

#ifdef __cplusplus
}
#endif

#endif
