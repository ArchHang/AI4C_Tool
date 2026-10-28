#include "ai_baseon_c/nn_platform.h"

#if NN_HAS_ARM_DSP
#define NN_PLATFORM_BACKEND_NAME "arm_dsp"
#define NN_PLATFORM_VECTOR_BITS 32U
#elif NN_HAS_NEON
#define NN_PLATFORM_BACKEND_NAME "neon"
#define NN_PLATFORM_VECTOR_BITS 128U
#elif NN_HAS_AVX2
#define NN_PLATFORM_BACKEND_NAME "avx2"
#define NN_PLATFORM_VECTOR_BITS 256U
#else
#define NN_PLATFORM_BACKEND_NAME "portable"
#define NN_PLATFORM_VECTOR_BITS 0U
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#define NN_PLATFORM_ARCHITECTURE_NAME "aarch64"
#elif defined(__ARM_ARCH_PROFILE) && (__ARM_ARCH_PROFILE == 'M')
#if defined(__ARM_ARCH_7EM__)
#define NN_PLATFORM_ARCHITECTURE_NAME "armv7e-m"
#else
#define NN_PLATFORM_ARCHITECTURE_NAME "arm-m-profile"
#endif
#elif defined(__arm__) || defined(_M_ARM)
#define NN_PLATFORM_ARCHITECTURE_NAME "aarch32"
#elif defined(__x86_64__) || defined(_M_X64)
#define NN_PLATFORM_ARCHITECTURE_NAME "x86_64"
#elif defined(__i386__) || defined(_M_IX86)
#define NN_PLATFORM_ARCHITECTURE_NAME "x86"
#else
#define NN_PLATFORM_ARCHITECTURE_NAME "generic"
#endif

static const nn_platform_info_t platform_info = {
    NN_PLATFORM_BACKEND_NAME,
    NN_PLATFORM_ARCHITECTURE_NAME,
    (nn_backend_t)NN_SELECTED_BACKEND,
    NN_PLATFORM_VECTOR_BITS,
    1U
};

nn_backend_t nn_get_backend(void)
{
    return platform_info.backend;
}

const char *nn_get_backend_name(void)
{
    return platform_info.backend_name;
}

const nn_platform_info_t *nn_get_platform_info(void)
{
    return &platform_info;
}
