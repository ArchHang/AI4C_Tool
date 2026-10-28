#include "ai_baseon_c/nn_platform.h"
#include "kernels/nn_dot.h"

#ifdef NN_TEST_EXPECTED_BACKEND
_Static_assert(NN_SELECTED_BACKEND == NN_TEST_EXPECTED_BACKEND, "Incorrect backend selection");
#endif

int32_t nn_backend_compile_probe(const uint8_t *input, const int8_t *weights, uint32_t size)
{
    return nn_dot_u8_s8(input, weights, size);
}

void nn_backend_compile_probe4(const uint8_t *input, const int8_t *const weights[4],
                               uint32_t size, int32_t zero_point, int32_t sums[4])
{
    nn_dot4_u8_s8_centered(input, weights, size, zero_point, sums);
}

#if NN_HAS_NEON
void nn_backend_compile_probe_tile(const uint8_t *input, int8_t weight, int32_t sums[8])
{
    nn_neon_mac8_u8_s8(input, weight, sums);
}
#endif
