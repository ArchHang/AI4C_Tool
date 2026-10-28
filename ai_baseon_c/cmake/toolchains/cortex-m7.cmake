set(NN_CPU "cortex-m7" CACHE STRING "Compiler CPU target")
set(NN_ARCH_BITS "32" CACHE STRING "Arm execution state")
# Single precision is a safe default for M7 implementations with an SP FPU.
set(NN_FPU "fpv5-sp-d16" CACHE STRING "FPU/NEON target, or none")
include("${CMAKE_CURRENT_LIST_DIR}/arm-gcc.cmake")
