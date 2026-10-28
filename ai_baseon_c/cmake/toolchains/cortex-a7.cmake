set(NN_CPU "cortex-a7" CACHE STRING "Compiler CPU target")
set(NN_ARCH_BITS "32" CACHE STRING "Arm execution state")
set(NN_FPU "neon-vfpv4" CACHE STRING "FPU/NEON target, or none")
include("${CMAKE_CURRENT_LIST_DIR}/arm-gcc.cmake")
