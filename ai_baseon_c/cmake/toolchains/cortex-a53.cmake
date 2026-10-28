set(NN_CPU "cortex-a53" CACHE STRING "Compiler CPU target")
set(NN_ARCH_BITS "64" CACHE STRING "Arm execution state")
set(NN_FPU "neon-fp-armv8" CACHE STRING "FPU/NEON target for AArch32")
include("${CMAKE_CURRENT_LIST_DIR}/arm-gcc.cmake")
