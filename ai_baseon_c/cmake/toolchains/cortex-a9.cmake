set(NN_CPU "cortex-a9" CACHE STRING "Compiler CPU target")
set(NN_ARCH_BITS "32" CACHE STRING "Arm execution state")
# This preset targets an A9 configured with its optional NEON unit.
set(NN_FPU "neon" CACHE STRING "FPU/NEON target, or none")
include("${CMAKE_CURRENT_LIST_DIR}/arm-gcc.cmake")
