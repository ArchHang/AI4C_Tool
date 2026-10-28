# Default builds keep objects and binaries below build/make/<backend>.
# Explicit legacy targets copy a binary to its historical root path.
CC = gcc
AR = $(shell "$(CC)" -print-prog-name=ar)
PYTHON ?= python
BACKEND ?= auto
BUILD_DIR ?= build/make/$(BACKEND)
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -Werror
NN_CPPFLAGS = -Iinclude -Iexamples/models -Isrc
ARCH_FLAGS ?=
NN_ARCH_FLAGS =
FP_FLAGS ?= -ffp-contract=off -fno-fast-math
LDLIBS += -lm

ifeq ($(BACKEND),auto)
BACKEND_DEFINE = NN_BACKEND_AUTO
else ifeq ($(BACKEND),portable)
BACKEND_DEFINE = NN_BACKEND_PORTABLE
else ifeq ($(BACKEND),arm_dsp)
BACKEND_DEFINE = NN_BACKEND_ARM_DSP
else ifeq ($(BACKEND),neon)
BACKEND_DEFINE = NN_BACKEND_NEON
else ifeq ($(BACKEND),avx2)
BACKEND_DEFINE = NN_BACKEND_AVX2
NN_ARCH_FLAGS += -mavx2
else
$(error BACKEND must be auto, portable, arm_dsp, neon or avx2)
endif
NN_CPPFLAGS += -DNN_BACKEND=$(BACKEND_DEFINE)

SOURCES = src/cnn.c src/rnn.c src/lstm.c src/platform.c
OBJECTS = $(patsubst src/%.c,$(BUILD_DIR)/%.o,$(SOURCES))
TEST_NAMES = kernels fused rnn lstm backend
TEST_EXES = $(addprefix $(BUILD_DIR)/test_,$(addsuffix .exe,$(TEST_NAMES)))
TEST_OBJECTS = $(addprefix $(BUILD_DIR)/test_,$(addsuffix .o,$(TEST_NAMES)))
LIBRARY = $(BUILD_DIR)/libai_baseon_c.a
CONFIG = $(BUILD_DIR)/build_config.json

.PHONY: all library tests check check-models legacy clean FORCE
all: library tests
library: $(LIBRARY)
tests: $(TEST_EXES)

$(CONFIG): FORCE
	$(PYTHON) tools/configure_make.py --directory "$(BUILD_DIR)" --cc "$(CC)" --cflags="$(CFLAGS) $(FP_FLAGS)" --cppflags="$(CPPFLAGS) $(NN_CPPFLAGS)" --arch-flags="$(ARCH_FLAGS) $(NN_ARCH_FLAGS)"

$(BUILD_DIR)/%.o: src/%.c $(CONFIG)
	"$(CC)" $(CPPFLAGS) $(NN_CPPFLAGS) $(CFLAGS) $(FP_FLAGS) $(ARCH_FLAGS) $(NN_ARCH_FLAGS) -MMD -MP -c $< -o "$@"

$(BUILD_DIR)/test_%.o: tests/test_%.c $(CONFIG)
	"$(CC)" $(CPPFLAGS) $(NN_CPPFLAGS) $(CFLAGS) $(FP_FLAGS) $(ARCH_FLAGS) $(NN_ARCH_FLAGS) $(TEST_FLAGS) -MMD -MP -c $< -o "$@"

$(BUILD_DIR)/test_kernels.o: TEST_FLAGS = -DTEST_HAS_SCRATCH=1

$(LIBRARY): $(OBJECTS)
	"$(AR)" rcs "$@" $(OBJECTS)

$(BUILD_DIR)/test_%.exe: $(BUILD_DIR)/test_%.o $(LIBRARY)
	"$(CC)" $(CFLAGS) $(FP_FLAGS) $(ARCH_FLAGS) $(NN_ARCH_FLAGS) -o "$@" $< "$(LIBRARY)" $(LDLIBS)

# Keep existing scripts usable when they explicitly request historical paths.
.PHONY: libai_baseon_c.a test_rnn.exe test_lstm.exe test_kernels.exe test_fused.exe test_backend.exe
libai_baseon_c.a: $(LIBRARY)
	$(PYTHON) -c "import shutil; shutil.copy2(r'$(LIBRARY)', 'libai_baseon_c.a')"

$(addprefix test_,$(addsuffix .exe,$(TEST_NAMES))): test_%.exe: $(BUILD_DIR)/test_%.exe
	$(PYTHON) -c "import shutil; shutil.copy2(r'$<', '$@')"

legacy: libai_baseon_c.a $(addprefix test_,$(addsuffix .exe,$(TEST_NAMES)))

check: tests
	"$(BUILD_DIR)/test_backend.exe"
	"$(BUILD_DIR)/test_kernels.exe"
	"$(BUILD_DIR)/test_fused.exe"
	$(PYTHON) tools/test_robustness.py --exe "$(BUILD_DIR)/test_kernels.exe" --scratch

check-models: tests
	$(PYTHON) tools/verify_models.py --n 1000 --exe-dir "$(BUILD_DIR)"
	$(PYTHON) tools/verify_models.py --n 5000 --accuracy-only --exe-dir "$(BUILD_DIR)"

clean:
	$(PYTHON) tools/configure_make.py --directory "$(BUILD_DIR)" --clean

.SECONDARY: $(TEST_OBJECTS)

FORCE:

-include $(OBJECTS:.o=.d) $(TEST_OBJECTS:.o=.d)
