# Copyright and licensing notices

The existing C implementation carries the BSD 3-Clause notice reproduced in
[LICENSE](LICENSE). Its original copyright notices remain in these files:

- `ai_baseon_c/src/cnn.c`
- `ai_baseon_c/src/rnn.c`
- `ai_baseon_c/src/lstm.c`
- `ai_baseon_c/include/ai_baseon_c/cnn.h`
- `ai_baseon_c/include/ai_baseon_c/rnn.h`
- `ai_baseon_c/include/ai_baseon_c/lstm.h`

The original notices identify VeriSilicon Holdings Co., Ltd., with 2025 or 2026
dates as stated in each file. Publishing this repository does not remove those
notices or transfer ownership.

The Python tools, other source additions, documentation, and generated examples
do not yet have a separately confirmed, repository-wide license grant. The
reproduced C license must not be interpreted as silently relicensing every
unannotated file. Check the applicable file notices and obtain permission for
uses that require an additional grant. A future license decision should be an
explicit, reviewed change.

PyTorch, NumPy, Tk, compilers, and build tools are external dependencies and are
not vendored here. Their own licenses apply to their distributions. The
repository does not include STM32 HAL/CMSIS, an RTOS SDK, CMSIS-NN, or NMSIS-NN
source trees. Documentation references to these projects describe background or
implementation ideas; they do not indicate an included dependency or endorsement.

The small RNN/LSTM parameter fixtures and the example checkpoint generator serve
synthetic correctness demonstrations. The generator's random, untrained models
have no speaker-recognition accuracy claim. Real recordings, speaker-model
checkpoints, firmware images, and machine-environment snapshots are excluded
from the source publication.
