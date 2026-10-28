# AI4C Tool

[![Portable checks](https://github.com/ArchHang/AI4C_Tool/actions/workflows/portable-check.yml/badge.svg)](https://github.com/ArchHang/AI4C_Tool/actions/workflows/portable-check.yml)

面向资源受限设备的 **C11 神经网络推理库**与 **PyTorch → C 模型转换工作台**。
支持已定义的 CNN、单层 tanh RNN、单层 LSTM，提供 FP32 与混合 INT8 推理。
C 运行时不依赖 MMU、Python、操作系统或动态堆分配；激活与工作区由调用方管理。

## 功能与边界

| 组件 | 提供的能力 |
|---|---|
| C 推理库 | CNN/RNN/LSTM、固定工作区、参数校验、FP32/INT8 数值路径 |
| 平台后端 | 编译目标自动选择 portable、Arm DSP、NEON 或 AVX2 |
| 转换器 | 已支持 schema 的 `.pth` → `model.c/.h`、manifest、调用示例与自包含 runtime |
| 桌面 GUI | 结构/参数/MAC 分析，量化设置，L1/FPGM/Taylor 重建与线性头 SVD，候选保存和导出 |
| 验证 | 合成模型的 PyTorch/C 对拍、整数参考、边界测试、GUI 控制器测试与可移植构建检查 |

**转换器不是任意 PyTorch 计算图编译器。** 未支持的二维、堆叠、双向或自定义图需要扩展
schema、算子和参考测试。MAC 是理论工作量，不是芯片实测时间；压缩也不保证准确率无损。
支持范围见[使用手册](docs/guide.md#supported-models)。

## 快速开始：只使用 C 库

需要 C11 编译器和 CMake 3.21+。从仓库根目录执行：

```sh
cmake -S ai_baseon_c -B build/portable -DNN_BACKEND=portable -DNN_BUILD_TESTS=ON
cmake --build build/portable --parallel
ctest --test-dir build/portable --output-on-failure
```

C 库本身不需要 PyTorch。上面的 CTest 不等于完整模型准确率验收；它的检查范围和额外
转换验证入口见[验证流程](docs/guide.md#validation)。

## 快速开始：转换器与 GUI

需要 Python 3.10+、NumPy、PyTorch 2.6+；GUI 还需要可用的 Tk。CPU 版 PyTorch 即可，
GPU 不是转换和示例的前提。建议新建虚拟环境，按[安装说明](docs/guide.md#installation)
安装依赖。

生成固定随机种子的**未训练合成模型**，不需要语音数据或下载权重：

```sh
python ai_baseon_c/tools/generate_example_models.py --output-dir build/example_models
python ai_baseon_c/tools/convert_model.py build/example_models/cnn.pth \
  --output-dir build/cnn_export --precision int8 --weight-scaling mse --bundle-runtime
python ai_baseon_c/tools/model_studio_gui.py build/example_models/cnn.pth
```

PowerShell 请把多行命令写为一行，或将续行符改为反引号。Windows 也可双击
`ai_baseon_c/start_model_studio.cmd`；解释器选择规则见手册。合成模型只演示转换与正确性，
没有说话人识别准确率承诺。

## 开发检查

```sh
python -m pip install -r ai_baseon_c/requirements-dev.txt
python ai_baseon_c/tools/check_repository.py --build-system cmake
```

默认CMake检查使用Ninja，需要它位于PATH；也可选`--build-system make`，或按手册指定其他生成器。
有显示环境时增加 `--require-gui`。
该检查使用临时构建和合成输入，不打开串口、不烧录开发板。依赖缺失、执行失败与跳过
应分别处理，不能把缺少私有数据的测试说成已通过。

## 实测报告

[F446 实测报告](docs/benchmark_report.md)区分参数个数、权重字节、模型常量、完整固件
Flash 和 RAM。每种架构的 FP32/INT8 使用同一 checkpoint；旧模型计时不混入主表。
报告同时保留宏召回、少数类退步和测试集限制。

![同一最终模型的参数和存储](docs/assets/model-resources.png)

公开仓库只包含脱敏的聚合结果和必要图片。真实语音、已训练说话人权重、原始串口记录、
板卡 SDK/RTOS 工程、工具链和本机环境快照不随源码分发；公开聚合结果不能代替完整原始审计。

## 仓库结构与文档

```text
ai_baseon_c/
  include/ai_baseon_c/   公共 C 接口
  src/                  C 实现、共享量化与平台内核
  tools/                转换器、GUI、合成示例与开发检查
  tests/                C 与 Python 测试
  examples/models/      合成回归参数头文件
  model/                小型合成回归参数

docs/
  guide.md              安装、量化、API、GUI、平台和验证
  benchmark_report.md   实验方法、结果和局限
  project_layout.md     目录归属、Git 边界和清理规则
```

- [使用手册](docs/guide.md)
- [目录管理](docs/project_layout.md)
- [贡献与验证要求](CONTRIBUTING.md)
- 问题反馈：[GitHub Issues](https://github.com/ArchHang/AI4C_Tool/issues)

## 许可与归属

保留现有 C 文件的 BSD 3-Clause 版权和许可声明，见 [LICENSE](LICENSE)。其他新增组件尚未
确认统一的授权范围；请按 [NOTICE](NOTICE.md) 和各文件声明判断，不将已有 C 许可理解为
对整个仓库的自动重新授权。外部依赖遵循各自许可证。
