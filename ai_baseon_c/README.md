# AI4C C runtime and model tools

本目录维护 C11 CNN/RNN/LSTM 推理库，以及已支持 PyTorch schema 的 CLI/GUI 转换工具。
请从[仓库 README](../README.md)开始。

- [安装、转换、量化公式、C 调用与 GUI](../docs/guide.md)
- [同模型 F446 实测与存储口径](../docs/benchmark_report.md)
- [目录、Git 边界和文件清理](../docs/project_layout.md)
- [贡献与验证](../CONTRIBUTING.md)
- [许可范围](../NOTICE.md)

只编译 `src/` 中的 canonical 实现；根目录 `.c/.h` 是历史兼容转发，不能把两套入口
同时链接。运行库不依赖 Python、MMU、RTOS 或堆分配。转换器仅支持明确的既有 schema，
不能根据权重名称推断任意 Python 计算图。
