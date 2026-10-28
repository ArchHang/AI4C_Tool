# 工程目录与Git管理

## 1. 版本管理边界

仓库根为AI4C_Tool；原研究电脑上可以仍位于更大的工作区目录中。Git只管理维护中的推理库、
转换软件、测试、合成示例和公开文档。根`.gitignore`先排除工作区顶层资料，再显式放行维护
入口；tools和Python tests也有进一步的发布名单。提交前仍须检查实际index，不能仅依赖ignore。

| 目录 | 维护规则 |
|---|---|
| `ai_baseon_c/include/ai_baseon_c/` | 公共C接口；接口变更需说明兼容性 |
| `ai_baseon_c/src/` | 唯一canonical实现，含私有量化/平台内核 |
| `ai_baseon_c/tools/model_converter/` | 已知图解析、量化与C生成 |
| `ai_baseon_c/tools/model_studio/` | GUI、分析、验证和压缩服务；历史板测编排不属于公共应用 |
| `ai_baseon_c/tests/` | C数值测试与可移植Python测试 |
| `ai_baseon_c/examples/models/`、`ai_baseon_c/model/` | 明确列出的合成回归参数；不能用真实录音替换 |
| `docs/` | 一份使用手册、一份实验报告、目录规范和小型聚合图/JSON |
| `.github/workflows/` | 源码构建与合成模型CI，不访问真实开发板 |

根兼容C/H文件只是转发，应用不能与`src/`同时链接。`ai_baseon_c/docs/model_conversion.md`兼容页仅将
旧导出路径导向当前手册，不维护第二份教程。

## 2. 本地研究资料

原研究工作区可能还保有下列目录。这些资料不属于公开源码分发，也不会因为被Git忽略而删除。

| 本地资料 | 保留原因 |
|---|---|
| `datasets/`、真实WAV/频域特征 | 训练/验证原始输入，可能含个人声音 |
| `model/`、`models/`、实际说话人checkpoint | 训练成果、优化器状态与实验对照 |
| `results/` | 原始测量、固件、恢复镜像、关键日志和历史摘要 |
| `cnn_C/`、`demo/`、`python/` | 历史集成与训练入口；仍有依赖时保留，不混入当前库 |
| `environment_migration/`、工具链/压缩包 | 换机恢复及第三方依赖，不当普通缓存删除 |
| 本机IDE配置、`.tools/` | 本地开发配置与工具，不随源码提交 |
| `ai_baseon_c/docs/guide.md` | 原工作区内部研究长手册，保留原字节，未分发；当前公开手册位于根`docs/` |

这些目录保持原路径，避免打断冻结实验及脚本引用。目录整理优先明确归属、建立Git边界和
清理可证明无用的缓存，不为视觉整齐移动原始证据或删除唯一旧实现。

## 3. 新工作应该放哪里

| 产物 | 推荐位置 | 入Git |
|---|---|---|
| 临时对象、DLL、例程、导出包 | 根`build/`或库`build/` | 否 |
| 合成checkpoint、临时校准输入 | `build/`或`artifacts/` | 否 |
| 一次任务的草稿/辅助文件 | `.work/<任务名>/` | 否；完成后核依赖并清理 |
| 必须保留的本地实验原始记录 | `results/<任务名>/` | 否；建立输入/源码/结果身份 |
| 公开聚合结果与报告图 | `docs/assets/`中的明确文件 | 是；检查匿名化与元数据 |
| 新源码/测试 | 对应维护目录 | 是；先检查许可与测试 |

生成器和导出器使用新的输出目录，不静默覆盖已验收checkpoint或固件。GUI与CLI共享模型
服务，不复制成另一套只在某个实验目录维护的实现。

## 4. 提交与推送

```sh
git status --short
git diff
git add README.md docs ai_baseon_c .github .gitignore .gitattributes .editorconfig LICENSE NOTICE.md CONTRIBUTING.md AGENTS.md
git diff --cached --stat
git diff --cached --check
git ls-files
git commit -m "Describe the concrete change"
git push origin main
```

实际提交前逐项检查stage清单：不包含真实录音/PTH、原始串口二进制、机器快照、凭据、编译器、
生成可执行文件和未知大文件。合成NPZ只按精确路径放行。不要使用force push覆盖他人的历史。
大模型/数据需要独立的授权、版本和分发方式，不能临时解除ignore将整个研究目录推上去。

## 5. 固定清理步骤

1. 识别任务拥有的临时目录，检查是否仍被最终产物、复现命令或验收流程引用。
2. 保留源码、输入、最终模型/固件、关键日志、原始测量、依赖及恢复点。
3. 缓存若缺少对应源码或可能是唯一旧实现，先保留并调查。
4. Windows删除/移动使用原生PowerShell与`-LiteralPath`；递归前解析绝对路径并拒绝越界、
   reparse point和未知子项，不能按扩展名在整个工作区批量清扫。
5. 删除后验证相关入口和归档仍存在；报告实际删除范围，没有可删项时直接说明。

源码发行的核验应来自Git index或commit导出的干净树，不能把整个原工作区复制后称为
“干净克隆”，因为本地数据和PYTHONPATH可能掩盖缺失依赖。

## 6. 文档维护规则

README说明用途和最短上手步骤；guide解释安装、API与数值契约；benchmark_report记录对象、
条件、结果、局限和来源。报告图由公开聚合JSON生成，明确参数个数、模型字节、完整固件与
RAM口径。历史实测不因源文件规范换行或文档改版而回写旧摘要。

本仓库的提交方式参考[GitHub README说明](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes)
与[大文件管理说明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)。
Git是源码版本管理工具，不是整个研究电脑的备份。
