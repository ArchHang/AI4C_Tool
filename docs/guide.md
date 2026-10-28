# AI4C Tool 使用手册

本手册说明公开源码的安装、转换、量化、调用和验证。实验结果集中在
[基准报告](benchmark_report.md)，目录与本地研究资料的边界集中在
[工程管理](project_layout.md)。不需要原作者的磁盘路径、语音数据或STM32工程也可以运行
下面的合成模型示例。

- [安装](#installation)
- [支持的模型](#supported-models)
- [生成示例并转换](#conversion)
- [GUI操作与压缩](#gui)
- [量化方法和误差](#quantization)
- [C调用和工作区](#c-api)
- [平台选择与编译](#platforms)
- [验证和问题定位](#validation)

<a id="installation"></a>
## 1. 安装

### C运行库

需要C11编译器。推荐CMake 3.21+，也提供GNU Make入口。运行库没有Python、MMU、RTOS或
动态堆分配依赖。开发测试中的Python辅助检查不属于目标设备运行时。

```sh
git clone https://github.com/ArchHang/AI4C_Tool.git
cd AI4C_Tool
cmake -S ai_baseon_c -B build/portable -DNN_BACKEND=portable -DNN_BUILD_TESTS=ON
cmake --build build/portable --parallel
ctest --test-dir build/portable --output-on-failure
```

### Python转换器和GUI

需要Python 3.10+、NumPy和PyTorch 2.6+。Tkinter是Python的标准接口，但图形库由Python发行版
或操作系统提供；不是执行`pip install tk`即可补齐。Linux通常需要系统的`python3-tk`；
没有桌面显示的机器可以使用CLI。

建议使用独立虚拟环境：

```sh
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r ai_baseon_c/requirements.txt
```

转换和示例只需要CPU版PyTorch。需要控制下载体积时，可以先按
[PyTorch官方安装选择器](https://pytorch.org/get-started/locally/)选择适合本机的CPU wheel，
再安装上述requirements。CUDA训练环境是可选项，不能把某台电脑的Conda环境名当作依赖规范。

开发验证和公开报告绘图使用额外依赖：

```sh
python -m pip install -r ai_baseon_c/requirements-dev.txt
python -c "import sys, torch, numpy, tkinter; print(sys.executable); print(torch.__version__); print(numpy.__version__)"
```

本项目实际研究环境使用过Python3.10.20、NumPy2.2.6和PyTorch2.12.1；最低版本声明不表示
所有版本组合均已穷举验证。GUI使用本机Tk，C数值测试还需要PATH中可用的GCC兼容编译器。

<a id="supported-models"></a>
## 2. 支持的模型与文件格式

| schema | 计算图契约 | 主要限制 |
|---|---|---|
| `cnn_voice_v1` | 三段Conv1d→BN→ReLU→MaxPool2，再Flatten→FC1→eval Dropout→FC2 | 已知5抽头结构；FC1后没有额外ReLU；维度必须满足池化和flatten |
| `rnn_tanh_v1` | 单层、单向tanh RNN，最终隐藏态接线性分类头 | 固定T/I/H，不能用relu RNN或堆叠/双向图替代 |
| `lstm_ifgo_v1` | 单层、单向LSTM，最终隐藏态接线性分类头 | PyTorch门序i/f/g/o；固定T/I/H；cell保留浮点 |

`.pth`可能是裸`state_dict`、含`state_dict`与metadata的checkpoint，或已知旧CNN整模型。
裸参数通常不包含完整前向语义，例如RNN的非线性类型、序列长度或额外激活函数；必须由
正确的schema声明和输入尺寸补齐。工具不会从张量名称推断任意Python控制流。

建议保存`state_dict + metadata`。CNN记录`schema`和`input_length`；循环模型记录
`schema`和`seq_length`。维度从参数形状校验，BN epsilon等非权重语义也应准确保存。
可运行示例由下一节生成器提供，不要求手工拼接所有键名。

未知图在GUI中可以进入张量清单分析，但不能因此宣称已支持C导出。分组、深度、膨胀专用卷积、
任意二维网络、ONNX/TFLite、动态控制流和QAT模型自动导出尚不属于现有转换范围。

加载优先使用受限的weights-only路径；已知旧CNN通过静态类型白名单兼容，不以关闭所有加载
限制来接纳未知整模型。加载成功仍需要图校验和数值验证。

<a id="conversion"></a>
## 3. 生成示例并转换

### 合成示例

```sh
python ai_baseon_c/tools/generate_example_models.py --output-dir build/example_models
```

该命令在CPU上用固定随机种子创建`cnn.pth`、`rnn.pth`、`lstm.pth`和examples.json。
模型未经训练，不包含语音，也不评估识别准确率。已有目录会被拒绝，避免覆盖重要checkpoint。

### CLI导出

```sh
python ai_baseon_c/tools/convert_model.py build/example_models/cnn.pth --output-dir build/cnn_export --precision int8 --weight-scaling mse --bundle-runtime
python ai_baseon_c/tools/convert_model.py build/example_models/rnn.pth --output-dir build/rnn_export --precision int8 --bundle-runtime
python ai_baseon_c/tools/convert_model.py build/example_models/lstm.pth --output-dir build/lstm_export --precision both --bundle-runtime
```

| 参数 | 含义 |
|---|---|
| `--architecture auto/cnn/rnn/lstm` | 自动识别已支持metadata，或显式声明受支持语义 |
| `--input-length` | CNN输入宽度；应与训练及flatten一致 |
| `--seq-length` | 循环序列长度；裸state_dict通常需要补充 |
| `--precision fp32/int8/both` | 选择部署路径；both用于对拍，会同时保留两套参数和缓冲区 |
| `--input-scale`、`--input-zero-point` | 输入反量化契约，默认1/255和0 |
| `--weight-scaling maxabs/mse` | 每输出行的权重尺度选择 |
| `--cache-policy none/packed` | 可选小卷积权重展开，增加workspace RAM，默认关闭 |
| `--bundle-runtime` | 把需要的C运行库一起导出，便于独立编译 |

CNN输入按连续CHW（带batch时NCHW）排列；循环输入按连续`[time][feature]`排列。元素总数相同
不代表布局相同。生成器中的CNN是1×64输入，不是研究报告的32×64说话人模型。

### 导出文件与独立编译

| 文件 | 用途 |
|---|---|
| `model.c`、`model.h` | 固定模型参数、图入口、workspace类型与容量常量 |
| `manifest.json` | 图维度、量化、参数字节、工作区成员、来源摘要 |
| `example.c` | 最小调用例子；默认输入只验证调用链 |
| `README.txt` | 与本次精度、组件相匹配的编译命令 |
| `runtime/` | `--bundle-runtime`导出的必要算子和头文件 |

在对应导出目录执行README.txt中的命令。例如CNN包可用：

```sh
cc -std=c11 -O3 -ffp-contract=off -fno-fast-math -Iruntime model.c example.c runtime/cnn.c runtime/platform.c -lm -o model_example
```

循环模型还需要相应的`runtime/rnn.c`或`runtime/lstm.c`。Windows可将输出名改为
`model_example.exe`。不要只换model.c而留下旧版本runtime；manifest与参数包共同构成部署对象。

<a id="gui"></a>
## 4. GUI操作与模型压缩

```sh
python ai_baseon_c/tools/model_studio_gui.py build/example_models/cnn.pth
```

Windows启动器依次使用`MODEL_STUDIO_PYTHON`指定的解释器、仓库根`.venv`、PATH中的Python，
不会要求某个私人Conda环境名。解释器需要具备前述依赖，启动器不自动安装包。

1. 导入checkpoint，确认schema与输入尺寸。未知图只能查看张量，不能导出或猜测连接。
2. 在结构与逐层分析中查看shape、参数、MAC及逻辑图。BN running statistics是buffer，
   不计入可训练参数；MAC不包含全部非线性和量化成本。
3. 选择FP32/INT8及MaxAbs/MSE，更新存储估计。参数个数、参数字节、workspace下界和芯片
   ELF实测是不同指标。
4. 根据目标选择压缩方法；涉及数据的方法使用独立校准集，保留原始模型作为本次基线。
5. 用代表性验证集比较准确率、逐类结果与预测变化，再另存候选并转换为C。
6. 修改checkpoint、前端、输入scale或压缩政策后，重新执行数值与目标设备验证。

| 方法 | 实现思路 | 必须注意 |
|---|---|---|
| L1结构化剪枝 | 按权重重要性物理移除通道/隐藏单元，切片依赖层 | 参数、MAC可减少；精度需要实际验证 |
| FPGM | 依据滤波器几何冗余选择候选，并物理缩维 | 不是对所有任务都优于L1 |
| Taylor＋重建 | 用校准数据估计敏感度并做局部重建 | 校准必须符合训练前端，不能用测试集反复选择 |
| 线性头SVD | 合并无中间非线性的FC链后做低秩分解 | 只在相应schema的线性链条件成立时适用 |
| 非结构化置零 | 将小权重置零 | 当前稠密内核不因零值自动获得稀疏加速 |

GUI中的验证是PyTorch比较，不等于已通过INT8 C或MCU验收；也没有自动GPU微调和GUI内一键烧录。
需要标签的识别准确率与无标签的预测一致性分别显示。有标签NPZ使用`inputs`和`labels`；
CNN形状为`[N,C,L]`，循环为`[N,T,I]`，输入预处理必须与checkpoint一致。

<a id="quantization"></a>
## 5. PTQ、定点实现与量化损失

### 5.1 基本表示

当前采用训练后量化PTQ，不修改训练图来模拟量化梯度。令实数值为x，尺度为s>0、整数零点为z：

$$
x\approx s_x(q_x-z_x),\qquad
q_x=\operatorname{clip}\bigl(\operatorname{RNE}(x/s_x)+z_x,0,255\bigr).
$$

RNE表示最近偶数舍入。对于非负输入或ReLU输出，常见特例为$z_x=0$、$s_x=x_{max}/255$。
负值激活需要仿射范围；本实现的CNN全连接量化会把实数零纳入普通范围，避免只截断零点造成
同号向量整体饱和。NaN、极值、常量向量和极小尺度由明确的校验/饱和/兜底规则处理。

权重逐输出行对称量化，8位有符号权重的常用范围为[-127,127]：

$$
s_{w,c}=\frac{\max_j|w_{c,j}|}{127},\qquad
q_{w,c,j}=\operatorname{clip}\bigl(\operatorname{RNE}(w_{c,j}/s_{w,c}),-127,127\bigr).
$$

分母不是255；255对应无符号激活的跨度。权重零点为0。不同输出行使用不同尺度，减少单个
大幅值输出行对其他行的精度影响。

### 5.2 BN融合与整数累加

推理BN可写为$y=\gamma(u-\mu)/\sqrt{\sigma^2+\epsilon}+\beta$。
若$u=Wx+b$，则在实数意义上：

$$
W'_c=\frac{\gamma_c}{\sqrt{\sigma_c^2+\epsilon}}W_c,\qquad
b'_c=\frac{\gamma_c(b_c-\mu_c)}{\sqrt{\sigma_c^2+\epsilon}}+\beta_c.
$$

先融合、再量化权重；BN epsilon必须与训练一致。浮点折叠改变计算顺序，不能把代数等价
直接当成逐位一致，应比较生成C与对应PyTorch定义。

对于权重零点0的卷积或线性输出：

$$
acc_c=\sum_j(q_{x,j}-z_x)q_{w,c,j},\qquad
y_c\approx s_xs_{w,c}\,acc_c+b_c.
$$

预计算权重行和可将零点修正移出热循环：

$$
acc_c=\sum_j q_{x,j}q_{w,c,j}-z_x\sum_jq_{w,c,j}.
$$

偏置的量纲必须明确。本库保留FP32 bias，并在相应输出/重定标路径中使用；不能未经推导
直接把$b/s_w$加入整数累加。若要使用accumulator域INT32 bias，通常应按$s_xs_w$缩放，
但这不是当前所有混合路径都采用的存储方式。

### 5.3 Q28重定标

本小节限定CNN动态ReLU输出、输出零点$z_y=0$。对输出尺度$s_y$，令
$a_c=s_xs_{w,c}/s_y$、$d_c=b_c/s_y$：

$$
q_{y,c}\approx a_c acc_c+d_c,\qquad
A_c=\operatorname{RNE}(a_c2^{28}),\quad D_c=\operatorname{RNE}(d_c2^{28}),
$$

$$
q_{y,c}=\operatorname{clip}\!\left(
\operatorname{RNE}\left(\frac{A_cacc_c+D_c}{2^{28}}\right),0,255\right).
$$

CNN动态ReLU输出在获取范围后准备这些系数。累加和系数乘法分别受INT32/INT64保守上界约束，
必须在转换/乘法前检查，而不是溢出后再检查；不满足定点安全界时使用既有安全路径。
Q28近似的是系数，仍存在权重、激活和系数误差，不能据此宣称任务准确率无损。
一般仿射输出还需计入输出零点$z_y$；循环状态和FC路径不能直接套用本节的ReLU特例。

### 5.4 循环网络和混合精度

RNN/LSTM矩阵权重使用INT8，隐藏态按时间步动态量化。tanh、sigmoid、尺度计算以及LSTM cell
仍有FP32运算，最终logit也是浮点；INT8是混合推理路径，不等于每一步都是整数。

循环隐藏态沿用确定的零点截断规则与常量状态幅值兜底，不能随意改成CNN的规则；跨主机和MCU
的非线性契约也需要一致。微小浮点误差在长递归和离散量化边界上可能改变后续状态。

### 5.5 怎样减少损失

- 使用逐输出行尺度，先正确融合BN，再处理权重。
- 可选MSE尺度搜索比较若干裁剪阈值，减少权重重建误差；它不保证分类分数优于MaxAbs。
- 校准与验证覆盖真实输入范围、常量、同号、极值和序列变化，避免前端/布局不匹配。
- 比较FP32→INT8的分类变化、逐类召回和logit差异，不能只看平均MSE。
- 压缩、训练和量化政策只在训练/validation上选择，固定后再评价独立测试。

参数个数不会因INT8而减少；矩阵字节通常降至1/4，但bias、scale、行和及算子代码另计。
完整固件Flash、workspace RAM和初始化成本需要分别测量，见基准报告。

<a id="c-api"></a>
## 6. C调用与内存契约

优先使用导出模型的高层接口。INT8-only导出包的一般用法如下，实际名称和常量以model.h为准：

```c
#include "model.h"

static ai_model_workspace_t workspace;
static uint8_t input[AI_MODEL_INPUT_ELEMENTS];
static float logits[AI_MODEL_OUTPUT_ELEMENTS];

int prepare_model(void)
{
    return ai_model_init(&workspace, sizeof(workspace));
}

int infer_frame(void)
{
    /* Fill input with the preprocessing/layout used to train this model. */
    return ai_model_run_int8(input, AI_MODEL_INPUT_ELEMENTS,
                            logits, AI_MODEL_OUTPUT_ELEMENTS, &workspace);
}
```

检查每次返回值；0为成功。workspace由应用拥有，静态对象可满足该类型自然对齐；自定义
原始存储需查询`ai_model_workspace_size()`与`ai_model_workspace_alignment()`并满足目标ABI。
不允许以错误容量、未对齐指针或不支持的重叠缓冲区调用。

同一模型不同任务并行时使用独立workspace；只读权重可共享。循环模型的高层run以固定序列为
一次推理并重置初始状态，不是流式状态延续接口。输入、输出和workspace的生命周期由调用方
保证；DMA/中断/RTOS共享缓冲区的同步也由应用负责。

只编译`src/`的canonical源码，或使用导出runtime。库根的兼容转发文件不能与canonical源码
同时编译，否则会产生重复定义。选择`both`会保留两精度参数与缓冲区；正式小体积部署选择
需要的单一精度。

<a id="platforms"></a>
## 7. 平台选择与编译

| 目标 | 编译时可能选择的后端 | 声明范围 |
|---|---|---|
| 不具备已支持SIMD特征的目标 | portable C | 提供回退，不自动推断板级资源 |
| Cortex-M4/M7且启用相应DSP目标特征 | Arm DSP | 实际FPU/ABI仍需编译器与芯片匹配 |
| Cortex-A7/A9/A35/A53的合适NEON编译目标 | NEON | CPU名本身不等于启用NEON或完成实板认证 |
| 支持AVX2的桌面目标 | AVX2 | 需要宿主CPU/OS支持；不能在未知CPU上强制执行 |

```sh
cmake -S ai_baseon_c -B build/auto -DNN_BACKEND=auto
cmake -S ai_baseon_c -B build/m4 -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/cortex-m4.cmake
```

交叉编译需要工具链可用，并设置实际CPU/FPU/ABI；各toolchain文件提供默认值。设备端不必有
MMU。DMA/cache/DTCM放置、时钟与链接地址由板级工程负责，库不会自动检测外设或移动链接段。

自动分派依据编译目标特征宏，不是运行时检测所有芯片。可显式选择portable用于对拍。
M4小卷积输入窗口复用和可选权重展开都有shape条件及额外栈/RAM代价；M4/M7硬浮点DSP配置下
有条件使用硬件RNE。`NN_M4_PATCH_MAX=64`和`NN_USE_ARM_HW_RNE=0`可恢复保守窗口/取整选择，
但任何改动都应针对自己的模型重新检查数值、栈和速度。

本项目F446有实际整网测量；其他ARM目标的交叉编译/指令语义覆盖不等于对应设备的吞吐、功耗
或长期可靠性验证。不要将架构理论向量宽度直接当成端到端加速倍数。

<a id="validation"></a>
## 8. 验证与问题定位

安装开发依赖后，从仓库根执行：

```sh
python ai_baseon_c/tools/check_repository.py --build-system cmake
python ai_baseon_c/tools/check_repository.py --build-system make --require-gui
```

可通过`--compiler`、`--cmake`、`--make`指定工具。CMake检查默认使用Ninja，需要Ninja位于PATH；
也可用`--generator "Unix Makefiles"`（Linux）或`--generator "MinGW Makefiles"`（对应Windows工具链）
指定实际可用的生成器，或直接选择`--build-system make`。
检查在临时树构建C测试、生成三种合成checkpoint、导出并编译运行独立C示例，再执行明确的
可移植Python测试名单。它不使用私有语音集、外部RTOS工程或串口。

开发检查需要可导入Tk模块，默认不要求显示器；`--require-gui`要求实际显示环境并执行GUI控制器检查。Linux CI可在Xvfb中
运行。检查报告分别记录测试数、错误和跳过；两项依赖私有语音的原始测试不在此发行门槛，
不能说它们已通过。小合成RNN/LSTM fixture只证明相应人工任务与数值回归。

部署新模型时再分层检查：

1. 验证图、维度、权重、布局及输入scale，确认与训练前端相同。
2. 在主机比较PyTorch与FP32 C，再比较INT8与独立整数/状态参考；检查边界和重复调用。
3. 固定模型与政策，在目标设备比较对应主机C输出并测完整forward。
4. 记录编译器、时钟、cache、计时边界、P95、workspace、实际栈和固件Flash。
5. 对真实标签统计总体准确率、逐类召回、宏召回、误接纳/误拒绝和独立录音数量。

出现误差时先定位第一层/第一时间步的分歧，区分浮点归约/非线性、量化参数与状态传播。
不为让测试通过而单独放宽容差或静默替换参考。没有标签时，只能报告数值差异和预测一致性。

提交问题时附模型schema、输入shape、工具版本、最小合成反例和错误信息；不要上传秘密、
未经授权的真实录音或本机凭据。公开实验的可复算聚合结果见[基准报告](benchmark_report.md)。
