"""Small, testable summaries for people; raw evidence stays in diagnostics."""

METHODS = {
    "线性头 SVD（CNN）": "svd",
    "重建引导通道剪枝（CNN，需校准）": "reconstruction",
    "Taylor + 重建（需校准）": "taylor",
    "FPGM 冗余通道": "fpgm",
    "L1 结构化（基线）": "structured",
    "非结构化置零（实验）": "unstructured",
}

PRECISIONS = {"INT8 · 量化权重": "int8", "FP32 · 浮点参考": "fp32", "INT8 + FP32 · 双版本": "both"}
SCALINGS = {"最大绝对值 · MaxAbs": "maxabs", "最小重建误差 · MSE": "mse"}
CACHE_POLICIES = {"常量直读 · 不增加权重缓存": "none", "SRAM 预打包 · Cortex-M DSP": "packed"}
ARCHITECTURES = {"自动识别": "auto", "CNN · 卷积网络": "cnn", "RNN · 循环网络": "rnn", "LSTM · 长短期记忆": "lstm"}
METHOD_HINTS = {
    "svd": "压缩 CNN 分类头；完整有效秩通常更稳妥。",
    "reconstruction": "用校准激活选择通道并补偿权重；仅支持当前 CNN。",
    "taylor": "根据校准输入评估单元敏感性，再重建分类头。",
    "fpgm": "根据权重相似性删除冗余单元，无需校准数据。",
    "structured": "按权重 L1 大小删除通道，用于建立对照。",
    "unstructured": "仅置零，不缩小稠密数组，也不会自动加速。",
}

HELP = {
    "precision": "INT8：逐输出通道量化权重，沿用本库混合整数推理；通常缩小常量数组。\n"
                 "FP32：保留浮点参数，适合作为数值参考。\nboth：同时导出两套，便于对拍，但体积更大。\n"
                 "改变精度不会改变稠密图的理论 MAC。",
    "maxabs": "maxabs · 最大绝对值\n每个输出通道 s=max(|w|)/127，q=clip(round(w/s),-127,127)。\n"
              "优点：计算快，覆盖该通道的权重范围。\n缺点：少数异常大值可能使大多数小权重的量化步长过粗。",
    "mse": "mse · 最小重建均方误差\n逐通道搜索 maxabs 范围的 50%～100%（101 个候选，含基线），"
           "选使 mean((w-q×s)²) 更小的尺度。\n优点：可降低权重重建误差。\n"
           "缺点：转换较慢，可能裁剪大权重；权重误差变小不保证分类准确率提高。",
    "pruning": "SVD：只压缩当前 CNN 无非线性的两层线性头；保留完整秩可近似保持浮点映射。\n"
               "重建引导：用下一层激活协方差评估删除代价，贪心删卷积通道并做岭回归补偿，再压缩分类头。仅当前 CNN。\n"
               "Taylor：利用独立校准输入估计敏感性，联动删除通道/隐藏单元，再重建分类头。\n"
               "FPGM：根据权重向量之间的距离删除冗余单元，无需数据。\n"
               "L1：按权重绝对值大小筛选，保留旧方法供对照。\n"
               "置零：不改变尺寸，本库稠密 C 内核不因此减少 MAC 或数组。",
    "amount": "结构化方法：每个可剪层请求删除此比例，至少保留 1 个单元；不是全网参数减少比例。\n"
              "SVD：只改变 CNN 线性头的秩，保留数受输出类别数限制；不剪卷积。\n"
              "重复执行会继续压缩当前候选。精度需用独立验证数据确认。",
    "calibration": "校准输入用于选择要保留的单元和重建权重，不是最终验收数据。\n"
                   "Taylor 与重建引导方法使用原模型输出，不使用文件中的类别标签；最多 128 个样本。\n"
                   "请从训练数据中准备具有代表性的 .npy/.npz 或 manifest.csv，"
                   "与验证集分开。输入格式、scale/zp 与验证数据相同。",
    "metrics": "MAC：batch=1 的稠密乘加次数，无法直接换算 MCU 耗时。\n"
               "常量负载：导出数组的字节数，不含代码、链接器布局，不等于 Flash。\n"
               "工作区下界：缓冲区成员之和，不含 ABI 对齐、栈和调用方 I/O。",
    "input": "uint8 输入按 x=(q-zero_point)×scale 还原。浮点数据应已完成训练所用预处理。\n"
             "scale 必须为有限正数，zero point 为 0～255 的整数；请勿凭默认值猜测训练预处理。",
    "compatibility": "自动读取 state_dict、常见 checkpoint 包装和本工程已知旧版 CNN 整模型。\n"
                     "旧模型仍通过受限白名单加载和计算图校验；未知 Python 类不会自动执行。\n"
                     "未知网络可查看张量，完整分析与 C 导出仍限于当前三种 schema。",
    "cache": "常量直读：权重留在只读常量区，不增加卷积权重缓存。\n"
             "SRAM 预打包：为合适的卷积预先扩展和排列权重，减少 Cortex-M4/M7 DSP 内核的重复读取与转换。\n"
             "代价：增加工作区 RAM，并有首次初始化成本；其他后端自动回退。\n"
             "速度取决于目标芯片与内存配置，必须分别实测；该选项不是所有平台的加速保证。",
    "structure": "结构图展示逻辑层；BN 在部署时折叠，融合层或 Flatten 不一定分配独立缓冲区。\n"
                 "理论 MAC 只统计乘加，不含偏置、激活、池化及量化开销。\n"
                 "未知 state_dict 不能仅凭权重形状可靠还原连接关系。",
    "reconstruction": "重建引导通道剪枝（仅本工程 CNN）\n"
                      "根据下一层校准激活的协方差，以组 OBS 方式贪心删除卷积通道，并用岭回归补偿中间层；"
                      "末端拟合原模型 logits，再用 SVD 合并连续线性分类头。\n"
                      "优点：真正缩小稠密形状，同时考虑通道相关性与输出重建。\n"
                      "限制：只拟合校准分布，免训练不代表无精度损失；必须使用独立验证集。",
}


def percent(value):
    return "未提供标签" if value is None else f"{100 * value:.2f}%"


def reduction(before, after):
    if before is None or after is None:
        return "未知"
    if before == 0:
        return "无变化" if after == 0 else "新增"
    ratio = 100 * (before - after) / before
    return f"减少 {ratio:.2f}%" if ratio >= 0 else f"增加 {-ratio:.2f}%"


def validation_view(result):
    if not result:
        return {"headline": "尚未验证当前候选", "tone": "pending", "reference": "—", "candidate": "—",
                "delta": "—", "agreement": "—", "scope": "选择独立验证数据后点击“比较原始与候选”。",
                "errors": ""}
    old, new = result.get("reference_accuracy"), result.get("candidate_accuracy")
    if old is None or new is None:
        headline, tone, delta = "已比较输出 · 任务准确率未知", "pending", "无标签"
    else:
        difference = (new - old) * 100
        delta = f"{difference:+.2f} 个百分点"
        headline = "此批样本准确率下降 · 请降低压缩强度或恢复模型" if difference < 0 else "此批样本准确率未下降 · 仍需独立验收"
        tone = "warning" if difference < 0 else "good"
    agreement = result.get("prediction_agreement")
    elapsed = result.get("elapsed_seconds")
    scope = f"{result.get('samples', 0):,} 个样本 · CPU / PyTorch FP32"
    if elapsed is not None:
        scope += f" · 比较用时 {elapsed:.2f} s（含准备，不是设备推理耗时）"
    context = result.get("evaluation_context", {})
    if context.get("calibration_overlap"):
        headline, tone = "验证数据与校准数据相同 · 仅供诊断", "warning"
    return {"headline": headline, "tone": tone, "reference": percent(old), "candidate": percent(new),
            "delta": delta, "agreement": "—" if agreement is None else percent(agreement), "scope": scope,
            "errors": f"输出误差：RMSE {result.get('rmse', 0):.4g} · 最大绝对误差 {result.get('max_abs_error', 0):.4g}"}


def change_description(change):
    method = change.get("method", "unknown")
    names = {value: key for key, value in METHODS.items()}
    title = names.get(method, method)
    old, new = change.get("weights_before"), change.get("weights_after")
    detail = f" · 矩阵权重 {old:,} → {new:,}" if old is not None and new is not None else ""
    if method == "svd" and "rank" in change:
        detail += f" · 保留秩 {change['rank']}"
        if change.get("exact_linear_rank_retained"):
            detail += "（完整有效秩，FP32 仅有舍入差异）"
        else:
            detail += "（截断秩，可能损失精度）"
    return title + detail
