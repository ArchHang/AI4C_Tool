"""Tk desktop frontend. Workers never access Tk; the UI owns all session changes."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
from pathlib import Path
import queue
import tkinter as tk
from tkinter import filedialog, font, messagebox, ttk
import traceback

from model_converter.schema import ConversionError

from .analysis import analyze_session, load_session
from .compression import prune_session, save_checkpoint
from .services import export_session, write_analysis_report
from .validation import evaluate_sessions
from .help_ui import help_icon
from .presentation import (ARCHITECTURES, CACHE_POLICIES, HELP, METHODS, METHOD_HINTS,
                           PRECISIONS, SCALINGS, change_description, reduction, validation_view)


BACKGROUND = "#f4f7fb"
INK = "#182b45"
ACCENT = "#087f8c"
SIDEBAR = "#172b46"


def number(value):
    if value is None:
        return "未知"
    return f"{value:,}" if isinstance(value, int) else f"{value:.4g}"


def size_text(value):
    if value is None:
        return "未知"
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:,.1f} {unit}" if unit != "B" else f"{value:,} B"
        value /= 1024


def shape_text(value):
    return " × ".join(map(str, value)) if value else "—"


class ModelStudio:
    def __init__(self, root):
        self.root = root
        self.original = self.current = None
        self.analysis = self.baseline = None
        self.validation = None
        self.import_identity = None
        self.busy = False
        self.closed = False
        self.events = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="model-studio")
        self.controls = []
        self.help_tips = []
        self.log_events = []
        self.diagnostic_events = []
        self.show_diagnostics = tk.BooleanVar(root, False)
        self.show_activity = tk.BooleanVar(root, False)
        self.activity_title = tk.StringVar(root, "查看操作记录")
        self.model_path = tk.StringVar(root)
        self.architecture = tk.StringVar(root, "auto")
        self.input_length = tk.StringVar(root)
        self.seq_length = tk.StringVar(root)
        self.trusted = tk.BooleanVar(root, False)
        self.precision = tk.StringVar(root, "int8")
        self.scaling = tk.StringVar(root, "maxabs")
        self.cache_policy = tk.StringVar(root, "none")
        self.prefix = tk.StringVar(root, "ai_model")
        self.scale = tk.StringVar(root, str(1 / 255))
        self.zero_point = tk.StringVar(root, "0")
        self.pruning = tk.StringVar(root, "线性头 SVD（CNN）")
        self.amount = tk.StringVar(root, "20")
        self.dataset = tk.StringVar(root)
        self.samples = tk.StringVar(root, "256")
        self.calibration = tk.StringVar(root)
        self.calibration_samples = tk.StringVar(root, "128")
        self.status = tk.StringVar(root, "就绪 · 导入模型开始分析")
        self.quality = tk.StringVar(root, "质量尚未评估")
        self._build()
        for variable in (self.scale, self.zero_point, self.dataset, self.samples):
            variable.trace_add("write", self._invalidate_validation)
        for variable in (self.precision, self.scaling, self.cache_policy):
            variable.trace_add("write", self._mark_estimate_pending)
        self.pruning.trace_add("write", self._sync_method_options)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.poll_id = self.root.after(80, self._poll)

    def _build(self):
        self.root.title("Model Studio · 模型分析与 C 部署")
        self.root.geometry("1420x900")
        self.root.minsize(1120, 740)
        self.root.configure(background=BACKGROUND)
        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont"):
            font.nametofont(name).configure(family="Microsoft YaHei UI", size=10)
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background=BACKGROUND)
        style.configure("TLabel", background=BACKGROUND, foreground=INK)
        style.configure("TButton", padding=(10, 7), background="white", foreground=INK,
                        bordercolor="#d9e2ee", lightcolor="white", darkcolor="white")
        style.map("TButton", background=[("active", "#e9f2f6"), ("disabled", "#e6ebf1")],
                  foreground=[("disabled", "#8491a3")])
        style.configure("Accent.TButton", background=ACCENT, foreground="white")
        style.map("Accent.TButton", background=[("active", "#096572"), ("disabled", "#88a8ac")])
        style.configure("Treeview", rowheight=31, background="white", fieldbackground="white", foreground=INK,
                        borderwidth=0)
        style.configure("Treeview.Heading", background="#eaf0f7", foreground=INK,
                        font=("Microsoft YaHei UI", 10, "bold"), padding=7, relief="flat")
        style.map("Treeview", background=[("selected", "#daeeef")], foreground=[("selected", INK)])
        style.configure("TNotebook", background=BACKGROUND, borderwidth=0)
        style.configure("TNotebook.Tab", padding=(16, 9))
        style.map("TNotebook.Tab", background=[("selected", "white"), ("!selected", "#e5ebf3")],
                  foreground=[("selected", ACCENT), ("!selected", "#586b83")])
        outer = ttk.Frame(self.root)
        outer.pack(fill="both", expand=True)
        self._build_sidebar(outer)

        main = ttk.Frame(outer, padding=(22, 17))
        main.pack(side="left", fill="both", expand=True)
        head = ttk.Frame(main)
        head.pack(fill="x")
        ttk.Label(head, text="模型工作台", font=("Microsoft YaHei UI", 21, "bold")).pack(side="left")
        ttk.Label(head, textvariable=self.quality, foreground=ACCENT, wraplength=310,
                  justify="right").pack(side="right")
        self.subtitle = ttk.Label(main, text="导入模型，查看结构和资源，再比较优化候选。", wraplength=950)
        self.subtitle.pack(fill="x", pady=(5, 15))
        cards = ttk.Frame(main)
        cards.pack(fill="x", pady=(0, 14))
        self.card_values = {}
        for index, (key, title, hint) in enumerate((
            ("total_parameters", "可训练参数", "不含批归一化运行统计"),
            ("macs", "理论乘加 / 次", "MAC · batch=1，非实测耗时"),
            ("deployed_parameter_bytes", "导出常量负载", "不含程序代码与链接器布局"),
            ("workspace_bytes_lower_bound", "工作区下界", "精确大小依赖目标 ABI"),
        )):
            cards.columnconfigure(index, weight=1, uniform="metrics")
            card = tk.Frame(cards, bg="white", highlightbackground="#dce4ef", highlightthickness=1,
                            padx=13, pady=12)
            card.grid(row=0, column=index, sticky="nsew", padx=(0, 8 if index < 3 else 0))
            tk.Label(card, text=title, bg="white", fg="#52657d", anchor="w").pack(fill="x")
            value = tk.StringVar(self.root, "—")
            self.card_values[key] = value
            tk.Label(card, textvariable=value, bg="white", fg=INK, font=("Segoe UI", 21, "bold"),
                     anchor="w").pack(fill="x", pady=5)
            tk.Label(card, text=hint, bg="white", fg="#75859a", anchor="w", font=("Microsoft YaHei UI", 8),
                     wraplength=160).pack(fill="x")
        self.tabs = ttk.Notebook(main)
        self.tabs.pack(fill="both", expand=True)
        graph_tab, layers_tab, tensors_tab, compare_tab, verify_tab = [ttk.Frame(self.tabs) for _ in range(5)]
        for tab, title in zip((graph_tab, layers_tab, tensors_tab, compare_tab, verify_tab),
                              ("模型结构", "逐层分析", "张量参数", "优化对比", "验证结果")):
            self.tabs.add(tab, text=title)
        self.graph_tab, self.verify_tab = graph_tab, verify_tab
        self.graph = tk.Canvas(graph_tab, bg="white", highlightthickness=0, width=440)
        graph_scroll = ttk.Scrollbar(graph_tab, orient="vertical", command=self.graph.yview)
        self.graph.configure(yscrollcommand=graph_scroll.set)
        self.graph.pack(side="left", fill="both", expand=True)
        graph_scroll.pack(side="left", fill="y")
        self.details = self._text(graph_tab, width=30)
        self.details.pack(side="right", fill="both", expand=True, padx=(10, 0))
        self.graph.bind("<MouseWheel>", lambda event: self.graph.yview_scroll(int(-event.delta / 120), "units"))
        self.layers = self._tree(layers_tab, (
            ("name", "层", 130), ("op", "算子", 150), ("input", "输入", 125),
            ("output", "输出", 125), ("params", "参数", 90), ("macs", "MAC", 110),
            ("share", "MAC占比", 85), ("weights", "常量负载", 110)))
        self.layers.bind("<<TreeviewSelect>>", self.select_layer)
        self.layer_details = self._text(layers_tab, height=5)
        self.layer_details.pack(fill="x", pady=7)
        self.tensors = self._tree(tensors_tab, (
            ("name", "张量", 220), ("shape", "维度", 140), ("dtype", "类型", 110),
            ("elements", "元素", 90), ("bytes", "字节", 100), ("zeros", "零比例", 85),
            ("min", "最小", 100), ("max", "最大", 100)))
        self.tensor_details = self._text(tensors_tab, height=6)
        self.tensor_details.pack(fill="x", pady=7)
        self.tensors.bind("<<TreeviewSelect>>", self.select_tensor)
        self.compare_headline = tk.StringVar(self.root, "尚未生成优化候选")
        comparison_head = ttk.Frame(compare_tab, padding=(10, 9))
        comparison_head.pack(fill="x")
        ttk.Label(comparison_head, textvariable=self.compare_headline,
                  font=("Microsoft YaHei UI", 12, "bold")).pack(side="left")
        button = ttk.Button(comparison_head, text="验证当前候选 →", command=lambda: self.tabs.select(self.verify_tab))
        button.pack(side="right")
        self.controls.append((button, "normal"))
        benefits = ttk.Frame(compare_tab, padding=(10, 0))
        benefits.pack(fill="x")
        self.benefit_values = {}
        self.benefit_labels = {}
        for index, (key, title) in enumerate((("deployed_parameter_bytes", "常量负载变化"),
                                             ("macs", "理论乘加变化"),
                                             ("workspace_bytes_lower_bound", "工作区下界变化"))):
            benefits.columnconfigure(index, weight=1, uniform="benefits")
            card = tk.Frame(benefits, bg="#e8f3f3", padx=11, pady=8)
            card.grid(row=0, column=index, sticky="nsew", padx=(0, 7 if index < 2 else 0))
            tk.Label(card, text=title, bg="#e8f3f3", fg="#617c88", font=("Microsoft YaHei UI", 9)).pack(anchor="w")
            variable = tk.StringVar(self.root, "—")
            self.benefit_values[key] = variable
            label = tk.Label(card, textvariable=variable, bg="#e8f3f3", fg=ACCENT,
                             font=("Microsoft YaHei UI", 15, "bold"))
            label.pack(anchor="w", pady=(4, 0))
            self.benefit_labels[key] = label
        self.compare_quality = tk.StringVar(self.root, "质量尚未验证")
        self.compare_quality_label = ttk.Label(compare_tab, textvariable=self.compare_quality,
                                               wraplength=850, padding=(12, 9))
        self.compare_quality_label.pack(fill="x")
        self.compare_table = self._tree(compare_tab, (
            ("metric", "资源指标", 140), ("original", "导入基线", 140),
            ("candidate", "当前候选", 140), ("change", "变化", 160)), height=4, expand=False)
        for column in self.compare_table["columns"]:
            self.compare_table.column(column, stretch=True)
        note = ttk.Frame(compare_tab, padding=(10, 6))
        note.pack(fill="x")
        self.deployment_note = tk.StringVar(self.root, "静态资源估算 · 设备速度、Flash 与整机 RAM 需编译和实测。")
        ttk.Label(note, textvariable=self.deployment_note, foreground="#65778c", wraplength=790).pack(side="left")
        self._help(note, HELP["metrics"])
        self.compare = self._text(compare_tab, height=3)
        self.compare.pack(fill="both", expand=True, padx=10, pady=(0, 9))
        self.compare_tab = compare_tab
        ttk.Label(verify_tab, text="独立数据验证", font=("Microsoft YaHei UI", 13, "bold"),
                  padding=(10, 12)).pack(anchor="w")
        data = ttk.Frame(verify_tab, padding=10)
        data.pack(fill="x")
        ttk.Label(data, text="验证数据 (.npy / .npz / manifest.csv)").grid(row=0, column=0, sticky="w")
        entry = ttk.Entry(data, textvariable=self.dataset)
        entry.grid(row=1, column=0, sticky="ew", padx=(0, 8))
        self.controls.append((entry, "normal"))
        data.columnconfigure(0, weight=1)
        button = ttk.Button(data, text="选择…", command=self.choose_dataset)
        button.grid(row=1, column=1)
        self.controls.append((button, "normal"))
        ttk.Label(data, text="最多样本").grid(row=0, column=2, padx=12)
        entry = ttk.Entry(data, textvariable=self.samples, width=8)
        entry.grid(row=1, column=2, padx=12)
        self.controls.append((entry, "normal"))
        button = ttk.Button(data, text="运行 FP32 对比", command=self.evaluate, style="Accent.TButton")
        button.grid(row=1, column=3)
        self.controls.append((button, "normal"))
        self.verdict = tk.StringVar(self.root)
        self.verdict_label = ttk.Label(verify_tab, textvariable=self.verdict,
                                      font=("Microsoft YaHei UI", 12, "bold"), padding=10, wraplength=900)
        self.verdict_label.pack(fill="x")
        quality_cards = ttk.Frame(verify_tab, padding=(10, 0))
        quality_cards.pack(fill="x")
        self.validation_values = {}
        for key, title in (("reference", "原始准确率"), ("candidate", "候选准确率"),
                           ("delta", "准确率变化"), ("agreement", "预测一致率")):
            item = tk.Frame(quality_cards, bg="white", padx=10, pady=10)
            item.pack(side="left", fill="both", expand=True, padx=(0, 6))
            tk.Label(item, text=title, bg="white", fg="#65778c").pack(anchor="w")
            variable = tk.StringVar(self.root, "—")
            self.validation_values[key] = variable
            tk.Label(item, textvariable=variable, bg="white", fg=INK,
                     font=("Microsoft YaHei UI", 15, "bold")).pack(anchor="w", pady=(7, 0))
        self.validation_scope = tk.StringVar(self.root)
        self.validation_errors = tk.StringVar(self.root)
        ttk.Label(verify_tab, textvariable=self.validation_scope, padding=(10, 8), wraplength=900).pack(fill="x")
        ttk.Label(verify_tab, textvariable=self.validation_errors, padding=(10, 0), foreground="#65778c").pack(fill="x")
        ttk.Label(verify_tab, text="验证范围：CPU 上的 PyTorch FP32 原模型 / 候选。INT8 C 输出与实板速度需另行验收。",
                  padding=10, foreground="#65778c").pack(fill="x")
        history_head = ttk.Frame(verify_tab, padding=(10, 3))
        history_head.pack(fill="x")
        ttk.Checkbutton(history_head, textvariable=self.activity_title, variable=self.show_activity,
                        command=self._toggle_activity).pack(side="left")
        ttk.Checkbutton(history_head, text="展开技术诊断", variable=self.show_diagnostics,
                        command=self._toggle_diagnostics).pack(side="right")
        self.activity_panel = ttk.Frame(verify_tab)
        self.log = self._text(self.activity_panel, height=4)
        self.log.pack(fill="x", padx=10, pady=(0, 5))
        self.diagnostics = self._text(self.activity_panel, height=7)
        self._render_validation()
        footer = ttk.Frame(main)
        footer.pack(side="bottom", fill="x", pady=(9, 0), before=self.tabs)
        self.progress = ttk.Progressbar(footer, mode="indeterminate", length=110)
        self.progress.pack(side="right")
        ttk.Label(footer, textvariable=self.status).pack(side="left")
        self._set_text(self.details, "选择文件后点击“导入并分析”。\n\n已支持三种项目图：\n• CNN voice v1\n• 单层 tanh RNN\n• 单层 IFGO LSTM\n\n其他 state_dict 可查看张量，完整结构与计算量需要已知图语义。")
        self._draw_empty("导入模型开始分析")
        # Tk does not bubble wheel events through parent frames. Give every
        # sidebar child a shared bindtag so entries/buttons also scroll it.
        wheel_tag = "StudioSidebarWheel"
        self.root.bind_class(wheel_tag, "<MouseWheel>", self._scroll_sidebar)
        def tag_children(widget):
            widget.bindtags((wheel_tag,) + widget.bindtags())
            for child in widget.winfo_children():
                tag_children(child)
        tag_children(self.side_host)
        self._select_step("import")

    def _build_sidebar(self, outer):
        """Only the current step scrolls; its primary actions stay visible."""
        self.side_host = tk.Frame(outer, bg=SIDEBAR, width=305)
        self.side_host.pack(side="left", fill="y")
        self.side_host.pack_propagate(False)
        brand = tk.Frame(self.side_host, bg=SIDEBAR, padx=18, pady=18)
        brand.pack(fill="x")
        tk.Label(brand, text="MODEL STUDIO", font=("Segoe UI", 18, "bold"),
                 bg=SIDEBAR, fg="white", anchor="w").pack(fill="x")
        tk.Label(brand, text="从模型到嵌入式 C", bg=SIDEBAR, fg="#99cbd2",
                 anchor="w").pack(fill="x", pady=(4, 0))
        navigation = tk.Frame(self.side_host, bg=SIDEBAR, padx=12)
        navigation.pack(fill="x", pady=(0, 12))
        self.step_buttons = {}
        for key, label in (("import", "01   导入与分析"), ("optimize", "02   压缩与验证"),
                           ("export", "03   量化与导出")):
            button = tk.Button(navigation, text=label, command=lambda step=key: self._select_step(step),
                               anchor="w", relief="flat", borderwidth=0, padx=14, pady=9,
                               bg=SIDEBAR, fg="#b7c9da", activebackground="#29435f", activeforeground="white",
                               font=("Microsoft YaHei UI", 10), cursor="hand2")
            button.pack(fill="x", pady=2)
            self.step_buttons[key] = button
            self.controls.append((button, "normal"))

        actions = tk.Frame(self.side_host, bg="#11243c", padx=16, pady=14)
        actions.pack(side="bottom", fill="x")
        self.step_action_hint = tk.StringVar(self.root)
        tk.Label(actions, textvariable=self.step_action_hint, bg="#11243c", fg="#a7c2d0",
                 font=("Microsoft YaHei UI", 9), anchor="w", justify="left", wraplength=265).pack(fill="x", pady=(0, 8))
        self.side_primary = self._button(actions, "导入并分析", self.import_model, accent=True)
        self.side_secondary = self._button(actions, "查看模型结构", lambda: self.tabs.select(self.graph_tab))
        tk.Frame(self.side_host, bg="#304359", height=1).pack(side="bottom", fill="x")
        canvas_host = tk.Frame(self.side_host, bg=SIDEBAR)
        canvas_host.pack(fill="both", expand=True)
        self.side_canvas = tk.Canvas(canvas_host, bg=SIDEBAR, highlightthickness=0, width=285)
        side_scroll = ttk.Scrollbar(canvas_host, orient="vertical", command=self.side_canvas.yview)
        self.side_canvas.configure(yscrollcommand=side_scroll.set)
        side_scroll.pack(side="right", fill="y")
        self.side_canvas.pack(side="left", fill="both", expand=True)
        side = tk.Frame(self.side_canvas, bg=SIDEBAR, padx=16, pady=4)
        side_window = self.side_canvas.create_window((0, 0), window=side, anchor="nw")
        side.bind("<Configure>", lambda event: self.side_canvas.configure(scrollregion=self.side_canvas.bbox("all")))
        self.side_canvas.bind("<Configure>", lambda event: self.side_canvas.itemconfigure(side_window, width=event.width))
        self.side_pages = {key: tk.Frame(side, bg=SIDEBAR) for key in self.step_buttons}
        self.advanced_sections = {}
        source = self.side_pages["import"]
        self._label(source, "模型文件")
        self._entry(source, self.model_path)
        self._button(source, "选择 .pth / .pt 文件…", self.choose_model)
        tk.Label(source, text="支持本工程 CNN、RNN、LSTM；\n未知网络可先查看张量。", bg=SIDEBAR,
                 fg="#a5bdce", justify="left", anchor="w", wraplength=258).pack(fill="x", pady=(8, 14))
        compatibility = tk.Frame(source, bg=SIDEBAR)
        compatibility.pack(fill="x", pady=(0, 10))
        tk.Label(compatibility, text="旧版 CNN 自动兼容", bg=SIDEBAR, fg="#9ed4cd").pack(side="left")
        self._help(compatibility, HELP["compatibility"], sidebar=True)
        import_options = self._foldout(source, "import", "高级导入设置")
        self._field(import_options, "模型类型", self.architecture, ARCHITECTURES)
        self._field(import_options, "CNN 输入宽度（留空自动）", self.input_length)
        self._field(import_options, "序列长度（留空自动）", self.seq_length)

        optimize = self.side_pages["optimize"]
        self._field(optimize, "压缩 / 剪枝方法", self.pruning, tuple(METHODS), help_text=HELP["pruning"])
        self.method_hint = tk.StringVar(self.root)
        tk.Label(optimize, textvariable=self.method_hint, bg=SIDEBAR, fg="#a5bdce", justify="left",
                 anchor="w", wraplength=253, font=("Microsoft YaHei UI", 9)).pack(fill="x", pady=(5, 9))
        self.calibration_anchor = self._field(optimize, "请求压缩比例 (%)", self.amount, help_text=HELP["amount"])
        self.calibration_panel = tk.Frame(optimize, bg=SIDEBAR)
        self._field(self.calibration_panel, "独立校准输入", self.calibration, help_text=HELP["calibration"])
        self._button(self.calibration_panel, "选择校准数据…", self.choose_calibration)
        calibration_options = self._foldout(self.calibration_panel, "calibration", "校准与输入预处理")
        self._field(calibration_options, "最多校准样本 (1～128)", self.calibration_samples)
        self._input_fields(calibration_options)
        tk.Label(optimize, text="生成候选后，在右侧查看收益，\n再用独立数据验证精度。", bg=SIDEBAR,
                 fg="#a5bdce", justify="left", anchor="w", font=("Microsoft YaHei UI", 9)).pack(fill="x", pady=13)

        deploy = self.side_pages["export"]
        self._field(deploy, "导出精度", self.precision, PRECISIONS, help_text=HELP["precision"])
        self._field(deploy, "权重量化方法", self.scaling, SCALINGS, help_text=lambda: HELP[self.scaling.get()])
        self._field(deploy, "卷积权重读取策略", self.cache_policy, CACHE_POLICIES, help_text=HELP["cache"])
        self._button(deploy, "更新资源估算", self.reanalyze)
        self._field(deploy, "C 符号前缀", self.prefix)
        input_options = self._foldout(deploy, "export", "输入预处理设置")
        self._input_fields(input_options)
        self._button(deploy, "保存分析 JSON / CSV…", self.save_report)
        tk.Label(deploy, text="常量负载 ≠ 最终 Flash\n工作区下界 ≠ 整机 RAM", justify="left",
                 bg=SIDEBAR, fg="#a5bdce", font=("Microsoft YaHei UI", 9), anchor="w").pack(fill="x", pady=10)
        self._sync_method_options()

    def _foldout(self, parent, key, title):
        host = tk.Frame(parent, bg=SIDEBAR)
        host.pack(fill="x", pady=(8, 5))
        content = tk.Frame(host, bg=SIDEBAR)
        toggle = tk.Button(host, text="▸  " + title, anchor="w", bg=SIDEBAR, fg="#9fc8d4",
                           activebackground=SIDEBAR, activeforeground="white", relief="flat",
                           font=("Microsoft YaHei UI", 9), padx=0, pady=5, cursor="hand2")
        toggle.pack(fill="x")
        def change():
            if content.winfo_manager():
                content.pack_forget()
                toggle.configure(text="▸  " + title)
            else:
                content.pack(fill="x", pady=(2, 3))
                toggle.configure(text="▾  " + title)
        toggle.configure(command=change)
        self.controls.append((toggle, "normal"))
        self.advanced_sections[key] = (toggle, content)
        return content

    def _input_fields(self, parent):
        self._field(parent, "输入量化步长 scale", self.scale, help_text=HELP["input"])
        self._field(parent, "输入零点 zero point", self.zero_point)

    def _select_step(self, step):
        self.current_step = step
        for key, page in self.side_pages.items():
            page.pack_forget()
            self.step_buttons[key].configure(bg="#294b66" if key == step else SIDEBAR,
                                             fg="white" if key == step else "#b7c9da")
        self.side_pages[step].pack(fill="x")
        primary, secondary, hint = {
            "import": (("导入并分析", self.import_model), ("查看模型结构", lambda: self.tabs.select(self.graph_tab)),
                       "先确认模型与输入尺寸，再开始分析。"),
            "optimize": (("生成优化候选", self.prune), ("恢复原始模型", self.restore),
                         "保留导入基线，可随时恢复。"),
            "export": (("导出 C + 独立运行库…", self.export_c), ("另存候选 .pth…", self.save_candidate),
                       "导出后仍需 C 对拍与目标板验收。"),
        }[step]
        self.side_primary.configure(text=primary[0], command=primary[1])
        self.side_secondary.configure(text=secondary[0], command=secondary[1])
        self.step_action_hint.set(hint)
        self.side_canvas.yview_moveto(0)
        self.tabs.select(self.graph_tab if step == "import" else self.compare_tab)

    def _sync_method_options(self, *args):
        method = METHODS.get(self.pruning.get(), self.pruning.get())
        self.method_hint.set(METHOD_HINTS.get(method, "请查看方法帮助，并验证候选精度。"))
        if method in ("taylor", "reconstruction"):
            self.calibration_panel.pack(fill="x", after=self.calibration_anchor)
        else:
            self.calibration_panel.pack_forget()

    def _scroll_sidebar(self, event):
        self.side_canvas.yview_scroll(int(-event.delta / 120), "units")
        return "break"

    def _label(self, parent, text, small=False, pady=(3, 2)):
        tk.Label(parent, text=text, anchor="w", bg=SIDEBAR, fg="#b8ccd9" if small else "white",
                 font=("Microsoft YaHei UI", 9 if small else 10)).pack(fill="x", pady=pady)

    def _entry(self, parent, variable, width=None):
        entry = ttk.Entry(parent, textvariable=variable, width=width)
        entry.pack(fill="x", pady=2)
        self.controls.append((entry, "normal"))
        return entry

    def _help(self, parent, text, sidebar=False):
        icon, tip = help_icon(parent, text, SIDEBAR if sidebar else BACKGROUND,
                              "#a9d6d9" if sidebar else ACCENT)
        icon.pack(side="right", padx=(5, 0))
        self.help_tips.append(tip)
        return icon

    def _field(self, parent, label, variable, values=None, help_text=None):
        row = tk.Frame(parent, bg=SIDEBAR)
        row.pack(fill="x", pady=(3, 2))
        tk.Label(row, text=label, anchor="w", bg=SIDEBAR, fg="#b8ccd9",
                 font=("Microsoft YaHei UI", 9)).pack(side="left")
        if help_text:
            self._help(row, help_text, sidebar=True)
        if values:
            display = variable
            if isinstance(values, dict):
                reverse = {value: label for label, value in values.items()}
                display = tk.StringVar(self.root, reverse.get(variable.get(), variable.get()))
                variable.trace_add("write", lambda *args: display.set(reverse.get(variable.get(), variable.get())))
            widget = ttk.Combobox(parent, textvariable=display, values=tuple(values), state="readonly")
            if isinstance(values, dict):
                widget.display_variable = display
                widget.bind("<<ComboboxSelected>>", lambda event: variable.set(values[display.get()]))
            widget.pack(fill="x", pady=2)
            self.controls.append((widget, "readonly"))
        else:
            widget = self._entry(parent, variable)
        return widget

    def _button(self, parent, text, command, accent=False):
        button = ttk.Button(parent, text=text, command=command, style="Accent.TButton" if accent else "TButton")
        button.pack(fill="x", pady=3)
        self.controls.append((button, "normal"))
        return button

    @staticmethod
    def _text(parent, **options):
        return tk.Text(parent, bg="white", fg=INK, relief="flat", padx=14, pady=12,
                       font=("Microsoft YaHei UI", 10), wrap="word", state="disabled", **options)

    @staticmethod
    def _set_text(widget, text):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("end", text)
        widget.configure(state="disabled")

    @staticmethod
    def _tree(parent, columns, height=None, expand=True):
        frame = ttk.Frame(parent)
        frame.pack(fill="both" if expand else "x", expand=expand)
        tree = ttk.Treeview(frame, columns=[c[0] for c in columns], show="headings", selectmode="browse",
                            **({"height": height} if height is not None else {}))
        for name, title, width in columns:
            tree.heading(name, text=title)
            tree.column(name, width=width, minwidth=70, stretch=False)
        vertical = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        horizontal = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        tree.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        return tree

    def _import_options(self):
        def optional(value, label):
            if not value.strip():
                return None
            result = int(value)
            if result <= 0:
                raise ValueError(f"{label}必须为正整数")
            return result
        return {"architecture": self.architecture.get(),
                "input_length": optional(self.input_length.get(), "CNN 输入宽度"),
                "seq_length": optional(self.seq_length.get(), "序列长度"),
                "trusted_module": self.trusted.get()}

    def _identity(self):
        return (str(Path(self.model_path.get()).expanduser().resolve()), self._import_options())

    def _require_model(self, supported=True):
        if self.current is None:
            raise ConversionError("请先导入模型。")
        if self._identity() != self.import_identity:
            raise ConversionError("导入设置已更改，请先点击“导入并分析”。")
        if supported and self.current.spec is None:
            raise ConversionError("当前为张量查看模式；需要受支持的图语义才能剪枝、验证或导出 C。")

    def _report_options(self):
        return {"precision": self.precision.get(), "weight_scaling": self.scaling.get(),
                "cache_policy": self.cache_policy.get()}

    def _start(self, title, work, complete):
        if self.busy or self.closed:
            return
        self.busy = True
        self.status.set(title + "…")
        for control, _ in self.controls:
            control.configure(state="disabled")
        self.progress.start(14)
        future = self.executor.submit(work)

        def finished(result):
            try:
                self.events.put((complete, result.result(), None))
            except Exception as error:
                self.events.put((None, error, traceback.format_exc()))
        future.add_done_callback(finished)

    def _poll(self):
        self.poll_id = None
        if self.closed:
            return
        try:
            complete, result, detail = self.events.get_nowait()
        except queue.Empty:
            self.poll_id = self.root.after(80, self._poll)
            return
        self.busy = False
        self.progress.stop()
        for control, state in self.controls:
            control.configure(state=state)
        try:
            if detail:
                self.status.set("操作失败 · 可修正设置后重试")
                self.append_log("操作失败：" + str(result), detail=detail)
                messagebox.showerror("操作未完成", str(result), parent=self.root)
            else:
                complete(result)
        except Exception as error:
            self.append_log("界面更新失败：" + str(error), detail=traceback.format_exc())
            self.status.set("界面更新失败")
            messagebox.showerror("界面更新失败", str(error), parent=self.root)
        self.poll_id = self.root.after(80, self._poll)

    def _guard(self, action):
        try:
            action()
        except (ValueError, OSError, ConversionError) as error:
            self.status.set(str(error))
            messagebox.showerror("请检查设置", str(error), parent=self.root)

    def choose_model(self):
        path = filedialog.askopenfilename(parent=self.root, title="选择模型 checkpoint",
                                         filetypes=[("PyTorch checkpoint", "*.pth *.pt"), ("所有文件", "*.*")])
        if path:
            self.model_path.set(path)

    def choose_dataset(self):
        path = filedialog.askopenfilename(parent=self.root, title="选择验证输入",
                                         filetypes=[("验证数据", "*.npy *.npz *.csv"), ("所有文件", "*.*")])
        if path:
            self.dataset.set(path)

    def choose_calibration(self):
        path = filedialog.askopenfilename(parent=self.root, title="选择独立校准输入（请勿选择最终验证集）",
                                         filetypes=[("校准数据", "*.npy *.npz *.csv"), ("所有文件", "*.*")])
        if path:
            self.calibration.set(path)

    def import_model(self):
        if self.busy:
            return
        def begin():
            self.original = self.current = self.analysis = self.baseline = self.validation = None
            self.import_identity = None
            self._clear()
            if not self.model_path.get().strip():
                raise ValueError("请选择模型文件。")
            identity = self._identity()
            path, options = identity
            report_options = self._report_options()

            def work():
                session = load_session(path, **options)
                return session, analyze_session(session, **report_options)

            def done(result):
                self.original = self.current = result[0]
                self.analysis = self.baseline = result[1]
                self.import_identity = identity
                if self.current.spec and self.current.spec.architecture != "cnn" and METHODS.get(self.pruning.get()) == "svd":
                    self.pruning.set("FPGM 冗余通道")
                self._render()
                self.tabs.select(self.graph_tab)
                self.append_log("导入成功：" + Path(path).name + (" · 仅张量查看" if not self.current.spec else ""),
                                detail={"warnings": self.current.warnings, "source": path})
                self.status.set("导入完成" if self.current.spec else "仅张量查看 · 图语义未确认")
            self._start("加载与分析模型", work, done)
        self._guard(begin)

    def _clear(self):
        for value in self.card_values.values():
            value.set("—")
        for value in self.benefit_values.values():
            value.set("—")
        for tree in (self.layers, self.tensors):
            tree.delete(*tree.get_children())
        self.quality.set("质量尚未评估")
        self.subtitle.configure(text="尚未载入当前模型")
        self._draw_empty("等待分析结果")
        self._set_text(self.details, "")
        self._set_text(self.layer_details, "")
        self._set_text(self.tensor_details, "")
        self._set_text(self.compare, "")
        self.compare_table.delete(*self.compare_table.get_children())
        self.compare_headline.set("尚未生成优化候选")
        self.compare_quality.set("请先导入模型。")
        self.log_events.clear()
        self.diagnostic_events.clear()
        self._set_text(self.log, "")
        self._set_text(self.diagnostics, "")
        self.activity_title.set("查看操作记录")
        self.show_activity.set(False)
        self.show_diagnostics.set(False)
        self._toggle_activity()
        self.deployment_note.set("静态资源估算 · 设备速度、Flash 与整机 RAM 需编译和实测。")
        self._render_validation()

    def _invalidate_validation(self, *args):
        if self.validation is not None:
            self.validation = None
            self.quality.set("验证设置已改变 · 请重新验证")
            if self.analysis:
                self._render()

    def _mark_estimate_pending(self, *args):
        if self.analysis and not self.busy:
            self._render()

    def reanalyze(self):
        def begin():
            self._require_model(False)
            current, original, options = self.current, self.original, self._report_options()
            def done(result):
                self.analysis, self.baseline = result
                self._render()
                self.status.set("当前精度估算已更新")
            self._start("更新资源估算", lambda: (analyze_session(current, **options),
                        analyze_session(original, **options)), done)
        self._guard(begin)

    def prune(self):
        def begin():
            self._require_model()
            amount = float(self.amount.get()) / 100
            method = METHODS.get(self.pruning.get(), self.pruning.get())
            calibration_options = {}
            if method == "reconstruction" and self.current.spec.architecture != "cnn":
                raise ValueError("重建引导通道剪枝当前仅支持本工程 CNN；RNN/LSTM 请使用其他结构化方法。")
            if method in ("taylor", "reconstruction"):
                if not self.calibration.get().strip():
                    raise ValueError("此方法需要独立校准输入，请先在左侧选择校准数据。")
                count = int(self.calibration_samples.get())
                if not 1 <= count <= 128:
                    raise ValueError("校准样本数必须在 1～128 之间。")
                calibration_options = {"data_path": self.calibration.get(),
                    "max_samples": count,
                    "input_scale": float(self.scale.get()), "input_zero_point": int(self.zero_point.get())}
                if self.dataset.get().strip() and self._same_dataset(self.calibration.get(), self.dataset.get()):
                    raise ValueError("校准数据与验证数据指向同一文件。请分别准备校准集和独立验证集。")
            current, original, options = self.current, self.original, self._report_options()
            def work():
                candidate, report = prune_session(current, amount, method, **calibration_options)
                return candidate, report, analyze_session(candidate, **options), analyze_session(original, **options)
            def done(result):
                self.current, report, self.analysis, self.baseline = result
                self.validation = None
                self.quality.set("剪枝候选 · 质量待验证")
                self._render()
                self.append_log("候选已生成：" + change_description(report), detail=report)
                self.tabs.select(self.compare_tab)
                self.status.set("候选已生成 · 原始模型保留，可恢复")
            self._start("生成剪枝候选", work, done)
        self._guard(begin)

    def restore(self):
        def begin():
            self._require_model(False)
            original, options = self.original, self._report_options()
            def done(analysis):
                self.current = original
                self.analysis = self.baseline = analysis
                self.validation = None
                self.quality.set("原始模型 · 质量尚未评估")
                self._render()
                self.append_log("已恢复原始模型。")
                self.status.set("已恢复原始模型")
            self._start("恢复原始模型", lambda: analyze_session(original, **options), done)
        self._guard(begin)

    def evaluate(self):
        def begin():
            self._require_model()
            if not self.dataset.get().strip():
                raise ValueError("请选择代表性验证数据。")
            reference, candidate = self.original, self.current
            path, count = self.dataset.get(), int(self.samples.get())
            scale, zp = float(self.scale.get()), int(self.zero_point.get())
            overlap = any(self._same_dataset(path, change.get("calibration", {}).get("path", ""))
                          for change in self._history(candidate) if isinstance(change.get("calibration"), dict))
            def done(result):
                self.validation = result
                self.validation["evaluation_context"] = {"scale": scale, "zero_point": zp,
                                                         "dataset": path, "calibration_overlap": overlap}
                self.quality.set(f"已比较 {result['samples']} 个样本 · 仅此数据范围")
                self._render()
                view = validation_view(result)
                self.append_log(f"验证完成：{result['samples']} 个样本 · 候选准确率 {view['candidate']} · {view['delta']}",
                                detail=result)
                self.tabs.select(self.verify_tab)
                self.status.set("原始/候选 FP32 比较完成；INT8 C 仍需单独验证")
            self._start("验证候选输出", lambda: evaluate_sessions(reference, candidate, path,
                        max_samples=count, input_scale=scale, input_zero_point=zp), done)
        self._guard(begin)

    def save_candidate(self):
        def begin():
            self._require_model()
            path = filedialog.asksaveasfilename(parent=self.root, title="另存候选（请选择新文件名）",
                    initialfile="candidate.pth", defaultextension=".pth", filetypes=[("Checkpoint", "*.pth")])
            if path:
                session = self.current
                self._start("保存候选", lambda: save_checkpoint(session, path),
                            lambda result: self._saved(result))
        self._guard(begin)

    def _saved(self, result):
        self.append_log("候选已保存：" + str(result.get("path", "")), detail=result)
        self.status.set("候选 checkpoint 已另存；源文件未修改")

    def export_c(self):
        def begin():
            self._require_model()
            parent = filedialog.askdirectory(parent=self.root, title="选择导出父目录（将创建独立子目录）")
            if not parent:
                return
            precision, scaling = self.precision.get(), self.scaling.get()
            cache_policy = self.cache_policy.get()
            name = f"model_{self.current.spec.architecture}_{precision}_{datetime.now():%Y%m%d_%H%M%S_%f}"
            destination = Path(parent) / name
            candidate, reference, validation = self.current, self.original, self.validation
            prefix, scale, zp = self.prefix.get(), float(self.scale.get()), int(self.zero_point.get())
            def work():
                result = export_session(candidate, destination, prefix=prefix,
                        precision=precision, input_scale=scale, input_zero_point=zp,
                        weight_scaling=scaling, validation=validation, cache_policy=cache_policy)
                result["baseline"] = analyze_session(reference, precision=precision, weight_scaling=scaling,
                                                     cache_policy=cache_policy)
                return result
            def done(result):
                self.analysis = result["analysis"]
                self.baseline = result["baseline"]
                self._render()
                self.append_log("导出目录：" + result["directory"] + "\n候选质量：" + result["manifest"]["quality_status"])
                self.status.set("C 与运行库已导出 · 不代表已完成 C/设备验收")
                messagebox.showinfo("导出完成", result["directory"] + "\n\n含 model.c / model.h / example.c / runtime / 分析与清单。", parent=self.root)
            self._start("生成候选 C 模型", work, done)
        self._guard(begin)

    def save_report(self):
        def begin():
            self._require_model(False)
            path = filedialog.asksaveasfilename(parent=self.root, title="保存当前分析报告（新文件名）",
                    initialfile="model_analysis.json", defaultextension=".json",
                    filetypes=[("JSON", "*.json"), ("逐层 CSV", "*.csv")])
            if not path:
                return
            session, original, options = self.current, self.original, self._report_options()
            def work():
                report = analyze_session(session, **options)
                return report, write_analysis_report(report, path), analyze_session(original, **options)
            def done(result):
                self.analysis = result[0]
                self.baseline = result[2]
                self._render()
                self.status.set("分析报告已保存：" + result[1])
            self._start("导出分析报告", work, done)
        self._guard(begin)

    def _draw_empty(self, message):
        self.graph.delete("all")
        self.graph.create_text(220, 150, text=message, fill="#75859a", font=("Microsoft YaHei UI", 13), width=360)
        self.graph.configure(scrollregion=(0, 0, 440, 450))

    def _render(self):
        report = self.analysis
        summary = report["summary"]
        for key, variable in self.card_values.items():
            variable.set(size_text(summary[key]) if "bytes" in key else number(summary[key]))
        title = Path(self.current.source).name if self.current.source else "内存模型"
        report_precision = report.get("deployment", {}).get("precision", "未标明")
        deployment = report.get("deployment", {})
        changed = report_precision != self.precision.get() or deployment.get("cache_policy", "none") != self.cache_policy.get()
        changed = changed or deployment.get("weight_scaling", self.scaling.get()) != self.scaling.get()
        pending = "（部署设置已改，请更新估算）" if changed else ""
        self.subtitle.configure(text=f"{title}  /  {report.get('schema') or '图未知 · 仅张量'}  /  "
                                     f"报告 {report_precision}{pending}  /  已应用 {len(self.current.changes)} 次优化")
        self._set_text(self.layer_details, "")
        self._set_text(self.tensor_details, "")
        self.layers.delete(*self.layers.get_children())
        self.tensors.delete(*self.tensors.get_children())
        total = summary["macs"] or 0
        for index, layer in enumerate(report["layers"]):
            macs = layer["macs"]
            share = f"{100 * macs / total:.1f}%" if total and macs is not None else "—"
            self.layers.insert("", "end", iid=str(index), values=(layer["name"], layer["op"],
                shape_text(layer["input_shape"]), shape_text(layer["output_shape"]),
                number(layer["parameters"]), number(macs), share, size_text(layer["weight_bytes"])))
        for index, tensor in enumerate(report["tensors"]):
            self.tensors.insert("", "end", iid=str(index), values=(tensor["name"], shape_text(tensor["shape"]),
                tensor["dtype"], number(tensor["elements"]), size_text(tensor["bytes"]),
                f"{100 * tensor['zero_fraction']:.1f}%", number(tensor["min"]), number(tensor["max"])))
        self._draw_graph()
        heavy = sorted((layer for layer in report["layers"] if layer.get("macs")),
                       key=lambda layer: layer["macs"], reverse=True)[:3]
        explanations = ("结构总览\n\n点击左侧节点查看这一层的输入、输出与参数。\n\n"
                        "理论计算量最大的层\n" + "\n".join(
                            f"• {layer['name']}：{100 * layer['macs'] / total:.1f}%"
                            for layer in heavy) +
                        "\n\nBN 部署时折叠，逻辑层不一定单独分配内存。\n"
                        "实际速度还受内核、缓存与芯片影响。") if report["supported"] else (
                            "当前为张量查看模式\n\n计算图语义尚未确认，暂不估算连接关系、计算量与部署内存。\n"
                            "请在“张量参数”中检查已加载权重。")
        if self.current.warnings:
            explanations += f"\n\n存在 {len(self.current.warnings)} 条导入提示，详见验证页的技术诊断。"
        self._set_text(self.details, explanations)
        self.compare_table.delete(*self.compare_table.get_children())
        for key, label in (("total_parameters", "可训练参数"), ("macs", "理论 MAC"),
                           ("deployed_parameter_bytes", "导出常量负载"),
                           ("workspace_bytes_lower_bound", "工作区下界")):
            old, new = self.baseline["summary"][key], summary[key]
            formatter = size_text if "bytes" in key else number
            self.compare_table.insert("", "end", iid=key,
                                      values=(label, formatter(old), formatter(new), reduction(old, new)))
        self.compare_headline.set("当前为导入基线" if not self.current.changes else
                                  f"已生成候选 · 累计 {len(self.current.changes)} 次压缩 / 剪枝")
        for key, variable in self.benefit_values.items():
            before, after = self.baseline["summary"][key], summary[key]
            variable.set(reduction(before, after))
            color = "#b04b22" if before is not None and after is not None and after > before else ACCENT
            self.benefit_labels[key].configure(foreground=color)
        view = validation_view(self.validation)
        self.compare_quality.set(view["headline"] + ("\n下一步：使用独立验证数据检查准确率与输出差异。" if not self.validation else
            f"\n原始 {view['reference']} → 候选 {view['candidate']} · {view['delta']}"))
        self.compare_quality_label.configure(foreground={"warning": "#b04b22", "good": ACCENT,
                                                        "pending": INK}[view["tone"]])
        cache = deployment.get("weight_cache") or {}
        if deployment.get("cache_policy") == "packed":
            cache_note = f"SRAM 预打包：卷积权重缓存 {size_text(cache.get('bytes'))} 已含在工作区下界。"
        else:
            cache_note = "常量直读：未增加卷积权重缓存。"
        self.deployment_note.set(cache_note + " 设备速度与完整 Flash / RAM 需编译、实测。")
        steps = [f"{index + 1}. {change_description(change)}" for index, change in enumerate(self.current.changes)]
        inherited = len(self._history(self.current)) - len(self.current.changes)
        history_note = f"导入文件另含 {inherited} 条历史压缩记录；上表以本次导入文件为基线。\n\n" if inherited else ""
        self._set_text(self.compare, history_note + "本次已应用的方法\n" + ("\n".join(steps) if steps else "尚未应用压缩。CNN 可先尝试线性头 SVD；通道剪枝需验证精度。"))
        self._render_validation()

    @staticmethod
    def _history(session):
        inherited = getattr(session, "metadata", {}).get("studio_history", [])
        inherited = [item for item in inherited if isinstance(item, dict)] if isinstance(inherited, list) else []
        return inherited + session.changes

    @staticmethod
    def _same_dataset(first, second):
        if not isinstance(first, (str, os.PathLike)) or not isinstance(second, (str, os.PathLike)) or not first or not second:
            return False
        a, b = Path(first).expanduser().resolve(), Path(second).expanduser().resolve()
        a = a / "manifest.csv" if a.is_dir() else a
        b = b / "manifest.csv" if b.is_dir() else b
        return a == b or (a.exists() and b.exists() and os.path.samefile(a, b))

    def _render_validation(self):
        view = validation_view(self.validation)
        self.verdict.set(view["headline"])
        self.verdict_label.configure(foreground={"warning": "#b04b22", "good": ACCENT, "pending": INK}[view["tone"]])
        for key, variable in self.validation_values.items():
            variable.set(view[key])
        self.validation_scope.set(view["scope"])
        self.validation_errors.set(view["errors"])

    def _draw_graph(self):
        layers = self.analysis["layers"]
        if not layers:
            self._draw_empty("无法从此权重文件还原计算图\n可在“张量参数”中查看已加载数据")
            return
        self.graph.delete("all")
        total = self.analysis["summary"]["macs"] or 1
        for index, layer in enumerate(layers):
            y = 20 + 115 * index
            tag = f"layer_{index}"
            if index:
                self.graph.create_line(210, y - 20, 210, y - 3, fill="#a9b9cc", width=2, arrow="last")
            color = "#e8f6f4" if layer["macs"] else "#f5f7fb"
            self.graph.create_rectangle(30, y, 390, y + 94, fill=color, outline="#d3e2e9", width=1, tags=tag)
            self.graph.create_text(44, y + 18, text=f"{index + 1:02d}   {layer['name']}  ·  {layer['op']}",
                                   anchor="w", fill=INK, font=("Microsoft YaHei UI", 10, "bold"), tags=tag)
            self.graph.create_text(44, y + 43, text=f"{shape_text(layer['input_shape'])}  →  {shape_text(layer['output_shape'])}",
                                   anchor="w", fill="#52657d", font=("Segoe UI", 10), tags=tag)
            self.graph.create_text(44, y + 70, text=f"MAC  {number(layer['macs'])}     参数  {number(layer['parameters'])}",
                                   anchor="w", fill=ACCENT, font=("Segoe UI", 10), tags=tag)
            width = min(80, 80 * (layer["macs"] or 0) / total)
            if width:
                self.graph.create_rectangle(290, y + 65, 290 + width, y + 75, fill=ACCENT, outline="", tags=tag)
            self.graph.tag_bind(tag, "<Button-1>", lambda event, i=index: self.show_layer(i))
        self.graph.configure(scrollregion=(0, 0, 420, 115 * len(layers) + 25))

    def show_layer(self, index):
        if self.analysis:
            layer = self.analysis["layers"][index]
            text = (f"{layer['name']} · {layer['op']}\n\n"
                    f"输入  {shape_text(layer['input_shape'])}\n输出  {shape_text(layer['output_shape'])}\n\n"
                    f"可训练参数  {number(layer['parameters'])}\n理论 MAC  {number(layer['macs'])}\n"
                    f"常量负载  {size_text(layer['weight_bytes'])}\n"
                    f"逻辑输出大小  {size_text(layer.get('activation_bytes'))}")
            if layer.get("attributes"):
                text += "\n\n算子设置\n" + "\n".join(f"{key}：{value}" for key, value in layer["attributes"].items())
            if layer.get("notes"):
                text += "\n\n部署备注\n" + "\n".join(layer["notes"])
            self._set_text(self.details, text)
            self._set_text(self.layer_details, text)

    def select_layer(self, event=None):
        selection = self.layers.selection()
        if selection:
            self.show_layer(int(selection[0]))

    def select_tensor(self, event=None):
        selection = self.tensors.selection()
        if selection and self.current:
            tensor = self.analysis["tensors"][int(selection[0])]
            values = self.current.state[tensor["name"]].detach().cpu().reshape(-1)[:32].tolist()
            self._set_text(self.tensor_details,
                           f"{tensor['name']}  ·  {tensor['dtype']}  ·  {shape_text(tensor['shape'])}\n"
                           f"{number(tensor['elements'])} 个元素，{size_text(tensor['bytes'])}，"
                           f"零值 {100 * tensor['zero_fraction']:.1f}%\n"
                           f"取值范围 {number(tensor['min'])} ～ {number(tensor['max'])}\n"
                           "前 32 个值（预览）：" + json.dumps(values, ensure_ascii=False))

    def append_log(self, text, detail=None):
        timestamp = f"[{datetime.now():%H:%M:%S}] "
        self.log_events.append(timestamp + text.replace("\n", " · "))
        self.log_events = self.log_events[-8:]
        self.activity_title.set(f"查看操作记录（最近 {len(self.log_events)} 条）")
        self._set_text(self.log, "\n".join(self.log_events))
        self.log.see("end")
        if detail is not None:
            if not isinstance(detail, str):
                detail = json.dumps(detail, ensure_ascii=False, indent=2)
            self.diagnostic_events.append(timestamp + text + "\n" + detail)
            self.diagnostic_events = self.diagnostic_events[-20:]
            self._set_text(self.diagnostics, "\n\n".join(self.diagnostic_events)[-100000:])

    def _toggle_diagnostics(self):
        if self.show_diagnostics.get():
            self.show_activity.set(True)
            self._toggle_activity()
            self.diagnostics.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        else:
            self.diagnostics.pack_forget()

    def _toggle_activity(self):
        if self.show_activity.get():
            self.activity_panel.pack(fill="both", expand=True)
        else:
            self.activity_panel.pack_forget()
            self.show_diagnostics.set(False)
            self.diagnostics.pack_forget()

    def close(self):
        if self.busy:
            messagebox.showinfo("任务正在运行", "请等待当前分析、验证或导出完成后关闭。", parent=self.root)
            return
        self.closed = True
        if self.poll_id is not None:
            self.root.after_cancel(self.poll_id)
            self.poll_id = None
        self.executor.shutdown(wait=False, cancel_futures=True)
        for tip in self.help_tips:
            tip.hide()
        self.root.destroy()


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Model Studio: analyze, prune, validate and export C")
    parser.add_argument("checkpoint", nargs="?")
    parser.add_argument("--architecture", default="auto", choices=("auto", "cnn", "rnn", "lstm"))
    parser.add_argument("--input-length", type=int)
    parser.add_argument("--seq-length", type=int)
    parser.add_argument("--trusted-module", action="store_true")
    args = parser.parse_args(argv)
    root = tk.Tk()
    application = ModelStudio(root)
    application.architecture.set(args.architecture)
    if args.input_length is not None:
        application.input_length.set(str(args.input_length))
    if args.seq_length is not None:
        application.seq_length.set(str(args.seq_length))
    application.trusted.set(args.trusted_module)
    if args.checkpoint:
        application.model_path.set(args.checkpoint)
        root.after(100, application.import_model)
    root.mainloop()
    return 0
