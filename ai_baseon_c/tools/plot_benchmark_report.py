"""Render public aggregate figures without private paths or raw experiment data.

Validation checks the consistency of the published snapshot, not independent
access to recordings, checkpoints or physical measurements. Only matplotlib is
needed for rendering. PNG metadata identifies the public JSON by basename/SHA.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import tempfile


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY = ROOT / "docs/assets/benchmark-summary.json"
OUTPUTS = ("model-resources.png", "same-model-comparison.png")
KINDS = ("cnn", "rnn", "lstm")
PRECISIONS = ("fp32", "int8")
CATEGORIES = ("fp32_weight", "int8_weight", "bias", "weight_scale", "row_sum",
              "bn_identity_gamma", "bn_zero", "descriptor")
COLORS = {"fp32": "#557da6", "int8": "#148e8a"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, name, *, integer=False, positive=False):
    require(type(value) is int if integer else type(value) in (int, float), "Invalid number: " + name)
    require(math.isfinite(value) and (value > 0 if positive else value >= 0), "Out-of-range number: " + name)
    return value


def close(actual, expected, name):
    number(actual, name)
    require(math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12), "Inconsistent " + name)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reject_paths(value):
    if isinstance(value, dict):
        for key, item in value.items():
            reject_paths(key);reject_paths(item)
    elif isinstance(value, list):
        for item in value:reject_paths(item)
    elif isinstance(value, str):
        require(not re.search(r"[A-Za-z]:[\\/]|\\\\|(?:^|\s)/(?:Users|home|mnt|tmp)/", value),
                "Public snapshot must not contain machine paths")


def validate_summary(report):
    require(report.get("schema") == "ai4c_public_benchmark_summary_v1", "Unsupported public summary schema")
    reject_paths(report)
    for name in ("source_audit_sha256", "source_resource_ledger_sha256"):
        require(re.fullmatch(r"[a-f0-9]{64}", report[name]) is not None, "Invalid source SHA")
    evaluation = report["evaluation"]
    support = evaluation["class_support"]
    require(len(support) == 4 and len(evaluation["classes"]) == len(set(evaluation["classes"])) == 4,
            "Exactly four distinct classes are required")
    for value in support:number(value, "class support", integer=True, positive=True)
    frames = number(evaluation["frames_per_precision"], "frames", integer=True, positive=True)
    require(sum(support) == frames and type(evaluation["final_cases"]) is int and evaluation["final_cases"] == 6
            and type(evaluation["physical_inferences"]) is int and evaluation["physical_inferences"] == 6 * frames,
            "Incomplete six-case aggregate")
    require(evaluation.get("host_latency_measured") is False
            and evaluation.get("raw_recordings_or_checkpoints_included") is False
            and evaluation.get("all_board_to_corresponding_host_logits_bit_exact") is True,
            "Unsupported timing/data/acceptance declaration")
    clock = number(report["conditions"]["clock_hz"], "clock_hz", integer=True, positive=True)
    conditions = report["conditions"]
    require(conditions["timing"] == "unprofiled single forward + argmax; interrupts retained"
            and set(conditions["excluded_from_timing"]) == {"audio frontend", "UART", "CRC", "printing", "initialization", "FP32 input dequantization"}
            and type(conditions["warmup_per_case"]) is int and conditions["warmup_per_case"] == 0,
            "Unsupported latency scope/IRQ/warmup contract")
    require(set(report["models"]) == set(KINDS), "Need the three final model pairs")
    for kind, model in report["models"].items():
        require(re.fullmatch(r"[a-f0-9]{64}", model["checkpoint_sha256"]) is not None, "Invalid checkpoint SHA")
        count = number(model["matrix_weight_elements"], "matrix elements", integer=True, positive=True)
        require(number(model["training_parameters"], "parameters", integer=True, positive=True) >= count,
                "Training parameters cannot be fewer than matrix weights")
        number(model["macs_per_inference"], "MACs", integer=True, positive=True)
        require(model["dimensions"]["num_classes"] == 4 and set(model["precisions"]) == set(PRECISIONS),
                "Each checkpoint requires both precisions and four outputs")
        require(len({p["elf_sha256"] for p in model["precisions"].values()}) == 2,
                "Single-precision measurements must identify two distinct ELFs")
        for precision, case in model["precisions"].items():
            require(re.fullmatch(r"[a-f0-9]{64}", case["elf_sha256"]) is not None, "Invalid ELF SHA")
            flash = case["flash"];categories = flash["category_bytes"]
            require(set(categories) == set(CATEGORIES), "Incomplete model constant categories")
            for name, value in categories.items():number(value, name, integer=True)
            for name in ("firmware_text_plus_data_bytes", "model_constants_bytes", "weight_bytes", "fp32_weight_bytes",
                         "int8_weight_bytes", "bias_bytes", "weight_scale_bytes", "row_sum_bytes", "bn_auxiliary_bytes",
                         "descriptor_bytes", "other_firmware_bytes"):
                number(flash[name], name, integer=True)
            require(sum(categories.values()) == flash["model_constants_bytes"]
                    and flash["model_constants_bytes"] + flash["other_firmware_bytes"] == flash["firmware_text_plus_data_bytes"]
                    <= 524288, "Flash categories do not sum to the actual firmware")
            require(flash["weight_bytes"] == count * (4 if precision == "fp32" else 1)
                    == categories[precision + "_weight"] and categories[("int8" if precision == "fp32" else "fp32") + "_weight"] == 0
                    and flash["fp32_weight_bytes"] == categories["fp32_weight"]
                    and flash["int8_weight_bytes"] == categories["int8_weight"], "Matrix weight storage is inconsistent")
            for field, names in (("bias_bytes", ("bias",)), ("weight_scale_bytes", ("weight_scale",)),
                                 ("row_sum_bytes", ("row_sum",)), ("descriptor_bytes", ("descriptor",)),
                                 ("bn_auxiliary_bytes", ("bn_identity_gamma", "bn_zero"))):
                require(flash[field] == sum(categories[name] for name in names), "Auxiliary model byte count differs")
            require(flash.get("elf_nm_c_sets_equal") is True and type(flash["alias_or_overlap_count"]) is int
                    and flash["alias_or_overlap_count"] == 0, "Aliased/unverified model constants")
            ram = case["ram"]
            for field, value in ram.items():number(value, field, integer=True)
            require(ram["input_staging_bytes"] == (4 * model["dimensions"]["input_elements"] if precision == "fp32" else 0)
                    and ram["workspace_bytes"] + ram["rtos_heap_arena_bytes"] + ram["input_staging_bytes"] <= ram["static_data_plus_bss_bytes"]
                    and ram["static_data_plus_bss_bytes"] + ram["linker_heap_stack_reserved_bytes"] <= 131072,
                    "RAM/subset/staging accounting differs")
            metrics = case["metrics"];matrix = metrics["confusion_matrix_label_rows"]
            require(len(matrix) == 4 and all(len(row) == 4 for row in matrix), "Need a complete four-class confusion matrix")
            for row in matrix:
                for value in row:number(value, "confusion count", integer=True)
            require([sum(row) for row in matrix] == metrics["class_support"] == support
                    and type(metrics["frames"]) is int and metrics["frames"] == frames
                    and type(metrics["correct"]) is int and metrics["correct"] == sum(matrix[i][i] for i in range(4)),
                    "Quality counts disagree with class supports")
            recalls = [matrix[i][i] / support[i] for i in range(4)]
            require(len(metrics["per_class_recall"]) == 4, "Missing per-class recalls")
            for observed, expected in zip(metrics["per_class_recall"], recalls):close(observed, expected, "class recall")
            close(metrics["accuracy"], metrics["correct"] / frames, "accuracy")
            close(metrics["macro_recall"], sum(recalls) / 4, "macro recall")
            median = number(metrics["median_cycles"], "median cycles", positive=True)
            high = number(metrics["p95_cycles"], "P95 cycles", integer=True, positive=True)
            require(median <= high <= 0xffffffff, "Invalid latency percentiles")
            close(metrics["median_ms"], median * 1000 / clock, "median milliseconds")
            close(metrics["p95_ms"], high * 1000 / clock, "P95 milliseconds")
            require(metrics.get("host_latency_measured") is False
                    and type(metrics["host_reference_bit_exact_frames"]) is int
                    and metrics["host_reference_bit_exact_frames"] == frames, "Incomplete exact reference agreement")
    return report


def load_summary(path):
    payload = Path(path).read_bytes()
    def invalid(value):raise ValueError("Nonfinite JSON token: " + value)
    return validate_summary(json.loads(payload.decode("utf-8-sig"), parse_constant=invalid)), hashlib.sha256(payload).hexdigest()


def png_metadata(path):
    """Read ordinary PNG tEXt metadata without requiring an image library."""
    data = Path(path).read_bytes();require(data[:8] == b"\x89PNG\r\n\x1a\n", "Not a PNG")
    offset, result = 8, {}
    while offset < len(data):
        length = struct.unpack_from(">I", data, offset)[0];kind = data[offset + 4:offset + 8]
        value = data[offset + 8:offset + 8 + length];offset += 12 + length
        if kind == b"tEXt":
            key, text = value.split(b"\0", 1);result[key.decode("latin-1")] = text.decode("latin-1")
        elif kind == b"IHDR":result["pixels"] = list(struct.unpack_from(">II", value))
    return result


def _style(axis):
    axis.set_facecolor("white");axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color("#c7d3df");axis.tick_params(length=0, pad=7)
    axis.grid(axis="x", color="#e4ebf2", linewidth=.7);axis.set_axisbelow(True)


def _footer(figure, filename, summary_sha):
    figure.text(.055, .018, "Public aggregate: " + filename + "  |  SHA256: " + summary_sha,
                fontsize=8.2, color="#637c93")


def _resources(plt, report, filename, summary_sha):
    figure = plt.figure(figsize=(16, 9.5), facecolor="#f5f8fc")
    figure.text(.055, .94, "Same parameters. Different storage.", fontsize=25, weight="bold", color="#20364d")
    figure.text(.055, .90, "Each architecture uses the same checkpoint in FP32 and INT8; no pruning.", fontsize=12, color="#50677e")
    for index, kind in enumerate(KINDS):
        model = report["models"][kind];x = .055 + .31 * index
        figure.text(x, .84, kind.upper(), fontsize=15, weight="bold", color="#20364d")
        figure.text(x, .80, f"{model['training_parameters']:,} trainable parameters", fontsize=16, color="#20364d")
        figure.text(x, .765, f"{model['matrix_weight_elements']:,} matrix-weight elements in both precisions", fontsize=10, color="#50677e")
    left = figure.add_axes((.10, .255, .35, .40));right = figure.add_axes((.575, .255, .35, .40))
    cases = [(kind, p, report["models"][kind]["precisions"][p]["flash"]) for kind in KINDS for p in PRECISIONS]
    colors = [COLORS[p] for _, p, _ in cases];labels = [k.upper() + " " + p.upper() for k, p, _ in cases]
    weights = [f["weight_bytes"] / 1024 for _, _, f in cases]
    constants = [f["model_constants_bytes"] / 1024 for _, _, f in cases]
    other = [f["other_firmware_bytes"] / 1024 for _, _, f in cases]
    total = [f["firmware_text_plus_data_bytes"] / 1024 for _, _, f in cases]
    bars = left.barh(range(6), weights, height=.62, color=colors)
    left.bar_label(bars, labels=[f"{x:.2f}" for x in weights], padding=6, fontsize=10)
    right.barh(range(6), constants, height=.62, color=colors)
    right.barh(range(6), other, left=constants, height=.62, color="#d4deea")
    for index, (model, rest, combined) in enumerate(zip(constants, other, total)):
        right.text(model / 2, index, f"{model:.1f}", va="center", ha="center", color="white", fontsize=10)
        right.text(model + rest / 2, index, f"{rest:.1f}", va="center", ha="center", color="#425a72", fontsize=10)
        right.text(combined + 3, index, f"{combined:.2f}", va="center", fontsize=10)
    for axis in (left, right):
        _style(axis);axis.set_yticks(range(6), labels, fontsize=10);axis.invert_yaxis();axis.set_xlabel("Flash / KiB", labelpad=12)
    left.set_xlim(0, max(weights) * 1.24);right.set_xlim(0, max(total) * 1.22)
    left.set_title("Matrix-weight arrays only", fontsize=15, loc="left", pad=22)
    right.set_title("Model constants + other firmware", fontsize=15, loc="left", pad=22)
    from matplotlib.patches import Patch
    figure.legend([Patch(color=COLORS["fp32"]), Patch(color=COLORS["int8"]), Patch(color="#d4deea")],
                  ["FP32 weights / model constants", "INT8 weights / model constants", "Other firmware (measured per ELF)"],
                  loc="lower center", bbox_to_anchor=(.52, .16), ncol=3, frameon=False, fontsize=10)
    figure.text(.055, .115, "Model constants also include bias, scales, row sums, BN helper arrays and descriptors. They are not just weights.", fontsize=10)
    figure.text(.055, .080, "Firmware = ELF text + data, not the ELF disk-file size. Other firmware includes code, RTOS and other constants.", fontsize=10)
    figure.text(.055, .047, "Weight arrays shrink exactly 4x; model constants and complete firmware have different ratios. 1 KiB = 1024 B.", fontsize=10, color="#50677e")
    _footer(figure, filename, summary_sha);return figure


def _performance(plt, report, filename, summary_sha):
    figure = plt.figure(figsize=(16, 9), facecolor="#f5f8fc")
    figure.text(.055, .94, "Final models: same-checkpoint FP32 / INT8", fontsize=24, weight="bold", color="#20364d")
    conditions = report["conditions"];evaluation = report["evaluation"]
    figure.text(.055, .89, f"{conditions['processor']}  |  {conditions['clock_hz'] / 1e6:g} MHz  |  {evaluation['frames_per_precision']} frames per case", fontsize=12, color="#50677e")
    axis = figure.add_axes((.10, .30, .40, .45));table = figure.add_axes((.60, .30, .35, .45));table.axis("off")
    cases = [(kind, p, report["models"][kind]["precisions"][p]["metrics"]) for kind in KINDS for p in PRECISIONS]
    medians = [m["median_ms"] for _, _, m in cases];highs = [m["p95_ms"] for _, _, m in cases]
    axis.barh(range(6), medians, height=.60, color=[COLORS[p] for _, p, _ in cases])
    axis.errorbar(medians, range(6), xerr=[[0] * 6, [b - a for a, b in zip(medians, highs)]], fmt="none", capsize=4, color="#20364d", linewidth=1)
    for index, (median, high) in enumerate(zip(medians, highs)):
        axis.text(high + max(highs) * .025, index, f"{median:.3f}\nP95 {high:.3f}", fontsize=10, va="center", color="#20364d")
    _style(axis);axis.set_yticks(range(6), [k.upper() + " " + p.upper() for k, p, _ in cases], fontsize=10)
    axis.invert_yaxis();axis.set_xlim(0, max(highs) * 1.35);axis.set_xlabel("Single forward + argmax / ms (lower is better)", labelpad=12)
    axis.set_title("Board latency: median / P95", fontsize=15, loc="left", pad=22)
    table.set_title("Board quality: four-class classification", fontsize=15, loc="left", pad=22)
    table.text(.02, .99, "Configuration", fontsize=10, color="#50677e")
    table.text(.56, .99, "Accuracy", ha="center", fontsize=10, color="#50677e")
    table.text(.87, .99, "Macro recall", ha="center", fontsize=10, color="#50677e")
    from matplotlib.patches import Rectangle
    for index, (kind, p, metrics) in enumerate(cases):
        y = .84 - index * .137
        table.add_patch(Rectangle((0, y - .055), 1, .12, color="#e8f0f8" if p == "fp32" else "#e2f2ef"))
        table.text(.025, y, kind.upper() + " " + p.upper(), va="center", color=COLORS[p], fontsize=11)
        table.text(.56, y, f"{100 * metrics['accuracy']:.2f}%", va="center", ha="center", fontsize=15, color="#20364d")
        table.text(.87, y, f"{100 * metrics['macro_recall']:.2f}%", va="center", ha="center", fontsize=15, weight="bold", color="#20364d")
    figure.text(.055, .20, "Macro recall = arithmetic mean of the four class recalls; each class has equal weight.", fontsize=11)
    figure.text(.055, .15, "Board DWT only. Host C provides numeric references; host latency was not measured.", fontsize=11, color="#50677e")
    figure.text(.055, .105, "IRQ retained. Excludes audio frontend, UART, printing, initialization and FP32 input dequantization.", fontsize=10)
    figure.text(.055, .063, "Published aggregates do not distribute the private recordings/checkpoints or establish general speaker recognition accuracy.", fontsize=10, color="#50677e")
    _footer(figure, filename, summary_sha);return figure


def _safe_output(directory):
    directory = Path(directory).absolute()
    require(directory.resolve() == directory, "Output directory redirects through a link")
    for entry in (directory, *directory.parents):
        if entry.exists():require(not entry.is_symlink() and not getattr(entry.lstat(), "st_file_attributes", 0) & 0x400, "Reparse output forbidden")
    return directory


def render_report(summary=DEFAULT_SUMMARY, output_directory=None, *, overwrite=False):
    require(type(overwrite) is bool, "overwrite must be boolean")
    summary = Path(summary).absolute();report, summary_sha = load_summary(summary)
    directory = _safe_output(output_directory if output_directory is not None else summary.parent)
    targets = [directory / name for name in OUTPUTS]
    for target in targets:
        _safe_output(target)
        require(not target.exists() or target.is_file(), "Output is not a regular file")
        if target.exists():
            require(target.stat().st_nlink == 1, "Refuse to overwrite a hard-linked figure")
            require(not os.path.samefile(target, summary), "Output aliases its input JSON")
            if not overwrite:raise FileExistsError("Choose fresh outputs or explicitly pass --overwrite")
    directory.mkdir(parents=True, exist_ok=True)
    prior = os.environ.get("MPLCONFIGDIR")
    metadata = {"Software": "AI4C_Tool public benchmark renderer", "SummaryFile": summary.name,
                "SummarySHA256": summary_sha, "GeneratorSHA256": digest(__file__), "Schema": report["schema"],
                "Scope": "Public aggregate snapshot; no host latency; private recordings/checkpoints not included"}
    result = {"summary_file": summary.name, "summary_sha256": summary_sha, "generator_sha256": metadata["GeneratorSHA256"], "figures": []}
    with tempfile.TemporaryDirectory(prefix=".public-report-", dir=directory) as temporary:
        try:
            os.environ["MPLCONFIGDIR"] = temporary
            import matplotlib
            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
            prepared = []
            with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11, "axes.unicode_minus": False}):
                for name, draw in zip(OUTPUTS, (_resources, _performance)):
                    figure = draw(plt, report, summary.name, summary_sha);path = Path(temporary) / name
                    try:figure.savefig(path, dpi=180, format="png", metadata=metadata)
                    finally:plt.close(figure)
                    actual = png_metadata(path)
                    require(all(actual.get(key) == value for key, value in metadata.items()) and min(actual["pixels"]) >= 1000,
                            "PNG metadata/resolution verification failed")
                    reject_paths({key: value for key, value in actual.items() if key != "pixels"})
                    prepared.append(path)
            require(digest(summary) == summary_sha, "Public input changed during rendering")
            for target, prepared_path in zip(targets, prepared):
                # Only these two explicitly named public figures may be replaced.
                target.write_bytes(prepared_path.read_bytes())
                result["figures"].append({"file": target.name, "sha256": digest(target), "metadata": png_metadata(target)})
        finally:
            if prior is None:os.environ.pop("MPLCONFIGDIR", None)
            else:os.environ["MPLCONFIGDIR"] = prior
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true", help="Replace only the two named public PNGs")
    args = parser.parse_args(argv)
    print(json.dumps(render_report(args.summary, args.output_dir, overwrite=args.overwrite), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
