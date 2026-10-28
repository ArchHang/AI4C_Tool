"""Transactional export of the current in-memory candidate and analysis reports."""

import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch

from model_converter.emitter import emit_model
from model_converter.schema import ConversionError

from .analysis import analyze_session


def state_digest(session):
    """Fingerprint tensor identity, not the source checkpoint or optimizer state."""
    digest = hashlib.sha256()
    for name, value in sorted(session.state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)],
                                 ensure_ascii=False).encode("utf-8"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def export_session(session, output_dir, prefix="ai_model", precision="int8",
                   input_scale=1 / 255, input_zero_point=0, weight_scaling="maxabs",
                   bundle_runtime=True, validation=None, cache_policy="none"):
    """Publish a complete new directory; never reload the original checkpoint."""
    if session.spec is None:
        raise ConversionError("当前仅能查看张量；缺少受支持的计算图，不能生成 C。")
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise ConversionError("导出目录已存在，请选择一个新的子目录；不会覆盖已有内容。")
    analysis = analyze_session(session, precision=precision, weight_scaling=weight_scaling,
                               cache_policy=cache_policy)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # The owned temporary directory is a sibling, so commit does not cross volumes.
    with tempfile.TemporaryDirectory(prefix=".studio-export-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        manifest = emit_model(session.spec, staging, prefix=prefix, precision=precision,
                              input_scale=input_scale, input_zero_point=input_zero_point,
                              checkpoint_sha256=session.sha256 if not session.changes else None,
                              weight_scaling=weight_scaling, cache_policy=cache_policy)
        if bundle_runtime:
            from library_sources import export_runtime
            components = ("cnn",) if session.spec.architecture == "cnn" else ("cnn", session.spec.architecture)
            runtime = export_runtime(staging / "runtime", components=components)
            source = staging / "model.c"
            source.write_text(source.read_text(encoding="utf-8").replace('"ai_baseon_c/', '"runtime/'),
                              encoding="utf-8")
            manifest["bundled_runtime"] = runtime
        manifest["studio_provenance"] = {
            "source_checkpoint": str(session.source) if session.source else None,
            "source_checkpoint_sha256": session.sha256,
            "candidate_state_sha256": state_digest(session),
            "changes": session.changes,
            "inherited_history": session.metadata.get("studio_history", []),
        }
        manifest["studio_analysis"] = analysis["summary"]
        manifest["candidate_fp32_comparison"] = validation
        manifest["quality_status"] = ("sample_comparison_only" if validation else "not_evaluated")
        manifest["target_performance"] = "not_measured"
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        (staging / "analysis.json").write_text(
            json.dumps(analysis, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        readme = staging / "README.txt"
        text = readme.read_text(encoding="utf-8").replace("docs/model_conversion.md", "docs/guide.md#model-studio")
        text += ("\nModel Studio: exports the current candidate. MAC counts are theoretical.\n"
                 "Pruning comparison, when present, compares PyTorch FP32 candidates only;\n"
                 "it does not validate generated INT8 C or target performance.\n")
        if bundle_runtime:
            sources = " ".join("runtime/" + name for name in manifest["bundled_runtime"]["sources"])
            text += ("\nSelf-contained hosted build:\n"
                     f"  cc -std=c99 -O3 -ffp-contract=off -fno-fast-math -Iruntime model.c example.c {sources} -lm -o model_example\n")
        readme.write_text(text, encoding="utf-8")
        if destination.exists() or destination.is_symlink():
            raise ConversionError("导出期间目标目录被创建，已停止，未覆盖该目录。")
        staging.rename(destination)
    return {"directory": str(destination), "manifest": manifest, "analysis": analysis}


def write_analysis_report(analysis, path, overwrite=False):
    """Export JSON or per-layer CSV; create exclusively unless explicitly allowed."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in (".json", ".csv"):
        raise ConversionError("分析报告格式必须为 .json 或 .csv。")
    source = analysis.get("source", {}).get("path")
    if source:
        same_path = path.resolve() == Path(source).resolve()
        same_file = path.exists() and Path(source).exists() and os.path.samefile(path, source)
        if same_path or same_file:
            raise ConversionError("报告不能覆盖源模型。")
    mode = "w" if overwrite else "x"
    if suffix == ".json":
        text = json.dumps(analysis, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        with path.open(mode, encoding="utf-8") as stream:
            stream.write(text)
    else:
        columns = ["name", "op", "input_shape", "output_shape", "parameters", "weight_bytes",
                   "macs", "activation_bytes", "notes"]
        with path.open(mode, encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for layer in analysis["layers"]:
                row = {name: layer.get(name) for name in columns}
                for name, value in row.items():
                    if isinstance(value, (list, dict)):
                        row[name] = json.dumps(value, ensure_ascii=False)
                    elif isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r")):
                        row[name] = "'" + value
                writer.writerow(row)
    return str(path)
