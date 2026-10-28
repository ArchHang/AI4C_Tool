"""Verify the existing trained VAD/speaker CNN against independent integer and float references.

Reuses dataset features, model/model_int8.npz and the existing native layer-dump format.
The scalar chain references originate in demo/scripts/verify_dynamic_layers.py; half-even
rounding is corrected with an independent quotient/remainder oracle. No model is trained.
Only NumPy is required. --exe must explicitly link ai_baseon_c/cnn.c.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).absolute().parents[2]

K = 28
LAYER_SHAPES = {
    "c1": (4, 2044),
    "s1": (4, 1022),
    "c2": (8, 1018),
    "s2": (8, 509),
    "c3": (16, 505),
    "s3": (16, 252),
    "f1": (32,),
    "q4": (32,),
    "logits": (4,),
}
FRAME_BYTES = (4 + 4 * 4 + 4 +  # frame + a1s/a2s/a3s/fc_scale + fc_zp
               4 * 2044 + 4 * 1022 + 8 * 1018 + 8 * 509 + 16 * 505 + 16 * 252 +
               32 * 4 + 32 + 4 * 4)


def rnd_half_even_shift64(v, bits):
    """复现 C 端 round_half_even_shift64：round-half-even 右移。"""
    # Quotient/remainder is independent of the library's shift implementation.
    a = np.abs(v)
    unit = 1 << bits
    quotient, remainder = np.divmod(a, unit)
    increment = (remainder > unit // 2) | ((remainder == unit // 2) & (quotient % 2 == 1))
    rounded = quotient + increment
    return np.where(v >= 0, rounded, -rounded).astype(np.int32)


def fixed_coeff(x):
    """复现 C 端 fixed_coeff：double 计算 + llrint（round-half-even）+ int64 饱和。"""
    x = np.asarray(x, dtype=np.float64)
    limit = 9.0e18
    return np.where(
        np.abs(x) > limit,
        np.where(x > 0, np.iinfo(np.int64).max, np.iinfo(np.int64).min),
        np.rint(x).astype(np.int64),
    )


def conv1d_acc(x, w):
    """int32 精确复现 conv1d_int8_acc（pad=0, stride=1）。"""
    out_ch, in_ch, k = w.shape
    out_col = x.shape[1] - k + 1
    xw = np.lib.stride_tricks.sliding_window_view(x, k, axis=1)  # (in_ch, out_col, k)
    acc = np.einsum("ipt,oit->op", xw.astype(np.int64), w.astype(np.int64),
                    dtype=np.int64, casting="unsafe").astype(np.int32)
    return acc


def dyn_conv_quant(x, w, w_scale, bias, in_act_scale):
    """复现 conv1d_int8_dynamic：整数域找 max + 定点重量化（K=28）。"""
    acc = conv1d_acc(x, w)
    mul = w_scale.astype(np.float32) * np.float32(in_act_scale)
    b = bias.astype(np.float32)
    y_ch = acc.max(axis=1).astype(np.float32) * mul + b
    y_relu = np.maximum(y_ch, np.float32(0.0))
    ymax = y_relu.max()
    scale = ymax / np.float32(255.0)
    if scale < np.float32(1e-9):
        scale = np.float32(1e-9)
    q = np.zeros(acc.shape, dtype=np.uint8)
    if ymax > 0:
        k2 = float(1 << K)
        a_fix = fixed_coeff(np.float64(mul) / np.float64(scale) * k2)
        b_fix = fixed_coeff(np.float64(b) / np.float64(scale) * k2)
        v = acc.astype(np.int64) * a_fix[:, None] + b_fix[:, None]
        q = np.clip(rnd_half_even_shift64(v, K), 0, 255).astype(np.uint8)
    return q, scale


def dyn_conv_quant_float(x, w, w_scale, bias, in_act_scale):
    """浮点参考：逐点 y = acc*mul+b -> ReLU -> 全局 ymax -> lrintf(y/scale)。"""
    acc = conv1d_acc(x, w)
    mul = w_scale.astype(np.float32) * np.float32(in_act_scale)
    b = bias.astype(np.float32)
    y = acc.astype(np.float32) * mul[:, None] + b[:, None]
    y = np.maximum(y, np.float32(0.0))
    ymax = y.max()
    scale = ymax / np.float32(255.0)
    if scale < np.float32(1e-9):
        scale = np.float32(1e-9)
    q = np.clip(np.rint((y / scale).astype(np.float64)), 0, 255).astype(np.uint8)
    return q, scale


def pool_u8(x):
    m = x.shape[1] // 2
    return np.maximum(x[:, : 2 * m : 2], x[:, 1 : 2 * m : 2])


def quantize_u8_py(x, scale, zp):
    """复现 quantize_u8：float32 除法 + lrintf（round-half-even）+ 零点 + clamp。"""
    q = np.rint((x.astype(np.float32) / np.float32(scale)).astype(np.float64)).astype(np.int64) + zp
    return np.clip(q, 0, 255).astype(np.uint8)


def py_chain(q_in, params):
    """完整动态链路，返回各层输出与标量。"""
    x0 = q_in.astype(np.uint8).reshape(1, 2048)
    c1, a1s = dyn_conv_quant(x0, params["w0_int8"].reshape(4, 1, 5), params["w0_scale"],
                             params["b0"], np.float32(1.0 / 255.0))
    s1 = pool_u8(c1)
    c2, a2s = dyn_conv_quant(s1, params["w1_int8"].reshape(8, 4, 5), params["w1_scale"],
                             params["b1"], a1s)
    s2 = pool_u8(c2)
    c3, a3s = dyn_conv_quant(s2, params["w2_int8"].reshape(16, 8, 5), params["w2_scale"],
                             params["b2"], a2s)
    s3 = pool_u8(c3)

    s3f = s3.reshape(-1)
    acc1 = (s3f.astype(np.int64) @ params["w3_int8"].reshape(32, 4032).T.astype(np.int64))
    f1 = (acc1.astype(np.float32) *
          (params["w3_scale"].astype(np.float32) * np.float32(a3s)) +
          params["b3"].astype(np.float32))
    mn = f1.min()
    mx = f1.max()
    fc_scale = (mx - mn) / np.float32(255.0)
    if fc_scale < np.float32(1e-9):
        fc_scale = np.float32(1e-9)
    fc_zp_f = np.clip(np.float32(-mn) / fc_scale, 0, 255)
    fc_zp = int(fc_zp_f)
    q4 = quantize_u8_py(f1, fc_scale, fc_zp)

    acc2 = ((q4.astype(np.int64) - fc_zp) @
            params["w4_int8"].reshape(4, 32).T.astype(np.int64))
    logits = (acc2.astype(np.float32) *
              (params["w4_scale"].astype(np.float32) * fc_scale) +
              params["b4"].astype(np.float32))
    return {
        "c1": c1, "s1": s1, "c2": c2, "s2": s2, "c3": c3, "s3": s3,
        "f1": f1, "q4": q4, "logits": logits,
        "a1s": a1s, "a2s": a2s, "a3s": a3s, "fc_scale": fc_scale, "fc_zp": fc_zp,
    }


def py_chain_float(q_in, params):
    """浮点参考完整链路（conv 用逐点 float 量化，fc 部分与 C 相同）。"""
    x0 = q_in.astype(np.uint8).reshape(1, 2048)
    c1, a1s = dyn_conv_quant_float(x0, params["w0_int8"].reshape(4, 1, 5), params["w0_scale"],
                                   params["b0"], np.float32(1.0 / 255.0))
    s1 = pool_u8(c1)
    c2, a2s = dyn_conv_quant_float(s1, params["w1_int8"].reshape(8, 4, 5), params["w1_scale"],
                                   params["b1"], a1s)
    s2 = pool_u8(c2)
    c3, a3s = dyn_conv_quant_float(s2, params["w2_int8"].reshape(16, 8, 5), params["w2_scale"],
                                   params["b2"], a2s)
    s3 = pool_u8(c3)

    s3f = s3.reshape(-1)
    acc1 = s3f.astype(np.int64) @ params["w3_int8"].reshape(32, 4032).T.astype(np.int64)
    f1 = (acc1.astype(np.float32) *
          (params["w3_scale"].astype(np.float32) * np.float32(a3s)) +
          params["b3"].astype(np.float32))
    mn = f1.min()
    mx = f1.max()
    fc_scale = (mx - mn) / np.float32(255.0)
    if fc_scale < np.float32(1e-9):
        fc_scale = np.float32(1e-9)
    fc_zp_f = np.clip(np.float32(-mn) / fc_scale, 0, 255)
    fc_zp = int(fc_zp_f)
    q4 = quantize_u8_py(f1, fc_scale, fc_zp)

    acc2 = ((q4.astype(np.int64) - fc_zp) @
            params["w4_int8"].reshape(4, 32).T.astype(np.int64))
    logits = (acc2.astype(np.float32) *
              (params["w4_scale"].astype(np.float32) * fc_scale) +
              params["b4"].astype(np.float32))
    return {
        "c1": c1, "s1": s1, "c2": c2, "s2": s2, "c3": c3, "s3": s3,
        "f1": f1, "q4": q4, "logits": logits,
        "a1s": a1s, "a2s": a2s, "a3s": a3s, "fc_scale": fc_scale, "fc_zp": fc_zp,
    }


def read_c_dump(path, nframes):
    """读取 C 端二进制逐层导出。"""
    raw = np.fromfile(path, dtype=np.uint8)
    assert raw.size == nframes * FRAME_BYTES, f"dump size {raw.size} != {nframes * FRAME_BYTES}"
    out = []
    for i in range(nframes):
        off = i * FRAME_BYTES
        frame, a1s, a2s, a3s, fc_scale, fc_zp = struct.unpack_from("<Iffffi", raw, off)
        p = off + 24
        layers = {}
        for name, shp in LAYER_SHAPES.items():
            n = int(np.prod(shp))
            if name in ("f1", "logits"):
                layers[name] = raw[p:p + 4 * n].view("<f4").reshape(shp).copy()
                p += 4 * n
            else:
                layers[name] = raw[p:p + n].reshape(shp).copy()
                p += n
        out.append((frame, a1s, a2s, a3s, fc_scale, fc_zp, layers))
    return out


def select_frames(manifest, count):
    candidates = []
    with manifest.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            feature_path = manifest.parent / Path(row["feat_bin"].replace("\\", "/"))
            raw = np.fromfile(feature_path, dtype=np.uint8)
            if raw.size == 0 or raw.size % 2048:
                raise ValueError(f"Invalid feature-file size: {feature_path}")
            for index, frame in enumerate(raw.reshape(-1, 2048)):
                candidates.append((str(feature_path), index, int(row["label_id"]), frame))
    if count < 1 or count > len(candidates):
        raise ValueError(f"--frames must be in [1,{len(candidates)}]")
    selected = np.random.RandomState(20261013).choice(len(candidates), count, replace=False)
    frames = np.stack([candidates[index][3] for index in selected])
    metadata = [{"path": candidates[index][0], "frame": candidates[index][1],
                 "label": candidates[index][2]} for index in selected]
    return frames, metadata


def compare_array(name, actual, reference, integer):
    if integer:
        if not np.array_equal(actual, reference):
            mismatches = int(np.count_nonzero(actual != reference))
            raise AssertionError(f"{name}: {mismatches} integer mismatches")
    elif not np.all(np.isfinite(actual)) or not np.allclose(actual, reference,
                                                           atol=1e-5, rtol=2e-6):
        raise AssertionError(f"{name}: floating error {np.abs(actual - reference).max():.9g}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--baseline-exe", type=Path)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "datasets/test_int8/manifest.csv")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "results/performance_20261013/cnn_model")
    args = parser.parse_args()
    frames, metadata = select_frames(args.manifest, args.frames)
    params = np.load(ROOT / "model/model_int8.npz")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs = args.output_dir / "selected_features.bin"
    dump = args.output_dir / "optimized_layers.bin"
    frames.tofile(inputs)
    subprocess.run([str(args.exe.absolute()), str(inputs.absolute()), str(dump.absolute()),
                    str(args.frames)], check=True, timeout=120)
    native = read_c_dump(dump, args.frames)
    baseline = None
    if args.baseline_exe:
        old_dump = args.output_dir / "baseline_layers.bin"
        subprocess.run([str(args.baseline_exe.absolute()), str(inputs.absolute()),
                        str(old_dump.absolute()), str(args.frames)], check=True, timeout=120)
        baseline = read_c_dump(old_dump, args.frames)
    report = {
        "frames": args.frames,
        "selection_seed": 20261013,
        "input_sha256": hashlib.sha256(frames.tobytes()).hexdigest(),
        "model_sha256": hashlib.sha256((ROOT / "model/model_int8.npz").read_bytes()).hexdigest(),
        "integer_layer_elements": 0,
        "float_max_error": {"f1": 0.0, "logits": 0.0},
        "float_reference_prediction_matches": 0,
        "strict_prediction_matches": 0,
        "correct_labels": 0,
        "baseline_prediction_matches": 0 if baseline else None,
        "baseline_changed_codes": 0 if baseline else None,
        "selected_frames": metadata,
    }
    for index, (frame_index, a1s, a2s, a3s, fc_scale, fc_zp, layers) in enumerate(native):
        reference = py_chain(frames[index], params)
        float_reference = py_chain_float(frames[index], params)
        if frame_index != index:
            raise AssertionError("Native dump frame indices are not sequential")
        for name, actual in layers.items():
            integer = name not in ("f1", "logits")
            compare_array(f"frame {index} {name}", actual, reference[name], integer)
            if integer:
                report["integer_layer_elements"] += actual.size
            else:
                error = float(np.abs(actual - reference[name]).max())
                report["float_max_error"][name] = max(report["float_max_error"][name], error)
        for name, actual in (("a1s", a1s), ("a2s", a2s), ("a3s", a3s),
                             ("fc_scale", fc_scale), ("fc_zp", fc_zp)):
            compare_array(f"frame {index} {name}", np.asarray(actual),
                          np.asarray(reference[name]), name == "fc_zp")
        prediction = int(np.argmax(layers["logits"]))
        strict_prediction = int(np.argmax(reference["logits"]))
        float_prediction = int(np.argmax(float_reference["logits"]))
        if prediction != strict_prediction or prediction != float_prediction:
            raise AssertionError(f"frame {index}: prediction disagreement")
        report["strict_prediction_matches"] += prediction == strict_prediction
        report["float_reference_prediction_matches"] += prediction == float_prediction
        report["correct_labels"] += prediction == metadata[index]["label"]
        if baseline:
            old_layers = baseline[index][-1]
            old_prediction = int(np.argmax(old_layers["logits"]))
            if old_prediction != prediction:
                raise AssertionError(f"frame {index}: baseline prediction changed")
            report["baseline_prediction_matches"] += old_prediction == prediction
            for name in layers:
                if name not in ("f1", "logits"):
                    report["baseline_changed_codes"] += int(np.count_nonzero(
                        old_layers[name] != layers[name]))
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "selected_frames"},
                     indent=2))
    print("PASS: trained CNN integer layers, scales, logits and predictions")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (AssertionError, ValueError, OSError, subprocess.SubprocessError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        sys.exit(1)
