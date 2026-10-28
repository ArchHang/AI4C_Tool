"""
RNN / LSTM C 推理库详细验证（fp32 与 int8 动态量化）。

对拍内容：
  1) C fp32 vs Python 严格复现（同一累加顺序，float32）：逐时间步、逐单元；
  2) C int8 vs Python 严格复现（整数 MAC + 每步动态量化）：h_q/scale/zp/logits；
  3) C fp32 vs PyTorch（导出权重重建模型）输出一致性；
  4) C fp32 / int8 在测试集上的分类准确率。

用法：
  python tools/verify_models.py [--n 1000] [--models rnn,lstm]
"""

import argparse
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_models import make_dataset  # noqa: E402
from speaker_benchmark.activation_reference import PortableMath  # noqa: E402

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
MODEL_DIR = os.path.join(ROOT, "model")
SEQ_LEN = 16
X_SCALE = np.float32(1.0 / 255.0)
X_ZP = 0
FP32_HIDDEN_ATOL = 2e-6
LOGITS_ATOL = 1e-5
TORCH_ATOL = 5e-4
PORTABLE_MATH = PortableMath()


def tanhf_reference(value, legacy_math=False):
    """Default matches the portable runtime; legacy NumPy libm stays diagnostic."""
    if legacy_math:
        return np.tanh(np.float32(value)).astype(np.float32)
    return PORTABLE_MATH.apply("tanh", value)


def require_close(name, actual, expected, atol, rtol=0.0):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(expected)):
        raise AssertionError(f"{name}: nonfinite values")
    if not np.allclose(actual, expected, atol=atol, rtol=rtol):
        error = float(np.max(np.abs(actual - expected)))
        raise AssertionError(f"{name}: max difference {error:.9g} exceeds tolerance")


def require_equal(name, actual, expected):
    if not np.array_equal(actual, expected):
        mismatches = int(np.count_nonzero(np.asarray(actual) != np.asarray(expected)))
        raise AssertionError(f"{name}: {mismatches} mismatches")


def hidden_quantization(raw):
    """Match the C constant-range fallback, preserving the normal min/max contract."""
    mn = np.float32(raw.min())
    mx = np.float32(raw.max())
    scale = np.float32((mx - mn) / np.float32(255.0))
    if scale < np.float32(1e-9):
        magnitude = np.float32(max(abs(mn), abs(mx)))
        if magnitude > np.float32(1e-9):
            scale = np.float32(magnitude / np.float32(128.0))
            zero_point = 255 if mn < np.float32(0.0) else 0
        else:
            scale = np.float32(1e-9)
            zero_point = 0
    else:
        zero_point = max(0, min(255, int(np.float32(-mn) / scale)))
    return scale, zero_point


def load_params(kind):
    return np.load(os.path.join(MODEL_DIR, f"{kind}_params.npz"))


def rnn_fp32_ref(x, p, legacy_math=False):
    """复现 C rnn_forward_fp32（float32，累加顺序一致）。返回 h_seq (T,H)。"""
    T, I = x.shape
    H = int(p["hidden_size"])
    w_ih = p["w_ih"].astype(np.float32)
    w_hh = p["w_hh"].astype(np.float32)
    b_ih = p["b_ih"].astype(np.float32)
    b_hh = p["b_hh"].astype(np.float32)
    h = np.zeros((T, H), dtype=np.float32)
    for t in range(T):
        hp = h[t - 1] if t > 0 else None
        for j in range(H):
            acc = np.float32(b_ih[j]) + np.float32(b_hh[j])
            for i in range(I):
                acc = acc + np.float32(w_ih[j, i]) * np.float32(x[t, i])
            if hp is not None:
                for i in range(H):
                    acc = acc + np.float32(w_hh[j, i]) * np.float32(hp[i])
            h[t, j] = tanhf_reference(acc, legacy_math)
    return h


def sigmoidf(v, legacy_math=False):
    if legacy_math:
        return (np.float32(1.0) / (np.float32(1.0) + np.exp(np.float32(-v)))).astype(np.float32)
    return PORTABLE_MATH.apply("sigmoid", v)


def lstm_fp32_ref(x, p, legacy_math=False):
    """复现 C lstm_forward_fp32。返回 h_seq (T,H)。"""
    T, I = x.shape
    H = int(p["hidden_size"])
    w_ih = p["w_ih"].astype(np.float32)
    w_hh = p["w_hh"].astype(np.float32)
    bias = p["bias"].astype(np.float32)
    h = np.zeros((T, H), dtype=np.float32)
    c = np.zeros(H, dtype=np.float32)
    for t in range(T):
        hp = h[t - 1] if t > 0 else None
        for j in range(H):
            rows = (j, H + j, 2 * H + j, 3 * H + j)
            y = [np.float32(bias[r]) for r in rows]
            for i in range(I):
                xv = np.float32(x[t, i])
                for k, r in enumerate(rows):
                    y[k] = y[k] + np.float32(w_ih[r, i]) * xv
            if hp is not None:
                for i in range(H):
                    hv = np.float32(hp[i])
                    for k, r in enumerate(rows):
                        y[k] = y[k] + np.float32(w_hh[r, i]) * hv
            gi = sigmoidf(y[0], legacy_math)
            gf = sigmoidf(y[1], legacy_math)
            gg = tanhf_reference(y[2], legacy_math)
            go = sigmoidf(y[3], legacy_math)
            cnew = gf * np.float32(c[j]) + gi * gg
            c[j] = cnew
            h[t, j] = go * tanhf_reference(cnew, legacy_math)
    return h


def quantize_u8_py(v, scale, zp):
    q = np.rint((v.astype(np.float32) / np.float32(scale)).astype(np.float64)).astype(np.int64) + zp
    return np.clip(q, 0, 255).astype(np.uint8)


def rnn_int8_ref(x_q, p, legacy_math=False):
    """复现 C rnn_forward_int8。返回 h_q (T,H), scale (T,), zp (T,), h_f32 (T,H)。"""
    T, I = x_q.shape
    H = int(p["hidden_size"])
    w_ih = p["w_ih_q"].astype(np.int64)
    w_hh = p["w_hh_q"].astype(np.int64)
    w_ih_s = (p["w_ih_scale"].astype(np.float32) * X_SCALE)
    w_hh_s = p["w_hh_scale"].astype(np.float32)
    b_ih = p["b_ih"].astype(np.float32)
    b_hh = p["b_hh"].astype(np.float32)
    h_q = np.zeros((T, H), dtype=np.uint8)
    h_f = np.zeros((T, H), dtype=np.float32)
    h_s = np.zeros(T, dtype=np.float32)
    h_z = np.zeros(T, dtype=np.int32)
    hprev_scale = np.float32(1.0)
    hprev_zp = 0
    for t in range(T):
        xt = x_q[t].astype(np.int64)
        hprev_q = h_q[t - 1].astype(np.int64) if t > 0 else None
        acc_ih = w_ih @ (xt - X_ZP)
        acc_hh = w_hh @ (hprev_q - hprev_zp) if hprev_q is not None else np.zeros(H, np.int64)
        y = (acc_ih.astype(np.float32) * w_ih_s + b_ih +
             acc_hh.astype(np.float32) * (w_hh_s * hprev_scale) + b_hh)
        raw = tanhf_reference(y, legacy_math)
        scale, zp = hidden_quantization(raw)
        h_q[t] = quantize_u8_py(raw, scale, zp)
        h_f[t] = raw
        h_s[t] = scale
        h_z[t] = zp
        hprev_scale = scale
        hprev_zp = zp
    return h_q, h_s, h_z, h_f


def lstm_int8_ref(x_q, p, legacy_math=False):
    """复现 C lstm_forward_int8。返回 h_q (T,H), scale (T,), zp (T,), h_f32 (T,H)。"""
    T, I = x_q.shape
    H = int(p["hidden_size"])
    w_ih = p["w_ih_q"].reshape(4, H, I).astype(np.int64)
    w_hh = p["w_hh_q"].reshape(4, H, H).astype(np.int64)
    w_ih_s = (p["w_ih_scale"].reshape(4, H).astype(np.float32) * X_SCALE)
    w_hh_s = p["w_hh_scale"].reshape(4, H).astype(np.float32)
    bias = p["bias"].reshape(4, H).astype(np.float32)
    h_q = np.zeros((T, H), dtype=np.uint8)
    h_f = np.zeros((T, H), dtype=np.float32)
    h_s = np.zeros(T, dtype=np.float32)
    h_z = np.zeros(T, dtype=np.int32)
    c = np.zeros(H, dtype=np.float32)
    hprev_scale = np.float32(1.0)
    hprev_zp = 0
    for t in range(T):
        xt = x_q[t].astype(np.int64)
        hprev_q = h_q[t - 1].astype(np.int64) if t > 0 else None
        acc_ih = np.tensordot(w_ih, xt - X_ZP, axes=1)  # (4,H)
        acc_hh = (np.tensordot(w_hh, hprev_q - hprev_zp, axes=1)
                  if hprev_q is not None else np.zeros((4, H), np.int64))
        y = (acc_ih.astype(np.float32) * w_ih_s +
             acc_hh.astype(np.float32) * (w_hh_s * hprev_scale) + bias)
        gi = sigmoidf(y[0], legacy_math)
        gf = sigmoidf(y[1], legacy_math)
        gg = tanhf_reference(y[2], legacy_math)
        go = sigmoidf(y[3], legacy_math)
        c = gf * c + gi * gg
        raw = (go * tanhf_reference(c, legacy_math)).astype(np.float32)
        scale, zp = hidden_quantization(raw)
        h_q[t] = quantize_u8_py(raw, scale, zp)
        h_f[t] = raw
        h_s[t] = scale
        h_z[t] = zp
        hprev_scale = scale
        hprev_zp = zp
    return h_q, h_s, h_z, h_f


def linear_int8_ref(h_q, h_scale, h_zp, p):
    acc = (h_q.astype(np.int64) - h_zp) @ p["out_w_q"].reshape(-1, h_q.size).T.astype(np.int64)
    return (acc.astype(np.float32) *
            (p["out_w_scale"].astype(np.float32) * np.float32(h_scale)) +
            p["out_b"].astype(np.float32))


def linear_fp32_ref(h_last, p):
    return (h_last.astype(np.float32) @ p["out_w"].astype(np.float32).T +
            p["out_b"].astype(np.float32))


def build_torch_model(kind, p):
    I = int(p["input_size"])
    H = int(p["hidden_size"])
    C = int(p["num_classes"])
    if kind == "rnn":
        rnn = nn.RNN(I, H, batch_first=True, nonlinearity="tanh")
        with torch.no_grad():
            rnn.weight_ih_l0.copy_(torch.from_numpy(p["w_ih"]))
            rnn.bias_ih_l0.copy_(torch.from_numpy(p["b_ih"]))
            rnn.weight_hh_l0.copy_(torch.from_numpy(p["w_hh"]))
            rnn.bias_hh_l0.copy_(torch.from_numpy(p["b_hh"]))
    else:
        rnn = nn.LSTM(I, H, batch_first=True)
        with torch.no_grad():
            rnn.weight_ih_l0.copy_(torch.from_numpy(p["w_ih"]))
            rnn.bias_ih_l0.copy_(torch.from_numpy(p["bias"] * 0.5))
            rnn.bias_hh_l0.copy_(torch.from_numpy(p["bias"] * 0.5))
            rnn.weight_hh_l0.copy_(torch.from_numpy(p["w_hh"]))
    head = nn.Linear(H, C)
    with torch.no_grad():
        head.weight.copy_(torch.from_numpy(p["out_w"]))
        head.bias.copy_(torch.from_numpy(p["out_b"]))
    rnn.eval()
    head.eval()
    return rnn, head


def run_c(kind, x_f32, x_q, n, exe_dir, tmp):
    xf = os.path.join(tmp, f"{kind}_x_f32.bin")
    xu = os.path.join(tmp, f"{kind}_x_u8.bin")
    df = os.path.join(tmp, f"{kind}_fp32.bin")
    di = os.path.join(tmp, f"{kind}_int8.bin")
    x_f32.astype(np.float32).tofile(xf)
    x_q.astype(np.uint8).tofile(xu)
    exe = os.path.join(exe_dir, f"test_{kind}.exe")
    r = subprocess.run([exe, xu, xf, str(n), df, di], capture_output=True, timeout=120)
    if r.returncode != 0:
        print(r.stdout.decode(errors="ignore"), r.stderr.decode(errors="ignore"))
        sys.exit(1)
    print(r.stdout.decode(errors="ignore").strip())
    return df, di


def read_dumps(kind, n, p, df, di):
    T = SEQ_LEN
    H = int(p["hidden_size"])
    C = int(p["num_classes"])
    h_f = np.fromfile(df, dtype=np.float32).reshape(n, T * H + C)
    h_seq = h_f[:, :T * H].reshape(n, T, H)
    logits_fp32 = h_f[:, T * H:].reshape(n, C)
    raw = np.fromfile(di, dtype=np.uint8)
    per = T * H + 4 * T + 4 * T + 4 * C
    raw = raw.reshape(n, per)
    h_q = raw[:, :T * H].reshape(n, T, H)
    off = T * H
    h_s = raw[:, off:off + 4 * T].view("<f4").reshape(n, T)
    off += 4 * T
    h_z = raw[:, off:off + 4 * T].view("<i4").reshape(n, T)
    off += 4 * T
    logits_int8 = raw[:, off:off + 4 * C].view("<f4").reshape(n, C)
    return h_seq, logits_fp32, h_q, h_s, h_z, logits_int8


def cmp_float(name, a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    exact = int((a == b).sum())
    diff = np.abs(a - b)
    return {"exact": exact, "total": a.size,
            "max": float(diff.max()) if a.size else 0.0,
            "le1e-6": int((diff <= 1e-6).sum())}


def verify_one(kind, n, exe_dir, use_torch=True, accuracy_only=False, device="cpu"):
    if kind not in ("rnn", "lstm"):
        raise ValueError(f"Unknown model: {kind}")
    p = load_params(kind)
    x, y = make_dataset(n, SEQ_LEN, int(p["input_size"]), int(p["num_classes"]),
                        seed=20260807 + 1)
    x_q = np.clip(np.round(x * 255.0), 0, 255).astype(np.uint8)

    with tempfile.TemporaryDirectory(prefix=f"verify_{kind}_", dir=ROOT) as tmp:
        df, di = run_c(kind, x, x_q, n, exe_dir, tmp)
        h_seq_c, lg_fp32_c, h_q_c, h_s_c, h_z_c, lg_int8_c = read_dumps(kind, n, p, df, di)
        # fromfile arrays own their memory, so temporary files can be removed here.
    require_close("fp32 logits finite", lg_fp32_c, lg_fp32_c, 0.0)
    require_close("int8 logits finite", lg_int8_c, lg_int8_c, 0.0)

    print(f"\n===== {kind.upper()} 验证 (n={n}) =====")

    if accuracy_only:
        pred_c = lg_fp32_c.argmax(1)
        pred_i_c = lg_int8_c.argmax(1)
        print(f"fp32 准确率(标签): {(pred_c == y).sum()}/{n}")
        print(f"int8 准确率(标签): {(pred_i_c == y).sum()}/{n}")
        print(f"int8 vs fp32 预测一致(C): {(pred_i_c == pred_c).sum()}/{n}")
        require_equal("fp32 marker-recall labels", pred_c, y)
        require_equal("int8 marker-recall labels", pred_i_c, y)
        require_equal("int8/fp32 predictions", pred_i_c, pred_c)
        return

    # 1) C fp32 vs Python 严格复现（逐样本逐时间步）
    h_seq_py = np.stack([rnn_fp32_ref(x[i], p) if kind == "rnn" else lstm_fp32_ref(x[i], p)
                         for i in range(n)])
    lg_py = np.stack([linear_fp32_ref(h_seq_py[i, -1], p) for i in range(n)])
    st = cmp_float("h_seq", h_seq_c, h_seq_py)
    stl = cmp_float("logits", lg_fp32_c, lg_py)
    pred_c = lg_fp32_c.argmax(1)
    pred_py = lg_py.argmax(1)
    print(f"fp32 h_seq : 逐元素一致 {st['exact']}/{st['total']}  "
          f"max_diff={st['max']:.3e}  <=1e-6: {st['le1e-6']}/{st['total']}")
    print(f"fp32 logits: 逐元素一致 {stl['exact']}/{stl['total']}  "
          f"max_diff={stl['max']:.3e}  <=1e-6: {stl['le1e-6']}/{stl['total']}")
    print(f"fp32 预测一致 C vs Python: {(pred_c == pred_py).sum()}/{n}")
    print(f"fp32 准确率(标签): C={(pred_c == y).sum()}/{n}  "
          f"Python={(pred_py == y).sum()}/{n}")

    require_close("fp32 hidden states", h_seq_c, h_seq_py, FP32_HIDDEN_ATOL)
    require_close("fp32 logits", lg_fp32_c, lg_py, LOGITS_ATOL)
    require_equal("fp32 Python predictions", pred_c, pred_py)
    require_equal("fp32 marker-recall labels", pred_c, y)

    # 2) C int8 vs Python 严格复现
    hq_py = np.zeros_like(h_q_c)
    hs_py = np.zeros_like(h_s_c)
    hz_py = np.zeros_like(h_z_c)
    for i in range(n):
        if kind == "rnn":
            hq, hs, hz, _ = rnn_int8_ref(x_q[i], p)
        else:
            hq, hs, hz, _ = lstm_int8_ref(x_q[i], p)
        hq_py[i] = hq
        hs_py[i] = hs
        hz_py[i] = hz
    q_eq = int((h_q_c == hq_py).sum())
    s_eq = int((h_s_c == hs_py).sum())
    z_eq = int((h_z_c == hz_py).sum())
    print(f"int8 h_q   : 逐元素一致 {q_eq}/{h_q_c.size}")
    print(f"int8 scale : 一致 {s_eq}/{h_s_c.size}")
    print(f"int8 zp    : 一致 {z_eq}/{h_z_c.size}")

    lg_i_py = np.stack([
        linear_int8_ref(hq_py[i, -1], hs_py[i, -1], int(hz_py[i, -1]), p) for i in range(n)
    ])
    sti = cmp_float("logits", lg_int8_c, lg_i_py)
    pred_i_c = lg_int8_c.argmax(1)
    pred_i_py = lg_i_py.argmax(1)
    print(f"int8 logits: 逐元素一致 {sti['exact']}/{sti['total']}  "
          f"max_diff={sti['max']:.3e}  <=1e-6: {sti['le1e-6']}/{sti['total']}")
    print(f"int8 预测一致 C vs Python: {(pred_i_c == pred_i_py).sum()}/{n}")
    print(f"int8 准确率(标签): C={(pred_i_c == y).sum()}/{n}  "
          f"Python={(pred_i_py == y).sum()}/{n}")
    print(f"int8 vs fp32 预测一致(C): {(pred_i_c == pred_c).sum()}/{n}")

    require_equal("int8 hidden codes", h_q_c, hq_py)
    require_equal("int8 zero points", h_z_c, hz_py)
    require_close("int8 scales", h_s_c, hs_py, 1e-9, 1e-6)
    require_close("int8 logits", lg_int8_c, lg_i_py, LOGITS_ATOL)
    require_equal("int8 Python predictions", pred_i_c, pred_i_py)
    require_equal("int8/fp32 predictions", pred_i_c, pred_c)
    require_equal("int8 marker-recall labels", pred_i_c, y)

    # 3) C fp32 vs PyTorch
    if use_torch:
        rnn, head = build_torch_model(kind, p)
        dev = torch.device(device)
        rnn.to(dev)
        head.to(dev)
        with torch.no_grad():
            xt = torch.from_numpy(x).to(dev)
            if kind == "rnn":
                h, _ = rnn(xt)
            else:
                h, _ = rnn(xt)
            lg = head(h[:, -1, :])
            h_t = h.cpu().numpy()
            lg_t = lg.cpu().numpy()
        stt = cmp_float("h_seq(torch)", h_seq_c, h_t)
        stlt = cmp_float("logits(torch)", lg_fp32_c, lg_t)
        print(f"torch h_seq : max_diff={stt['max']:.3e}  <=1e-5: "
              f"{int((np.abs(h_seq_c - h_t) <= 1e-5).sum())}/{h_seq_c.size}")
        print(f"torch logits: max_diff={stlt['max']:.3e}  "
              f"预测一致: {(lg_t.argmax(1) == pred_c).sum()}/{n}")
        require_close("PyTorch hidden states", h_seq_c, h_t, TORCH_ATOL)
        require_close("PyTorch logits", lg_fp32_c, lg_t, TORCH_ATOL)
        require_equal("PyTorch predictions", lg_t.argmax(1), pred_c)
    print(f"PASS {kind}: all correctness contracts satisfied")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--models", default="rnn,lstm")
    ap.add_argument("--exe-dir", default=ROOT)
    ap.add_argument("--no-torch", action="store_true")
    ap.add_argument("--accuracy-only", action="store_true")
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu",
                    help="PyTorch reference device; CPU is portable and deterministic")
    args = ap.parse_args()
    if args.n <= 0:
        ap.error("--n must be positive")
    failures = []
    for kind in args.models.split(","):
        try:
            verify_one(kind.strip(), args.n, args.exe_dir, use_torch=not args.no_torch,
                       accuracy_only=args.accuracy_only, device=args.device)
        except (AssertionError, ValueError, RuntimeError, OSError, subprocess.TimeoutExpired) as error:
            failures.append(f"{kind}: {error}")
            print(f"FAIL {failures[-1]}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
