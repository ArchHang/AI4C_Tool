"""
训练 RNN / LSTM（GPU）并导出 fp32 与 int8 动态量化参数。

任务：标记回忆（marker recall）。
  输入序列长度 T=16，每步 4 维；序列中随机一个位置出现值为 1 的标记
  （所在维度即类别），其余位置为 0。模型需要记住标记维度直到序列结束，
  输出 4 分类 logits。

流程：
  1) 生成训练/测试集（固定随机种子，可复现）；
  2) GPU 训练 nn.RNN / nn.LSTM + 分类头；
  3) 权重 per-row 对称 int8 量化，导出：
       - model/rnn_params.npz / lstm_params.npz（Python 验证用）
       - examples/models/{rnn,lstm}_model.h（兼容旧测试入口）
       - model/{rnn,lstm}.pth（张量 checkpoint + schema 元数据）
       - model/generated/{rnn,lstm}/（完整 C 模型与独立运行库）

用法：
  python tools/train_models.py [--epochs 30] [--hidden 32] [--seq-len 16]
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn

from model_converter.cli import convert_checkpoint
from model_converter.emitter import c_array
from model_converter.quantization import quantize_rows

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
MODEL_DIR = os.path.join(ROOT, "model")


def make_dataset(n, seq_len, input_dim, num_classes, seed):
    """生成标记回忆任务数据。x: (n, seq_len, input_dim)，标签为标记所在维度。"""
    rng = np.random.RandomState(seed)
    x = np.zeros((n, seq_len, input_dim), dtype=np.float32)
    y = rng.randint(0, num_classes, size=n)
    pos = rng.randint(0, seq_len, size=n)
    for i in range(n):
        x[i, pos[i], y[i]] = 1.0
    return x, y


def build_model(kind, input_dim, hidden, num_classes):
    if kind == "rnn":
        rnn = nn.RNN(input_dim, hidden, batch_first=True, nonlinearity="tanh")
    else:
        rnn = nn.LSTM(input_dim, hidden, batch_first=True)
    head = nn.Linear(hidden, num_classes)
    return rnn, head


def forward(kind, rnn, head, x):
    if kind == "rnn":
        h, _ = rnn(x)
    else:
        h, _ = rnn(x)
    return head(h[:, -1, :])


def train(kind, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{kind}] device={device}")

    x_tr, y_tr = make_dataset(args.train_n, args.seq_len, args.input_dim, args.num_classes,
                              seed=args.seed)
    x_te, y_te = make_dataset(args.test_n, args.seq_len, args.input_dim, args.num_classes,
                              seed=args.seed + 1)

    x_tr_t = torch.from_numpy(x_tr).to(device)
    y_tr_t = torch.from_numpy(y_tr).long().to(device)
    x_te_t = torch.from_numpy(x_te).to(device)
    y_te_t = torch.from_numpy(y_te).long().to(device)

    rnn, head = build_model(kind, args.input_dim, args.hidden, args.num_classes)
    rnn.to(device)
    head.to(device)
    opt = torch.optim.Adam(list(rnn.parameters()) + list(head.parameters()), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    n_batch = (x_tr_t.size(0) + args.batch - 1) // args.batch
    for ep in range(args.epochs):
        rnn.train()
        head.train()
        total_loss = 0.0
        perm = torch.randperm(x_tr_t.size(0), device=device)
        for b in range(n_batch):
            idx = perm[b * args.batch:(b + 1) * args.batch]
            xb, yb = x_tr_t[idx], y_tr_t[idx]
            opt.zero_grad()
            out = forward(kind, rnn, head, xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        rnn.eval()
        head.eval()
        with torch.no_grad():
            out = forward(kind, rnn, head, x_te_t)
            acc = float((out.argmax(1) == y_te_t).float().mean())
        if (ep + 1) % 5 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep + 1:>3}/{args.epochs}  loss={total_loss / n_batch:.4f}  "
                  f"test_acc={acc * 100:.2f}%")
    return rnn, head


def quantize_row_wise(w):
    """per-row（每个输出单元）对称 int8 量化。w: (rows, cols) float32。"""
    wq, scale, _, _ = quantize_rows(w)
    return wq, scale


def export_params(kind, rnn, head, args):
    os.makedirs(MODEL_DIR, exist_ok=True)
    sd = {}
    if kind == "rnn":
        w_ih = rnn.weight_ih_l0.detach().cpu().numpy()
        b_ih = rnn.bias_ih_l0.detach().cpu().numpy()
        w_hh = rnn.weight_hh_l0.detach().cpu().numpy()
        b_hh = rnn.bias_hh_l0.detach().cpu().numpy()
        bias = None
    else:
        w_ih = rnn.weight_ih_l0.detach().cpu().numpy()
        b_ih = rnn.bias_ih_l0.detach().cpu().numpy()
        w_hh = rnn.weight_hh_l0.detach().cpu().numpy()
        b_hh = rnn.bias_hh_l0.detach().cpu().numpy()
        bias = (b_ih + b_hh).astype(np.float32)  # LSTM 合并两个偏置
        b_ih = None
        b_hh = None
    out_w = head.weight.detach().cpu().numpy()
    out_b = head.bias.detach().cpu().numpy()

    w_ih_q, w_ih_s = quantize_row_wise(w_ih)
    w_hh_q, w_hh_s = quantize_row_wise(w_hh)
    out_w_q, out_w_s = quantize_row_wise(out_w)
    w_ih_rowsum = w_ih_q.astype(np.int32).sum(axis=1)
    w_hh_rowsum = w_hh_q.astype(np.int32).sum(axis=1)

    d = {
        "input_size": np.int32(args.input_dim),
        "hidden_size": np.int32(args.hidden),
        "num_classes": np.int32(args.num_classes),
        "w_ih": w_ih.astype(np.float32),
        "w_hh": w_hh.astype(np.float32),
        "out_w": out_w.astype(np.float32),
        "out_b": out_b.astype(np.float32),
        "w_ih_q": w_ih_q,
        "w_ih_scale": w_ih_s,
        "w_ih_rowsum": w_ih_rowsum,
        "w_hh_q": w_hh_q,
        "w_hh_scale": w_hh_s,
        "w_hh_rowsum": w_hh_rowsum,
        "out_w_q": out_w_q,
        "out_w_scale": out_w_s,
    }
    if kind == "rnn":
        d["b_ih"] = b_ih.astype(np.float32)
        d["b_hh"] = b_hh.astype(np.float32)
    else:
        d["bias"] = bias.astype(np.float32)

    npz_path = os.path.join(MODEL_DIR, f"{kind}_params.npz")
    np.savez(npz_path, **d)
    legacy_model_dir = os.path.join(ROOT, "examples", "models")
    os.makedirs(legacy_model_dir, exist_ok=True)
    write_c_header(kind, d, os.path.join(legacy_model_dir, f"{kind}_model.h"))

    # Store graph semantics explicitly; tensor shapes alone do not encode RNN nonlinearity.
    state = {"rnn." + name: value.detach().cpu() for name, value in rnn.state_dict().items()}
    state.update({"head." + name: value.detach().cpu() for name, value in head.state_dict().items()})
    checkpoint_path = os.path.join(MODEL_DIR, f"{kind}.pth")
    torch.save({"state_dict": state, "metadata": {
        "schema": "rnn_tanh_v1" if kind == "rnn" else "lstm_ifgo_v1",
        "seq_length": args.seq_len, "nonlinearity": "tanh"}}, checkpoint_path)
    generated_dir = os.path.join(MODEL_DIR, "generated", kind)
    convert_checkpoint(checkpoint_path, generated_dir, prefix=f"{kind}_model", precision="int8",
                       bundle_runtime=True)
    print(f"exported: {npz_path}, {checkpoint_path}, {generated_dir}, "
          f"examples/models/{kind}_model.h (legacy harness compatibility)")


def fmt_float_array(name, arr):
    return c_array(name, np.asarray(arr, dtype=np.float32)).splitlines()


def fmt_int8_array(name, arr):
    return c_array(name, np.asarray(arr, dtype=np.int8)).splitlines()


def fmt_int32_array(name, arr):
    return c_array(name, np.asarray(arr, dtype=np.int32)).splitlines()


def write_c_header(kind, d, path):
    h = d["hidden_size"].item()
    i = d["input_size"].item()
    c = d["num_classes"].item()
    lines = [
        "/* 自动生成：tools/train_models.py，请勿手改 */",
        f"#ifndef AI_BASEON_C_{kind.upper()}_MODEL_H_",
        f"#define AI_BASEON_C_{kind.upper()}_MODEL_H_",
        "",
        "#include <stdint.h>",
        "",
        f"#define {kind.upper()}_INPUT_SIZE {i}",
        f"#define {kind.upper()}_HIDDEN_SIZE {h}",
        f"#define {kind.upper()}_NUM_CLASSES {c}",
        "",
    ]
    for name in ("w_ih", "w_hh", "out_w", "out_b"):
        lines += fmt_float_array(f"{kind}_{name}", d[name])
        lines.append("")
    for name in ("w_ih_q", "w_hh_q", "out_w_q"):
        lines += fmt_int8_array(f"{kind}_{name}", d[name])
        lines.append("")
    for name in ("w_ih_scale", "w_hh_scale", "out_w_scale"):
        lines += fmt_float_array(f"{kind}_{name}", d[name])
        lines.append("")
    for name in ("w_ih_rowsum", "w_hh_rowsum"):
        lines += fmt_int32_array(f"{kind}_{name}", d[name])
        lines.append("")
    if kind == "rnn":
        for name in ("b_ih", "b_hh"):
            lines += fmt_float_array(f"{kind}_{name}", d[name])
            lines.append("")
    else:
        lines += fmt_float_array("lstm_bias", d["bias"])
        lines.append("")
    lines.append(f"#endif  // AI_BASEON_C_{kind.upper()}_MODEL_H_")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--input-dim", type=int, default=4)
    ap.add_argument("--num-classes", type=int, default=4)
    ap.add_argument("--train-n", type=int, default=20000)
    ap.add_argument("--test-n", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=20260807)
    args = ap.parse_args()

    for kind in ("rnn", "lstm"):
        rnn, head = train(kind, args)
        export_params(kind, rnn, head, args)


if __name__ == "__main__":
    main()
