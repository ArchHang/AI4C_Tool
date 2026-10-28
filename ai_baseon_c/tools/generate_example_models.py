"""Generate small, untrained CNN/RNN/LSTM checkpoints on CPU for GUI/CLI demos.

No recordings, pretrained speaker weights, GPU or network are used. These are
schema/functionality examples and must not be described as recognition models.
"""

import argparse
import hashlib
import json
from pathlib import Path


def generate_examples(output_dir, seed=20261028):
    if type(seed) is not int or not 0 <= seed < 2 ** 63:
        raise ValueError("seed must be an integer in [0,2**63)")
    output = Path(output_dir).absolute()
    if output.exists():
        raise FileExistsError("Use a new output directory; existing models are preserved")
    import torch
    from model_converter.schema import parse_spec
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def weights(*shape):
        return torch.randn(shape, generator=generator, device="cpu") * .05

    examples = []
    state, channels, width = {}, 1, 64
    for index, outputs in zip((0, 4, 8), (2, 3, 4)):
        state[f"model.{index}.weight"] = weights(outputs, channels, 5)
        state[f"model.{index}.bias"] = torch.zeros(outputs)
        for suffix, value in (("weight", 1.), ("bias", 0.), ("running_mean", 0.), ("running_var", 1.)):
            state[f"model.{index + 1}.{suffix}"] = torch.full((outputs,), value)
        state[f"model.{index + 1}.num_batches_tracked"] = torch.tensor(0, dtype=torch.int64)
        channels, width = outputs, (width - 4) // 2
    state.update({"fc1.weight": weights(5, channels * width), "fc1.bias": torch.zeros(5),
                  "fc2.weight": weights(3, 5), "fc2.bias": torch.zeros(3)})
    examples.append(("cnn", state, {"schema": "cnn_voice_v1", "input_length": 64}))
    for kind, gates in (("rnn", 1), ("lstm", 4)):
        state = {"rnn.weight_ih_l0": weights(gates * 5, 3), "rnn.weight_hh_l0": weights(gates * 5, 5),
                 "rnn.bias_ih_l0": torch.zeros(gates * 5), "rnn.bias_hh_l0": torch.zeros(gates * 5),
                 "head.weight": weights(3, 5), "head.bias": torch.zeros(3)}
        examples.append((kind, state, {"schema": "rnn_tanh_v1" if kind == "rnn" else "lstm_ifgo_v1",
                                       "seq_length": 7}))
    prepared = []
    for kind, state, metadata in examples:
        metadata = {**metadata, "example_only": True, "training_status": "untrained_synthetic"}
        spec = parse_spec(state, metadata)
        prepared.append((kind, state, metadata, spec.dimensions))
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "synthetic_example_models_v1", "trained": False, "contains_voice_data": False,
              "device": "cpu", "seed": seed, "torch_version": str(torch.__version__), "models": []}
    for kind, state, metadata, dimensions in prepared:
        path = output / (kind + ".pth")
        torch.save({"state_dict": state, "metadata": metadata}, path)
        report["models"].append({"architecture": kind, "checkpoint": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "metadata": metadata,
            "dimensions": dimensions})
    (output / "examples.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261028)
    args = parser.parse_args(argv)
    report = generate_examples(**vars(args))
    print(json.dumps({"generated": len(report["models"]), "trained": False, "device": "cpu"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
