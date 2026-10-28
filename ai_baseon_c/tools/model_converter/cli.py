"""Command-line frontend for known-schema model conversion."""

import argparse
from pathlib import Path
import sys

from .emitter import emit_model
from .loaders import load_checkpoint
from .schema import ConversionError, parse_spec


def convert_checkpoint(checkpoint, output_dir, architecture="auto", precision="int8",
                       prefix="ai_model", input_length=None, seq_length=None,
                       input_scale=1 / 255, input_zero_point=0, trusted_module=False,
                       bundle_runtime=False, weight_scaling="maxabs", mse_min_ratio=0.5,
                       mse_steps=101, cache_policy="none"):
    state, metadata, digest = load_checkpoint(checkpoint, trusted_module)
    spec = parse_spec(state, metadata, architecture, input_length, seq_length)
    manifest = emit_model(spec, output_dir, prefix, precision, input_scale, input_zero_point, digest,
                          weight_scaling, mse_min_ratio, mse_steps, cache_policy=cache_policy)
    if bundle_runtime:
        import json
        from library_sources import export_runtime
        components = ("cnn",) if spec.architecture == "cnn" else ("cnn", spec.architecture)
        runtime = export_runtime(Path(output_dir) / "runtime", components=components)
        source = Path(output_dir) / "model.c"
        source.write_text(source.read_text(encoding="utf-8").replace('"ai_baseon_c/', '"runtime/'),
                          encoding="utf-8")
        manifest["bundled_runtime"] = runtime
        (Path(output_dir) / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                                        encoding="utf-8")
        with (Path(output_dir) / "README.txt").open("a", encoding="utf-8") as stream:
            sources = " ".join("runtime/" + name for name in runtime["sources"])
            stream.write("\nSelf-contained bundled runtime build (no library installation required):\n"
                         f"  cc -std=c99 -O3 -ffp-contract=off -fno-fast-math -Iruntime model.c example.c {sources} -lm -o model_example\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export known CNN/RNN/LSTM .pth checkpoints to heap-free C")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--architecture", choices=("auto", "cnn", "rnn", "lstm"), default="auto",
                        help="Explicit architecture declares the supported graph semantics for a plain state_dict")
    parser.add_argument("--precision", choices=("int8", "fp32", "both"), default="int8")
    parser.add_argument("--prefix", default="ai_model")
    parser.add_argument("--input-length", type=int, help="CNN input width; cnn_voice_v1 defaults to 2048")
    parser.add_argument("--seq-length", type=int, help="Fixed recurrent sequence length; required without metadata")
    parser.add_argument("--input-scale", type=float, default=1 / 255)
    parser.add_argument("--input-zero-point", type=int, default=0)
    parser.add_argument("--weight-scaling", choices=("maxabs", "mse"), default="maxabs",
                        help="Per-row scale selection; mse may clip weights to reduce reconstruction MSE")
    parser.add_argument("--mse-min-ratio", type=float, default=0.5,
                        help="Smallest maxabs scale multiplier searched by mse (default 0.5)")
    parser.add_argument("--mse-steps", type=int, default=101,
                        help="Number of candidate ratios including the exact maxabs baseline (default 101)")
    parser.add_argument("--cache-policy", choices=("none", "packed"), default="none",
                        help="Optional small Conv weight prepacking in workspace RAM; accelerates ARM DSP, "
                             "requires model_init and additional RAM; other backends retain their native path")
    parser.add_argument("--trusted-module", action="store_true",
                        help="Allow only the known legacy CNNVoiceClassifier and audited torch module types")
    parser.add_argument("--bundle-runtime", action="store_true",
                        help="Copy the required C runtime into the export for a self-contained build")
    args = parser.parse_args(argv)
    try:
        manifest = convert_checkpoint(**vars(args))
    except (ConversionError, OSError) as error:
        print(f"conversion failed: {error}", file=sys.stderr)
        return 2
    print(f"exported {manifest['schema']} ({manifest['precision']}) to {args.output_dir.absolute()}")
    print(f"input={manifest['dimensions']['input_elements']}, "
          f"output={manifest['dimensions']['num_classes']}, parameters={manifest['parameter_bytes']} bytes")
    return 0
