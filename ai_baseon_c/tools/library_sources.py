"""Canonical source inventory and deterministic, flat embedded runtime export.

Export changes only quoted include paths, retaining copyright notices and kernels.
Historical deployment snapshots are independent of this evolving source tree.
"""

import hashlib
from pathlib import Path


LIBRARY = Path(__file__).absolute().parents[1]
COMPONENTS = ("cnn", "rnn", "lstm")


def source_files(library=LIBRARY, include_tests=False):
    files = []
    for directory in ("include", "src"):
        files.extend(path for path in (library / directory).rglob("*") if path.is_file())
    if include_tests:
        files.extend(path for path in (library / "tests").rglob("*")
                     if path.is_file() and path.suffix in (".c", ".h", ".py"))
    for name in ("Makefile", "CMakeLists.txt"):
        if (library / name).exists():
            files.append(library / name)
    return sorted(files)


def source_hashes(library=LIBRARY, include_tests=False):
    return {path.relative_to(library).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_files(library, include_tests)}


def export_runtime(destination, components=COMPONENTS, library=LIBRARY):
    """Copy a freestanding runtime; compile listed .c files, link libm as needed.

    destination must be a new or caller-controlled directory. This function never
    deletes files. Returns canonical and exported hashes for reproducibility.
    """
    destination = Path(destination).absolute()
    components = tuple(dict.fromkeys(components))
    if any(name not in COMPONENTS for name in components):
        raise ValueError("components must be cnn, rnn, or lstm")
    if "cnn" not in components:
        components = ("cnn",) + components  # recurrent heads and quantizer use CNN APIs
    destination.mkdir(parents=True, exist_ok=True)
    paths = [(library / "src" / (name + ".c"), Path(name + ".c")) for name in components]
    paths += [(library / "include/ai_baseon_c" / (name + ".h"), Path(name + ".h"))
              for name in components]
    paths += [(library / "src/platform.c", Path("platform.c")),
              (library / "include/ai_baseon_c/nn_platform.h", Path("nn_platform.h")),
              (library / "src/quantization.h", Path("quantization.h")),
              (library / "src/kernels/nn_dot.h", Path("kernels/nn_dot.h"))]
    recurrent = any(name in components for name in ("rnn", "lstm"))
    if recurrent:
        paths.append((library / "src/nn_activations.h", Path("nn_activations.h")))
    paths += [(path, Path("kernels/backends") / path.name)
              for path in sorted((library / "src/kernels/backends").glob("*.h"))]
    hashes = {}
    for source, relative in paths:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        text = source.read_text(encoding="utf-8-sig")
        for name in components:
            text = text.replace('"../include/ai_baseon_c/' + name + '.h"',
                                '"' + name + '.h"')
        text = text.replace('"../../include/ai_baseon_c/nn_platform.h"', '"../nn_platform.h"')
        text = text.replace('"../include/ai_baseon_c/nn_platform.h"', '"nn_platform.h"')
        text = text.replace('"ai_baseon_c/nn_platform.h"', '"nn_platform.h"')
        with target.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        hashes[relative.as_posix()] = {
            "canonical": source.relative_to(library).as_posix(),
            "canonical_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "exported_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        }
    result = {"sources": [name + ".c" for name in components] + ["platform.c"], "hashes": hashes}
    if recurrent:
        result["activation_math_contract"] = "deterministic_float32_v1"
        result["required_compile_flags"] = ["-ffp-contract=off", "-fno-fast-math"]
        result["required_fp_environment"] = "binary32_nearest_even_gradual_underflow"
    return result
