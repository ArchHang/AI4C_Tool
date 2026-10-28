"""Record Make flags and remove only validated, project-local build products."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys


LIBRARY = Path(os.path.abspath(__file__)).parents[1]
LEGACY_PRODUCTS = (
    "cnn.o", "rnn.o", "lstm.o", "platform.o", "libai_baseon_c.a",
    "test_kernels.exe", "test_fused.exe", "test_rnn.exe", "test_lstm.exe",
    "test_backend.exe",
)


def validated_directory(path, for_cleanup=False):
    directory = Path(os.path.abspath(path))
    build_root = LIBRARY / "build"
    if directory == build_root or build_root not in directory.parents:
        raise ValueError("Build directory must be strictly below ai_baseon_c/build")
    # Physical path resolution is essential for deletion; ordinary compilation
    # only creates products under a lexically validated project build directory.
    if for_cleanup:
        physical_build_root = build_root.resolve()
        if physical_build_root != build_root:
            raise ValueError("The project build root must not be a junction or symbolic link")
        build_root = physical_build_root
        resolved = directory.resolve()
    else:
        resolved = directory
    if resolved == build_root or build_root not in resolved.parents:
        raise ValueError("Build directory must be strictly below ai_baseon_c/build")
    # Resolve and validate the absolute target before any recursive deletion.
    current = directory
    while current != LIBRARY and current.parent != current:
        if current.is_symlink():
            raise ValueError("Symbolic links are not allowed in the build directory path")
        current = current.parent
    return resolved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--cc", default="gcc")
    parser.add_argument("--cflags", default="")
    parser.add_argument("--cppflags", default="")
    parser.add_argument("--arch-flags", default="")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    try:
        directory = validated_directory(args.directory, for_cleanup=args.clean)
    except ValueError as error:
        parser.error(str(error))
    if args.clean:
        if directory.exists():
            shutil.rmtree(directory)
        for name in LEGACY_PRODUCTS:
            path = LIBRARY / name
            if path.is_file() or path.is_symlink():
                path.unlink()
        return 0
    directory.mkdir(parents=True, exist_ok=True)
    configuration = {"cc": args.cc, "cflags": args.cflags,
                     "cppflags": args.cppflags, "arch_flags": args.arch_flags}
    text = json.dumps(configuration, indent=2, sort_keys=True) + "\n"
    path = directory / "build_config.json"
    if not path.exists() or path.read_text(encoding="utf-8") != text:
        path.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
