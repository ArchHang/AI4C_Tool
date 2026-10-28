"""Check a clean repository using explicit portable tests and CPU-only examples.

Requires Python3.10+, numpy, torch2.6+, Tkinter and a native GCC-compatible
toolchain. No datasets, historical results, GPU, STM32 tools or devices are used.
CMake is the default; --build-system make verifies the independent Make entry.
All generated binaries/checkpoints live in one automatically cleaned build tree.
"""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
CORE_TEST_MODULES = (
    "test_model_converter", "test_weight_cache_converter", "test_model_studio_analysis",
    "test_model_studio_compatibility", "test_model_studio_compression", "test_model_studio_advanced",
    "test_model_studio_reconstruction", "test_model_studio_services", "test_model_studio_cache",
    "test_model_studio_calibration_data", "test_model_studio_system", "test_repository_tools",
)
EXTERNAL_TESTS = frozenset((
    "test_model_converter.ConverterTests.test_real_cnn_all_dataset_frames",
    "test_model_converter.ConverterTests.test_real_repository_cnn_checkpoint",
))
REQUIRED_FIXTURES = ("examples/models/rnn_model.h", "examples/models/lstm_model.h",
                     "model/rnn_params.npz", "model/lstm_params.npz")


def _flatten(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from _flatten(item)
        else:
            yield item


def core_suite(root, require_gui=False):
    """No wildcard discovery: hardware/corpus experiment tests are excluded."""
    library = Path(root) / "ai_baseon_c"
    sys.path.insert(0, str(library / "tools"))
    sys.path.insert(0, str(library / "tests/python"))
    modules = CORE_TEST_MODULES + (("test_model_studio_gui",) if require_gui else ())
    if (library / "tests/python/test_public_report.py").is_file():
        modules += ("test_public_report",)
    tests = [test for name in modules for test in _flatten(unittest.defaultTestLoader.loadTestsFromName(name))
             if test.id() not in EXTERNAL_TESTS]
    return unittest.TestSuite(tests)


def _python_tests(root, output, require_gui):
    import torch
    torch.set_num_threads(1)
    suite = core_suite(root, require_gui)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    record = {"passed": result.wasSuccessful() and not result.skipped and result.testsRun > 0,
              "tests": result.testsRun, "skipped": [(test.id(), reason) for test, reason in result.skipped],
              "failures": len(result.failures), "errors": len(result.errors),
              "external_tests_intentionally_excluded": sorted(EXTERNAL_TESTS),
              "gui_controller_tests_requested": require_gui, "log": stream.getvalue()}
    Path(output).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(stream.getvalue(), end="")
    return 0 if record["passed"] else 1


def _executable(value, name):
    path = shutil.which(str(value))
    if path is None:
        raise ValueError(f"{name} executable unavailable: {value}")
    return Path(path).resolve()


def _print_failed_process(result, name):
    """Keep failed child tracebacks visible even when the JSON artifact is lost."""
    if result.returncode == 0:
        return
    for channel in ("stdout", "stderr"):
        value = getattr(result, channel, None)
        if value:
            print(f"[{name}] captured {channel}:\n{value}", file=sys.stderr,
                  end="" if value.endswith("\n") else "\n", flush=True)


def check_repository(root=ROOT, *, build_system="cmake", compiler="gcc", cmake="cmake", make="make",
                     generator="Ninja", require_gui=False):
    root = Path(root).resolve()
    library, tools = root / "ai_baseon_c", root / "ai_baseon_c/tools"
    if build_system not in ("cmake", "make"):
        raise ValueError("build_system must be cmake or make")
    for relative in ("CMakeLists.txt", "Makefile", *REQUIRED_FIXTURES):
        if not (library / relative).is_file():
            raise ValueError("Missing public source/fixture: ai_baseon_c/" + relative)
    compiler = _executable(compiler, "C compiler")
    executable = _executable(cmake if build_system == "cmake" else make, build_system)
    environment = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
                       PYTHONHASHSEED="0", CUDA_VISIBLE_DEVICES="")
    environment["PATH"] = os.pathsep.join((str(compiler.parent), str(Path(sys.executable).parent), environment.get("PATH", "")))
    # Do not let a parent checkout's PYTHONPATH hide a missing public dependency.
    environment["PYTHONPATH"] = os.pathsep.join((str(tools), str(library / "tests/python")))
    gcc = _executable(shutil.which("gcc", path=environment["PATH"]) or "gcc", "Native Python-test GCC")
    inputs = [path for directory in ("include", "src") for path in (library / directory).rglob("*") if path.is_file()]
    inputs += [library / relative for relative in ("CMakeLists.txt", "Makefile", *REQUIRED_FIXTURES)]
    input_hashes = {path.relative_to(library).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted(inputs)}
    report = {"schema": "ai4c_repository_check_v1", "passed": False, "hardware_accessed": False,
        "network_accessed": False, "scope": "Portable C and synthetic CNN/RNN/LSTM converter/GUI contracts; no real corpus, historical experiments or MCU performance qualification.",
        "repository_root": str(root), "python": sys.executable, "build_system": build_system,
        "compiler": str(compiler), "python_test_compiler": str(gcc), "steps": [],
        "source_and_fixture_sha256": input_hashes,
        "core_test_modules": list(CORE_TEST_MODULES), "required_fixtures": list(REQUIRED_FIXTURES),
        "public_report_tests_included": (library / "tests/python/test_public_report.py").is_file(),
        "gui_display_tested": require_gui, "temporary_files_cleaned": False}
    python = [sys.executable, "-s", "-B", "-X", "utf8"]

    def checked(name, arguments, timeout=300, allow_failure=False):
        result = subprocess.run([str(argument) for argument in arguments], cwd=root, env=environment,
                                text=True, capture_output=True, timeout=timeout)
        report["steps"].append({"name": name, "argv": [str(argument) for argument in arguments],
            "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
        _print_failed_process(result, name)
        if result.returncode and not allow_failure:
            raise RuntimeError(name + " failed:\n" + result.stdout + result.stderr)
        print(name + (": passed" if result.returncode == 0 else ": failed"), flush=True)
        return result

    build_root = library / "build"
    build_root.mkdir(exist_ok=True)
    if build_root.resolve() != build_root or build_root.is_symlink():
        raise ValueError("The repository build directory must not redirect through a link")
    legacy_report = build_root / "converter_recurrent_accuracy.json"
    previous_legacy = legacy_report.read_bytes() if legacy_report.is_file() else None
    converter_temporary = build_root / "converter-tests"
    converter_temporary_existed = converter_temporary.exists()
    temporary_path = None
    try:
        with tempfile.TemporaryDirectory(prefix=".repository-check-", dir=build_root) as temporary:
            temporary = Path(temporary)
            temporary_path = temporary
            probe = ("import json,sys,torch,numpy,tkinter; import model_converter; import model_studio.gui; "
                     "assert sys.version_info >= (3,10); "
                     "assert callable(getattr(torch.serialization,'get_unsafe_globals_in_checkpoint',None)); "
                     "print(json.dumps({'python':sys.version,'torch':str(torch.__version__),'numpy':numpy.__version__,'Tk':tkinter.TkVersion}))")
            checked("python_dependencies_and_cli_imports", python + ["-c", probe])
            for script in ("convert_model.py", "model_studio_gui.py", "generate_example_models.py"):
                checked(script + "_help", python + [tools / script, "--help"])
            if require_gui:
                checked("tk_display", python + ["-c", "import tkinter as tk; r=tk.Tk(); r.withdraw(); r.update(); r.destroy()"])
            binary_dir = temporary / build_system
            if build_system == "cmake":
                checked("cmake_configure", [executable, "-S", library, "-B", binary_dir, "-G", generator,
                    "-DNN_BUILD_TESTS=ON", "-DNN_BACKEND=portable", "-DCMAKE_BUILD_TYPE=Release",
                    "-DCMAKE_C_COMPILER=" + str(compiler), "-DPython3_EXECUTABLE=" + sys.executable])
                checked("cmake_build", [executable, "--build", binary_dir, "--parallel", "2"])
                checked("ctest", [executable.parent / ("ctest.exe" if os.name == "nt" else "ctest"),
                    "--test-dir", binary_dir, "--output-on-failure"])
            else:
                checked("make_build", [executable, "-C", library, "all", "BACKEND=portable",
                    "BUILD_DIR=" + binary_dir.relative_to(library).as_posix(),
                    "CC=" + compiler.name, "PYTHON=" + Path(sys.executable).name])
                for test in ("kernels", "fused", "backend"):
                    checked("native_" + test, [binary_dir / ("test_" + test + ".exe")])
                checked("native_robustness", python + [tools / "test_robustness.py", "--exe",
                    binary_dir / "test_kernels.exe", "--scratch"])
            examples = temporary / "examples"
            checked("generate_untrained_cpu_examples", python + [tools / "generate_example_models.py", "--output-dir", examples])
            for kind in ("cnn", "rnn", "lstm"):
                exported = temporary / ("export_" + kind)
                checked("convert_example_" + kind, python + [tools / "convert_model.py", examples / (kind + ".pth"),
                    "--output-dir", exported, "--precision", "both", "--bundle-runtime"])
                manifest = json.loads((exported / "manifest.json").read_text(encoding="utf-8"))
                binary = exported / ("example.exe" if os.name == "nt" else "example")
                checked("build_example_" + kind, [gcc, "-std=c11", "-O3", "-ffp-contract=off", "-fno-fast-math",
                    "-I" + str(exported / "runtime"), exported / "model.c", exported / "example.c",
                    *[exported / "runtime" / source for source in manifest["bundled_runtime"]["sources"]], "-lm", "-o", binary])
                checked("run_example_" + kind, [binary])
            test_report = temporary / "python_tests.json"
            arguments = python + [tools / "check_repository.py", "--internal-python-tests", test_report, "--root", root]
            if require_gui:
                arguments.append("--require-gui")
            checked("portable_python_test_whitelist", arguments, timeout=600, allow_failure=True)
            report["python_tests"] = json.loads(test_report.read_text(encoding="utf-8"))
            report["passed"] = report["python_tests"]["passed"]
            if not report["passed"]:
                report["error"] = "Portable Python tests failed or skipped; see python_tests for exact cases"
    except (ValueError, RuntimeError, OSError, subprocess.TimeoutExpired) as error:
        report["error"] = str(error)
        report["passed"] = False
    finally:
        # One legacy synthetic regression writes this fixed build-only report.
        # Preserve a pre-existing result; otherwise remove only our new output.
        if previous_legacy is not None:
            legacy_report.write_bytes(previous_legacy)
        else:
            legacy_report.unlink(missing_ok=True)
        if not converter_temporary_existed and converter_temporary.is_dir() and not any(converter_temporary.iterdir()):
            converter_temporary.rmdir()
    report["temporary_files_cleaned"] = temporary_path is None or not temporary_path.exists()
    report["source_and_fixtures_unchanged"] = all(
        hashlib.sha256((library / name).read_bytes()).hexdigest() == expected for name, expected in input_hashes.items())
    if not report["source_and_fixtures_unchanged"]:
        report["passed"] = False
        report["error"] = "Canonical source or synthetic fixture changed during verification"
    if not report["temporary_files_cleaned"]:
        report["passed"] = False
        report["error"] = "Owned temporary build directory was not removed"
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--build-system", choices=("cmake", "make"), default="cmake")
    parser.add_argument("--compiler", default=os.environ.get("CC", "gcc"))
    parser.add_argument("--cmake", default="cmake")
    parser.add_argument("--make", default="make")
    parser.add_argument("--generator", default="Ninja")
    parser.add_argument("--require-gui", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--internal-python-tests", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.internal_python_tests:
        return _python_tests(args.root, args.internal_python_tests, args.require_gui)
    output = args.output
    if output is not None and output.exists():
        parser.error("Output already exists; preserve earlier check evidence")
    options = vars(args);options.pop("output");options.pop("internal_python_tests")
    report = check_repository(**options)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
    print(json.dumps({"passed": report["passed"], "python_tests": report.get("python_tests", {}).get("tests"),
                      "temporary_files_cleaned": report["temporary_files_cleaned"], "error": report.get("error")}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
