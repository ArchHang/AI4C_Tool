"""Compile supported CPU/backend combinations and retain reproducible evidence.

Desktop rows execute exact arithmetic and CNN tests. Bare-metal rows only compile:
they do not claim target correctness, measured throughput, or hardware execution.
No serial port, debugger, programmer, or board is accessed.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


LIBRARY = Path(os.path.abspath(__file__)).parents[1]
SOURCES = ("src/cnn.c", "src/rnn.c", "src/lstm.c", "src/platform.c")
BACKENDS = {"portable": 1, "arm_dsp": 2, "neon": 3, "avx2": 4}
ARM32_CPUS = (
    ("cortex-m4", "fpv4-sp-d16", "arm_dsp"),
    ("cortex-m7", "fpv5-sp-d16", "arm_dsp"),
    ("cortex-a7", "neon-vfpv4", "neon"),
    ("cortex-a9", "neon", "neon"),
    ("cortex-a35", "neon-fp-armv8", "neon"),
    ("cortex-a53", "neon-fp-armv8", "neon"),
)


def source_hashes():
    files = []
    for directory in ("include", "src", "tests"):
        files.extend(path for path in (LIBRARY / directory).rglob("*")
                     if path.is_file() and path.suffix in (".c", ".h"))
    files += [LIBRARY / name for name in ("CMakeLists.txt", "Makefile",
              "tools/test_platform_matrix.py", "tools/configure_make.py")]
    files.extend((LIBRARY / "cmake").rglob("*.cmake"))
    return {path.relative_to(LIBRARY).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(files)}


def logged(command, directory, name, environment, timeout=180, expect_failure=False):
    log = directory / (name + ".txt")
    with log.open("wb") as stream:
        try:
            result = subprocess.run(command, cwd=directory, env=environment,
                                    stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
            code = result.returncode
            executed = True
        except (OSError, subprocess.TimeoutExpired) as error:
            stream.write(("ERROR: " + str(error) + "\n").encode("utf-8"))
            code = -1
            executed = False
    content = log.read_text(encoding="utf-8", errors="replace")
    passed = executed and ((code != 0 and "error" in content.lower())
                           if expect_failure else code == 0)
    return {"command": command, "returncode": code, "passed": passed,
            "expected_failure": expect_failure, "log": str(log)}


def compiler_information(compiler, directory, environment):
    information = {"path": compiler}
    for name, flags in (("version", ["--version"]), ("machine", ["-dumpmachine"])):
        result = logged([compiler] + flags, directory, "compiler_" + name, environment)
        information[name] = Path(result["log"]).read_text(errors="replace").strip()
        information[name + "_passed"] = result["passed"]
    return information


def compile_row(name, compiler, flags, backend, expected, output, environment, execute=False):
    directory = output / name
    directory.mkdir(parents=True, exist_ok=True)
    defines = ["-DNN_BACKEND=NN_BACKEND_" + backend.upper()]
    common = [compiler, "-std=c11", "-Wall", "-Wextra", "-Werror", "-ffp-contract=off",
              "-fno-fast-math", "-I" + str(LIBRARY / "include"),
              "-I" + str(LIBRARY / "src"), "-I" + str(LIBRARY / "examples/models")]
    common += flags + defines
    row = {"name": name, "backend_requested": backend, "expected_backend": expected,
           "evidence": "desktop_execution" if execute else "compile_only", "commands": []}
    objects = []
    for source in SOURCES:
        destination = directory / (Path(source).stem + ".o")
        result = logged(common + ["-c", str(LIBRARY / source), "-o", str(destination)],
                        directory, "compile_" + destination.stem, environment)
        row["commands"].append(result)
        if not result["passed"]:
            row["passed"] = False
            print("FAIL " + name + " (" + destination.stem + ")", flush=True)
            return row
        objects.append(str(destination))
    ar_result = logged([compiler, "-print-prog-name=ar"], directory,
                       "locate_archiver", environment)
    row["commands"].append(ar_result)
    if ar_result["passed"]:
        archiver = Path(ar_result["log"]).read_text(errors="replace").strip()
        archive = directory / "libai_baseon_c.a"
        archive_result = logged([archiver, "rcs", str(archive)] + objects,
                                directory, "archive_library", environment)
        row["commands"].append(archive_result)
        if archive_result["passed"]:
            row["archive"] = str(archive)
            row["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    probe = [str(LIBRARY / "tests/backend_compile.c"),
             "-DNN_TEST_EXPECTED_BACKEND=" + str(BACKENDS[expected])]
    result = logged(common + ["-S"] + probe + ["-o", str(directory / "backend.s")],
                    directory, "compile_selection_probe", environment)
    row["commands"].append(result)
    if result["passed"]:
        assembly = (directory / "backend.s").read_text(errors="replace").lower()
        row["assembly_observations"] = {
            "smlad": "smlad" in assembly,
            "neon_widening_mac": any(opcode in assembly for opcode in ("vmlal.s16", "smlal")),
            "avx2_integer_madd": "vpmaddwd" in assembly,
        }
    if execute:
        for test in ("backend", "kernels", "fused", "rnn", "lstm"):
            executable = directory / ("test_" + test + ".exe")
            extra = ["-DTEST_HAS_SCRATCH=1"] if test == "kernels" else []
            result = logged(common + extra + [str(LIBRARY / ("tests/test_" + test + ".c"))]
                            + objects + ["-lm", "-o", str(executable)], directory,
                            "build_test_" + test, environment)
            row["commands"].append(result)
            if result["passed"] and test in ("backend", "kernels", "fused"):
                row["commands"].append(logged([str(executable)], directory,
                                              "run_test_" + test, environment))
        kernels = directory / "test_kernels.exe"
        if kernels.is_file():
            row["commands"].append(logged(
                [sys.executable, str(LIBRARY / "tools/test_robustness.py"),
                 "--exe", str(kernels), "--scratch", "--json", str(directory / "robustness.json")],
                directory, "run_robustness", environment))
    row["passed"] = all(command["passed"] for command in row["commands"])
    print(("PASS " if row["passed"] else "FAIL ") + name, flush=True)
    return row


def can_execute_avx2(compiler, output, environment):
    source = output / "host_avx2_probe.c"
    source.write_text("int main(void) {\n"
                      "#if (defined(__i386__) || defined(__x86_64__)) && defined(__GNUC__)\n"
                      "__builtin_cpu_init(); return __builtin_cpu_supports(\"avx2\") ? 0 : 1;\n"
                      "#else\nreturn 1;\n#endif\n}\n", encoding="utf-8")
    executable = output / "host_avx2_probe.exe"
    build = logged([compiler, str(source), "-o", str(executable)], output,
                   "host_avx2_probe_build", environment)
    if not build["passed"]:
        return False
    result = logged([str(executable)], output, "host_avx2_probe_run", environment)
    return result["passed"]


def cxx_smoke(compiler, output, environment):
    directory = output / "host_cxx_integration"
    directory.mkdir(exist_ok=True)
    source = directory / "integration.cpp"
    source.write_text(
        '#include "ai_baseon_c/cnn.h"\n#include "ai_baseon_c/rnn.h"\n'
        '#include "ai_baseon_c/lstm.h"\n#include "ai_baseon_c/nn_platform.h"\n'
        'int main() {\n'
        '    auto volatile cnn = &conv2d_int8_run;\n'
        '    auto volatile rnn = &rnn_forward_int8;\n'
        '    auto volatile lstm = &lstm_forward_int8;\n'
        '    (void)cnn; (void)rnn; (void)lstm;\n'
        '    const nn_platform_info_t *info = nn_get_platform_info();\n'
        '    return info != nullptr && info->backend == nn_get_backend() ? 0 : 1;\n}\n',
        encoding="utf-8")
    executable = directory / "integration.exe"
    row = {"name": "host_cxx_integration", "evidence": "desktop_execution", "commands": []}
    archive = output / "host_portable_O3/libai_baseon_c.a"
    row["commands"].append(logged(
        [compiler, "-std=c++17", "-O3", "-Wall", "-Wextra", "-Werror",
         "-I" + str(LIBRARY / "include"), "-DNN_BACKEND=NN_BACKEND_PORTABLE", str(source),
         str(archive), "-lm", "-o", str(executable)], directory, "build", environment))
    if row["commands"][0]["passed"]:
        row["commands"].append(logged([str(executable)], directory, "run", environment))
    row["passed"] = all(command["passed"] for command in row["commands"])
    print(("PASS " if row["passed"] else "FAIL ") + row["name"], flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cc", default=shutil.which("gcc"), help="Desktop GCC/Clang")
    parser.add_argument("--cxx", default=shutil.which("g++"), help="Matching desktop C++ compiler")
    parser.add_argument("--arm-gcc", default=shutil.which("arm-none-eabi-gcc"))
    parser.add_argument("--aarch64-gcc", default=shutil.which("aarch64-none-elf-gcc"))
    parser.add_argument("--output-dir", type=Path,
                        default=LIBRARY.parent / "results/multiplatform_20261014/matrix")
    parser.add_argument("--require-arm32", action="store_true")
    parser.add_argument("--require-aarch64", action="store_true")
    args = parser.parse_args()
    output = Path(os.path.abspath(args.output_dir))
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "temp"
    temporary.mkdir(exist_ok=True)
    environment = {**os.environ, "TEMP": str(temporary), "TMP": str(temporary), "PYTHONUTF8": "1"}
    report = {"started_utc": datetime.now(timezone.utc).isoformat(), "host": platform.platform(),
              "source_hashes_before": source_hashes(), "compilers": {}, "rows": [], "skipped": [],
              "serial_access": False, "flash_access": False}
    if args.cc:
        information = compiler_information(args.cc, output, environment)
        report["compilers"]["desktop"] = information
        machine = information["machine"].lower()
        expected_host = "neon" if ("aarch64" in machine or "arm64" in machine) else "portable"
        for optimization in ("-O0", "-O3"):
            report["rows"].append(compile_row("host_portable_" + optimization[1:], args.cc,
                [optimization], "portable", "portable", output, environment, execute=True))
        report["rows"].append(compile_row("host_auto_O3", args.cc, ["-O3"], "auto",
            expected_host, output, environment, execute=True))
        if "x86" in machine or "mingw" in machine:
            supported = can_execute_avx2(args.cc, output, environment)
            for backend in ("avx2", "auto"):
                report["rows"].append(compile_row("host_" + backend + "_avx2_O3", args.cc,
                    ["-O3", "-mavx2"], backend, "avx2", output, environment, execute=supported))
            if not supported:
                report["skipped"].append({"name": "host_avx2_execution",
                                          "reason": "Host OS/CPU AVX2 probe failed"})
        if args.cxx:
            report["compilers"]["cxx"] = compiler_information(args.cxx, output, environment)
            report["rows"].append(cxx_smoke(args.cxx, output, environment))
    else:
        report["skipped"].append({"name": "desktop", "reason": "Desktop GCC/Clang not found"})
    if args.arm_gcc:
        report["compilers"]["arm32"] = compiler_information(args.arm_gcc, output, environment)
        for cpu, fpu, expected in ARM32_CPUS:
            flags = ["-O3", "-mcpu=" + cpu, "-mfpu=" + fpu, "-mfloat-abi=hard",
                     "-mthumb" if "cortex-m" in cpu else "-marm"]
            for backend in ("auto", "portable"):
                report["rows"].append(compile_row(cpu + "_arm32_" + backend, args.arm_gcc,
                    flags, backend, expected if backend == "auto" else "portable", output, environment))
        report["rows"].append(compile_row("cortex-a9_arm32_no_neon", args.arm_gcc,
            ["-O3", "-mcpu=cortex-a9", "-mfpu=vfpv3-d16", "-mfloat-abi=hard", "-marm"],
            "auto", "portable", output, environment))
    else:
        report["skipped"].append({"name": "arm32", "reason": "Arm32 compiler not provided/found"})
    if args.aarch64_gcc:
        report["compilers"]["aarch64"] = compiler_information(args.aarch64_gcc, output, environment)
        for cpu in ("cortex-a35", "cortex-a53"):
            for backend in ("auto", "portable"):
                report["rows"].append(compile_row(cpu + "_aarch64_" + backend, args.aarch64_gcc,
                    ["-O3", "-mcpu=" + cpu], backend, "neon" if backend == "auto" else "portable",
                    output, environment))
    else:
        report["skipped"].append({"name": "aarch64", "reason": "AArch64 compiler not provided/found"})
    report["source_hashes_after"] = source_hashes()
    report["source_files_unchanged"] = report["source_hashes_before"] == report["source_hashes_after"]
    report["passed"] = (bool(report["rows"]) and all(row["passed"] for row in report["rows"])
                        and report["source_files_unchanged"]
                        and (not args.require_arm32 or bool(args.arm_gcc))
                        and (not args.require_aarch64 or bool(args.aarch64_gcc)))
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    destination = output / "report.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "rows": len(report["rows"]),
                      "skipped": report["skipped"], "report": str(destination)}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
