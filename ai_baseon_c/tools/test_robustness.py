"""Run invalid CNN inputs in separate processes; crashes and timeouts are failures.

Usage: python tools/test_robustness.py --exe test_kernels.exe [--json report.json]
The executable is built from test_kernels.c and the selected library version.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys


CASES = (
    "conv_int8_null_input",
    "conv_int8_zero_stride",
    "conv_fp32_zero_stride",
    "pool_u8_large_kernel",
    "conv_int8_overflow_padding",
    "conv_int8_mac_limit",
    "cal_conv_zero_stride",
    "cal_conv_overflow_padding",
    "conv_fp32_zero_channels",
    "conv_fp32_zero_kernel",
    "conv_fp32_bn_nan",
    "conv_fp32_overflow_shape",
    "conv_int8_overflow_shape",
    "conv_int8_nan_scale",
    "conv_int8_inf_scale",
    "conv_int8_negative_zp",
    "conv_int8_large_output_zp",
    "conv_int8_negative_weight_scale",
    "conv_int8_inf_bias",
    "conv_int8_bad_dynamic",
    "conv_dynamic_null_scale",
    "linear_fp32_zero_inputs",
    "linear_int8_nan_scale",
    "linear_int8_large_zp",
    "linear_int8_overflow_shape",
    "linear_int8_overflow_output_bytes",
    "linear_int8_nonfinite_output_tail",
    "linear_int8_nonfinite_output_tile",
    "linear_fp32_overflow_shape",
    "linear_fp32_nonfinite_output_tail",
    "linear_fp32_nonfinite_output_tile",
    "linear_fp32_nan_input",
    "pool_fp32_zero_channels",
    "pool_u8_zero_channels",
    "pool_u8_overflow_shape",
    "quant_nan_scale",
    "quant_inf_scale",
    "quant_large_zp",
    "quant_extreme_finite",
    "softmax_nan_input",
)
SCRATCH_CASES = ("scratch_small", "scratch_unaligned", "scratch_null_nonzero",
                 "scratch_invalid_query")


def run_case(executable, name, timeout):
    try:
        result = subprocess.run(
            [str(executable), "--robust-case", name],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "case": name,
            "passed": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "timeout": False,
        }
    except subprocess.TimeoutExpired:
        return {"case": name, "passed": False, "returncode": None, "timeout": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--scratch", action="store_true",
                        help="Include optional API tests; compile with -DTEST_HAS_SCRATCH")
    args = parser.parse_args()
    executable = args.exe.absolute()
    reports = []
    for case in CASES + (SCRATCH_CASES if args.scratch else ()):
        report = run_case(executable, case, args.timeout)
        reports.append(report)
        detail = "timeout" if report["timeout"] else str(report["returncode"])
        print(f"{'PASS' if report['passed'] else 'FAIL'} {case} exit={detail}", flush=True)
    passed = sum(report["passed"] for report in reports)
    print(f"RESULT {passed}/{len(reports)} passed")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    return 0 if passed == len(reports) else 1


if __name__ == "__main__":
    sys.exit(main())
