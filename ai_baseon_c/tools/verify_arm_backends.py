#!/usr/bin/env python3
"""Execute ARM32/AArch64 backend probes in Unicorn against Python integer arithmetic.

This validates instruction semantics and strict read bounds, never MCU cycle counts.
The optional dependencies are pyelftools and Unicorn; the inference library needs neither.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import struct
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("elf", type=Path)
    parser.add_argument("--python-deps", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.python_deps:
        sys.path.insert(0, str(args.python_deps.resolve()))

    from elftools.elf.elffile import ELFFile
    import unicorn
    from unicorn import Uc, UC_ARCH_ARM, UC_ARCH_ARM64, UC_MODE_ARM, UC_HOOK_MEM_READ
    from unicorn.arm_const import (
        UC_ARM_REG_R0, UC_ARM_REG_R1, UC_ARM_REG_R2, UC_ARM_REG_R3,
        UC_ARM_REG_SP, UC_ARM_REG_LR, UC_ARM_REG_PC, UC_ARM_REG_FPEXC, UC_ARM_REG_C1_C0_2,
        UC_CPU_ARM_CORTEX_A9,
    )

    with args.elf.open("rb") as stream:
        elf = ELFFile(stream)
        is_aarch64 = elf.header["e_machine"] == "EM_AARCH64"
        if is_aarch64:
            from unicorn.arm64_const import (
                UC_ARM64_REG_X0, UC_ARM64_REG_X1, UC_ARM64_REG_X2, UC_ARM64_REG_X3,
                UC_ARM64_REG_X4, UC_ARM64_REG_SP, UC_ARM64_REG_X30, UC_ARM64_REG_PC,
                UC_ARM64_REG_CPACR_EL1, UC_CPU_ARM64_A53,
            )
            uc = Uc(UC_ARCH_ARM64, UC_MODE_ARM)
            uc.ctl_set_cpu_model(UC_CPU_ARM64_A53)
            registers = (UC_ARM64_REG_X0, UC_ARM64_REG_X1, UC_ARM64_REG_X2, UC_ARM64_REG_X3)
            register_sp, register_lr, register_pc = \
                UC_ARM64_REG_SP, UC_ARM64_REG_X30, UC_ARM64_REG_PC
            pointer_format = "<4Q"
            cpu_name = "cortex-a53 AArch64 instruction emulator"
            uc.reg_write(UC_ARM64_REG_CPACR_EL1, 3 << 20)
        elif elf.header["e_machine"] == "EM_ARM":
            uc = Uc(UC_ARCH_ARM, UC_MODE_ARM)
            uc.ctl_set_cpu_model(UC_CPU_ARM_CORTEX_A9)
            registers = (UC_ARM_REG_R0, UC_ARM_REG_R1, UC_ARM_REG_R2, UC_ARM_REG_R3)
            register_sp, register_lr, register_pc = UC_ARM_REG_SP, UC_ARM_REG_LR, UC_ARM_REG_PC
            pointer_format = "<4I"
            cpu_name = "cortex-a9 instruction emulator"
            uc.reg_write(UC_ARM_REG_C1_C0_2, 0xF << 20)
            uc.reg_write(UC_ARM_REG_FPEXC, 1 << 30)
        else:
            raise ValueError("An ARM32 or AArch64 ELF is required")
        uc.mem_map(0, 0x200000)
        for segment in elf.iter_segments():
            if segment["p_type"] == "PT_LOAD":
                uc.mem_write(segment["p_vaddr"], segment.data())
        symbols = {
            symbol.name: int(symbol["st_value"])
            for symbol in elf.get_section_by_name(".symtab").iter_symbols()
            if symbol.name
        }
    uc.mem_map(0x200000, 0x600000)
    uc.mem_map(0xF00000, 0x1000)
    input_base = 0x200000
    weight_bases = [0x300000, 0x400000, 0x500000, 0x600000]
    table_address = 0x710000
    output_address = 0x720000
    stack_address = 0x7FF000
    active_ranges = []
    read_errors = []

    def check_read(instance, access, address, size, value, user_data):
        del access, value, user_data
        for base, begin, end in active_ranges:
            if base <= address < base + 0x100000:
                if address < begin or address + size > end:
                    read_errors.append((hex(address), size, hex(begin), hex(end)))
                    instance.emu_stop()
                break

    uc.hook_add(UC_HOOK_MEM_READ, check_read)

    def call(name, call_args):
        if len(call_args) > 5:
            raise ValueError("Probe supports at most five ABI arguments")
        for register, value in zip(registers, call_args):
            uc.reg_write(register, value & 0xFFFFFFFF)
        uc.reg_write(register_sp, stack_address)
        if len(call_args) == 5:
            if is_aarch64:
                uc.reg_write(UC_ARM64_REG_X4, call_args[4])
            else:
                uc.mem_write(stack_address, struct.pack("<I", call_args[4]))
        uc.reg_write(register_lr, 0xF00000)
        uc.emu_start(symbols[name], 0xF00000, count=10000000)
        if read_errors:
            raise AssertionError(f"Backend read outside an exact buffer: {read_errors}")
        if uc.reg_read(register_pc) != 0xF00000:
            raise AssertionError(f"Probe did not return: {name}")
        result = uc.reg_read(registers[0]) & 0xFFFFFFFF
        return result if result < 0x80000000 else result - 0x100000000

    rng = random.Random(20261014)
    zero_points = (0, 1, 127, 128, 254, 255)
    lengths = tuple(range(66)) + (127, 128, 129, 255, 256, 257, 1023, 1024, 1025, 4032)
    cases = 0
    tile_cases = 0

    def verify_case(inputs, weights, offset, size, point):
        nonlocal cases, active_ranges
        input_address = input_base + offset
        weight_addresses = [base + (offset + row * 7) % 32
                            for row, base in enumerate(weight_bases)]
        uc.mem_write(input_address, bytes(inputs))
        for address, values in zip(weight_addresses, weights):
            uc.mem_write(address, bytes(value & 255 for value in values))
        uc.mem_write(table_address, struct.pack(pointer_format, *weight_addresses))
        active_ranges = [(input_base, input_address, input_address + size)] + [
            (base, address, address + size)
            for base, address in zip(weight_bases, weight_addresses)
        ]
        expected = [sum((value - point) * weight for value, weight in zip(inputs, row))
                    for row in weights]
        call("nn_backend_compile_probe4",
             [input_address, table_address, size, point, output_address])
        actual = struct.unpack("<4i", uc.mem_read(output_address, 16))
        if tuple(expected) != actual:
            raise AssertionError(f"Centered dot mismatch: size={size} zp={point} "
                                 f"offset={offset}, expected={expected}, actual={actual}")
        for address, row in zip(weight_addresses, weights):
            expected_raw = sum(value * weight for value, weight in zip(inputs, row))
            actual_raw = call("nn_backend_compile_probe", [input_address, address, size])
            if actual_raw != expected_raw:
                raise AssertionError(f"Raw dot mismatch: size={size} offset={offset}")
        cases += 1

    for offset in range(32):
        for size in lengths:
            inputs = [rng.randrange(256) for _ in range(size)]
            weights = [[rng.randrange(-128, 128) for _ in range(size)] for _ in range(4)]
            for point in zero_points:
                verify_case(inputs, weights, offset, size, point)
    for size in (65279, 65280):
        weights = [[value] * size for value in (-128, 127, -1, 1)]
        for input_value in (0, 255):
            for point in zero_points:
                verify_case([input_value] * size, weights, 1, size, point)

    if "nn_backend_compile_probe_tile" in symbols:
        for offset in range(32):
            for weight in range(-128, 128):
                inputs = bytes(rng.randrange(256) for _ in range(8))
                start = (1, -2, 3, -4, 5, -6, 7, -8)
                uc.mem_write(input_base + offset, inputs)
                uc.mem_write(output_address, struct.pack("<8i", *start))
                active_ranges = [(input_base, input_base + offset, input_base + offset + 8)]
                call("nn_backend_compile_probe_tile", [input_base + offset, weight, output_address])
                actual = struct.unpack("<8i", uc.mem_read(output_address, 32))
                expected = tuple(value + coefficient * weight
                                 for value, coefficient in zip(start, inputs))
                if actual != expected:
                    raise AssertionError("NEON spatial tile mismatch")
                tile_cases += 1

    report = {
        "elf": str(args.elf.resolve()),
        "engine": "Unicorn",
        "engine_version": unicorn.__version__,
        "cpu_model": cpu_name,
        "dot_cases": cases,
        "tile_cases": tile_cases,
        "exact_integer_output": True,
        "reads_strictly_in_bounds": True,
        "performance_measurement": False,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
