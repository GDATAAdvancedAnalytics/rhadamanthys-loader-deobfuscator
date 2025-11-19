#!/usr/bin/env python3
"""
Main script that orchestrates the deobfuscation of a binary
by calling the other scripts.

Copyright (c) 2025 G DATA Advanced Analytics GmbH
"""

import pefile
import sys

from capstone.x86 import X86_INS_JMP, X86_INS_PALIGNR, X86_INS_RET, X86_OP_REG

from deob_jumps import deobfuscate_function as deobfuscate_function_jumps
from deob_cflow import deobfuscate_function as deobfuscate_function_cflow
from deob_consts import deobfuscate as deobfuscate_constants
from deob_util import ContinuousDisassembler


def discover_functions(blob: bytearray, pe: pefile.PE) -> list[int]:
    raw = pe.sections[0].PointerToRawData
    raw_size = pe.sections[0].SizeOfRawData
    virt = pe.sections[0].VirtualAddress + pe.OPTIONAL_HEADER.ImageBase

    dis = ContinuousDisassembler()
    addr_insn_map = dis.disassemble_all(blob[raw:raw+raw_size], virt)

    result = []

    current_function = virt
    function_ended = False
    is_candidate = False
    unsupported = False
    insns_since_setcc = 0

    for insn in addr_insn_map.values():
        # Simplistic check that works for this sample.
        if insn.id == X86_INS_RET or insn.bytes == b"\xCC":
            function_ended = True
        elif function_ended and insn.bytes != b"\xCC":
            if is_candidate and not unsupported:
                result.append(current_function)

            current_function = insn.address
            function_ended = False
            is_candidate = False
            unsupported = False

        if insn.id == X86_INS_JMP and insn.operands[0].type == X86_OP_REG:
            is_candidate = True
            insns_since_setcc = 0  # reset (new basic block)
        elif insn.id == X86_INS_PALIGNR:
            # Static C runtime hit, stop.
            break

        if insn.mnemonic.startswith("set"):
            # setcc instructions clustered closely together indicate cases where the compiler mashed things together.
            # This usually affects small functions that have 2-3 conditions at most.
            # They're not supported for deobfuscation.
            if 0 < insns_since_setcc < 10:
                unsupported = True
            insns_since_setcc = 1
        elif insns_since_setcc > 0:
            insns_since_setcc += 1

    return result


def do_steps(blob: bytearray, pe: pefile.PE) -> None:
    """Applies all deobfuscation steps to the given binary."""
    funcs = discover_functions(blob, pe)

    print("Applicable functions:")
    for func in funcs:
        print(hex(func))

    print("\n=== BEGIN JUMP DEOBFUSCATION ===")

    for func in funcs:
        deobfuscate_function_jumps(blob, pe, func)

    print("\n=== BEGIN CONST DEOBFUSCATION ===")

    deobfuscate_constants(blob, pe)

    print("\n=== BEGIN CFLOW DEOBFUSCATION ===")

    for func in funcs:
        deobfuscate_function_cflow(blob, pe, func)

    print("Done.")


def deobfuscate(filename: str) -> bytes:
    """Loads and deobfuscates the given file. Returns patched file data."""
    with open(filename, "rb") as fp:
        blob = bytearray(fp.read())

    pe = pefile.PE(data=blob, fast_load=True)

    do_steps(blob, pe)

    return blob


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <filename> [<result filename; by default, .patched is appended>]")
        return

    new_bin = deobfuscate(sys.argv[1])

    if len(sys.argv) < 3:
        fn_out = sys.argv[1] + ".patched"
    else:
        fn_out = sys.argv[2]

    with open(fn_out, "wb") as fp:
        fp.write(new_bin)


if __name__ == "__main__":
    main()
