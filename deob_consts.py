"""
Copyright (c) 2025 G DATA Advanced Analytics GmbH
"""

import pefile
from capstone import CsInsn
from capstone.x86 import (X86_OP_MEM, X86_OP_REG,
                          X86_INS_ADD, X86_INS_MOV, X86_INS_MOVZX,
                          X86_INS_SBB, X86_INS_SUB, X86_INS_XOR)

from deob_util import ContinuousDisassembler


def filter_outliers(addrs: list[int], max_step: int = 0x10, min_cluster_size: int = 10) -> list[int]:
    """Does an outlier analysis on the sorted addrs list. Only addresses that are part of clusters are kept."""
    n = len(addrs)
    if n < 2:
        return []

    # Compute diffs
    diffs = [addrs[i] - addrs[i-1] for i in range(1, n)]
    # Find stable windows: intervals [s, e] on diffs such that for k in [s..e] diffs[k] <= max_step
    # and length (e - s + 1) >= min_cluster_size
    stable_diff_windows = []
    s = 0
    while s < len(diffs):
        # Find run of consecutive diffs <= max_step starting at s
        if diffs[s] <= max_step and diffs[s] >= 0:
            e = s
            while e+1 < len(diffs) and diffs[e+1] <= max_step and diffs[e+1] >= 0:
                e += 1
            if e - s + 1 >= min_cluster_size:
                stable_diff_windows.append((s, e))
            s = e + 1
        else:
            s += 1

    # Convert diff-windows to address-windows (addresses covered are indices [start ... end+1])
    stable_addr_windows = []
    for ds, de in stable_diff_windows:
        stable_addr_windows.append((ds, de + 1))  # inclusive indexes into addrs

    # Merge overlapping/adjacent address windows
    stable_addr_windows.sort()
    merged: list[list[int]] = []
    for win in stable_addr_windows:
        if not merged:
            merged.append(list(win))
        else:
            if win[0] <= merged[-1][1] + 1:  # overlap or adjacent
                merged[-1][1] = max(merged[-1][1], win[1])
            else:
                merged.append(list(win))
    stable_addr_windows = [(a, b) for a, b in merged]

    valid = [False] * n
    for a, b in stable_addr_windows:
        for i in range(a, b+1):
            valid[i] = True

    # Allow single jumps between two stable windows: if windows are separated by exactly one address
    # e.g. window1 = [0..10], window2 = [12..22] -> index 11 is single-jump; mark 10,11,12 valid
    for i in range(len(stable_addr_windows)-1):
        end1 = stable_addr_windows[i][1]
        start2 = stable_addr_windows[i+1][0]
        if start2 - end1 == 2:  # exactly one address in between
            # Mark end1, end1+1, start2 valid (some may already be valid)
            for idx in (end1, end1+1, start2):
                if 0 <= idx < n:
                    valid[idx] = True

    filtered = [a for a, v in zip(addrs, valid) if v]
    outliers = [a for a, v in zip(addrs, valid) if not v]
    print("Outliers:")
    for addr in outliers:
        print(hex(addr))
    return filtered


class ConstantInliner:

    def __init__(self, blob: bytearray, insn_map: dict[int, CsInsn], pe: pefile.PE):
        self.blob = blob
        self.insn_map = insn_map

        imagebase = pe.OPTIONAL_HEADER.ImageBase
        self.raw_virt_delta_text = imagebase + pe.sections[0].VirtualAddress - pe.sections[0].PointerToRawData
        self.raw_virt_delta_data = imagebase + pe.sections[2].VirtualAddress - pe.sections[2].PointerToRawData
        self.data_start = imagebase + pe.sections[2].VirtualAddress
        # End doesn't matter too much, just bound it somewhere.
        self.data_end = imagebase + pe.OPTIONAL_HEADER.SizeOfImage

    def is_acceptable_load(self, insn: CsInsn) -> bool:
        """Returns True if we have operands 'reg, [imm32]' where imm32 is in the PE's data range."""
        return (insn.id in (X86_INS_MOV, X86_INS_MOVZX, X86_INS_ADD, X86_INS_SBB, X86_INS_SUB, X86_INS_XOR)
                and insn.operands[0].type == X86_OP_REG
                and insn.operands[1].type == X86_OP_MEM
                and insn.operands[1].value.mem.base == 0
                and insn.operands[1].value.mem.index == 0
                and insn.operands[1].value.mem.segment == 0
                and self.data_start <= insn.operands[1].value.mem.disp <= self.data_end)

    def transform_mov_mem_to_imm(self, instr_bytes: bytes, imm_value: int) -> bytes:
        """
        Transforms 'mov reg, [imm32]' into 'mov reg, imm32_value'.
        """
        b = instr_bytes

        # mov eax, [imm32]
        if b[0] == 0xA1 and len(b) == 5:
            imm32_bytes = imm_value.to_bytes(4, 'little')
            return b"\xB8" + imm32_bytes  # mov eax, imm32

        # mov r32, [imm32]
        if b[0] == 0x8B and len(b) == 6:
            modrm = b[1]
            if (modrm & 0b11000111) == 0b00000101:  # [imm32]
                reg = (modrm >> 3) & 7
                imm32_bytes = imm_value.to_bytes(4, 'little')
                return bytes([0x90, 0xB8 + reg]) + imm32_bytes  # mov r32, imm32

        # movzx r32, byte/word ptr [imm32]
        if len(b) >= 7 and b[0] == 0x0F and b[1] in (0xB6, 0xB7):
            modrm = b[2]
            if (modrm & 0b11000111) == 0b00000101:  # [imm32]
                reg = (modrm >> 3) & 7
                if b[1] == 0xB6:
                    imm_value &= 0xFF
                else:
                    imm_value &= 0xFFFF
                imm32_bytes = imm_value.to_bytes(4, 'little')
                return bytes([0xB8 + reg]) + imm32_bytes + b"\x90\x90"  # mov r32, imm32

        raise AssertionError(b.hex())

    def transform_arith_mem_to_imm(self, instr_bytes: bytes, imm_value: int) -> bytes:
        """
        Transforms 'add/sub/etc reg, [imm32]' into 'add/sub/etc reg, imm32_value'.
        """
        b = instr_bytes

        opcode = b[0]
        modrm = b[1]

        # verify addressing mode [imm32]
        if (modrm & 0b11000111) == 0b00000101:
            reg = (modrm >> 3) & 7

            if opcode == 0x03:  # add r32, [imm32]
                imm_bytes = imm_value.to_bytes(4, 'little')
                return bytes([0x81, 0xC0 + reg]) + imm_bytes  # add r32, imm32

            elif opcode == 0x02:  # add r8, [imm32]
                imm_byte = imm_value & 0xFF
                return bytes([0x80, 0xC0 + reg, imm_byte, 0x90, 0x90, 0x90])  # add r8, imm8

            elif opcode == 0x1B:  # sbb r32, [imm32]
                imm_bytes = imm_value.to_bytes(4, 'little')
                return bytes([0x81, 0xD8 + reg]) + imm_bytes  # sbb r32, imm32

            elif opcode == 0x2B:  # sub r32, [imm32]
                imm_bytes = imm_value.to_bytes(4, 'little')
                return bytes([0x81, 0xE8 + reg]) + imm_bytes  # sub r32, imm32

            elif opcode == 0x2A:  # sub r8, [imm32]
                imm_byte = imm_value & 0xFF
                return bytes([0x80, 0xE8 + reg, imm_byte, 0x90, 0x90, 0x90])  # sub r8, imm8

            elif opcode == 0x32:  # xor r8, [imm32]
                imm_byte = imm_value & 0xFF
                return bytes([0x80, 0xF0 + reg, imm_byte, 0x90, 0x90, 0x90])  # xor r8, imm8

        raise AssertionError(b.hex())

    def deobfuscate(self) -> None:
        candidates = []
        loads: list[CsInsn] = []
        for insn in self.insn_map.values():
            if self.is_acceptable_load(insn):
                loads.append(insn)
                candidates.append(insn.operands[1].value.mem.disp)

        candidates = list(set(candidates))
        candidates.sort()
        obfuscator_addrs = filter_outliers(candidates)

        for insn in loads:
            address = insn.operands[1].value.mem.disp
            if address not in obfuscator_addrs:
                continue

            data = self.blob[address - self.raw_virt_delta_data:address - self.raw_virt_delta_data+4]
            const_val = int.from_bytes(data, 'little')

            if insn.id in (X86_INS_MOV, X86_INS_MOVZX):
                patched_insn = self.transform_mov_mem_to_imm(insn.bytes, const_val)
            elif insn.id in (X86_INS_ADD, X86_INS_SBB, X86_INS_SUB, X86_INS_XOR):
                patched_insn = self.transform_arith_mem_to_imm(insn.bytes, const_val)
            else:
                raise AssertionError()

            assert len(patched_insn) == insn.size
            offset = insn.address - self.raw_virt_delta_text
            self.blob[offset:offset + insn.size] = patched_insn

        # for addr in addresses:
        #     print(hex(addr))


def deobfuscate(blob: bytearray, pe: pefile.PE) -> None:
    """Applies constant deobfuscation to the given blob. It is modified in-place."""
    raw = pe.sections[0].PointerToRawData
    raw_size = pe.sections[0].SizeOfRawData
    virt = pe.sections[0].VirtualAddress + pe.OPTIONAL_HEADER.ImageBase

    dis = ContinuousDisassembler()
    addr_insn_map = dis.disassemble_all(blob[raw:raw+raw_size], virt)

    inliner = ConstantInliner(blob, addr_insn_map, pe)
    inliner.deobfuscate()


if __name__ == "__main__":
    with open("bin.dat.patched", "rb") as fp:
        blob = bytearray(fp.read())

    pe = pefile.PE(data=blob, fast_load=True)

    deobfuscate(blob, pe)

    with open("bin.dat.patchedc", "wb") as fp2:
        fp2.write(blob)
