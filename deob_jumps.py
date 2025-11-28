"""
Copyright (c) 2025 G DATA Advanced Analytics GmbH
"""

from typing import Optional

import pefile
from capstone import CsInsn
from capstone.x86 import (X86_OP_IMM, X86_OP_MEM, X86_OP_REG,
                          X86_REG_EFLAGS,
                          X86_GRP_CMOV, X86_GRP_JUMP,
                          X86_INS_ADD, X86_INS_JMP, X86_INS_LEA, X86_INS_MOV, X86_INS_SHL,
                          X86_INS_XOR)

from deob_util import FunctionDisassembler, Rewriter
from deob_util import branch_sources, get_register_sub_slice, prev_insn, register_intersection


def find_assignment(slice: list[CsInsn], reg: int) -> Optional[CsInsn]:
    """Returns the first lea/mov in slice that overwrites reg, or None."""
    for insn in slice:
        if insn.id in (X86_INS_LEA, X86_INS_MOV) and insn.operands[0].value.reg == reg:
            return insn
    return None


class JumpSite:
    """
    Represents a single indirect-jmp site. It holds the register that is jumped to and
    instruction slices extracted by data_slice().
    """

    def __init__(self, reg: int) -> None:
        self.reg = reg
        """Jump destination register."""
        self.main_slice: list[CsInsn] = []
        """Slice with all instructions influencing the jump on the main path. Ends with jmp <reg>."""
        self.incoming_slices: list[list[CsInsn]] = []
        """Slices of incoming branches into the block that can also influence the jump destination.
           They usually only contain 1-2 mov/lea and end with a jcc or cmov."""

    def fuse_slices(self, extras: list[CsInsn] = []) -> list[CsInsn]:
        """Takes instructions from all slices and orders them.
           Especially useful to determine what can be safely overwritten.
           * extras (optional): Instructions that should furthermore be added to the result."""
        result = list(self.main_slice)
        for incoming in self.incoming_slices + [extras]:
            for insn in incoming:
                if insn not in result:
                    result.append(insn)

        result.sort(key=lambda insn: insn.address)
        return result

    def __str__(self):
        res = str(self.main_slice)
        if self.incoming_slices is not None:
            res += "; " + str(self.incoming_slices)
        return res


class JumpInliner:

    def __init__(self, blob: bytearray, insn_map: dict[int, CsInsn], branches: list[tuple[int, int]], pe: pefile.PE):
        self.blob = blob
        self.insn_map = insn_map
        self.branches = branches
        self.init_consts: dict[int, int] = {}

        imagebase = pe.OPTIONAL_HEADER.ImageBase
        self.raw_virt_delta_text = imagebase + pe.sections[0].VirtualAddress - pe.sections[0].PointerToRawData
        self.raw_virt_delta_data = imagebase + pe.sections[2].VirtualAddress - pe.sections[2].PointerToRawData

    def data_slice(self, va_start: int, target_regs: set[int], slices: list[list[CsInsn]],
                   in_branch: bool = False) -> None:
        """
        Performs static data slicing analysis for target_reg. Incoming branches are followed for a depth of 1.

        - slices: List will be filled by this method with lists of instructions influencing target_regs.
                  The first list contains instructions executed on all paths, the others only on some paths.
        """
        result: list[CsInsn] = []
        result.append(self.insn_map[va_start])
        slices.append(result)
        insn = prev_insn(self.insn_map, va_start)

        # We go upwards and figure out all instructions that influence target_regs.
        tracked_regs = set(target_regs)
        while insn.id != X86_INS_JMP:
            read, written = insn.regs_access()
            tracked_and_written = register_intersection(tracked_regs, written)
            if tracked_and_written:
                result.insert(0, insn)

                assert len(tracked_and_written) == 1
                written_reg = next(iter(tracked_and_written))

                new_tracked = set(read) - {X86_REG_EFLAGS}
                tracked_regs.update(new_tracked)

                # Check if overwritten without using same reg (i.e., not 'mov eax, [eax]').
                if (insn.id in (X86_INS_LEA, X86_INS_MOV) and written_reg not in new_tracked) or \
                        (insn.id == X86_INS_XOR and insn.operands[0].value.reg == insn.operands[1].value.reg):
                    tracked_regs.remove(written_reg)
                    if not tracked_regs:
                        break

                # For cmov reg1, reg2 instructions, we have two cases:
                # Either reg1 is overwritten with reg2, or reg1 is left untouched.
                if insn.group(X86_GRP_CMOV):
                    other_regs = set(tracked_regs)
                    # On one path, assume we've overwritten it...
                    tracked_regs.remove(written_reg)

                    # ...on the other path, continue tracking the target register.
                    other_regs.difference_update(new_tracked)
                    self.data_slice(insn.address, other_regs, slices, True)

            sources = branch_sources(self.branches, insn.address)
            if sources and in_branch:
                if not result[-1].group(X86_GRP_CMOV):
                    print("multi target ", hex(va_start))
                sources.clear()

            if sources:
                # Create a new slice, since from this point, data may diverge.
                result = []
                slices.append(result)

            for src in sources:
                self.data_slice(src, tracked_regs, slices, True)

            insn = prev_insn(self.insn_map, insn.address)

        if not result:  # Remove if nothing further was added
            slices.remove(result)

    def deobfuscate(self) -> None:
        """Finds all indirect jumps in the function and patches them."""
        # Collect all "mov/lea reg, imm" from the start of the function.
        for insn in self.insn_map.values():
            if insn.id == X86_INS_JMP:
                break

            if insn.id == X86_INS_MOV:
                if insn.operands[0].type == X86_OP_REG and insn.operands[1].type == X86_OP_IMM:
                    self.init_consts[insn.operands[0].value.reg] = insn.operands[1].value.imm
            elif insn.id == X86_INS_LEA:
                if insn.operands[0].type == X86_OP_REG and insn.operands[1].value.mem.disp != 0 \
                        and insn.operands[1].value.mem.base == 0:
                    self.init_consts[insn.operands[0].value.reg] = insn.operands[1].value.mem.disp

        # Analyze jump sites & patch them.
        for insn in list(self.insn_map.values()):
            if insn.id != X86_INS_JMP or insn.operands[0].type != X86_OP_REG:
                continue

            jmp_reg = insn.operands[0].value.reg
            site = JumpSite(jmp_reg)
            slices: list[list[CsInsn]] = []
            self.data_slice(insn.address, {jmp_reg}, slices)
            site.main_slice = slices[0]
            site.incoming_slices = slices[1:]
            print(site)
            self.fixup(site)

    def _get_value(self, insn: CsInsn) -> int:
        """Returns the integer value of the second operand, with some magic in case it is a register."""
        if insn.id == X86_INS_LEA or (insn.id == X86_INS_MOV and insn.operands[1].type == X86_OP_MEM):
            assert insn.operands[1].value.mem.disp != 0
            assert insn.operands[1].value.mem.base == 0
            assert insn.operands[1].value.mem.index == 0
            return insn.operands[1].value.mem.disp

        elif insn.id == X86_INS_MOV:
            if insn.operands[1].type == X86_OP_IMM:
                return insn.operands[1].value.imm
            elif insn.operands[1].type == X86_OP_REG:
                # Slight hack, but if a register wasn't assigned a concrete value on a path,
                # for this obfuscator, it always seems to have the initial value assigned at function start.
                assert insn.operands[1].value.reg in self.init_consts
                return self.init_consts[insn.operands[1].value.reg]
            else:
                raise AssertionError()

        elif insn.group(X86_GRP_CMOV):
            # If we get a cmov in this function, it's the first block and the reg was loaded in the prologue.
            assert insn.operands[1].type == X86_OP_REG
            assert insn.operands[1].value.reg in self.init_consts
            return self.init_consts[insn.operands[1].value.reg]
        else:
            raise AssertionError()

    def _find_imm_write(self, slice: list[CsInsn], reg: int) -> Optional[int]:
        """Checks if the given register is written to with a mov reg, imm in the slice."""
        for insn in slice[::-1]:
            if insn.id == X86_INS_MOV and insn.operands[0].value.reg == reg:
                if insn.operands[1].type == X86_OP_IMM:
                    return insn.operands[1].value.imm
        return None

    def _compute_dest(self, site: JumpSite, load: CsInsn | int) -> int:
        """Computes target address the same way the obfuscator does."""
        load_addr = self._get_value(load) if type(load) is CsInsn else load
        assert load_addr != 0
        # mov eax, [eax]
        load_offset = load_addr - self.raw_virt_delta_data
        data = int.from_bytes(self.blob[load_offset:load_offset+4], 'little')

        # add eax, esi
        assert site.main_slice[-2].id == X86_INS_ADD
        key_reg = site.main_slice[-2].operands[1].value.reg
        # In rare cases, key_reg is written in the function.
        written = self._find_imm_write(site.main_slice[:-2], key_reg)
        if written is None:
            to_add = self.init_consts[key_reg]
        else:
            to_add = written

        return (data + to_add) & 0xffffffff

    def fixup(self, site: JumpSite) -> None:
        """Analyzes a single site and patches it."""
        for i in site.main_slice:
            if i.group(X86_GRP_CMOV):
                self._fixup_cmov(site)
                return
            if i.id == X86_INS_MOV and i.operands[1].type == X86_OP_MEM and i.operands[1].value.mem.disp != 0 \
                    and (i.operands[1].value.mem.base != 0 or i.operands[1].value.mem.index != 0):
                self._fixup_index_access(site)
                return

        # Default case: Assume simple jmp that may also have an incoming branch.
        self._fixup_jmp_and_possible_jcc(site)

    def _move_commonalities(self, site: JumpSite) -> None:
        """Checks if higher-up blocks have certain instructions that are present in all of them,
           and pulls them down (0x40E43F)."""
        of_interest = site.incoming_slices[2:]
        if len(of_interest) < 2:
            return

        for insn in of_interest[0]:
            if all(insn in insns for insns in of_interest):
                for i in (0, 1):
                    # This check is wonky as hell and will likely cause incorrectness if this ever happens elsewhere...
                    if site.incoming_slices[i][0].address > insn.address:
                        site.incoming_slices[i].insert(0, insn)

        del site.incoming_slices[2:]

    def _fixup_jmp_and_possible_jcc(self, site: JumpSite) -> None:
        """Fixes indirect jmp with a possible incoming branch influencing the destination."""
        self._move_commonalities(site)

        # Locate mov reg, dword ptr [reg]
        for insn in site.main_slice[::-1]:
            if insn.id == X86_INS_MOV and insn.operands[1].type == X86_OP_MEM:
                load_reg = insn.operands[1].value.mem.base
                search_slice = site.main_slice[:site.main_slice.index(insn)]
                break
        else:
            raise AssertionError("mov reg, [reg] not found")

        has_branch = False
        if search_slice:  # if this has elements, we have assignments on the main path (and thus a single destination).
            # Extract the specific assignment chain for our register of interest.
            reg_slice = get_register_sub_slice(search_slice, load_reg)
            value_load = reg_slice[0]

        elif insn.operands[1].value.mem.base != 0:
            # Load is in another basic block, meaning we likely have multiple values.
            """
            lea     eax, off_48B6D0   ; site.incoming_slices[1][0]  (branch_value_load)
            cmp     ebx, 65203D55h
            jl      short loc_40A6A0  ; site.incoming_slices[1][1]
            lea     eax, off_48B4A8   ; site.incoming_slices[0][0]  (value_load)
            loc_40A6A0:                             ; CODE XREF: .text:0040A698↑j
            mov     eax, [eax]        ; insn
            """
            reg_slice = get_register_sub_slice(site.incoming_slices[0], load_reg)
            value_load = reg_slice[0]
            has_branch = True

        else:
            # No register in mem operand: mov eax, [123]; add eax, ebx; jmp eax
            value_load = insn

        jmp_dest = self._compute_dest(site, value_load)
        print(hex(jmp_dest))

        jmp_dest_jcc = -1
        jcc_insn = None
        if has_branch:
            # incoming[0] contains instructions from when the branch is not taken.
            # incoming[1] contains instructions from above (and including) the jcc.
            branch_slice = site.incoming_slices[1]

            if len(branch_slice) == 1:
                # Special case at 0x41598A: reg is not assigned, so we assume it's from the prologue.
                assert branch_slice[0].group(X86_GRP_JUMP)
                jcc_insn = branch_slice[0]
                assert load_reg in self.init_consts
                jmp_dest_jcc = self._compute_dest(site, self.init_consts[load_reg])
                print(hex(jmp_dest_jcc))
                branch_value_load = None

            elif len(branch_slice) == 2:  # Normal case
                branch_value_load = branch_slice[0]
                main0 = value_load
                if branch_value_load.id == main0.id and branch_value_load.op_str == main0.op_str:
                    # If these are equal, the destinations are equal anyway OR it's a bogus incoming branch.
                    branch_value_load = None
                    site.incoming_slices.remove(branch_slice)

            else:  # extremely rare
                # mov     eax, edi          ; [0] still happens to be right; find() is here in case they were swapped
                # mov     ebp, 8A5BF346h    ; key restore
                # jl      short loc_408BE4
                reg_slice = get_register_sub_slice(branch_slice, load_reg)
                branch_value_load = reg_slice[0]

            if branch_value_load is not None:
                jcc_insn = branch_slice[-1]
                jmp_dest_jcc = self._compute_dest(site, branch_value_load)
                print(hex(jmp_dest_jcc))

        dispensable = site.fuse_slices(extras=self._get_gap_fill_instructions(site.main_slice))
        rew = Rewriter(self.raw_virt_delta_text, self.insn_map, self.branches, dispensable)
        rew.plan_branch(b"\xE9", jmp_dest, site.main_slice[-1])

        if jmp_dest_jcc != -1 and jmp_dest_jcc != jmp_dest:
            assert jcc_insn is not None
            jcc_offset = jcc_insn.address - self.raw_virt_delta_text
            assert jcc_insn.size == 2  # else +0x10 needs changing below
            rew.plan_branch(bytes([0x0F, self.blob[jcc_offset] + 0x10]), jmp_dest_jcc, jcc_insn)

        rew.rewrite(self.blob)

    def _assemble_jcc_and_jmp(self, site: JumpSite, true_dest: int, false_dest: int, x86_cc: int, original_cond: CsInsn) -> None:  # noqa:E501
        """Helper function that calls the rewriter to create a jcc and jmp with the given specifications."""
        dispensable = site.fuse_slices(extras=self._get_gap_fill_instructions(site.main_slice))

        # Ensure moving the condition check down won't cause logic bugs.
        final_jmp = site.main_slice[-1]
        if flag_write := self._get_flags_interference(original_cond.address, final_jmp.address, dispensable):
            if not self._fix_interference(original_cond, flag_write):
                raise Exception(f"Flags written between {original_cond} and jmp: {flag_write}")

        rew = Rewriter(self.raw_virt_delta_text, self.insn_map, self.branches, dispensable)
        rew.plan_branch(bytes([0x0F, x86_cc + 0x80]), true_dest, final_jmp)
        rew.plan_branch(b"\xE9", false_dest, final_jmp)
        rew.rewrite(self.blob)

    def _get_flags_interference(self, va_from: int, va_to: int, dispensable: list[CsInsn]) -> Optional[CsInsn]:
        """Returns first instruction that writes eflags in range [va_from, va_to) and is not dispensable."""
        va = va_from
        while va < va_to:
            insn = self.insn_map[va]
            if insn not in dispensable:
                _, written = insn.regs_access()
                if X86_REG_EFLAGS in written:
                    return insn
            va += insn.size

        return None

    def _fix_interference(self, flag_dependent: CsInsn, flag_write: CsInsn) -> bool:
        """Ugly hack for a very specific case."""
        # Assume that everything is neatly lined up, else it gets quite complicated.
        # .text:00408EBD  cmp     edx, 0F5BEC16Ch   ; prev
        # .text:00408EC3  cmovl   eax, edi          ; flag_dependent
        # .text:00408EC6  xor     ecx, ecx          ; flag_write (going to be moved above cmp)
        if flag_write.address != flag_dependent.address + flag_dependent.size:
            return False

        prev = prev_insn(self.insn_map, flag_dependent.address)
        if X86_REG_EFLAGS not in prev.regs_access()[1]:
            return False

        # Ensure prev doesn't touch anything that flag_write overwrites
        written = set(flag_write.regs_access()[1]) - {X86_REG_EFLAGS}
        read = set(prev.regs_access()[0]) - {X86_REG_EFLAGS}
        if len(written & read) != 0:
            return False

        top_offset = prev.address - self.raw_virt_delta_text
        top_size = prev.size + flag_dependent.size
        move_down = self.blob[top_offset:top_offset + top_size]
        bottom_offset = flag_write.address - self.raw_virt_delta_text
        bottom_size = flag_write.size
        move_up = self.blob[bottom_offset:bottom_offset + bottom_size]

        self.blob[top_offset:top_offset + bottom_size] = move_up
        self.blob[top_offset + bottom_size:top_offset + bottom_size + top_size] = move_down

        self.insn_map[prev.address] = flag_write
        del self.insn_map[flag_dependent.address]
        del self.insn_map[flag_write.address]

        flag_write._raw.address -= top_size
        prev._raw.address += bottom_size
        flag_dependent._raw.address += bottom_size

        self.insn_map[flag_dependent.address] = flag_dependent
        self.insn_map[prev.address] = prev

        return True

    def _fixup_cmov(self, site: JumpSite) -> None:
        """Rewrites the cmov pattern with a jcc and jmp."""
        # assert len(site.incoming_slices) == 1

        # Due to how data_slice() works, main_slice is always the path where the cmovcc DOES move (cc true),
        # while incoming_slices[0] is what happens if the condition is false.
        # Thus, if we turn cmovcc into jcc, the jcc goes to the destination of main_slice.
        slice_true = site.main_slice
        slice_false = site.incoming_slices[0]
        cmov = slice_false[-1]  # incoming slices always end with the cmov
        assert cmov.group(X86_GRP_CMOV) and cmov.operands[1].type == X86_OP_REG

        # Usually the final value is in the first element, except when there's some unrelated register load...
        index = 0 if slice_true[0] != slice_false[0] else 1
        true_dest = self._compute_dest(site, slice_true[index])
        false_dest = self._compute_dest(site, slice_false[index])
        print("T: " + hex(true_dest))
        print("F: " + hex(false_dest))

        # cmovcc is 0F40~0F4F
        cc = cmov.bytes[1] - 0x40
        self._assemble_jcc_and_jmp(site, true_dest, false_dest, cc, cmov)

    def _fixup_index_access(self, site: JumpSite) -> None:
        """Rewrites the index access pattern (setcc followed by mov with scale/shl) with a jcc and jmp."""
        set_insn = site.main_slice[1]
        assert set_insn.mnemonic.startswith("set")  # doesn't have an X86_GRP

        for i in site.main_slice[::-1]:
            if i.id == X86_INS_MOV and i.operands[1].type == X86_OP_MEM:
                mov_insn = i
                break
        else:
            raise AssertionError("mov not found")

        if mov_insn.operands[1].value.mem.scale != 1:
            mul_factor = mov_insn.operands[1].value.mem.scale
        else:
            shl = site.main_slice[2]
            assert shl.id == X86_INS_SHL and shl.operands[1].type == X86_OP_IMM
            mul_factor = 2 ** shl.operands[1].value.imm

        disp = mov_insn.operands[1].value.mem.disp
        true_dest = self._compute_dest(site, disp + 1 * mul_factor)
        false_dest = self._compute_dest(site, disp + 0 * mul_factor)
        print("T (I): " + hex(true_dest))
        print("F (I): " + hex(false_dest))

        # setcc is 0F90~0F9F
        cc = set_insn.bytes[1] - 0x90
        self._assemble_jcc_and_jmp(site, true_dest, false_dest, cc, set_insn)

    def _get_gap_fill_instructions(self, slice: list[CsInsn]) -> list[CsInsn]:
        """The obfuscator sometimes needs to restore registers that hold values from the function start.
           This method adds such instructions to the slice, so that free space analysis has better chances."""
        result = []

        slice_addrs = {insn.address for insn in slice}
        for insn in slice[::-1][1:]:
            next_addr = insn.address + insn.size
            if next_addr not in slice_addrs:
                insn = self.insn_map[next_addr]
                # Check if this looks like a restore (lea with known value).
                if insn.id == X86_INS_LEA and insn.operands[0].value.reg in self.init_consts:
                    if insn.operands[1].value.mem.disp == self.init_consts[insn.operands[0].value.reg]:
                        result.append(insn)
                        break  # it's only ever one, apparently

        return result


def deobfuscate_function(blob: bytearray, pe: pefile.PE, func: int) -> None:
    imagebase = pe.OPTIONAL_HEADER.ImageBase
    raw_virt_delta = imagebase + pe.sections[0].VirtualAddress - pe.sections[0].PointerToRawData

    dis = FunctionDisassembler()
    addr_insn_map, branches = dis.disassemble_function(blob[func-raw_virt_delta:], func)

    inliner = JumpInliner(blob, addr_insn_map, branches, pe)
    inliner.deobfuscate()


if __name__ == "__main__":
    with open("bin.dat", "rb") as fp:
        blob = bytearray(fp.read())

    pe = pefile.PE(data=blob, fast_load=True)

    for func in (0x40A560, 0x40A140,
                 0x407510, 0x407810, 0x407AF0, 0x407DB0,
                 0x408090, 0x408380, 0x4086A0, 0x4089B0, 0x408CB0, 0x408F90,
                 0x409280, 0x409580, 0x409860, 0x409B50, 0x409E50,
                 0x40C8B0, 0x40CDA0, 0x40D200,
                 0x40F830, 0x40FBE0, 0x40FE40,
                 0x411850, 0x412470, 0x412D50, 0x413050, 0x413EF0,
                 0x4142D0, 0x414890, 0x4150E0, 0x416790, 0x4185E0):
        deobfuscate_function(blob, pe, func)

    with open("bin.dat.patched", "wb") as fp2:
        fp2.write(blob)
