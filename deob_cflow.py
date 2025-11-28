"""
Copyright (c) 2025 G DATA Advanced Analytics GmbH
"""

from dataclasses import dataclass
from typing import Optional

import pefile
from capstone import CsInsn
from capstone.x86 import (X86_OP_IMM, X86_OP_MEM, X86_OP_REG,
                          X86_REG_EAX, X86_REG_EBP, X86_REG_ESP, X86_REG_EFLAGS,
                          X86_GRP_CMOV, X86_GRP_JUMP,
                          X86_INS_ADD, X86_INS_CALL, X86_INS_CMP, X86_INS_JE, X86_INS_JNE, X86_INS_JMP, X86_INS_MOV)
from keystone import Ks, KS_ARCH_X86, KS_MODE_32

from deob_util import FunctionDisassembler, Rewriter
from deob_util import branch_sources, find_above, find_above_s, get_register_sub_slice, prev_insn


def assemble(insn_str: str) -> bytes:
    assembled, _ = Ks(KS_ARCH_X86, KS_MODE_32).asm(insn_str, as_bytes=True)
    assert assembled is not None
    return assembled


def eval_imm(operations: list[CsInsn]) -> int:
    value = 0
    for insn in operations:
        if insn.id == X86_INS_MOV:
            value = insn.operands[1].value.imm
        elif insn.id == X86_INS_ADD:
            value += insn.operands[1].value.imm
        else:
            raise Exception(f"Operation {insn.mnemonic} not handled")

    return value & 0xffffffff


@dataclass
class MemAnalysis:
    x86_cc: int
    target_true: int
    target_false: int
    dispensable_insns: list[CsInsn]
    branch_condition: bytes
    """Instruction bytes for cmp and such"""
    need_mov: bool = False
    """Did we patch branch_condition in such a way that the (original) dispatcher mov is NOT dispensable?"""


class CflowUnflattener:

    def __init__(self, blob: bytearray, insn_map: dict[int, CsInsn], branches: list[tuple[int, int]],
                 raw_virt_delta: int):
        self.blob = blob
        self.raw_virt_delta = raw_virt_delta
        self.insn_map = insn_map
        self.branches = branches
        self.flow_reg = self._find_flow_register()

    def _find_flow_register(self) -> int:
        """Figures out which register holds the control flow key."""
        candidates: dict[int, int] = {}
        for insn in self.insn_map.values():
            if insn.id == X86_INS_CMP and insn.operands[0].type == X86_OP_REG and insn.operands[1].type == X86_OP_IMM \
                    and int.bit_count(insn.operands[1].value.imm) > 5:
                reg = insn.operands[0].value.reg
                candidates[reg] = candidates.get(reg, 0) + 1
                if candidates[reg] == 3:
                    return reg
        raise Exception("Flow register not found")

    def _is_flow_op(self, insn: CsInsn) -> bool:
        """Returns True if the given instruction has the flow reg as first operand."""
        # Note: Cannot check for op[1].type == IMM here. It's sometimes valid that a reg is assigned to the flow reg.
        return insn.operands[0].type == X86_OP_REG and insn.operands[0].value.reg == self.flow_reg

    def _find_targets(self) -> dict[int, int]:
        """Analyzes all dispatchers to figure out which key value goes to which block. Returns key -> va mapping."""
        targets: dict[int, int] = {}
        for cmp in self.insn_map.values():
            if cmp.id == X86_INS_CMP and self._is_flow_op(cmp):
                if cmp.operands[1].type != X86_OP_IMM:
                    continue  # Especially if flow reg is eax, this can be a call result etc.

                next_insn = self.insn_map[cmp.address + cmp.size]
                while not next_insn.group(X86_GRP_JUMP):
                    next_insn = self.insn_map[next_insn.address + next_insn.size]

                flow_value = cmp.operands[1].value.imm
                if next_insn.id == X86_INS_JE:
                    targets[flow_value] = next_insn.operands[0].value.imm
                    print(f"{flow_value:x} -> {targets[flow_value]:x}")

                elif next_insn.id == X86_INS_JNE:
                    next_insn = self.insn_map[next_insn.address + next_insn.size]
                    if next_insn.id != X86_INS_JMP:
                        raise Exception("Not jmp: " + str(next_insn))

                    targets[flow_value] = next_insn.operands[0].value.imm
                    print(f"(n) {flow_value:x} -> {targets[flow_value]:x}")

                # ... else it's gonna be something like jl, not interesting.

        return targets

    def _cmov_analysis(self, cmov: CsInsn, targets: dict[int, int]) -> tuple[int, int, int, list[CsInsn]]:
        """Analyzes a cmov dispatcher type. Returns x86_cc, true_dest, false_dest, involved insns."""
        assert cmov.operands[1].type == X86_OP_REG
        looky1 = cmov.operands[0].value.reg  # this is used if cc is false
        looky2 = cmov.operands[1].value.reg  # this is used if cc is true
        value1, value2 = None, None

        insns = [cmov]

        # Luckily, the movs always seem to be in the same block shortly above the cmov.
        va = cmov.address
        i = 0
        while i < 30:  # some function prologues have a lot of stuff in it, e.g. 40C99AA
            i += 1
            insn = prev_insn(self.insn_map, va)
            va = insn.address

            if insn.id == X86_INS_JMP:
                raise Exception("cmov analysis failed (hit jmp)")
            if insn.id != X86_INS_MOV or insn.operands[0].type != X86_OP_REG or insn.operands[1].type != X86_OP_IMM:
                continue

            insn_reg = insn.operands[0].value.reg
            if value1 is None and insn_reg == looky1:
                value1 = insn.operands[1].value.imm
                insns.append(insn)
            elif value2 is None and insn_reg == looky2:
                value2 = insn.operands[1].value.imm
                insns.append(insn)

            if value1 is not None and value2 is not None:
                break
        else:
            raise Exception("cmov analysis failed")

        return cmov.bytes[1] - 0x40, targets[value2], targets[value1], insns

    def _mem_analysis(self, mem_insn: CsInsn, targets: dict[int, int]) -> MemAnalysis:
        """When dealing with a memory operand, figures out where it is assigned and what path decision is behind it."""
        mem = mem_insn.operands[1].value.mem
        assert mem.base == X86_REG_ESP

        prologue = sorted(self.insn_map.values(), key=lambda insn: insn.address)
        for i in range(len(prologue)):
            if prologue[i].group(X86_GRP_JUMP):
                prologue = prologue[:i]
                break

        for mov_insn in prologue:
            if mov_insn.id == X86_INS_MOV and mov_insn.operands[0].type == X86_OP_MEM \
                    and mov_insn.operands[0].value.mem.base == mem.base \
                    and mov_insn.operands[0].value.mem.disp == mem.disp:
                break
        else:
            raise Exception("mov to stack mem not found")

        assert mov_insn.operands[1].type == X86_OP_REG
        src_reg = mov_insn.operands[1].value.reg

        reg_assign = self._find_reg_assign_above(mov_insn, src_reg)
        if reg_assign is None:
            raise Exception("reg assignment not found above reference")

        assert reg_assign.group(X86_GRP_CMOV), "Expected cmov, got " + str(reg_assign)
        cmov_res = self._cmov_analysis(reg_assign, targets)

        # Find out what sets the flag that cmov sets (we check all flags, but that's fine).
        def is_flag_write(i: CsInsn):
            return X86_REG_EFLAGS in i.regs_access()[1]

        branch_condition = find_above(self.insn_map, reg_assign, is_flag_write)
        if branch_condition is None:
            raise Exception("flag write not found above assignment")

        analysis = MemAnalysis(cmov_res[0], cmov_res[1], cmov_res[2], cmov_res[3], branch_condition.bytes)
        self._condition_analysis(branch_condition, prologue, mov_insn,
                                 mem_insn.operands[0].value.reg, analysis)

        return analysis

    def _trace_reg(self, reg: int, prologue: list[CsInsn], anchor: CsInsn):
        writes = get_register_sub_slice(prologue[:prologue.index(anchor)], reg)

        if reg == X86_REG_EAX:
            # Special case: function result (and hope we don't run into an edx:eax situation)
            call = find_above_s(prologue, anchor, lambda insn: insn.id == X86_INS_CALL)
            if call is not None and (not writes or writes[-1].address < call.address):
                writes = []

        if not writes:
            save = find_above_s(prologue, anchor, lambda insn: insn.id == X86_INS_MOV
                                and insn.operands[0].type == X86_OP_MEM
                                and insn.operands[1].type == X86_OP_REG
                                and insn.operands[1].value.reg == reg)
            if save is not None:
                print(anchor.reg_name(reg), ":", "MEM via ", save, f" ({save.op_str.split(',')[0]})")
                return (X86_OP_MEM, save.op_str.split(",")[0])

        elif writes[-1].operands[1].type == X86_OP_MEM and writes[-1].operands[1].value.mem.base in (X86_REG_ESP, X86_REG_EBP):  # noqa:E501
            print(anchor.reg_name(reg), ":", "MEM via ", writes[-1], f" ({writes[-1].op_str.split(', ')[1]})")
            return (X86_OP_MEM, writes[-1].op_str.split(", ")[1])

        elif all(insn.operands[1].type == X86_OP_IMM for insn in writes):
            value = eval_imm(writes)
            print(anchor.reg_name(reg), ":", f"IMM {value:x}")
            return (X86_OP_IMM, hex(value))

        # print(anchor.reg_name(reg), ":", writes)
        print(anchor.reg_name(reg), ": stays reg")
        return (X86_OP_REG, reg)

    def _condition_analysis(self, condition_insn: CsInsn, prologue: list[CsInsn],
                            mov_in_prologue: CsInsn, reg_at_site: int, analysis: MemAnalysis):
        """MEM and IMM operands in the instruction are fine. REG is critical because regs tend to
           be reassigned further down in the prologue block, making copying the insn elsewhere problematic.
           This function modifies the given analysis object with patched (safe) branch_condition bytes.
           It may also modify mov_insn to store a register on the stack (rather than a cmov result)."""

        # non-esp/ebp in MEM could also be problematic, but hasn't occurred so far
        reg_in_0 = condition_insn.operands[0].type == X86_OP_REG
        reg_in_1 = condition_insn.operands[1].type == X86_OP_REG
        if not reg_in_0 and not reg_in_1:
            return None

        print(condition_insn)

        same = reg_in_0 and condition_insn.operands[0].value.reg == condition_insn.operands[1].value.reg
        if reg_in_0:
            trace_0 = self._trace_reg(condition_insn.operands[0].value.reg, prologue, condition_insn)
        if reg_in_1 and not same:
            trace_1 = self._trace_reg(condition_insn.operands[1].value.reg, prologue, condition_insn)

        # Nop cmov stuff so save_reg has less interference.
        for insn in analysis.dispensable_insns:
            prologue.remove(insn)
            insn_offset = insn.address - self.raw_virt_delta
            self.blob[insn_offset:insn_offset+insn.size] = b"\x90" * insn.size

        def save_reg(reg: int):
            """Modifies the prologue mov to save our needed data (residing in a register; we move it to the stack)."""
            i1 = prologue.index(condition_insn)
            i2 = prologue.index(mov_in_prologue)
            assert i1 < i2
            assert not any(reg in insn.regs_access()[1] for insn in prologue[i1+1:i2])

            insn_str = f"mov {mov_in_prologue.op_str.split(', ')[0]}, {condition_insn.reg_name(reg)}"
            print("(top) ", insn_str)
            asm = assemble(insn_str)
            assert len(asm) == mov_in_prologue.size
            offset = mov_in_prologue.address - self.raw_virt_delta
            self.blob[offset:offset + len(asm)] = asm
            # TODO should adjust insn_map probably...

        if reg_in_0 and not reg_in_1:
            # cmp reg, 123
            assert trace_0[0] == X86_OP_MEM  # only because other cases have not been encountered/tested
            # --> cmp [stackvar], 123
            insn_str = f"{condition_insn.mnemonic} {trace_0[1]}, {condition_insn.op_str.split(', ')[1]}"
            print(insn_str)
            print()
            analysis.branch_condition = assemble(insn_str)

        elif not reg_in_0:
            # cmp [stackvar], reg
            assert trace_1[0] == X86_OP_MEM  # only because other cases have not been encountered/tested
            # --> mov freereg, [stackvar1]; cmp [stackvar], freereg
            freereg = condition_insn.reg_name(self.flow_reg)
            insn_str = f"mov {freereg}, {trace_1[1]}\n"
            insn_str += f"{condition_insn.mnemonic} {condition_insn.op_str.split(', ')[0]}, {freereg}"
            print(insn_str)
            print()
            analysis.branch_condition = assemble(insn_str)

        elif same:
            # test reg, reg
            if trace_0[0] == X86_OP_MEM:
                # --> mov freereg, [stackvar]; test freereg, freereg
                freereg = condition_insn.reg_name(self.flow_reg)
                insn_str = f"mov {freereg}, {trace_0[1]}\n"
                insn_str += f"{condition_insn.mnemonic} {freereg}, {freereg}"
                print(insn_str)
                print()
                analysis.branch_condition = assemble(insn_str)
                return

            assert trace_0[0] == X86_OP_REG
            # --> mov [existing_stackvar], reg
            save_reg(trace_0[1])
            # --> test reg_at_site, reg_at_site
            reg_at_site_name = condition_insn.reg_name(reg_at_site)
            insn_str = f"{condition_insn.mnemonic} {reg_at_site_name}, {reg_at_site_name}"
            print(insn_str)
            print()
            analysis.need_mov = True
            analysis.branch_condition = assemble(insn_str)

        else:
            if trace_0[0] == X86_OP_REG and trace_1[0] == X86_OP_REG:
                raise Exception("Would have to store two registers")

            if trace_0[0] == X86_OP_REG:
                save_reg(trace_0[1])
                right_side = trace_1[1]
            else:
                # We need to swap the operands to be able to encode them...
                assert condition_insn.id == X86_INS_CMP
                save_reg(trace_1[1])
                right_side = trace_0[1]
                # Reverse direction of the condition code as a result of the swap.
                analysis.x86_cc = (0, 1, 7, 6, 4, 5, 7, 6, 8, 9, 10, 11, 15, 14, 13, 12)[analysis.x86_cc]

            # --> cmp reg_at_site, whatever
            reg_at_site_name = condition_insn.reg_name(reg_at_site)
            insn_str = f"{condition_insn.mnemonic} {reg_at_site_name}, {right_side}"
            print(insn_str)
            print()
            analysis.need_mov = True
            analysis.branch_condition = assemble(insn_str)

    def _find_reg_assign_above(self, above_what: CsInsn, find_reg: int) -> Optional[CsInsn]:
        def is_assign(i: CsInsn):
            return len(i.operands) > 0 and i.operands[0].type == X86_OP_REG and i.operands[0].value.reg == find_reg and i.id != X86_INS_CMP  # noqa:E501

        return find_above(self.insn_map, above_what, is_assign)

    def deobfuscate(self) -> None:
        """Main cflow deobufscation loop. Finds assignments to the flow register and inserts
           code for jumping to the destination(s) rather than entering the dispatcher.
           This whole method is a rather messy state machine, sorry about that."""

        targets = self._find_targets()

        # "staging" variables
        reg_assign = None
        saved_reg_assign = None
        saved_branch_insn = None
        # final "confirmed" variables
        branch_reg_assign = None
        branch_insn = None
        target: Optional[int | tuple[int, int, int, list[CsInsn]] | MemAnalysis] = None
        had_cmp = False
        dispensable: list[CsInsn] = []

        for insn in list(self.insn_map.values()):
            if saved_reg_assign is not None and branch_sources(self.branches, insn.address):
                print("HAS BRANCH:", hex(targets[saved_reg_assign.operands[1].value.imm]))
                branch_reg_assign = saved_reg_assign
                branch_insn = saved_branch_insn

            # Find a mov to the flow reg.
            if (insn.id == X86_INS_MOV or insn.group(X86_GRP_CMOV)) and self._is_flow_op(insn):
                reg_assign = insn
                target = None
                had_cmp = False
                dispensable.clear()

            # If we see a jcc, store our current assignment because it may be overwritten.
            if insn.group(X86_GRP_JUMP) and insn.id != X86_INS_JMP:
                saved_reg_assign = reg_assign
                saved_branch_insn = insn

            # Ensure it's not just some random assignment to the register.
            # Basically we ignore all reg_assign until the last one before a flow cmp or jmp.
            if (insn.id == X86_INS_CMP and self._is_flow_op(insn) and insn.operands[1].type == X86_OP_IMM) or insn.group(X86_GRP_JUMP):  # noqa:E501
                if reg_assign is not None:
                    print(reg_assign)
                    if reg_assign.group(X86_GRP_CMOV):
                        # Conditional branch
                        target = self._cmov_analysis(reg_assign, targets)
                        print(f"{target[0]:x} : {target[1]:x} else {target[2]:x}")

                    elif reg_assign.operands[1].type == X86_OP_IMM:
                        # Single destination
                        target = targets[reg_assign.operands[1].value.imm]
                        # print(f"----> {target:x}")

                    else:
                        # Likely referencing a stack value
                        if reg_assign.operands[1].type == X86_OP_REG:
                            # Special case with an intermediary mov
                            follow = self._find_reg_assign_above(reg_assign, reg_assign.operands[1].value.reg)
                            if follow is not None:
                                # cmp usually wasn't captured in this case because it's against a different reg
                                cmp = find_above(self.insn_map, reg_assign,
                                                 lambda insn: insn.id == X86_INS_CMP and insn.operands[1].type == X86_OP_IMM)  # noqa:E501
                                if cmp is not None:
                                    dispensable.append(cmp)
                                reg_assign = follow

                        assert reg_assign.operands[1].type == X86_OP_MEM
                        target = self._mem_analysis(reg_assign, targets)

                    dispensable.append(reg_assign)
                    dispensable.append(insn)
                    reg_assign = None

            # The following part only executes if we saw a valid target assignment.
            if target is None:
                continue

            if insn.id == X86_INS_CMP and self._is_flow_op(insn):
                had_cmp = True

            # We can place our jump if:
            # a) we found an unconditional jump (usually only at function start)
            # b) we found a conditional jump and saw a cmp against the flow reg before
            if insn.id == X86_INS_JMP or (had_cmp and insn.group(X86_GRP_JUMP)):
                if insn not in dispensable:
                    dispensable.append(insn)

                print("Go from", insn)
                if isinstance(target, int) and branch_reg_assign is None:
                    self._rewrite_simple(target, dispensable)
                elif isinstance(target, tuple):
                    self._rewrite_cmov(target[0], target[1], target[2], dispensable)
                elif isinstance(target, MemAnalysis):
                    self._rewrite_mem_ref(target, dispensable)
                else:  # with branch
                    assert branch_reg_assign is not None and branch_insn is not None
                    # Branch taken: The mov further up is in action.
                    target_true = targets[branch_reg_assign.operands[1].value.imm]
                    # Branch not taken: The mov further down is used.
                    target_false = target
                    self._rewrite_branch(branch_insn, target_true, target_false, dispensable)

                print()
                target = None
                saved_reg_assign = None
                branch_reg_assign = None
                branch_insn = None

    def _rewrite_simple(self, target: int, dispensable: list[CsInsn]) -> None:
        """Most common case where it goes to a single destination."""
        rew = Rewriter(self.raw_virt_delta, self.insn_map, self.branches, dispensable)
        rew.plan_branch(b"\xE9", target, dispensable[-1])
        rew.rewrite(self.blob)

    def _rewrite_cmov(self, x86_cc: int, target_true: int, target_false: int, dispensable: list[CsInsn]) -> None:
        """cmov case where it goes to one of two destinations based on a condition code."""
        cmov = dispensable[0]
        cmp = dispensable[1]
        assert cmov.group(X86_GRP_CMOV)
        assert cmp.id == X86_INS_CMP

        # Patch at end of dispatch (jmp). Sometimes there are instructions beween dispatcher jcc and jmp (4162BE).
        insn = dispensable[-1]
        if insn.id != X86_INS_JMP:
            while True:
                insn = self.insn_map[insn.address + insn.size]
                if insn.id == X86_INS_JMP:
                    dispensable.append(insn)
                    break

        if self._has_interference(cmov.address, dispensable[-1].address, dispensable):
            raise Exception("Flags written between cmov and end of dispatch")

        rew = Rewriter(self.raw_virt_delta, self.insn_map, self.branches, dispensable)
        # Replace cmp with jcc
        rew.plan_branch(b"\x0F" + bytes([0x80 + x86_cc]), target_true, dispensable[-1])
        # Replace original dispatcher jcc with jmp
        rew.plan_branch(b"\xE9", target_false, dispensable[-1])
        rew.rewrite(self.blob)

    def _rewrite_branch(self, branch: CsInsn, target_true: int, target_false: int, dispensable: list[CsInsn]) -> None:
        """Similar to cmov but implemented with a jcc instruction."""
        cmp = dispensable[-2]
        assert cmp.id == X86_INS_CMP
        assert branch.address < cmp.address
        assert branch.size == 2  # cc extraction code below only supports short jcc

        # Patch at end of dispatch (jmp). Sometimes there are instructions beween dispatcher jcc and jmp (4162BE).
        insn = dispensable[-1]
        if insn.id != X86_INS_JMP:
            while True:
                insn = self.insn_map[insn.address + insn.size]
                if insn.id == X86_INS_JMP:
                    dispensable.append(insn)
                    break

        if self._has_interference(branch.address, dispensable[-1].address, dispensable):
            raise Exception("Flags written between jcc and end of dispatch")

        rew = Rewriter(self.raw_virt_delta, self.insn_map, self.branches, dispensable)
        # Place jcc
        rew.plan_branch(b"\x0F" + bytes([branch.bytes[0] + 0x10]), target_true, dispensable[-1])
        # Place jmp
        rew.plan_branch(b"\xE9", target_false, dispensable[-1])
        rew.rewrite(self.blob)

    def _rewrite_mem_ref(self, analysis: MemAnalysis, dispensable: list[CsInsn]) -> None:
        """Flow reg is set to a (stack) memory reference, which is usually computed at the top of the function."""
        dispensable.sort(key=lambda insn: insn.address)

        has_jmp = dispensable[-1].id == X86_INS_JMP
        cmp = dispensable[-3 if has_jmp else -2]
        disp_jcc = dispensable[-2 if has_jmp else -1]
        assert cmp.id == X86_INS_CMP
        assert disp_jcc.group(X86_GRP_JUMP) and disp_jcc.id != X86_INS_JMP

        if not has_jmp:
            disp_jmp = self.insn_map[disp_jcc.address + disp_jcc.size]
            assert disp_jmp.id == X86_INS_JMP
            dispensable.append(disp_jmp)
        else:
            disp_jmp = dispensable[-1]

        if analysis.need_mov:
            assert dispensable[0].id == X86_INS_MOV
            del dispensable[0]

        rew = Rewriter(self.raw_virt_delta, self.insn_map, self.branches, dispensable, allow_claim_nops=True)
        # Replace cmp with branch condition
        rew.plan_instruction(analysis.branch_condition, cmp)
        # Jcc for condition
        rew.plan_branch(b"\x0F" + bytes([0x80 + analysis.x86_cc]), analysis.target_true, disp_jcc)
        # Jmp for condition
        rew.plan_branch(b"\xE9", analysis.target_false, disp_jmp)
        rew.rewrite(self.blob)

    def _has_interference(self, va_from: int, va_to: int, dispensable: list[CsInsn] = []) -> bool:
        """Returns True if eflags are written to in range [va_from, va_to)."""
        va = va_from
        while va < va_to:
            insn = self.insn_map[va]
            if insn not in dispensable:
                _, written = insn.regs_access()
                if X86_REG_EFLAGS in written:
                    return True
            va += insn.size

        return False


def deobfuscate_function(blob: bytearray, pe: pefile.PE, func: int) -> None:
    imagebase = pe.OPTIONAL_HEADER.ImageBase
    raw_virt_delta = imagebase + pe.sections[0].VirtualAddress - pe.sections[0].PointerToRawData

    dis = FunctionDisassembler()
    addr_insn_map, branches = dis.disassemble_function(blob[func-raw_virt_delta:], func)

    inliner = CflowUnflattener(blob, addr_insn_map, branches, raw_virt_delta)
    inliner.deobfuscate()


if __name__ == "__main__":
    with open("bin.dat.patchedc", "rb") as fp:
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

    with open("bin.dat.patchedf", "wb") as fp2:
        fp2.write(blob)
