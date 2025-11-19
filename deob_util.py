"""
Copyright (c) 2025 G DATA Advanced Analytics GmbH
"""

from typing import Callable, Optional

from capstone import Cs, CsInsn
from capstone import CS_ARCH_X86, CS_MODE_32
from capstone import CS_GRP_JUMP, CS_GRP_RET
from capstone.x86 import (X86_OP_IMM, X86_OP_MEM, X86_OP_REG,
                          X86_REG_AL, X86_REG_AH, X86_REG_AX, X86_REG_EAX,
                          X86_REG_BL, X86_REG_BH, X86_REG_BX, X86_REG_EBX,
                          X86_REG_CL, X86_REG_CH, X86_REG_CX, X86_REG_ECX,
                          X86_REG_DL, X86_REG_DH, X86_REG_DX, X86_REG_EDX,
                          X86_REG_DI, X86_REG_EDI, X86_REG_SI, X86_REG_ESI,
                          X86_REG_BP, X86_REG_EBP, X86_REG_EFLAGS,
                          X86_GRP_JUMP,
                          X86_INS_CALL, X86_INS_LEA, X86_INS_MOV, X86_INS_NOP, X86_INS_XOR)
from capstone.x86 import X86Op


X86_REG_NORMALIZE = {
    X86_REG_AL:  X86_REG_EAX,
    X86_REG_AH:  X86_REG_EAX,
    X86_REG_AX:  X86_REG_EAX,
    X86_REG_EAX: X86_REG_EAX,

    X86_REG_BL:  X86_REG_EBX,
    X86_REG_BH:  X86_REG_EBX,
    X86_REG_BX:  X86_REG_EBX,
    X86_REG_EBX: X86_REG_EBX,

    X86_REG_CL:  X86_REG_ECX,
    X86_REG_CH:  X86_REG_ECX,
    X86_REG_CX:  X86_REG_ECX,
    X86_REG_ECX: X86_REG_ECX,

    X86_REG_DL:  X86_REG_EDX,
    X86_REG_DH:  X86_REG_EDX,
    X86_REG_DX:  X86_REG_EDX,
    X86_REG_EDX: X86_REG_EDX,

    X86_REG_SI:  X86_REG_ESI,
    X86_REG_ESI: X86_REG_ESI,

    X86_REG_DI:  X86_REG_EDI,
    X86_REG_EDI: X86_REG_EDI,

    X86_REG_BP:  X86_REG_EBP,
    X86_REG_EBP: X86_REG_EBP,
}


def register_intersection(to_find: set[int], haystack: list[int]) -> set[int]:
    """Returns all registers from haystack that are in to_find. Everything is normalized to the full 32-bit register."""
    result = set()
    for reg in haystack:
        norm = X86_REG_NORMALIZE.get(reg, reg)
        if norm in to_find:
            result.add(norm)

    return result


def prev_insn(insn_map: dict[int, CsInsn], va: int) -> CsInsn:
    """Returns the instruction preceding the one at va, or throws."""
    distance = 0
    while distance < 15:
        va -= 1
        distance += 1
        if va in insn_map:
            return insn_map[va]
    raise Exception("prev_insn failed")


def branch_sources(branches: list[tuple[int, int]], va: int) -> list[int]:
    """Returns VAs that jump to the given va."""
    result = []
    for b in branches:
        if b[1] == va:
            result.append(b[0])
    return result


def find_above(insn_map: dict[int, CsInsn], start: CsInsn, predicate: Callable[[CsInsn], bool],
               limit: int = 10) -> Optional[CsInsn]:
    """Tests the given predicate (taking a CsInsn) against each instruction above 'start', for up to 'limit'."""
    va = start.address
    i = 0
    while i < limit:
        insn = prev_insn(insn_map, va)
        if insn.id != X86_INS_NOP:
            i += 1
        va = insn.address

        if predicate(insn):
            return insn

    return None


def find_above_s(slice: list[CsInsn], start: CsInsn, predicate: Callable[[CsInsn], bool],
                 limit: int = 10) -> Optional[CsInsn]:
    """Tests the given predicate (taking a CsInsn) against each instruction above 'start', for up to 'limit'."""
    i = 0
    pos = slice.index(start)
    while i < limit and pos > 0:
        pos -= 1
        i += 1

        if predicate(slice[pos]):
            return slice[pos]

    return None


def get_register_sub_slice(slice: list[CsInsn], reg: int) -> list[CsInsn]:
    """Filters slice for instructions that influence reg."""
    result: list[CsInsn] = []

    tracked_regs = {reg}
    for insn in slice[::-1]:
        read, written = insn.regs_access()
        tracked_and_written = register_intersection(tracked_regs, written)
        if tracked_and_written:
            result.insert(0, insn)

            assert len(tracked_and_written) == 1
            written_reg = next(iter(tracked_and_written))

            new_tracked = set(read) - {X86_REG_EFLAGS}
            tracked_regs.update(new_tracked)

            if (insn.id in (X86_INS_LEA, X86_INS_MOV) and written_reg not in new_tracked) or \
                    (insn.id == X86_INS_XOR and insn.operands[0].value.reg == insn.operands[1].value.reg):
                tracked_regs.remove(written_reg)
                if not tracked_regs:
                    break

    return result


def operands_equal(o1: X86Op, o2: X86Op) -> bool:
    if o1.type != o2.type:
        return False
    if o1.type == X86_OP_REG:
        return o1.value.reg == o2.value.reg
    if o1.type == X86_OP_MEM:
        m1 = o1.value.mem
        m2 = o2.value.mem
        return m1.base == m2.base and m1.disp == m2.disp and \
            m1.index == m2.index and m1.scale == m2.scale and \
            m1.segment == m2.segment
    if o1.type == X86_OP_IMM:
        return o1.value.imm == o2.value.imm
    raise AssertionError()


class ContinuousDisassembler:

    def __init__(self, arch: int = CS_ARCH_X86, mode: int = CS_MODE_32) -> None:
        self.cs = Cs(arch, mode)
        self.cs.detail = True

    def disassemble_all(self, code: bytes, va: int) -> dict[int, CsInsn]:
        """
        Disassembles all given bytes.

        Returns:
          dict[address, CsInsn]
        """
        addr_insn: dict[int, CsInsn] = {}

        for insn in self.cs.disasm(code, va):
            addr_insn[insn.address] = insn

        return addr_insn


class FunctionDisassembler:
    """
    Disassembles a single function from start_addr until the first 'ret'.
    This works because the obfuscator never seems to produce ret blocks in the middle of a function.

    Produces:
      - addr_insn: dict[address, CsInsn]
      - branch_targets: list[tuple[address, address]] (source, target)

    Parameters:
      - arch, mode: capstone architecture/mode constants (defaults to IA-32).
    """

    def __init__(self, arch: int = CS_ARCH_X86, mode: int = CS_MODE_32) -> None:
        self.cs = Cs(arch, mode)
        self.cs.detail = True

    def _extract_imm_target(self, insn: CsInsn) -> Optional[int]:
        """
        Tries to extract an immediate target for branch/call/jump instructions.
        Returns the target address if found, else None.
        """
        ops = insn.operands
        first = ops[0]
        if first.type == X86_OP_IMM:
            return int(first.imm)
        return None

    def disassemble_function(self, code: bytes, start_addr: int) -> tuple[dict[int, CsInsn], list[tuple[int, int]]]:
        """
        Disassembles bytes starting at start_addr until a 'ret' instruction is encountered.

        Returns:
          (addr_insn, branch_targets)
        """
        addr_insn: dict[int, CsInsn] = {}
        branches: list[tuple[int, int]] = []

        for insn in self.cs.disasm(code, start_addr):
            addr_insn[insn.address] = insn

            # In case of branch, record target
            if CS_GRP_JUMP in insn.groups:
                tgt = self._extract_imm_target(insn)
                if tgt is not None:
                    branches.append((insn.address, tgt))

            # In case of return, stop disassembling
            if (CS_GRP_RET in insn.groups) or insn.mnemonic.lower().startswith("ret"):
                break

        return addr_insn, branches


class Extent:
    """
    Represents an extent of virtual address space that is considered
    unoccupied and may be overwritten with new instructions.
    """

    def __init__(self, first_insn: CsInsn) -> None:
        self.start = first_insn.address
        self.size = first_insn.size
        self.end = self.start + self.size
        self.allocated_size = 0

    def contains(self, address: int) -> bool:
        return self.start <= address < self.end

    def get_remainder(self) -> tuple[int, int]:
        """Returns (va, size) of unallocated space."""
        return self.start + self.allocated_size, self.size - self.allocated_size

    def try_add(self, insn: CsInsn) -> bool:
        """Expands this extent if the instruction at its current end is provided."""
        if insn.address == self.end:
            self.size += insn.size
            self.end += insn.size
            return True
        return False

    def try_place(self, size: int) -> int:
        """Allocates space for overwriting in this extent.
           Returns the addess of the placement, or -1 if there was no space."""
        va = self.start + self.allocated_size
        if va + size <= self.end:
            self.allocated_size += size
            return va
        return -1


class Rewriter:
    """
    This class is capable of rewriting a stretch of instructions, i.e., replacing instructions that
    are not required anymore with new instructions (currently only branches).

    It's able to move around instructions that are still needed in order to
    make room for the new instructions.
    """

    def __init__(self, imagebase: int, insn_map: dict[int, CsInsn],
                 all_branches: list[tuple[int, int]], dispensable_insns: list[CsInsn],
                 allow_claim_nops: bool = False) -> None:
        """
        - all_branches: List of src,dest va tuples.
        - dispensable_insns: List of instructions that may be overwritten. Assumed to be sorted by address!
        - allow_claim_nops: If True, nops below the dispensable area may be overwritten. Has implications
                            for branches targeting the nops; they're adjusted.
        """
        self.new_instructions: list[tuple[bytes, Optional[int], CsInsn]] = []
        self.imagebase = imagebase
        self.insn_map = insn_map
        self.all_branches = all_branches
        self.dispensable_insns = dispensable_insns
        self.allow_claim_nops = allow_claim_nops

        self.extents = [Extent(self.dispensable_insns[0])]
        for insn in self.dispensable_insns[1:]:
            if not self.extents[-1].try_add(insn):
                self.extents.append(Extent(insn))

        self._work_blob: Optional[bytearray] = None

    @staticmethod
    def _contains_rel(insn: CsInsn) -> bool:
        """Returns True if the given instruction depends on its current position."""
        return (insn.group(X86_GRP_JUMP) or insn.id == X86_INS_CALL) and insn.operands[0].type == X86_OP_IMM

    def _is_target(self, va: int) -> bool:
        """Returns True if the given va is a branch target."""
        return any(b[1] == va for b in self.all_branches)

    def plan_branch(self, opcode: bytes, target_va: int, anchor: CsInsn) -> None:
        """
        Enqueues insertion of a branch instruction.

        opcode: Bytes for the branch opcode.
        target_va: Destination of the branch (virtual address).
        anchor: Location around which the branch should be inserted.
        """
        if anchor not in self.dispensable_insns:
            raise Exception(f"The given anchor {anchor} is not listed as overwritable")
        self.new_instructions.append((opcode, target_va, anchor))

    def plan_instruction(self, opcode: bytes, anchor: CsInsn) -> None:
        """
        Enqueues insertion of a non-branch instruction.

        opcode: Bytes for instruction.
        anchor: Location around which the instruction should be inserted.
        """
        if anchor not in self.dispensable_insns:
            raise Exception(f"The given anchor {anchor} is not listed as overwritable")
        self.new_instructions.append((opcode, None, anchor))

    def get_extent(self, va: int) -> Extent:
        """Returns the extent containing va or the next one below (since _push_up accumulates free space at the bottom).
           Throws ValueError if va seems completely out of range."""
        for extent in self.extents:
            if extent.contains(va) or extent.start > va:  # extents list is ordered by start
                return extent
        raise ValueError()

    def _reset_extents(self) -> None:
        """Clears all allocated space."""
        for ext in self.extents:
            ext.allocated_size = 0

    def _push_up(self, blob: bytes, space_required: int) -> bool:
        """Makes space by moving unrelated instructions between extents upwards.
           Returns False if no more space could be created.
           It's also not guaranteed space_required is actually met in a single call."""
        if len(self.extents) < 2:
            return False

        ext_upper = self.extents[-2]
        ext_lower = self.extents[-1]

        # Collect everything in the "gap".
        affected_insns: list[CsInsn] = []
        va = ext_upper.end  # Points to the first instruction in the gap between extents
        move_start = va
        move_size = 0
        while va != ext_lower.start:
            insn = self.insn_map[va]
            if Rewriter._contains_rel(insn) or self._is_target(insn.address):
                raise AssertionError()  # just out of interest
                return False

            affected_insns.append(insn)
            move_size += insn.size
            va = insn.address + insn.size

        assert len(affected_insns) > 0

        # We move them into the upper extent, thus increasing space in the lower one.
        move_distance = 0
        va = ext_upper.end
        while move_distance < space_required and va != ext_upper.start:
            prev = prev_insn(self.insn_map, va)
            move_distance += prev.size
            va = prev.address

        target = ext_upper.end - move_distance
        assert ext_upper.contains(target)

        print(f"Move by {move_distance} from {move_start:x} to {target:x} (len {move_size})")

        for insn in affected_insns:
            del self.insn_map[insn.address]
            insn._raw.address -= move_distance
            self.insn_map[insn.address] = insn

        ext_upper.end -= move_distance
        ext_upper.size -= move_distance
        ext_lower.start -= move_distance
        ext_lower.size += move_distance

        if ext_upper.size == 0:
            self.extents.remove(ext_upper)

        # Work in a temp blob because we can't be sure the rewrite as a whole will succeed.
        if self._work_blob is None:
            self._work_blob = bytearray(blob)

        data = self._work_blob[move_start - self.imagebase:move_start - self.imagebase + move_size]
        self._work_blob[target - self.imagebase:target - self.imagebase + len(data)] = data

        return True

    def rewrite(self, blob: bytearray) -> None:
        """Attempts to execute the planned rewrites. blob is modified in-place."""
        self.new_instructions.sort(key=lambda t: t[2].address)

        to_write = []
        for opcode, target_va, anchor in self.new_instructions:
            extent = self.get_extent(anchor.address)
            new_insn_length = len(opcode) + (4 if target_va is not None else 0)
            new_insn_va = extent.try_place(new_insn_length)

            if new_insn_va == -1:
                moved = self._push_up(blob, new_insn_length - (extent.size - extent.allocated_size))
                if moved or (self.allow_claim_nops and self._extend_extent(extent, blob)):
                    # Retry placement in changed layout.
                    self._reset_extents()
                    self.rewrite(blob)
                else:
                    raise Exception(f"Placement failed for {opcode.hex()} near {anchor}")
                return
            # else:
            #     print(f"Placement: {new_insn_va:x}")

            new_insn_data = opcode
            if target_va is not None:
                new_insn_data += int.to_bytes(target_va - new_insn_va - new_insn_length, 4, 'little', signed=True)
            to_write.append((new_insn_va, new_insn_data))

        for extent in self.extents:
            if extent.allocated_size != 0 and extent.allocated_size != extent.size:
                va, size = extent.get_remainder()
                to_write.append((va, b"\x90" * size))
            elif extent.allocated_size == 0:
                # Help deob_cflow by getting rid of unneeded instructions in deob_jumps.
                # Overwriting everything is not safe; jumps have to be filtered.
                va = extent.start
                while va in self.insn_map and va < extent.end:
                    insn = self.insn_map[va]
                    if not insn.group(X86_GRP_JUMP) and blob[va-self.imagebase:va-self.imagebase+insn.size] == insn.bytes:  # noqa:E501
                        to_write.append((insn.address, b"\x90" * insn.size))
                    va += insn.size

        # Everything seems in order; now do the actual modifications to the bytearray.
        if self._work_blob is not None:
            blob[:] = self._work_blob

        for va, data in to_write:
            blob[va-self.imagebase:va-self.imagebase+len(data)] = data

    def _extend_extent(self, extent: Extent, blob: bytearray) -> bool:
        """Checks if the space below the extent happens to be nops, then claims it."""
        space = 0
        while blob[extent.end - self.imagebase + space] == 0x90:
            space += 1

        if space == 0:
            return False

        for va in range(extent.end, extent.end + space):
            for i in range(len(self.all_branches)):
                src, dest = self.all_branches[i]
                # Jumping into nops? Modify it to skip nops
                if dest == va:
                    jmpoff = src - self.imagebase
                    if blob[jmpoff] != 0xE9:
                        return False  # not supported
                    oldrel = int.from_bytes(blob[jmpoff+1:jmpoff+5], 'little', signed=True)
                    blob[jmpoff+1:jmpoff+5] = int.to_bytes(oldrel + space, 4, 'little', signed=True)
                    self.all_branches[i] = (src, dest + space)
                    self.insn_map[src].operands[0].value.imm += space

        extent.end += space
        extent.size += space

        return True
