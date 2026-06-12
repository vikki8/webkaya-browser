"""Userspace eBPF VM for the WebKaya Python client.

A faithful port of the TypeScript ``src/ebpf/vm.ts``: it executes standard
eBPF bytecode (64-bit registers r0-r10, 512-byte stack, little-endian
encoding) with a static verifier pass and dynamic bounds / instruction-count
enforcement. Because the encoding matches, a probe assembled for the browser
SDK runs unchanged here — and on kernel eBPF / ubpf on a native tier.
"""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

MASK64 = (1 << 64) - 1

# Helper ABI (maps addressed by fd index in r1, not by pointer).
MAP_GET = 1
MAP_SET = 2
MAP_ADD = 3
TRACE = 4
KTIME_GET_NS = 5
_KNOWN_HELPERS = {MAP_GET, MAP_SET, MAP_ADD, TRACE, KTIME_GET_NS}

MAX_PROGRAM_INSNS = 4096
DEFAULT_MAX_INSTRUCTIONS = 100_000
HARD_MAX_INSTRUCTIONS = 1_000_000

_CTX_BASE = 0x10000000
_STACK_BASE = 0x20000000
_STACK_SIZE = 512


def _u64(value: int) -> int:
    return value & MASK64


def _trunc(value: int, bits: int) -> int:
    return value & ((1 << bits) - 1)


def _to_signed(value: int, bits: int) -> int:
    value = _trunc(value, bits)
    return value - (1 << bits) if value & (1 << (bits - 1)) else value


class EbpfMap:
    """BPF-style u64->u64 map: the shared channel between programs and host."""

    def __init__(self, max_entries: int = 1024) -> None:
        self.max_entries = max_entries
        self._data: dict[int, int] = {}

    def get(self, key: int) -> Optional[int]:
        return self._data.get(_u64(key))

    def set(self, key: int, value: int) -> bool:
        key = _u64(key)
        if key not in self._data and len(self._data) >= self.max_entries:
            return False
        self._data[key] = _u64(value)
        return True

    def add(self, key: int, delta: int) -> bool:
        key = _u64(key)
        if key not in self._data and len(self._data) >= self.max_entries:
            return False
        self._data[key] = _u64(self._data.get(key, 0) + delta)
        return True

    def delete(self, key: int) -> bool:
        return self._data.pop(_u64(key), None) is not None

    def entries(self):
        return list(self._data.items())

    def clear(self) -> None:
        self._data.clear()


@dataclass(frozen=True)
class _Insn:
    opcode: int
    dst: int
    src: int
    off: int
    imm: int


def _decode(program: bytes) -> List[_Insn]:
    if len(program) == 0 or len(program) % 8 != 0:
        raise ValueError("eBPF program must be a non-empty multiple of 8 bytes.")
    insns: List[_Insn] = []
    for i in range(0, len(program), 8):
        reg = program[i + 1]
        off = struct.unpack_from("<h", program, i + 2)[0]
        imm = struct.unpack_from("<i", program, i + 4)[0]
        insns.append(_Insn(program[i], reg & 0x0F, (reg >> 4) & 0x0F, off, imm))
    return insns


def _check(insn: _Insn, index: int, count: int, is_lddw_second: Sequence[bool]) -> None:
    opcode, dst, src, off, imm = insn.opcode, insn.dst, insn.src, insn.off, insn.imm
    cls = opcode & 0x07
    if src > 10:
        raise ValueError(f"eBPF verify: invalid source register r{src} at insn {index}.")

    if cls in (0x07, 0x04):  # ALU64 / ALU32
        if (opcode >> 4) > 0x0C:
            raise ValueError(f"eBPF verify: unsupported ALU op 0x{opcode:x} at insn {index}.")
        if dst > 9:
            raise ValueError(f"eBPF verify: write to read-only r{dst} at insn {index}.")
    elif cls == 0x05:  # JMP
        jop = opcode >> 4
        if jop > 0x0D:
            raise ValueError(f"eBPF verify: unsupported JMP op 0x{opcode:x} at insn {index}.")
        if jop == 0x08:
            if imm not in _KNOWN_HELPERS:
                raise ValueError(f"eBPF verify: unknown helper {imm} at insn {index}.")
        elif jop != 0x09:
            target = index + 1 + off
            if target < 0 or target >= count or is_lddw_second[target]:
                raise ValueError(f"eBPF verify: jump to invalid target {target} at insn {index}.")
    elif cls == 0x06:  # JMP32 — conditionals only
        jop = opcode >> 4
        if jop in (0x00, 0x08, 0x09) or jop > 0x0D:
            raise ValueError(f"eBPF verify: unsupported JMP32 op 0x{opcode:x} at insn {index}.")
        target = index + 1 + off
        if target < 0 or target >= count or is_lddw_second[target]:
            raise ValueError(f"eBPF verify: jump to invalid target {target} at insn {index}.")
    elif cls == 0x01:  # LDX
        if (opcode & 0xE0) != 0x60:
            raise ValueError(f"eBPF verify: unsupported load mode at insn {index}.")
        if dst > 9:
            raise ValueError(f"eBPF verify: write to read-only r{dst} at insn {index}.")
    elif cls in (0x02, 0x03):  # ST / STX (dst is address base, r10 allowed)
        if (opcode & 0xE0) != 0x60:
            raise ValueError(f"eBPF verify: unsupported store mode at insn {index}.")
    else:  # LD — only lddw
        if opcode != 0x18:
            raise ValueError(f"eBPF verify: unsupported LD op 0x{opcode:x} at insn {index}.")
        if dst > 9:
            raise ValueError(f"eBPF verify: write to read-only r{dst} at insn {index}.")


def verify_program(program: bytes) -> List[_Insn]:
    insns = _decode(program)
    if len(insns) > MAX_PROGRAM_INSNS:
        raise ValueError(f"eBPF verify: program too large ({len(insns)} insns, max {MAX_PROGRAM_INSNS}).")

    is_lddw_second = [False] * len(insns)
    i = 0
    while i < len(insns):
        if not is_lddw_second[i] and insns[i].opcode == 0x18:
            if i + 1 >= len(insns) or insns[i + 1].opcode != 0:
                raise ValueError(f"eBPF verify: lddw at insn {i} is missing its second slot.")
            is_lddw_second[i + 1] = True
        i += 1

    for idx, insn in enumerate(insns):
        if not is_lddw_second[idx]:
            _check(insn, idx, len(insns), is_lddw_second)

    last = len(insns) - 1
    if is_lddw_second[last] or insns[last].opcode != 0x95:
        raise ValueError("eBPF verify: program must end with exit.")
    return insns


def _size_of(opcode: int) -> int:
    return {0x00: 4, 0x08: 2, 0x10: 1}.get(opcode & 0x18, 8)


@dataclass
class EbpfEnv:
    maps: Sequence[EbpfMap] = ()
    trace: Optional[Callable[[int], None]] = None


class EbpfVm:
    def __init__(self, program: bytes, max_instructions: int = DEFAULT_MAX_INSTRUCTIONS) -> None:
        self.insns = verify_program(program)
        self.max_instructions = min(max(max_instructions, 1), HARD_MAX_INSTRUCTIONS)

    def run(self, ctx: bytes = b"", env: Optional[EbpfEnv] = None) -> int:
        env = env or EbpfEnv()
        maps = list(env.maps)
        stack = bytearray(_STACK_SIZE)
        regs = [0] * 11
        regs[1] = _CTX_BASE
        regs[10] = _STACK_BASE + _STACK_SIZE

        def resolve(addr: int, size: int, write: bool):
            end = addr + size
            if _CTX_BASE <= addr and end <= _CTX_BASE + len(ctx):
                if write:
                    raise ValueError("eBPF: context memory is read-only.")
                return ("ctx", addr - _CTX_BASE)
            if _STACK_BASE <= addr and end <= _STACK_BASE + _STACK_SIZE:
                return ("stack", addr - _STACK_BASE)
            raise ValueError(f"eBPF: out-of-bounds memory access at 0x{addr:x} ({size} bytes).")

        def load(addr: int, size: int) -> int:
            region, offset = resolve(addr, size, False)
            buf = ctx if region == "ctx" else stack
            return int.from_bytes(buf[offset:offset + size], "little")

        def store(addr: int, size: int, value: int) -> None:
            resolve(addr, size, True)  # ctx is read-only; only stack reaches here
            offset = addr - _STACK_BASE
            stack[offset:offset + size] = _trunc(value, size * 8).to_bytes(size, "little")

        def map_for(fd: int) -> EbpfMap:
            if fd < 0 or fd >= len(maps):
                raise ValueError(f"eBPF: no map at fd {fd}.")
            return maps[fd]

        def call_helper(helper_id: int) -> int:
            if helper_id == MAP_GET:
                return map_for(regs[1]).get(regs[2]) or 0
            if helper_id == MAP_SET:
                return 0 if map_for(regs[1]).set(regs[2], regs[3]) else MASK64
            if helper_id == MAP_ADD:
                return 0 if map_for(regs[1]).add(regs[2], regs[3]) else MASK64
            if helper_id == TRACE:
                if env.trace:
                    env.trace(regs[1])
                return 0
            if helper_id == KTIME_GET_NS:
                return _u64(time.perf_counter_ns())
            raise ValueError(f"eBPF: unknown helper {helper_id}.")

        pc = 0
        steps = 0
        while True:
            steps += 1
            if steps > self.max_instructions:
                raise ValueError(f"eBPF: instruction limit exceeded ({self.max_instructions}).")
            insn = self.insns[pc]
            opcode, dst, src, off, imm = insn.opcode, insn.dst, insn.src, insn.off, insn.imm
            cls = opcode & 0x07

            if cls in (0x07, 0x04):  # ALU64 / ALU32
                width = 64 if cls == 0x07 else 32
                use_reg = (opcode & 0x08) != 0
                alu_op = opcode >> 4
                a = _trunc(regs[dst], width)
                b = _trunc(regs[src] if use_reg else imm, width)
                shift = b & (width - 1)
                if alu_op == 0x0:
                    out = a + b
                elif alu_op == 0x1:
                    out = a - b
                elif alu_op == 0x2:
                    out = a * b
                elif alu_op == 0x3:
                    out = 0 if b == 0 else a // b
                elif alu_op == 0x4:
                    out = a | b
                elif alu_op == 0x5:
                    out = a & b
                elif alu_op == 0x6:
                    out = a << shift
                elif alu_op == 0x7:
                    out = a >> shift
                elif alu_op == 0x8:
                    out = -a
                elif alu_op == 0x9:
                    out = a if b == 0 else a % b
                elif alu_op == 0xA:
                    out = a ^ b
                elif alu_op == 0xB:
                    out = b
                else:  # 0xC ARSH
                    out = _to_signed(a, width) >> shift
                regs[dst] = _trunc(out, width)
                pc += 1
            elif cls in (0x05, 0x06):  # JMP / JMP32
                jop = opcode >> 4
                if cls == 0x05 and jop == 0x08:  # CALL
                    regs[0] = _u64(call_helper(imm))
                    regs[1] = regs[2] = regs[3] = regs[4] = regs[5] = 0
                    pc += 1
                    continue
                if cls == 0x05 and jop == 0x09:  # EXIT
                    return regs[0]
                if cls == 0x05 and jop == 0x00:  # JA
                    pc += 1 + off
                    continue
                width = 64 if cls == 0x05 else 32
                use_reg = (opcode & 0x08) != 0
                ua = _trunc(regs[dst], width)
                ub = _trunc(regs[src] if use_reg else imm, width)
                sa = _to_signed(ua, width)
                sb = _to_signed(ub, width)
                if jop == 0x1:
                    take = ua == ub
                elif jop == 0x2:
                    take = ua > ub
                elif jop == 0x3:
                    take = ua >= ub
                elif jop == 0x4:
                    take = (ua & ub) != 0
                elif jop == 0x5:
                    take = ua != ub
                elif jop == 0x6:
                    take = sa > sb
                elif jop == 0x7:
                    take = sa >= sb
                elif jop == 0xA:
                    take = ua < ub
                elif jop == 0xB:
                    take = ua <= ub
                elif jop == 0xC:
                    take = sa < sb
                else:  # 0xD JSLE
                    take = sa <= sb
                pc += (1 + off) if take else 1
            elif cls == 0x00:  # lddw
                lo = imm & 0xFFFFFFFF
                hi = self.insns[pc + 1].imm & 0xFFFFFFFF
                regs[dst] = lo | (hi << 32)
                pc += 2
            elif cls == 0x01:  # LDX
                regs[dst] = load(_u64(regs[src] + off), _size_of(opcode))
                pc += 1
            elif cls == 0x02:  # ST imm
                store(_u64(regs[dst] + off), _size_of(opcode), imm)
                pc += 1
            else:  # 0x03 STX
                store(_u64(regs[dst] + off), _size_of(opcode), regs[src])
                pc += 1
