"""eBPF instruction builder and default fabric programs (Python port of
``src/ebpf/asm.ts`` and ``src/net/hooks.ts``). Emits standard eBPF encoding,
so programs built here also load into the browser SDK and kernel eBPF.
"""

from __future__ import annotations

import struct
from typing import List

from .ebpf import KTIME_GET_NS, MAP_ADD, MAP_GET, MAP_SET, TRACE  # noqa: F401

# Reserved address for ingress / load balancers; sandboxes get addresses >= 1.
INGRESS_ADDR = 0

NETWORK_POLICY_LAYOUT = ("srcAddr", "dstAddr", "port", "length", "protocol")
LOAD_BALANCER_LAYOUT = ("srcAddr", "srcPort", "dstPort", "requestHash", "backendCount")


def insn(opcode: int, dst: int, src: int, offset: int, imm: int) -> bytes:
    return struct.pack("<BBhi", opcode, ((src & 0x0F) << 4) | (dst & 0x0F), offset, _i32(imm))


def _i32(value: int) -> int:
    value &= 0xFFFFFFFF
    return value - (1 << 32) if value & 0x80000000 else value


# ALU64
def mov_imm(dst, imm): return insn(0xB7, dst, 0, 0, imm)
def mov_reg(dst, src): return insn(0xBF, dst, src, 0, 0)
def add_imm(dst, imm): return insn(0x07, dst, 0, 0, imm)
def add_reg(dst, src): return insn(0x0F, dst, src, 0, 0)
def sub_imm(dst, imm): return insn(0x17, dst, 0, 0, imm)
def mul_imm(dst, imm): return insn(0x27, dst, 0, 0, imm)
def div_reg(dst, src): return insn(0x3F, dst, src, 0, 0)
def mod_reg(dst, src): return insn(0x9F, dst, src, 0, 0)
def and_imm(dst, imm): return insn(0x57, dst, 0, 0, imm)
# ALU32
def mov32_imm(dst, imm): return insn(0xB4, dst, 0, 0, imm)


def lddw(dst: int, value: int) -> bytes:
    value &= (1 << 64) - 1
    return insn(0x18, dst, 0, 0, value & 0xFFFFFFFF) + insn(0x00, 0, 0, 0, value >> 32)


# Memory
def ldxdw(dst, src, offset): return insn(0x79, dst, src, offset, 0)
def ldxw(dst, src, offset): return insn(0x61, dst, src, offset, 0)
def stxdw(dst, offset, src): return insn(0x7B, dst, src, offset, 0)
def stdw(dst, offset, imm): return insn(0x7A, dst, 0, offset, imm)
# Jumps
def ja(offset): return insn(0x05, 0, 0, offset, 0)
def jeq_imm(dst, imm, offset): return insn(0x15, dst, 0, offset, imm)
def jne_imm(dst, imm, offset): return insn(0x55, dst, 0, offset, imm)
def jgt_imm(dst, imm, offset): return insn(0x25, dst, 0, offset, imm)
def jge_imm(dst, imm, offset): return insn(0x35, dst, 0, offset, imm)
def jlt_imm(dst, imm, offset): return insn(0xA5, dst, 0, offset, imm)
def jle_imm(dst, imm, offset): return insn(0xB5, dst, 0, offset, imm)
# Control
def call(helper_id): return insn(0x85, 0, 0, 0, helper_id)
def exit_(): return insn(0x95, 0, 0, 0, 0)


def assemble(chunks: List[bytes]) -> bytes:
    return b"".join(chunks)


def deny_east_west_policy() -> bytes:
    """Drop a request only when both endpoints are sandboxes (addr != 0)."""
    return assemble([
        ldxdw(2, 1, 0),                 # r2 = srcAddr
        ldxdw(3, 1, 8),                 # r3 = dstAddr
        mov_imm(0, 0),                  # verdict = allow
        jeq_imm(2, INGRESS_ADDR, 2),    # src is ingress -> allow
        jeq_imm(3, INGRESS_ADDR, 1),    # dst is ingress -> allow
        mov_imm(0, 1),                  # both sandboxes -> deny
        exit_(),
    ])


def round_robin_balancer() -> bytes:
    """Counter in map fd 0; returns counter % backendCount."""
    return assemble([
        mov_reg(9, 1),                  # r9 = ctx ptr (preserved across calls)
        mov_imm(1, 0), mov_imm(2, 0),
        call(MAP_GET),                  # r0 = counter
        mov_reg(6, 0),                  # r6 = counter (preserved across calls)
        mov_imm(1, 0), mov_imm(2, 0), mov_reg(3, 6), add_imm(3, 1),
        call(MAP_SET),                  # counter += 1
        ldxdw(2, 9, 32),                # r2 = backendCount
        mov_reg(0, 6), mod_reg(0, 2),   # index = counter % backendCount
        exit_(),
    ])


def hash_balancer() -> bytes:
    """Sticky: returns requestHash % backendCount."""
    return assemble([
        ldxdw(0, 1, 24),                # r0 = requestHash
        ldxdw(2, 1, 32),                # r2 = backendCount
        mod_reg(0, 2),
        exit_(),
    ])
