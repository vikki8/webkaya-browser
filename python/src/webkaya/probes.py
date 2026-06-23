"""Sandbox tracepoints and the eBPF probe registry (Python port of
``src/sandbox/probes.ts``).

Verified eBPF programs attach to sandbox tracepoints. A ``run:start`` probe that
returns nonzero denies the run (admission control); probes at other tracepoints
observe and update their maps. This brings the Python sandbox to governance
parity with the TypeScript SDK: the same standard bytecode runs in both.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

from .ebpf import EbpfEnv, EbpfMap, EbpfVm

# Field order of the u64 context struct passed to probes at each tracepoint.
# Field N lives at byte offset N * 8 (little-endian); a probe reads it with
# ``ldxdw rX, [r1 + N*8]``.
TRACEPOINT_LAYOUTS = {
    "run:start": ("run_index", "code_length", "estimated_memory_mb", "has_args", "timestamp_ms"),
    "run:end": ("run_index", "ok", "duration_us", "log_count", "code_length"),
    "snapshot": ("run_count", "state_bytes"),
    "log": ("run_index", "message_length"),
}

_MASK64 = (1 << 64) - 1


@dataclass
class _AttachedProbe:
    id: str
    tracepoint: str
    name: str
    vm: EbpfVm
    maps: List[EbpfMap]
    fail_closed: bool


class ProbeRegistry:
    """Holds probes and fires them at tracepoints. Probe errors never escape to
    the sandbox; a crashing probe denies only when ``fail_closed`` is set."""

    def __init__(self, on_log: Callable[[str], None]) -> None:
        self._probes: List[_AttachedProbe] = []
        self._on_log = on_log
        self._counter = 0

    def attach(
        self,
        tracepoint: str,
        program: bytes,
        name: Optional[str] = None,
        maps: Optional[Sequence[EbpfMap]] = None,
        fail_closed: bool = False,
        max_instructions: Optional[int] = None,
    ) -> str:
        if tracepoint not in TRACEPOINT_LAYOUTS:
            raise ValueError(f"Unknown tracepoint: {tracepoint!r}")
        self._counter += 1
        probe_id = f"probe-{self._counter}"
        vm = EbpfVm(program) if max_instructions is None else EbpfVm(program, max_instructions)
        self._probes.append(
            _AttachedProbe(probe_id, tracepoint, name or probe_id, vm, list(maps or []), fail_closed)
        )
        return probe_id

    def detach(self, probe_id: str) -> bool:
        before = len(self._probes)
        self._probes = [p for p in self._probes if p.id != probe_id]
        return len(self._probes) != before

    def fire(self, tracepoint: str, fields: Sequence[int]) -> Optional[str]:
        """Run every probe on ``tracepoint`` against the encoded context.
        Returns the name of the first probe that returns nonzero (or that errors
        while ``fail_closed``); the caller decides whether that vetoes the op."""
        matching = [p for p in self._probes if p.tracepoint == tracepoint]
        if not matching:
            return None
        ctx = b"".join(struct.pack("<Q", int(f) & _MASK64) for f in fields)
        for probe in matching:
            try:
                env = EbpfEnv(
                    maps=probe.maps,
                    trace=lambda v, n=probe.name: self._on_log(f"[probe {n}] trace: {v}"),
                )
                if probe.vm.run(ctx, env) != 0:
                    return probe.name
            except Exception as error:  # noqa: BLE001 — probe failures never crash the sandbox
                self._on_log(f"[probe {probe.name}] error: {error}")
                if probe.fail_closed:
                    return probe.name
        return None
