"""Sandbox fabric and eBPF load balancer (Python port of ``src/net``).

Network policy and load balancing are the same verified eBPF bytecode used in
the browser SDK, executed by the ported VM. The fabric and load balancer are
in-process models of TCP/SDN, not implementations; the same bytecode attaches
to kernel eBPF (tc/XDP) on the server tier.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .asm import INGRESS_ADDR, round_robin_balancer
from .ebpf import EbpfEnv, EbpfMap, EbpfVm
from .sandbox import Sandbox

_DEFAULT_HANDLER = "return {'echo': ctx.args['payload'], 'from': ctx.args['from']}"


@dataclass
class NetResponse:
    ok: bool
    status: int
    from_addr: int
    to_addr: int
    body: Any = None
    error: Optional[str] = None
    denied: bool = False


@dataclass
class _Endpoint:
    address: int
    name: str
    sandbox: Sandbox
    handler: str


def _pack_ctx(fields: List[int]) -> bytes:
    return b"".join(struct.pack("<Q", f & ((1 << 64) - 1)) for f in fields)


class SandboxFabric:
    """In-process virtual network: addressing, east-west delivery, eBPF policy."""

    def __init__(self, on_log: Optional[Callable[[str], None]] = None,
                 policy_program: Optional[bytes] = None) -> None:
        self._endpoints: Dict[int, _Endpoint] = {}
        self._next_address = INGRESS_ADDR + 1
        self._policy: Optional[EbpfVm] = None
        self._on_log = on_log or (lambda _msg: None)
        self.delivered_by_dst = EbpfMap()
        self.dropped_by_src = EbpfMap()
        if policy_program is not None:
            self.set_policy(policy_program)

    def set_policy(self, program: Optional[bytes]) -> None:
        self._policy = EbpfVm(program) if program is not None else None

    def join(self, sandbox: Sandbox, handler: Optional[str] = None, name: Optional[str] = None) -> int:
        address = self._next_address
        self._next_address += 1
        self._endpoints[address] = _Endpoint(
            address=address,
            name=name or f"sandbox-{address}",
            sandbox=sandbox,
            handler=handler or _DEFAULT_HANDLER,
        )
        return address

    def leave(self, address: int) -> bool:
        return self._endpoints.pop(address, None) is not None

    def addresses(self) -> List[int]:
        return list(self._endpoints.keys())

    def endpoint_name(self, address: int) -> str:
        if address == INGRESS_ADDR:
            return "ingress"
        endpoint = self._endpoints.get(address)
        return endpoint.name if endpoint else f"addr-{address}"

    def request(self, from_addr: int, to_addr: int, payload: Any,
                port: int = 0, protocol: int = 0) -> NetResponse:
        target = self._endpoints.get(to_addr)
        if target is None:
            self.dropped_by_src.add(from_addr, 1)
            return NetResponse(False, 404, from_addr, to_addr, error=f"No endpoint at address {to_addr}.")

        if self._policy is not None:
            verdict = self._run_policy(from_addr, to_addr, port, self._length(payload), protocol)
            if verdict != 0:
                self.dropped_by_src.add(from_addr, 1)
                self._on_log(
                    f"[fabric] policy dropped {self.endpoint_name(from_addr)} -> "
                    f"{self.endpoint_name(to_addr)} (verdict {verdict})."
                )
                return NetResponse(False, 403, from_addr, to_addr,
                                   error="Denied by network policy.", denied=True)

        result = target.sandbox.run(
            target.handler,
            name=f"request:{self.endpoint_name(from_addr)}->{target.name}",
            args={"from": from_addr, "payload": payload, "port": port},
        )
        self.delivered_by_dst.add(to_addr, 1)
        return NetResponse(
            ok=result.ok,
            status=200 if result.ok else 500,
            from_addr=from_addr,
            to_addr=to_addr,
            body=result.value if result.ok else None,
            error=None if result.ok else result.error,
        )

    def _run_policy(self, from_addr, to_addr, port, length, protocol) -> int:
        ctx = _pack_ctx([from_addr, to_addr, port, length, protocol])
        try:
            return self._policy.run(ctx, EbpfEnv(trace=lambda v: self._on_log(f"[policy] trace: {v}")))
        except Exception as error:  # noqa: BLE001 — a crashing policy fails closed
            self._on_log(f"[fabric] policy error (failing closed): {error}")
            return 1

    @staticmethod
    def _length(payload: Any) -> int:
        try:
            import json
            return len(json.dumps(payload))
        except Exception:  # noqa: BLE001
            return 0


def _fnv1a(text: str) -> int:
    h = 0x811C9DC5
    for ch in text:
        h ^= ord(ch)
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


_MAX_U64 = (1 << 64) - 1


class LoadBalancer:
    """eBPF load balancer that also serves static routes (web-server mode)."""

    address = INGRESS_ADDR

    def __init__(self, fabric: SandboxFabric, program: Optional[bytes] = None,
                 maps: Optional[List[EbpfMap]] = None,
                 on_log: Optional[Callable[[str], None]] = None) -> None:
        self._fabric = fabric
        self._backends: List[int] = []
        self._maps = maps if maps is not None else [EbpfMap()]
        self._vm = EbpfVm(program or round_robin_balancer())
        self._static: Dict[str, Any] = {}
        self._on_log = on_log or (lambda _msg: None)

    def add_backend(self, address: int) -> None:
        if address not in self._backends:
            self._backends.append(address)

    def remove_backend(self, address: int) -> None:
        if address in self._backends:
            self._backends.remove(address)

    def backend_pool(self) -> List[int]:
        return list(self._backends)

    def set_program(self, program: bytes) -> None:
        self._vm = EbpfVm(program)

    def serve_static(self, path: str, body: Any) -> None:
        self._static[path] = body

    def handle(self, path: str = "/", payload: Any = None,
               src_port: int = 0, request_hash: Optional[int] = None) -> NetResponse:
        if path in self._static:
            return NetResponse(True, 200, self.address, self.address, body=self._static[path])
        if not self._backends:
            return NetResponse(False, 503, self.address, self.address, error="No backends available.")

        import json
        try:
            payload_str = json.dumps(payload)
        except Exception:  # noqa: BLE001
            payload_str = ""
        h = request_hash if request_hash is not None else _fnv1a(f"{path}:{payload_str}")
        index = self._select_backend(src_port, _fnv1a(path), h)
        if index is None:
            return NetResponse(False, 503, self.address, self.address,
                               error="Load balancer dropped the request.")

        backend = self._backends[index]
        self._on_log(f"[lb] {path} -> {self._fabric.endpoint_name(backend)} (backend {index}).")
        return self._fabric.request(self.address, backend, payload, port=src_port)

    def _select_backend(self, src_port: int, dst_port_hash: int, request_hash: int) -> Optional[int]:
        ctx = _pack_ctx([self.address, src_port, dst_port_hash, request_hash, len(self._backends)])
        try:
            ret = self._vm.run(ctx, EbpfEnv(maps=self._maps,
                                            trace=lambda v: self._on_log(f"[lb] trace: {v}")))
        except Exception as error:  # noqa: BLE001
            self._on_log(f"[lb] program error: {error}")
            return None
        if ret == _MAX_U64:
            return None
        return ret % len(self._backends)
