"""WebKaya — client-side AI agent sandboxes.

Python client mirroring the TypeScript SDK: governed sandboxes with snapshot /
fork / replay, a userspace eBPF VM, an eBPF-governed fabric and load balancer,
and Redis-shaped tiered memory. The eBPF bytecode is standard, so probe and
policy programs are portable across the browser SDK, this client, and kernel
eBPF on the server tier.
"""

from .policy import (
    DEFAULT_SANDBOX_POLICY,
    DISALLOWED_GUEST_TOKENS,
    SandboxPolicy,
    assert_guest_code_safety,
    normalize_policy,
    validate_policy,
)
from .memory import KVStore, MemoryBinding, MemoryTier, TieredMemory
from .redis_memory import RedisMemoryTier
from .sandbox import (
    GuestContext,
    MemorySnapshotStore,
    RunRecord,
    RunResult,
    Sandbox,
    Snapshot,
)
from .ebpf import (
    DEFAULT_MAX_INSTRUCTIONS,
    EbpfEnv,
    EbpfMap,
    EbpfVm,
    KTIME_GET_NS,
    MAP_ADD,
    MAP_GET,
    MAP_SET,
    TRACE,
    verify_program,
)
from . import asm
from .asm import (
    INGRESS_ADDR,
    LOAD_BALANCER_LAYOUT,
    NETWORK_POLICY_LAYOUT,
    assemble,
    deny_east_west_policy,
    hash_balancer,
    round_robin_balancer,
)
from .net import LoadBalancer, NetResponse, SandboxFabric
from .probes import TRACEPOINT_LAYOUTS, ProbeRegistry
from .llm import (
    AgentAttempt,
    AgentOutcome,
    ClaudeProvider,
    CodeAgent,
    CodeGenResult,
    LlmProvider,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "SandboxPolicy",
    "DEFAULT_SANDBOX_POLICY",
    "DISALLOWED_GUEST_TOKENS",
    "normalize_policy",
    "validate_policy",
    "assert_guest_code_safety",
    "MemoryTier",
    "TieredMemory",
    "MemoryBinding",
    "KVStore",
    "RedisMemoryTier",
    "Sandbox",
    "GuestContext",
    "RunResult",
    "RunRecord",
    "Snapshot",
    "MemorySnapshotStore",
    "EbpfVm",
    "EbpfMap",
    "EbpfEnv",
    "verify_program",
    "MAP_GET",
    "MAP_SET",
    "MAP_ADD",
    "TRACE",
    "KTIME_GET_NS",
    "DEFAULT_MAX_INSTRUCTIONS",
    "asm",
    "assemble",
    "INGRESS_ADDR",
    "NETWORK_POLICY_LAYOUT",
    "LOAD_BALANCER_LAYOUT",
    "deny_east_west_policy",
    "round_robin_balancer",
    "hash_balancer",
    "SandboxFabric",
    "LoadBalancer",
    "NetResponse",
    "ProbeRegistry",
    "TRACEPOINT_LAYOUTS",
    "LlmProvider",
    "ClaudeProvider",
    "CodeAgent",
    "CodeGenResult",
    "AgentAttempt",
    "AgentOutcome",
]
