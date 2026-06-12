"""Sandbox governance policy (Python port of ``src/runtime/policy.ts``)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List, Optional

MAX_POLICY_CODE_LENGTH = 20_000

# The guest sees no ambient I/O: it interacts only through ``ctx``. These tokens
# are rejected to keep runs local, replayable, and free of hidden dependencies.
DISALLOWED_GUEST_TOKENS = (
    "import ",
    "__",
    "open(",
    "eval(",
    "exec(",
    "compile(",
    "globals(",
    "locals(",
    "getattr(",
    "setattr(",
    "delattr(",
    "input(",
    "breakpoint(",
    "vars(",
)


@dataclass(frozen=True)
class SandboxPolicy:
    entrypoint: str = "main"
    timeout_ms: int = 10_000
    retry_count: int = 1
    memory_budget_mb: int = 512
    cold_start_ms: int = 0
    max_guest_code_length: int = 20_000
    snapshot_every_n_runs: int = 0


DEFAULT_SANDBOX_POLICY = SandboxPolicy()


def _clamp(value: int, low: int, high: int) -> int:
    return min(high, max(low, value))


def _to_int(value, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _sanitize_entrypoint(raw) -> str:
    if not isinstance(raw, str):
        return DEFAULT_SANDBOX_POLICY.entrypoint
    cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in raw.strip())
    return cleaned or DEFAULT_SANDBOX_POLICY.entrypoint


def normalize_policy(policy: Optional[dict] = None) -> SandboxPolicy:
    p = dict(policy or {})
    return SandboxPolicy(
        entrypoint=_sanitize_entrypoint(p.get("entrypoint", DEFAULT_SANDBOX_POLICY.entrypoint)),
        timeout_ms=_clamp(_to_int(p.get("timeout_ms", 10_000), 10_000), 100, 120_000),
        retry_count=_clamp(_to_int(p.get("retry_count", 1), 1), 0, 8),
        memory_budget_mb=_clamp(_to_int(p.get("memory_budget_mb", 512), 512), 64, 16_384),
        cold_start_ms=_clamp(_to_int(p.get("cold_start_ms", 0), 0), 0, 10_000),
        max_guest_code_length=_clamp(_to_int(p.get("max_guest_code_length", 20_000), 20_000), 256, 200_000),
        snapshot_every_n_runs=_clamp(_to_int(p.get("snapshot_every_n_runs", 0), 0), 0, 1_000),
    )


def validate_policy(policy: SandboxPolicy) -> List[str]:
    errors: List[str] = []
    if not policy.entrypoint.strip():
        errors.append("Entrypoint name is required.")
    if not 100 <= policy.timeout_ms <= 120_000:
        errors.append("Timeout must be between 100 and 120000 ms.")
    if not 0 <= policy.retry_count <= 8:
        errors.append("Retry count must be between 0 and 8.")
    if not 64 <= policy.memory_budget_mb <= 16_384:
        errors.append("Memory budget must be between 64MB and 16384MB.")
    if not 0 <= policy.cold_start_ms <= 10_000:
        errors.append("Cold start delay must be between 0 and 10000 ms.")
    if not 256 <= policy.max_guest_code_length <= 200_000:
        errors.append("Max guest code length must be between 256 and 200000 characters.")
    if not 0 <= policy.snapshot_every_n_runs <= 1_000:
        errors.append("Snapshot cadence must be between 0 and 1000 runs.")
    return errors


def assert_guest_code_safety(code: str, max_length: int = MAX_POLICY_CODE_LENGTH) -> None:
    if len(code) > max_length:
        raise ValueError(f"Guest code too large (max {max_length} characters).")
    for token in DISALLOWED_GUEST_TOKENS:
        if token in code:
            raise ValueError(f'Guest code uses disallowed token: "{token.strip()}".')


__all__ = [
    "SandboxPolicy",
    "DEFAULT_SANDBOX_POLICY",
    "DISALLOWED_GUEST_TOKENS",
    "normalize_policy",
    "validate_policy",
    "assert_guest_code_safety",
    "replace",
]
