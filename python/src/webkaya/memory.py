"""Redis-shaped tiered memory (Python port of ``src/memory/tiered-memory.ts``).

The in-process tiers below are synchronous; a server deployment can back the
global tier with a real Redis (redis-py) behind the same method names.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional


def _glob_to_regex(pattern: str) -> "re.Pattern[str]":
    escaped = re.escape(pattern).replace(r"\*", ".*").replace(r"\?", ".")
    return re.compile(f"^{escaped}$")


class MemoryTier:
    """One key/value tier with lazy TTL expiry. ``del`` is spelled ``delete``."""

    def __init__(self, now: Callable[[], float] = lambda: time.time() * 1000) -> None:
        self._now = now
        self._data: Dict[str, dict] = {}

    def _live(self, key: str) -> Optional[dict]:
        entry = self._data.get(key)
        if entry is None:
            return None
        if entry.get("expires_at") is not None and entry["expires_at"] <= self._now():
            self._data.pop(key, None)
            return None
        return entry

    def get(self, key: str) -> Optional[str]:
        entry = self._live(key)
        return entry["value"] if entry else None

    def set(self, key: str, value, ttl_ms: Optional[float] = None) -> None:
        entry = {"value": str(value), "expires_at": None}
        if ttl_ms is not None and ttl_ms > 0:
            entry["expires_at"] = self._now() + ttl_ms
        self._data[key] = entry

    def delete(self, key: str) -> bool:
        return self._data.pop(key, None) is not None

    def incr(self, key: str, by: int = 1) -> int:
        entry = self._live(key)
        base = int(entry["value"]) if entry else 0
        nxt = base + by
        self._data[key] = {"value": str(nxt), "expires_at": entry["expires_at"] if entry else None}
        return nxt

    def expire(self, key: str, ttl_ms: float) -> bool:
        entry = self._live(key)
        if not entry:
            return False
        entry["expires_at"] = self._now() + ttl_ms
        return True

    def ttl(self, key: str) -> float:
        entry = self._live(key)
        if not entry:
            return -2
        if entry["expires_at"] is None:
            return -1
        return max(0, entry["expires_at"] - self._now())

    def keys(self, pattern: Optional[str] = None) -> List[str]:
        matcher = _glob_to_regex(pattern) if pattern and pattern != "*" else None
        out: List[str] = []
        for key in list(self._data.keys()):
            if self._live(key) and (matcher is None or matcher.match(key)):
                out.append(key)
        return out

    def flush(self) -> None:
        self._data.clear()


@dataclass
class MemoryBinding:
    """Handed to a sandbox: ``local`` is private, ``shared`` is the global tier.

    Named ``shared`` rather than ``global`` because ``global`` is a Python
    keyword; it maps to ``ctx.global`` in the TypeScript SDK.
    """

    local: MemoryTier
    shared: MemoryTier


class TieredMemory:
    def __init__(self, now: Callable[[], float] = lambda: time.time() * 1000) -> None:
        self._now = now
        self.shared = MemoryTier(now)
        self._locals: Dict[str, MemoryTier] = {}

    def local_for(self, identifier: str) -> MemoryTier:
        tier = self._locals.get(identifier)
        if tier is None:
            tier = MemoryTier(self._now)
            self._locals[identifier] = tier
        return tier

    def binding_for(self, identifier: str) -> MemoryBinding:
        return MemoryBinding(local=self.local_for(identifier), shared=self.shared)

    def drop_local(self, identifier: str) -> None:
        self._locals.pop(identifier, None)
