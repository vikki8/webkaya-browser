"""Distributed global memory backed by Redis.

Implements the same ``KVStore`` interface as the in-process ``MemoryTier``, so
the global tier can move from one process to a Redis instance shared by a whole
fleet without touching guest code. Crucially, ``incr`` maps to Redis ``INCRBY``,
which is **atomic across processes** — that is what makes a shared counter or
budget race-safe when many workers hit it concurrently.

Redis is an optional dependency:  pip install webkaya[redis]
"""

from __future__ import annotations

from typing import Any, List, Optional


class RedisMemoryTier:
    """A ``KVStore`` over a Redis client. Keys are namespaced so one Redis
    instance can host many independent stores and ``flush`` only clears this
    namespace (never the whole database)."""

    def __init__(
        self,
        client: Any = None,
        namespace: str = "webkaya",
        url: str = "redis://localhost:6379/0",
    ) -> None:
        if client is None:
            try:
                import redis  # noqa: PLC0415
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "RedisMemoryTier needs the redis package. Install it with: "
                    "pip install webkaya[redis]"
                ) from exc
            client = redis.Redis.from_url(url, decode_responses=True)
        self._c = client
        self._ns = namespace

    def _k(self, key: str) -> str:
        return f"{self._ns}:{key}"

    def get(self, key: str) -> Optional[str]:
        value = self._c.get(self._k(key))
        return value if value is None else str(value)

    def set(self, key: str, value, ttl_ms: Optional[float] = None) -> None:
        if ttl_ms is not None and ttl_ms > 0:
            self._c.set(self._k(key), str(value), px=int(ttl_ms))
        else:
            self._c.set(self._k(key), str(value))

    def delete(self, key: str) -> bool:
        return bool(self._c.delete(self._k(key)))

    def incr(self, key: str, by: int = 1) -> int:
        # Atomic across processes — the reason to use Redis for a shared counter.
        return int(self._c.incrby(self._k(key), by))

    def expire(self, key: str, ttl_ms: float) -> bool:
        return bool(self._c.pexpire(self._k(key), int(ttl_ms)))

    def ttl(self, key: str) -> float:
        # Redis PTTL already returns -2 (missing) / -1 (no expiry) / ms remaining,
        # matching MemoryTier.ttl exactly.
        return int(self._c.pttl(self._k(key)))

    def keys(self, pattern: Optional[str] = None) -> List[str]:
        match = self._k(pattern if pattern else "*")
        prefix = self._k("")
        return [k[len(prefix):] for k in self._c.scan_iter(match=match)]

    def flush(self) -> None:
        for k in list(self._c.scan_iter(match=self._k("*"))):
            self._c.delete(k)
