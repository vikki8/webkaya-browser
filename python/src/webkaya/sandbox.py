"""Client-side sandbox (Python port of ``src/sandbox/sandbox.ts``).

Runs guest Python against an isolated state dict under a governance policy,
records every run, and supports snapshot / fork / restore / replay. Guest code
runs behind a token-scanned, restricted-builtins boundary — an honesty note,
not a hard security claim. CPython cannot preempt a running thread, so the
policy ``timeout_ms`` is advisory in this local engine (enforced on the browser
and server tiers).
"""

from __future__ import annotations

import copy
import textwrap
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .memory import MemoryBinding
from .policy import SandboxPolicy, assert_guest_code_safety, normalize_policy

_SAFE_BUILTIN_NAMES = (
    # Data / iteration helpers
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter",
    "float", "int", "len", "list", "map", "max", "min", "range", "reversed",
    "round", "set", "sorted", "str", "sum", "tuple", "zip",
    # Exception types so guests can raise and catch normally
    "Exception", "ValueError", "RuntimeError", "KeyError", "IndexError",
    "TypeError", "ZeroDivisionError", "ArithmeticError", "AssertionError",
    "StopIteration", "OverflowError",
)


def _resolve_builtin(name: str):
    source = __builtins__ if isinstance(__builtins__, dict) else vars(__builtins__)
    return source[name]


_SAFE_BUILTINS = {name: _resolve_builtin(name) for name in _SAFE_BUILTIN_NAMES}


def _new_id() -> str:
    return str(uuid.uuid4())


@dataclass
class GuestContext:
    state: Dict[str, Any]
    args: Any
    log: Callable[[str], None]
    local: Any = None   # MemoryTier when memory is bound
    shared: Any = None  # MemoryTier (global tier) when memory is bound


@dataclass
class RunResult:
    id: str
    name: str
    ok: bool
    value: Any = None
    error: Optional[str] = None
    logs: List[str] = field(default_factory=list)
    started_at: float = 0.0
    duration_ms: float = 0.0


@dataclass
class RunRecord:
    id: str
    name: str
    code: str
    args: Any
    estimated_memory_mb: float
    ok: bool


@dataclass
class Snapshot:
    id: str
    label: Optional[str]
    created_at: float
    sandbox_id: str
    parent_snapshot_id: Optional[str]
    run_count: int
    state: Dict[str, Any]


class MemorySnapshotStore:
    def __init__(self) -> None:
        self._snapshots: Dict[str, Snapshot] = {}

    def save(self, snapshot: Snapshot) -> None:
        self._snapshots[snapshot.id] = copy.deepcopy(snapshot)

    def load(self, snapshot_id: str) -> Optional[Snapshot]:
        found = self._snapshots.get(snapshot_id)
        return copy.deepcopy(found) if found else None

    def list(self) -> List[Snapshot]:
        return sorted((copy.deepcopy(s) for s in self._snapshots.values()), key=lambda s: s.created_at)

    def remove(self, snapshot_id: str) -> None:
        self._snapshots.pop(snapshot_id, None)


def _compile_guest(code: str):
    body = textwrap.indent(code if code.strip() else "pass", "    ")
    source = "def _guest(ctx):\n" + body
    namespace: Dict[str, Any] = {}
    exec(compile(source, "<guest>", "exec"), {"__builtins__": _SAFE_BUILTINS}, namespace)  # noqa: S102
    return namespace["_guest"]


class Sandbox:
    def __init__(
        self,
        policy: Optional[dict] = None,
        initial_state: Optional[Dict[str, Any]] = None,
        store: Optional[MemorySnapshotStore] = None,
        on_log: Optional[Callable[[str], None]] = None,
        memory: Optional[MemoryBinding] = None,
        _parent_snapshot_id: Optional[str] = None,
    ) -> None:
        self.id = _new_id()
        self.policy: SandboxPolicy = normalize_policy(policy)
        self.parent_snapshot_id = _parent_snapshot_id
        self._initial_state = copy.deepcopy(initial_state or {})
        self._state = copy.deepcopy(self._initial_state)
        self._store = store or MemorySnapshotStore()
        self._on_log = on_log or (lambda _msg: None)
        self._memory = memory
        self._run_log: List[RunRecord] = []
        self._disposed = False

    # --- Parity aliases with the TS async API (create/restore are factories) ---
    @classmethod
    def create(cls, **kwargs) -> "Sandbox":
        return cls(**kwargs)

    @classmethod
    def restore(cls, snapshot_id: str, store: MemorySnapshotStore, **kwargs) -> "Sandbox":
        snapshot = store.load(snapshot_id)
        if snapshot is None:
            raise ValueError(f'Snapshot "{snapshot_id}" not found.')
        return cls(initial_state=snapshot.state, store=store, _parent_snapshot_id=snapshot.id, **kwargs)

    def get_state(self) -> Dict[str, Any]:
        return copy.deepcopy(self._state)

    def get_run_log(self) -> List[RunRecord]:
        return list(self._run_log)

    def run(self, code: str, name: Optional[str] = None, args: Any = None,
            estimated_memory_mb: float = 0) -> RunResult:
        self._assert_usable()
        run_id = _new_id()
        run_index = len(self._run_log)
        name = name or f"{self.policy.entrypoint}#{run_index + 1}"
        logs: List[str] = []
        started_at = time.time()
        started_clock = time.perf_counter()

        def finish(ok: bool, value: Any = None, error: Optional[str] = None) -> RunResult:
            duration_ms = (time.perf_counter() - started_clock) * 1000
            self._run_log.append(RunRecord(run_id, name, code, args, estimated_memory_mb, ok))
            return RunResult(run_id, name, ok, value, error, logs, started_at, duration_ms)

        try:
            assert_guest_code_safety(code, self.policy.max_guest_code_length)
            if estimated_memory_mb > self.policy.memory_budget_mb:
                raise ValueError(
                    f"Invocation requires ~{estimated_memory_mb}MB, above sandbox budget "
                    f"{self.policy.memory_budget_mb}MB."
                )
            working_state = copy.deepcopy(self._state)
            guest = _compile_guest(code)
        except Exception as error:  # noqa: BLE001
            return finish(False, error=str(error))

        def log(message) -> None:
            logs.append(str(message))
            self._on_log(str(message))

        ctx = GuestContext(
            state=working_state,
            args=args,
            log=log,
            local=self._memory.local if self._memory else None,
            shared=self._memory.shared if self._memory else None,
        )

        retries = max(0, self.policy.retry_count)
        last_error: Optional[str] = None
        for _attempt in range(retries + 1):
            try:
                value = guest(ctx)
                self._state = working_state  # commit only on success
                result = finish(True, value=value)
                self._maybe_auto_snapshot()
                return result
            except Exception as error:  # noqa: BLE001
                last_error = str(error)
                working_state = copy.deepcopy(self._state)
                ctx.state = working_state
        return finish(False, error=last_error)

    def snapshot(self, label: Optional[str] = None) -> Snapshot:
        self._assert_usable()
        snapshot = Snapshot(
            id=_new_id(),
            label=label,
            created_at=time.time(),
            sandbox_id=self.id,
            parent_snapshot_id=self.parent_snapshot_id,
            run_count=len(self._run_log),
            state=copy.deepcopy(self._state),
        )
        self._store.save(snapshot)
        return snapshot

    def fork(self, **overrides) -> "Sandbox":
        self._assert_usable()
        snapshot = self.snapshot("fork-point")
        return Sandbox(
            policy=overrides.get("policy") or self.policy.__dict__,
            store=overrides.get("store") or self._store,
            on_log=overrides.get("on_log") or self._on_log,
            memory=overrides.get("memory") or self._memory,
            initial_state=copy.deepcopy(self._state),
            _parent_snapshot_id=snapshot.id,
        )

    def replay(self):
        self._assert_usable()
        replay_policy = dict(self.policy.__dict__)
        replay_policy.update(cold_start_ms=0, snapshot_every_n_runs=0)
        replay_box = Sandbox(
            policy=replay_policy,
            store=self._store,
            memory=self._memory,
            initial_state=copy.deepcopy(self._initial_state),
        )
        results = [
            replay_box.run(r.code, name=r.name, args=r.args, estimated_memory_mb=r.estimated_memory_mb)
            for r in self._run_log
        ]
        return results, replay_box.get_state()

    def dispose(self) -> None:
        self._disposed = True
        self._state = {}

    def _assert_usable(self) -> None:
        if self._disposed:
            raise RuntimeError("Sandbox has been disposed.")

    def _maybe_auto_snapshot(self) -> None:
        cadence = self.policy.snapshot_every_n_runs
        if cadence > 0 and len(self._run_log) % cadence == 0:
            self.snapshot("auto")
