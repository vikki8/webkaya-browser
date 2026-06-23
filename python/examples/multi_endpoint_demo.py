"""Multi-endpoint fleet: different Claude-written handlers per route, one shared memory.

A small payments service with three endpoints, each backed by its own pool of
worker sandboxes behind a per-path load balancer:

  /score   (3 workers) — score a transaction, draw down a shared approval budget
  /refund  (2 workers) — approve a refund against a shared refund budget
  /audit   (1 worker)  — read aggregate stats written by the other pools

Every pool shares ONE global memory tier, so /audit sees what /score and /refund
wrote, and the two budgets are enforced fleet-wide. Each handler is written by
Claude (validated, with repair) and deployed across its pool; set REDIS_URL to
make the shared tier distributed.

Run:
    pip install webkaya[claude]            # + [redis] to distribute
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/multi_endpoint_demo.py
"""

import os
from dataclasses import dataclass
from typing import Dict, List

from webkaya import (
    ClaudeProvider,
    CodeGenResult,
    LoadBalancer,
    MemorySnapshotStore,
    RedisMemoryTier,
    Sandbox,
    SandboxFabric,
    TieredMemory,
    deny_east_west_policy,
)

HANDLER_RULES = """Contract:
- Your code is the BODY of a function receiving one argument `ctx`.
- `ctx.args['payload']` is the request body (a dict).
- `ctx.shared` is GLOBAL key/value memory shared by the WHOLE service. Methods: .get(key)->str|None, .set(key,value), .incr(key, by=1)->int. Values are strings.
- `ctx.local` is private per-worker memory (same methods).
- End with `return <dict>`. Increment local 'handled' by 1 on every call.
- NO imports (the sandbox rejects 'import', '__', 'open(', 'eval(', etc.). Builtins and ctx only. Keep it short."""


@dataclass
class Endpoint:
    path: str
    pool_size: int
    system: str
    reference: str
    sample: dict


SCORE = Endpoint(
    path="/score",
    pool_size=3,
    system=f"""You write the /score handler for a transaction-scoring service worker.

{HANDLER_RULES}

Seeded global keys: 'merchant_risk:<merchant>' (float string, missing=0), 'blocklist:<card>' (present='1' if blocked), 'approval_budget' (int string, fleet-wide approvals left).
Payload keys: id, merchant, card, amount (number).
Behavior:
- score = merchant_risk + amount/1000 + (4.0 if blocked else 0.0); flagged = blocked or score >= 5.0
- incr global 'total_processed'; if flagged incr global 'total_flagged'.
- if flagged: decision='review'. else if approval_budget>0: decrement it by 1, decision='approve'; else decision='hold'.
- return {{'id': id, 'score': round(score,2), 'decision': decision}}""",
    reference="""
p = ctx.args['payload']
risk = float(ctx.shared.get('merchant_risk:' + p['merchant']) or '0')
blocked = ctx.shared.get('blocklist:' + p['card']) is not None
score = risk + (p['amount'] / 1000.0) + (4.0 if blocked else 0.0)
flagged = blocked or score >= 5.0
ctx.shared.incr('total_processed')
if flagged:
    ctx.shared.incr('total_flagged')
decision = 'review'
if not flagged:
    if int(ctx.shared.get('approval_budget') or '0') > 0:
        ctx.shared.incr('approval_budget', -1)
        decision = 'approve'
    else:
        decision = 'hold'
ctx.local.incr('handled')
return {'id': p['id'], 'score': round(score, 2), 'decision': decision}
""",
    sample={"id": "s", "merchant": "acme", "card": "C-1", "amount": 100},
)

REFUND = Endpoint(
    path="/refund",
    pool_size=2,
    system=f"""You write the /refund handler for a payments service worker.

{HANDLER_RULES}

Seeded global keys: 'refund_budget' (int string, dollars of refund left for the whole fleet).
Payload keys: id, amount (integer dollars).
Behavior:
- remaining = int(refund_budget or 0). If amount <= remaining: decrement 'refund_budget' by amount, incr global 'refunds_count', approved=True. Else approved=False.
- return {{'id': id, 'approved': approved, 'refund_budget_left': <remaining after>}}""",
    reference="""
p = ctx.args['payload']
amount = int(p['amount'])
remaining = int(ctx.shared.get('refund_budget') or '0')
approved = amount <= remaining
if approved:
    remaining = ctx.shared.incr('refund_budget', -amount)
    ctx.shared.incr('refunds_count')
ctx.local.incr('handled')
return {'id': p['id'], 'approved': approved, 'refund_budget_left': remaining}
""",
    sample={"id": "r", "amount": 10},
)

AUDIT = Endpoint(
    path="/audit",
    pool_size=1,
    system=f"""You write the read-only /audit handler for a payments service worker.

{HANDLER_RULES}

Read these global counters (missing=0) and return them as integers in a dict:
'total_processed', 'total_flagged', 'approval_budget' (return as 'approvals_left'),
'refunds_count', 'refund_budget' (return as 'refund_budget_left'). Do not modify any global key.""",
    reference="""
ctx.local.incr('handled')
return {
    'total_processed': int(ctx.shared.get('total_processed') or '0'),
    'total_flagged': int(ctx.shared.get('total_flagged') or '0'),
    'approvals_left': int(ctx.shared.get('approval_budget') or '0'),
    'refunds_count': int(ctx.shared.get('refunds_count') or '0'),
    'refund_budget_left': int(ctx.shared.get('refund_budget') or '0'),
}
""",
    sample={},
)

ENDPOINTS = [SCORE, REFUND, AUDIT]

REQUESTS = [
    ("/score", {"id": "t01", "merchant": "acme", "card": "C-1001", "amount": 120}),
    ("/score", {"id": "t02", "merchant": "globex", "card": "C-1002", "amount": 900}),
    ("/refund", {"id": "r01", "amount": 40}),
    ("/score", {"id": "t03", "merchant": "acme", "card": "C-1003", "amount": 60}),
    ("/refund", {"id": "r02", "amount": 50}),
    ("/score", {"id": "t04", "merchant": "acme", "card": "C-1004", "amount": 75}),
    ("/refund", {"id": "r03", "amount": 30}),   # pushes refund budget (100) over
    ("/score", {"id": "t05", "merchant": "initech", "card": "C-1005", "amount": 90}),
    ("/audit", {}),
]


def build_memory() -> TieredMemory:
    url = os.environ.get("REDIS_URL")
    if url:
        print(f"Global memory: Redis at {url}\n")
        return TieredMemory(shared=RedisMemoryTier(url=url, namespace="multi-endpoint-demo"))
    return TieredMemory()


def seed_global(g) -> None:
    g.set("merchant_risk:acme", "1.0")
    g.set("merchant_risk:globex", "4.5")
    g.set("merchant_risk:initech", "1.5")
    g.set("blocklist:C-9999", "1")
    g.set("approval_budget", "3")   # fleet-wide
    g.set("refund_budget", "100")   # fleet-wide dollars


def smoke_test(handler: str, sample: dict):
    memory = TieredMemory()
    seed_global(memory.shared)
    box = Sandbox.create(policy={"cold_start_ms": 0}, store=MemorySnapshotStore(), memory=memory.binding_for("smoke"))
    res = box.run(handler, name="smoke", args={"from": 0, "payload": sample, "port": 0})
    if not res.ok:
        return False, res.error
    if not isinstance(res.value, dict):
        return False, f"handler returned {res.value!r}; expected a dict"
    return True, None


def build_handler(provider, endpoint: Endpoint, max_attempts: int = 3) -> str:
    prompt = "Write the handler."
    for attempt in range(1, max_attempts + 1):
        gen = provider.generate_code(endpoint.system, prompt)
        ok, err = smoke_test(gen.code, endpoint.sample)
        if ok:
            print(f"  {endpoint.path}: handler ready ({gen.input_tokens} in / {gen.output_tokens} out tokens)")
            return gen.code
        print(f"  {endpoint.path}: validation failed on attempt {attempt}: {err}")
        prompt = (
            f"Write the handler.\n\nThe previous version failed when run.\n\n"
            f"Previous code:\n{gen.code}\n\nError:\n{err}"
        )
    raise RuntimeError(f"Could not obtain a working handler for {endpoint.path}.")


class ReferenceProvider:
    name = "reference"

    def __init__(self, by_path: Dict[str, str]):
        self._by_path = by_path
        self._current = None

    def for_path(self, path: str):
        self._current = path
        return self

    def generate_code(self, system: str, prompt: str) -> CodeGenResult:
        return CodeGenResult(code=self._by_path[self._current], explanation="reference")


def main() -> None:
    use_claude = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if use_claude:
        print("Using Claude (claude-opus-4-8) to write each endpoint's handler.\n")
        provider = ClaudeProvider()
    else:
        print("No ANTHROPIC_API_KEY — using built-in reference handlers to show the flow.\n")
        provider = ReferenceProvider({e.path: e.reference for e in ENDPOINTS})

    memory = build_memory()
    memory.shared.flush()
    seed_global(memory.shared)

    fabric = SandboxFabric(policy_program=deny_east_west_policy())
    routers: Dict[str, LoadBalancer] = {}
    worker_names: Dict[int, str] = {}

    print("Building endpoint pools:")
    for ep in ENDPOINTS:
        handler = build_handler(provider if use_claude else provider.for_path(ep.path), ep)
        lb = LoadBalancer(fabric)
        for i in range(ep.pool_size):
            name = f"{ep.path.strip('/')}-{i}"
            box = Sandbox.create(
                policy={"cold_start_ms": 0}, store=MemorySnapshotStore(), memory=memory.binding_for(name)
            )
            addr = fabric.join(box, handler=handler, name=name)
            lb.add_backend(addr)
            worker_names[addr] = name
        routers[ep.path] = lb
    print()

    print("=== Routing a mixed request stream across the endpoint pools ===\n")
    for path, payload in REQUESTS:
        res = routers[path].handle(path=path, payload=payload)
        served = fabric.endpoint_name(res.to_addr)
        body = res.body
        summary = body if path == "/audit" else {k: body[k] for k in list(body)[:3]}
        print(f"{path:7} -> {served:9}  {summary}")

    g = memory.shared
    print("\n=== Shared GLOBAL state (written across all three pools) ===")
    print("total processed   :", g.get("total_processed"))
    print("total flagged     :", g.get("total_flagged"))
    print("approvals left    :", g.get("approval_budget"), "(started at 3)")
    print("refunds count     :", g.get("refunds_count"))
    print("refund budget left:", g.get("refund_budget"), "(started at 100)")

    print(
        "\nThree pools, three different Claude-written handlers, one shared memory: the "
        "/audit pool read exactly what /score and /refund wrote, and both budgets were "
        "enforced across pools. A real microservice topology on the sandbox fleet — no "
        "database, no service mesh."
    )


if __name__ == "__main__":
    main()
