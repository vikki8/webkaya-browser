"""Cluster + LLM: Claude writes the fleet's handler, the fleet runs it.

Same scenario as cluster_demo.py, but the per-request scoring handler is
written by Claude instead of by hand. The flow:

  1. Claude writes a handler that scores transactions using GLOBAL memory.
  2. We smoke-test it in a throwaway sandbox; if it errors, the error is fed
     back to Claude for a fix (the generate -> run -> repair loop).
  3. The validated handler is deployed to every worker in the fleet.
  4. Requests are load-balanced across the workers, which coordinate through
     the shared global memory tier exactly as before.

So the LLM writes the code once; an isolated, load-balanced fleet runs it and
coordinates through shared state — the combination that's hard to get from
per-VM sandbox services.

Run:
    pip install webkaya[claude]
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/cluster_agent_demo.py

Without a key it falls back to a built-in reference handler so you can see the
flow; set the key to have Claude actually write it.
"""

import os

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


def build_memory() -> TieredMemory:
    """In-process by default; set REDIS_URL to distribute the global tier."""
    url = os.environ.get("REDIS_URL")
    if url:
        print(f"Global memory: Redis at {url}\n")
        return TieredMemory(shared=RedisMemoryTier(url=url, namespace="cluster-agent-demo"))
    return TieredMemory()

HANDLER_SYSTEM = """You write a Python request handler that runs inside a WebKaya sandbox worker — one of several workers behind a load balancer.

Contract:
- Your code is the BODY of a function receiving one argument `ctx`.
- `ctx.args['payload']` is the transaction dict with keys: id (str), merchant (str), card (str), amount (number).
- `ctx.shared` is GLOBAL key/value memory shared by the WHOLE fleet. Methods: .get(key) -> str or None, .set(key, value), .incr(key, by=1) -> int. All values are strings.
- `ctx.local` is private per-worker memory (same methods).
- `ctx.log(message)` records a line of output.
- End with `return <dict>`.

Seeded global keys you can rely on:
- 'merchant_risk:<merchant>' -> a float string (risk weight); may be missing, treat missing as 0.
- 'blocklist:<card>' -> present (the string '1') if the card is blocked, otherwise missing.
- 'approval_budget' -> an integer string: approvals remaining for the WHOLE fleet.

Required behavior, exactly:
- score = merchant_risk + amount / 1000 + (4.0 if the card is blocked else 0.0)
- flagged = (card is blocked) OR (score >= 5.0)
- Always increment global 'total_processed' by 1. When flagged, also increment global 'total_flagged' by 1.
- decision: if flagged -> 'review'. Otherwise, if global 'approval_budget' > 0, decrement it by 1 and set decision 'approve'; else decision 'hold:budget_exhausted'.
- Always increment local 'handled' by 1.
- return {'id': <tx id>, 'score': round(score, 2), 'flagged': <bool>, 'decision': <str>}

Hard constraints: NO imports (the sandbox rejects code containing 'import', '__', 'open(', 'eval(', 'exec(', and similar). Use only Python builtins and `ctx`. Keep it short."""

# Used when no API key is set, so the flow is demonstrable offline.
REFERENCE_HANDLER = """
tx = ctx.args['payload']
risk = float(ctx.shared.get('merchant_risk:' + tx['merchant']) or '0')
blocked = ctx.shared.get('blocklist:' + tx['card']) is not None
score = risk + (tx['amount'] / 1000.0) + (4.0 if blocked else 0.0)
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
        decision = 'hold:budget_exhausted'
ctx.local.incr('handled')
return {'id': tx['id'], 'score': round(score, 2), 'flagged': flagged, 'decision': decision}
"""

TRANSACTIONS = [
    {"id": "t01", "merchant": "acme", "card": "C-1001", "amount": 120},
    {"id": "t02", "merchant": "globex", "card": "C-1002", "amount": 900},
    {"id": "t03", "merchant": "initech", "card": "C-1003", "amount": 50},
    {"id": "t04", "merchant": "acme", "card": "C-9999", "amount": 30},
    {"id": "t05", "merchant": "initech", "card": "C-1005", "amount": 200},
    {"id": "t06", "merchant": "acme", "card": "C-1006", "amount": 75},
    {"id": "t07", "merchant": "umbrella", "card": "C-1007", "amount": 400},
    {"id": "t08", "merchant": "acme", "card": "C-1008", "amount": 60},
    {"id": "t09", "merchant": "initech", "card": "C-1009", "amount": 90},
    {"id": "t10", "merchant": "acme", "card": "C-1010", "amount": 110},
    {"id": "t11", "merchant": "initech", "card": "C-1011", "amount": 40},
    {"id": "t12", "merchant": "globex", "card": "C-9999", "amount": 1500},
]


class ReferenceProvider:
    """Stand-in used when no API key is set; returns the reference handler."""

    name = "reference"

    def generate_code(self, system: str, prompt: str) -> CodeGenResult:
        return CodeGenResult(code=REFERENCE_HANDLER, explanation="reference handler")


def seed_global(g) -> None:
    g.set("merchant_risk:acme", "1.0")
    g.set("merchant_risk:globex", "4.5")
    g.set("merchant_risk:initech", "1.5")
    g.set("merchant_risk:umbrella", "5.5")
    g.set("blocklist:C-9999", "1")
    g.set("approval_budget", "5")


def smoke_test(handler: str):
    """Run the handler once in a throwaway sandbox; return (ok, error)."""
    memory = TieredMemory()
    seed_global(memory.shared)
    box = Sandbox.create(
        policy={"cold_start_ms": 0}, store=MemorySnapshotStore(), memory=memory.binding_for("smoke")
    )
    sample = {"id": "smoke", "merchant": "acme", "card": "C-1001", "amount": 120}
    result = box.run(handler, name="smoke", args={"from": 0, "payload": sample, "port": 0})
    if not result.ok:
        return False, result.error
    value = result.value
    if not isinstance(value, dict) or "decision" not in value or "id" not in value:
        return False, f"handler returned {value!r}; expected a dict with id/score/flagged/decision"
    return True, None


def build_handler(provider, max_attempts: int = 3) -> str:
    """Ask the model for a handler, validate it, repair on failure."""
    prompt = "Write the handler."
    for attempt in range(1, max_attempts + 1):
        print(f"[builder] asking {provider.name} for a handler (attempt {attempt}/{max_attempts})")
        gen = provider.generate_code(HANDLER_SYSTEM, prompt)
        ok, err = smoke_test(gen.code)
        if ok:
            print(f"[builder] handler validated ({gen.input_tokens} in / {gen.output_tokens} out tokens)\n")
            return gen.code
        print(f"[builder] validation failed: {err}")
        prompt = (
            f"Write the handler.\n\nThe previous version failed when run.\n\n"
            f"Previous code:\n{gen.code}\n\nError:\n{err}"
        )
    raise RuntimeError("Could not obtain a working handler after retries.")


def main() -> None:
    if os.environ.get("ANTHROPIC_API_KEY"):
        provider = ClaudeProvider()
        print("Using Claude (claude-opus-4-8) to write the fleet's handler.\n")
    else:
        provider = ReferenceProvider()
        print("No ANTHROPIC_API_KEY set — using a built-in reference handler to show the flow.")
        print("Set the key to have Claude actually write it.\n")

    handler = build_handler(provider)
    print("=== Handler the model produced (deployed to every worker) ===")
    print(handler.strip() + "\n")

    # One shared memory for the whole fleet.
    memory = build_memory()
    memory.shared.flush()   # idempotent across runs when backed by a persistent Redis
    seed_global(memory.shared)

    fabric = SandboxFabric(policy_program=deny_east_west_policy())
    lb = LoadBalancer(fabric)
    workers = {}
    for i in range(3):
        box = Sandbox.create(
            policy={"cold_start_ms": 0}, store=MemorySnapshotStore(), memory=memory.binding_for(f"worker-{i}")
        )
        addr = fabric.join(box, handler=handler, name=f"worker-{i}")
        lb.add_backend(addr)
        workers[addr] = f"worker-{i}"

    print("=== Scoring transactions across the 3-worker fleet ===\n")
    for tx in TRANSACTIONS:
        res = lb.handle(path="/score", payload=tx)
        body = res.body
        print(
            f"{tx['id']}  -> {fabric.endpoint_name(res.to_addr):9}  "
            f"score={body['score']:5}  flagged={str(body['flagged']):5}  {body['decision']}"
        )

    g = memory.shared
    print("\n=== Shared GLOBAL state (written by all 3 workers) ===")
    print("total processed :", g.get("total_processed"))
    print("total flagged   :", g.get("total_flagged"))
    print("approvals left  :", g.get("approval_budget"), "(started at 5 — shared across the fleet)")

    print("\n=== Per-worker LOCAL state ===")
    for name in workers.values():
        print(f"{name}: handled {memory.local_for(name).get('handled')} requests")

    print(
        "\nThe model wrote the handler once; an isolated, load-balanced fleet ran it and "
        "enforced a single shared budget through global memory — no external database in sight."
    )


if __name__ == "__main__":
    main()
