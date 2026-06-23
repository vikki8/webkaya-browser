"""Cluster demo: a load-balanced fleet of sandboxes coordinating through GLOBAL memory.

This is the capability that sets WebKaya apart from per-VM / ephemeral sandbox
services: many sandboxes behind a load balancer, sharing one tiered memory, so
they can hold *fleet-wide* state (reference data, counters, a shared budget)
without each call shipping data to an external datastore. The workers are
isolated from each other (east-west traffic denied) yet still coordinate, purely
through the shared global tier.

Scenario: a transaction-scoring service. Each request is round-robined to one of
three worker sandboxes. Every worker:
  - reads shared reference data from GLOBAL memory (merchant risk, a card blocklist),
  - updates fleet-wide counters in GLOBAL memory (processed, flagged),
  - draws down a single fleet-wide approval budget in GLOBAL memory,
  - keeps its own per-worker tally in LOCAL memory.

Run:  python examples/cluster_demo.py
(No API key needed — fully deterministic. The scoring handler below is plain
guest code; it could equally be written by Claude via CodeAgent, which we've
shown works.)
"""

from webkaya import (
    LoadBalancer,
    MemorySnapshotStore,
    Sandbox,
    SandboxFabric,
    TieredMemory,
    deny_east_west_policy,
)

# The handler each worker runs per request. It sees `ctx.args['payload']` (the
# transaction), `ctx.shared` (GLOBAL memory, one store for the whole fleet), and
# `ctx.local` (private to this worker). No imports — stdlib + ctx only.
SCORING_HANDLER = """
tx = ctx.args['payload']
merchant = tx['merchant']
card = tx['card']
amount = tx['amount']

# Shared reference data — read from GLOBAL memory, identical for every worker.
risk = float(ctx.shared.get('merchant_risk:' + merchant) or '0')
blocked = ctx.shared.get('blocklist:' + card) is not None

score = risk + (amount / 1000.0) + (4.0 if blocked else 0.0)
flagged = blocked or score >= 5.0

# Fleet-wide counters live in GLOBAL memory, so every worker increments the same total.
ctx.shared.incr('total_processed')
if flagged:
    ctx.shared.incr('total_flagged')

# A SINGLE approval budget shared across the whole fleet — the hard part for
# isolated sandboxes. Workers collectively cannot approve more than the budget.
decision = 'review'
if not flagged:
    remaining = int(ctx.shared.get('approval_budget') or '0')
    if remaining > 0:
        ctx.shared.incr('approval_budget', -1)
        decision = 'approve'
    else:
        decision = 'hold:budget_exhausted'

ctx.local.incr('handled')   # private per-worker tally
return {'id': tx['id'], 'score': round(score, 2), 'flagged': flagged, 'decision': decision}
"""

TRANSACTIONS = [
    {"id": "t01", "merchant": "acme", "card": "C-1001", "amount": 120},
    {"id": "t02", "merchant": "globex", "card": "C-1002", "amount": 900},   # high-risk merchant
    {"id": "t03", "merchant": "initech", "card": "C-1003", "amount": 50},
    {"id": "t04", "merchant": "acme", "card": "C-9999", "amount": 30},      # blocklisted card
    {"id": "t05", "merchant": "initech", "card": "C-1005", "amount": 200},
    {"id": "t06", "merchant": "acme", "card": "C-1006", "amount": 75},
    {"id": "t07", "merchant": "umbrella", "card": "C-1007", "amount": 400}, # very-high-risk merchant
    {"id": "t08", "merchant": "acme", "card": "C-1008", "amount": 60},
    {"id": "t09", "merchant": "initech", "card": "C-1009", "amount": 90},
    {"id": "t10", "merchant": "acme", "card": "C-1010", "amount": 110},
    {"id": "t11", "merchant": "initech", "card": "C-1011", "amount": 40},   # past the budget of 5
    {"id": "t12", "merchant": "globex", "card": "C-9999", "amount": 1500},  # high-risk AND blocked
]


def main() -> None:
    # One shared memory for the whole fleet. Seed the global reference data.
    memory = TieredMemory()
    g = memory.shared
    g.set("merchant_risk:acme", "1.0")
    g.set("merchant_risk:globex", "4.5")
    g.set("merchant_risk:initech", "1.5")
    g.set("merchant_risk:umbrella", "5.5")
    g.set("blocklist:C-9999", "1")
    g.set("approval_budget", "5")   # the whole fleet may approve at most 5

    # Deny east-west traffic: workers cannot reach each other. Only the load
    # balancer (ingress) can reach them. They coordinate solely via global memory.
    fabric = SandboxFabric(policy_program=deny_east_west_policy())
    lb = LoadBalancer(fabric)   # round-robin across backends by default

    worker_addrs = {}
    for i in range(3):
        box = Sandbox.create(
            policy={"cold_start_ms": 0},
            store=MemorySnapshotStore(),
            memory=memory.binding_for(f"worker-{i}"),   # same global tier, own local tier
        )
        addr = fabric.join(box, handler=SCORING_HANDLER, name=f"worker-{i}")
        lb.add_backend(addr)
        worker_addrs[addr] = f"worker-{i}"

    print("=== Scoring a stream of transactions across a 3-worker fleet ===\n")
    for tx in TRANSACTIONS:
        res = lb.handle(path="/score", payload=tx)
        served_by = fabric.endpoint_name(res.to_addr)
        body = res.body
        print(
            f"{tx['id']}  -> {served_by:9}  "
            f"score={body['score']:5}  flagged={str(body['flagged']):5}  {body['decision']}"
        )

    # Fleet-wide state — written by ALL three workers into ONE global store.
    print("\n=== Shared GLOBAL state (written by all 3 workers) ===")
    print("total processed :", g.get("total_processed"))
    print("total flagged   :", g.get("total_flagged"))
    print("approvals left  :", g.get("approval_budget"), "(started at 5 — shared across the fleet)")

    # Per-worker LOCAL state — private to each sandbox, showing load distribution.
    print("\n=== Per-worker LOCAL state (private to each sandbox) ===")
    for addr, name in worker_addrs.items():
        print(f"{name}: handled {memory.local_for(name).get('handled')} requests")

    # Isolation proof: a worker cannot talk to another worker directly.
    addrs = list(worker_addrs)
    denied = fabric.request(addrs[0], addrs[1], payload={"probe": True})
    print("\n=== East-west isolation ===")
    print(f"worker-0 -> worker-1 direct call: status {denied.status}, denied={denied.denied}")

    print(
        "\nWhy this is the differentiator: three isolated sandboxes just enforced a "
        "single shared approval budget and shared counters with no external database — "
        "the global memory tier IS the coordination layer. Per-VM sandbox services give "
        "you isolation but make you bolt on Redis/a DB to share any state across the fleet."
    )


if __name__ == "__main__":
    main()
