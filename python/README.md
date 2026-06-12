# webkaya (Python client)

Python client for **WebKaya** — client-side AI agent sandboxes. It mirrors the
TypeScript SDK's programming model for orchestration and local prototyping from
Python: governed sandboxes with snapshot / fork / replay, a userspace eBPF VM,
an eBPF-governed fabric and load balancer, and Redis-shaped tiered memory.

The eBPF bytecode is standard, so a probe or network-policy program is portable
across the browser SDK, this client, and kernel eBPF on the server tier — write
it once, run it anywhere the workload lands.

## Install

```bash
pip install webkaya
```

No third-party dependencies (standard library only); Python 3.9+.

## Quickstart

```python
from webkaya import Sandbox, SandboxFabric, LoadBalancer, TieredMemory, deny_east_west_policy

# A governed sandbox: guest Python runs against isolated state behind a
# token-scanned, restricted-builtins boundary. Failed runs never commit state.
box = Sandbox.create(initial_state={"rows": []})
result = box.run(
    "ctx.state['rows'].append(ctx.args)\nctx.log('added')\nreturn len(ctx.state['rows'])",
    args={"id": 1},
)
print(result.ok, result.value, result.logs)

snap = box.snapshot("after-import")   # persist state
fork = box.fork()                      # branch and diverge independently
results, final_state = box.replay()    # re-run the recorded sequence

# A cluster: eBPF load balancer + shared/local memory, default-deny east-west.
fabric = SandboxFabric(policy_program=deny_east_west_policy())
memory = TieredMemory()
lb = LoadBalancer(fabric)  # round-robin eBPF program by default
handler = "n = ctx.shared.incr('requests')\nreturn {'served_by': ctx.args['from'], 'total': n}"
for i in range(3):
    b = Sandbox.create(memory=memory.binding_for(f"backend-{i}"))
    lb.add_backend(fabric.join(b, name=f"backend-{i}", handler=handler))

lb.serve_static("/health", {"status": "green"})        # web-server mode
print(lb.handle(path="/api", payload={"q": "hi"}).body) # round-robined to a backend
```

## API parity notes

This client tracks the TypeScript SDK with a few language-driven differences:

- `Sandbox.create(...)` / `Sandbox.restore(...)` are synchronous factories (the TS API is async).
- Guest code is a Python snippet wrapped into a function body, so `return` works as in the TS `Function` boundary. Guests get `ctx.state`, `ctx.args`, `ctx.log`, and — when memory is bound — `ctx.local` and `ctx.shared` (the global tier; named `shared` because `global` is a Python keyword).
- KV `delete(...)` replaces the TS `del` (a Python keyword).
- `policy.timeout_ms` is **advisory** in this local engine: CPython cannot preempt a running call, so the timeout is enforced on the browser and server tiers, not here.

## Run the tests

```bash
cd python
PYTHONPATH=src python -m unittest discover -s tests
```

## License

MIT
