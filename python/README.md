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
pip install webkaya            # core: no third-party deps, Python 3.9+
pip install webkaya[claude]    # + the Anthropic SDK, to let Claude write the code
```

## Let Claude write the code (the agent loop)

`CodeAgent` runs the generate → run → repair loop in Python: Claude writes guest
Python, the sandbox runs it under full policy, and any failure (a raised
exception, a token-scan rejection) is fed back for another attempt. Only the
task text and error reach the model — the data stays in the sandbox.

```python
from webkaya import Sandbox, ClaudeProvider, CodeAgent

sandbox = Sandbox.create(initial_state={"rows": [{"region": "EMEA", "revenue": 95}]})
agent = CodeAgent(ClaudeProvider(), sandbox, max_attempts=3)   # ANTHROPIC_API_KEY from env

outcome = agent.run("Sum revenue per region from ctx.state['rows'] and return {region: total}.")
print(outcome.ok, outcome.result.value)          # True {'EMEA': 95}
print(outcome.code)                               # the Python Claude wrote
print(outcome.input_tokens, outcome.output_tokens)
```

A runnable version is in [`examples/agent_demo.py`](examples/agent_demo.py):

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python examples/agent_demo.py
```

Note: the Python sandbox runs Python guest code with **no imports allowed**
(stdlib + `ctx` only), so Claude writes plain-Python aggregation over
`ctx.state`, not pandas.

## Governing runs with eBPF probes

Attach verified eBPF programs to sandbox tracepoints (`run:start`, `run:end`,
`snapshot`, `log`). A `run:start` probe that returns nonzero **denies** the run —
admission control as bytecode, independent of the guest:

```python
from webkaya import Sandbox, EbpfMap
from webkaya.asm import assemble, call, exit_, mov_imm, MAP_ADD

box = Sandbox.create()
counter = EbpfMap()
# map[0] += 1 on every run; return 0 = allow.
box.attach_probe("run:start", assemble([
    mov_imm(1, 0), mov_imm(2, 0), mov_imm(3, 1), call(MAP_ADD), mov_imm(0, 0), exit_(),
]), maps=[counter])

box.run("return 1")
print(counter.get(0))   # 1
```

The same standard bytecode runs in the browser SDK, this client, and (later)
kernel eBPF — write the policy once, enforce it wherever the workload lands.

## Distributed global memory (Redis)

The shared/global tier is pluggable. In-process by default; swap in
`RedisMemoryTier` and the whole fleet coordinates through one real Redis —
guest code unchanged. `incr` maps to Redis `INCRBY`, which is atomic across
processes, so a shared counter or budget stays correct under real concurrency.

```python
from webkaya import TieredMemory, RedisMemoryTier

memory = TieredMemory(shared=RedisMemoryTier(url="redis://localhost:6379/0"))
# hand memory.binding_for(worker_id) to each sandbox exactly as before
```

```bash
pip install webkaya[redis]
REDIS_URL=redis://localhost:6379/0 python examples/cluster_demo.py   # now distributed
```

`MemoryTier` (in-process) and `RedisMemoryTier` both satisfy the `KVStore`
protocol, so they're interchangeable wherever a tier is expected.

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

## Examples

Runnable scripts in [`examples/`](examples) (each falls back to a built-in
reference handler when no API key is set, so they run offline):

| Script | Shows |
|---|---|
| `agent_demo.py` | Claude writes Python, the sandbox runs it, repair on failure |
| `cluster_demo.py` | A load-balanced fleet coordinating through global memory (set `REDIS_URL` to distribute) |
| `cluster_agent_demo.py` | Claude writes the fleet's handler; it's validated, then deployed across the workers |
| `multi_endpoint_demo.py` | Different Claude-written handlers per route (`/score`, `/refund`, `/audit`), each its own worker pool, one shared memory |

## Run the tests

```bash
cd python
PYTHONPATH=src python -m unittest discover -s tests
```

## License

MIT
