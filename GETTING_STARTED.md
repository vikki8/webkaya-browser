# Getting started with WebKaya

This guide walks you from zero to running everything in the repo. Every snippet
here has been run as-is.

## What this actually is (the 30-second version)

WebKaya runs **AI-generated code in a browser tab** instead of a cloud server.
The one idea: an LLM writes code → a `Sandbox` runs it under rules you control →
the user's data never leaves their device. Everything else is a feature on top
of that:

| Piece | What it does | Where |
|---|---|---|
| **Sandbox** | Runs guest code against isolated state; snapshot / fork / replay | `src/sandbox/` |
| **eBPF probes** | Programmable rules: count/meter runs, or veto them before they run | `src/ebpf/`, `src/sandbox/probes.ts` |
| **Fabric + LoadBalancer** | Many sandboxes that talk to each other behind a balancer | `src/net/` |
| **Tiered memory** | Redis-style shared + per-sandbox key/value store | `src/memory/` |
| **Python runtime** | Run real Python (pandas) in the browser via Pyodide | `src/python/` |
| **LLM layer** | Claude writes the guest code; agent loops that self-repair | `src/llm/` |
| **Python client** | `pip install` version of the same model | `python/` |

---

## Step 0 — Setup (once)

```bash
npm install        # restore dependencies
npm run build      # compile src/ -> dist/ (needed before running anything)
npm test           # 114 tests — proof the whole thing works
```

If `npm test` passes, everything in this guide will work.

---

## Step 1 — A sandbox in 6 lines (Node)

Create `try.mjs` in the repo root:

```js
import { Sandbox } from './dist/index.js';

const box = await Sandbox.create({ initialState: { count: 0 } });
const result = await box.run("ctx.state.count += 41; ctx.log('hi'); return ctx.state.count;");

console.log(result.ok, result.value);   // true 42
console.log(box.getState());            // { count: 42 }
```

```bash
node try.mjs
```

The string you pass to `run()` is the "guest" — code an AI would normally write.
It only sees `ctx`: `ctx.state` (persistent data), `ctx.args` (input), `ctx.log()`.
A failed run never corrupts state, and dangerous tokens (`fetch`, `eval`, …) are
rejected.

---

## Step 2 — Add a rule with an eBPF probe

Probes are little programs that run *before* (or after) each guest run. A
`run:start` probe that returns nonzero **blocks** the run.

```js
import { Sandbox, EbpfMap, assemble, op, HELPERS } from './dist/index.js';

const box = await Sandbox.create({ initialState: {} });

// Count every run in an eBPF map.
const counter = new EbpfMap();
box.attachProbe('run:start', {
  name: 'counter',
  maps: [counter],
  program: assemble([
    op.movImm(1, 0), op.movImm(2, 0), op.movImm(3, 1),
    op.call(HELPERS.MAP_ADD),   // map[0] += 1
    op.movImm(0, 0), op.exit(), // return 0 = allow
  ]),
});

await box.run('return 1;');
await box.run('return 2;');
console.log('runs so far:', counter.get(0n));   // 2n
```

This is the governance layer: metering, rate limits, and admission control as
verified bytecode instead of host-side `if` statements.

---

## Step 3 — Snapshot, fork, replay

```js
const snap = await box.snapshot('checkpoint');   // save state
const fork = await box.fork();                    // branch it
await fork.run('ctx.state.x = 999;');             // fork diverges; parent untouched
const { results } = await box.replay();           // re-run the whole history
```

---

## Step 4 — The headline demo: analyze a CSV with Claude, in your browser

This is the product in one page: pick a CSV, ask a question in English, Claude
writes the pandas, it runs **in your browser** over your file, and the "bytes of
your data sent to a server" counter stays at 0.

```bash
npm run build           # if you haven't already
npx serve .             # serves the repo over http (any static server works)
# open the printed URL + /examples/local-data-analyst/
```

In the page:
1. Click **Use sample data** (or pick your own CSV).
2. Ask something: *"average revenue by region"*.

It works with **no API key** (a built-in planner handles a few question shapes).
To use Claude for real: paste an Anthropic API key into the key field, and it
switches to the full **generate → run → repair** loop (`claude-opus-4-8`). Only
the question text reaches Claude; your data never does. Use a throwaway,
revocable key — a browser key is visible to the page.

---

## Step 5 — The agent loop in code

`CodeAgent` is the loop the product is built around: the model writes code, the
sandbox runs it, and failures are fed back for another attempt — all governed by
the sandbox (probes, timeouts, etc.).

```js
import { Sandbox } from './dist/index.js';
import { ClaudeProvider, CodeAgent } from './dist/llm/index.js';

const provider = new ClaudeProvider();   // reads ANTHROPIC_API_KEY from env
const box = await Sandbox.create();
const agent = new CodeAgent(provider, box, { maxAttempts: 3 });

const outcome = await agent.run('compute the 10th Fibonacci number');
console.log(outcome.ok, outcome.result.value, `(${outcome.attempts.length} attempts)`);
```

```bash
ANTHROPIC_API_KEY=sk-ant-... node agent.mjs
```

For Python/pandas over a DataFrame, the equivalent is `DataAgent` from
`./dist/python/index.js` (browser/Pyodide).

---

## Step 6 — Do it all in Python instead

A `pip`-installable mirror of the same model, including the Claude agent loop:

```bash
cd python
pip install ".[claude]"                                # core + Anthropic SDK
PYTHONPATH=src python -m unittest discover -s tests     # 49 tests
```

A plain sandbox run:

```python
from webkaya import Sandbox
box = Sandbox.create(initial_state={"n": 0})
print(box.run("ctx.state['n'] += 41\nreturn ctx.state['n']").value)   # 41
```

The full agent loop — Claude writes the Python, the sandbox runs it, repair on
failure (`python/examples/agent_demo.py`):

```python
from webkaya import Sandbox, ClaudeProvider, CodeAgent

sandbox = Sandbox.create(initial_state={"rows": [{"region": "EMEA", "revenue": 95}]})
agent = CodeAgent(ClaudeProvider(), sandbox)            # ANTHROPIC_API_KEY from env
outcome = agent.run("Sum revenue per region from ctx.state['rows']; return {region: total}.")
print(outcome.ok, outcome.result.value)                 # True {'EMEA': 95}
```

```bash
export ANTHROPIC_API_KEY=sk-ant-... && python examples/agent_demo.py
```

The Python sandbox runs Python with no imports (stdlib + `ctx` only), so Claude
writes plain-Python aggregation over `ctx.state`, not pandas. For pandas over a
DataFrame, use the browser/Pyodide path (Step 4).

---

## Mental model recap

- **Inline mode** (default): guest runs in-process. Simplest; use it to learn.
- **Worker mode** (`runtime: 'worker'`): guest runs on its own thread, can be
  killed on timeout. Use for real isolation.
- The **LLM layer** is optional and provider-agnostic — `ClaudeProvider` is one
  implementation; the sandbox governs whatever code comes out.
- **Nothing requires a cloud account** except the optional Claude calls.

Start at Step 1, get one `run()` working, then add pieces.
