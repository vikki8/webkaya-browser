# WebKaya Sandbox

**Client-side AI agent sandboxes.** Run agent-generated code in the end user's browser — next to the user's data, at zero marginal cost — with policy governance, snapshots, fork, and deterministic-leaning replay built in.

Cloud sandbox vendors run agent code in server-side microVMs, priced per second, far from the user's data. WebKaya inverts that: the browser tab is already an isolated, capable runtime. This SDK turns it into a governed execution environment for AI products — code interpreters, data-analysis agents, "chat with your files" features — where the data never leaves the device and every run is recorded, snapshottable, and replayable.

This is **not** a security product. The differentiators are **cost** (no per-second VM billing), **locality** (compute happens where the data lives), and **reproducibility** (fork/replay turns agent debugging into something like git).

Stack: **TypeScript**, **Vitest**. No runtime dependencies.

> A Python client mirroring this SDK lives in [`python/`](python/README.md) — `pip install webkaya` for orchestration and local prototyping. The same eBPF bytecode runs in both.

---

## Install & develop

```bash
npm install        # restore dev dependencies
npm test           # Vitest unit tests
npm run typecheck  # tsc --noEmit
npm run build      # emit dist/ (ESM + .d.ts)
```

---

## Quickstart

```ts
import { Sandbox } from '@webkaya/sandbox';

const box = await Sandbox.create({
  policy: { timeoutMs: 5_000, memoryBudgetMB: 512, snapshotEveryNRuns: 5 },
  initialState: { rows: [] },
});

// Run agent-generated code. Guests see only `ctx` — no network, no DOM.
const result = await box.run(
  'ctx.state.rows.push(ctx.args); ctx.log("added row"); return ctx.state.rows.length;',
  { args: { id: 1, value: 'hello' } }
);
console.log(result.ok, result.value, result.logs);

// Snapshot, fork, replay.
const snap = await box.snapshot('after-import');
const fork = await box.fork();                    // branch state, try something else
const { results, finalState } = await box.replay(); // re-execute the recorded run sequence
const restored = await Sandbox.restore(snap.id);  // resume from a persisted snapshot
```

---

## Concepts

| Concept | What it does |
|---------|--------------|
| **Policy** | Clamped governance config per sandbox: timeout, retries, memory budget, guest code size, auto-snapshot cadence. Supports "policy as code" via template/eject modes (`resolvePolicy`). |
| **Guest context** | Guest code receives a single `ctx` object (`state`, `args`, `log`). Code is token-scanned (no `fetch`, `eval`, DOM, ambient globals) and state commits only on success. |
| **Execution mode** | `runtime: 'inline'` (default) runs guests in the host realm and exposes `ctx.local`/`ctx.global`. `runtime: 'worker'` runs them over the host↔sandbox message protocol — off the main thread with a real `Worker` for true isolation and hard timeout enforcement, or an in-process loopback as a fallback. |
| **Snapshots** | Sandbox state persists to OPFS (`OpfsSnapshotStore`) with an in-memory fallback. Snapshots carry lineage (`parentSnapshotId`). |
| **Fork** | Branch a sandbox at its current state; parent and fork diverge independently. |
| **Replay** | Re-execute the recorded run sequence from initial state in a fresh sandbox — debug a run, verify reproducibility. |
| **eBPF probes** | Sandboxes expose tracepoints (`run:start`, `run:end`, `snapshot`, `log`) that verified eBPF programs attach to — admission control, metering, and custom observability that travel with the workload. |
| **Fabric & load balancer** | Sandboxes `join` a `SandboxFabric` to get an address and talk to each other; an eBPF network policy gates east-west traffic, and an eBPF `LoadBalancer` distributes ingress requests (and serves static routes like a web server). |
| **Tiered memory** | Redis-shaped `TieredMemory` gives each sandbox a private `local` tier and a shared `global` tier, exposed to guest code as `ctx.local` / `ctx.global`. |
| **Capabilities & backends** | `detectCapabilities()` tiers the device (WebGPU → WebGL2 → WASM SIMD → CPU); `selectComputeBackend()` picks the compute path. |
| **Hardware monitor** | Samples activity, heap, and thermal pressure (Compute Pressure API) so hosts can throttle or show a resource HUD. |

---

## eBPF probes: the runtime's instrumentation plane

WebKaya treats eBPF as the portable bytecode for governing and observing sandboxes. A userspace eBPF VM (`EbpfVm`) executes probe programs in the browser today; because the bytecode is standard eBPF, the same programs can attach to kernel tracepoints (or ubpf) on the planned server tier — write a probe once, run it wherever the workload lands.

Probes attach to sandbox tracepoints and communicate with the host through BPF-style maps. At `run:start`, a nonzero return value **denies the run** — policy as verified bytecode:

```ts
import { Sandbox, EbpfMap, op, assemble, HELPERS } from '@webkaya/sandbox';

const box = await Sandbox.create();

// Count every run: map_add(fd 0, key 0, 1)
const counters = new EbpfMap();
box.attachProbe('run:start', {
  name: 'run-counter',
  maps: [counters],
  program: assemble([
    op.movImm(1, 0),            // r1 = map fd
    op.movImm(2, 0),            // r2 = key
    op.movImm(3, 1),            // r3 = delta
    op.call(HELPERS.MAP_ADD),
    op.movImm(0, 0),            // return 0 = allow
    op.exit(),
  ]),
});

// Admission control: deny guest programs longer than 4 KB
box.attachProbe('run:start', {
  name: 'max-code-length',
  program: assemble([
    op.ldxdw(2, 1, 8),          // r2 = ctx.codeLength
    op.movImm(0, 0),
    op.jleImm(2, 4096, 1),
    op.movImm(0, 1),            // nonzero = deny
    op.exit(),
  ]),
});
```

The VM enforces bounded execution: a static verifier pass (register bounds, jump targets, known helpers, must end in `exit`) plus dynamic memory bounds and an instruction budget. Each tracepoint's context struct is documented in `TRACEPOINT_LAYOUTS`. Helpers: `MAP_GET`, `MAP_SET`, `MAP_ADD`, `TRACE`, `KTIME_GET_NS` (maps addressed by fd index — see `src/ebpf/vm.ts` for the ABI note).

---

## Sandbox clusters: fabric, load balancer, tiered memory

Sandboxes are not just isolated runtimes — they form a cluster. They join a fabric, address each other, and sit behind an eBPF load balancer that doubles as a web server. Network policy and load balancing are the same verified eBPF bytecode that governs runs, so the cluster's data plane is programmable end to end:

```
        ingress request
              |
        LoadBalancer  ── serves static routes itself (web-server mode)
        (eBPF pick)
          /      \
   Sandbox A    Sandbox B      east-west A<->B gated by eBPF network policy
       \           /
     TieredMemory (global tier, shared)   +   per-sandbox local tier
```

```ts
import {
  Sandbox, SandboxFabric, LoadBalancer, TieredMemory, denyEastWestPolicy,
} from '@webkaya/sandbox';

const fabric = new SandboxFabric({ policyProgram: denyEastWestPolicy() }); // default-deny between sandboxes
const memory = new TieredMemory();
const lb = new LoadBalancer(fabric); // round-robin by default

for (let i = 0; i < 3; i++) {
  const box = await Sandbox.create({ memory: memory.bindingFor(`backend-${i}`) });
  const addr = fabric.join(box, {
    name: `backend-${i}`,
    // A request handler is a governed run: its probes, timeout, and memory budget all apply.
    handler: 'const total = ctx.global.incr("requests"); return { servedBy: ctx.args.from, total };',
  });
  lb.addBackend(addr);
}

lb.serveStatic('/health', { status: 'green' });        // edge-terminated, no backend hit
const res = await lb.handle({ path: '/api', payload: { q: 'hello' } }); // -> round-robined to a backend
```

- **Fabric** (`SandboxFabric`) — `join` assigns an address (ingress is the reserved address `0`); `request(from, to, …)` delivers traffic and runs the destination's handler. Delivered/dropped counts live in eBPF maps (`deliveredByDst`, `droppedBySrc`).
- **Network policy** — one eBPF program returns a verdict per hop (`0` allow, nonzero drop) over `[srcAddr, dstAddr, port, length, protocol]`. `denyEastWestPolicy()` blocks sandbox↔sandbox traffic while still permitting ingress — the browser-tier analogue of Cilium default-deny. A crashing policy fails closed.
- **Load balancer** (`LoadBalancer`) — an eBPF program picks a backend from `[srcAddr, srcPort, dstPort, requestHash, backendCount]`. `roundRobinBalancer()` (default, map-backed counter) and `hashBalancer()` (sticky) ship built in; swap in your own. Static routes are served at the edge, so the LB also acts as a web server / TLS-terminating ingress would.
- **Tiered memory** (`TieredMemory`) — Redis-shaped (`get/set/del/incr/expire/ttl/keys`) with a shared `global` tier and per-sandbox `local` tiers, surfaced to guests as `ctx.global` / `ctx.local`. In the browser these are synchronous in-memory tiers; the server deployment backs `global` with a real Redis behind the same method names.

These are **in-process models** of TCP/SDN/Redis, not implementations of them — the value is that the same policy/LB bytecode and the same memory API move onto kernel eBPF and real Redis on the server tier without changing application code.

---

## Project layout

```
src/
├── index.ts            # Public API
├── sandbox/
│   ├── sandbox.ts      # Sandbox: create/run/snapshot/fork/restore/replay
│   ├── executor.ts     # Inline vs worker execution seam
│   ├── probes.ts       # Tracepoints + probe registry (eBPF attach points)
│   └── snapshot-store.ts  # OPFS + in-memory snapshot persistence
├── ebpf/
│   ├── vm.ts           # Userspace eBPF VM: verifier + interpreter + helpers
│   ├── maps.ts         # BPF-style u64->u64 maps (probe <-> host channel)
│   └── asm.ts          # Instruction builder for authoring probe programs
├── net/
│   ├── fabric.ts       # SandboxFabric: addressing + east-west delivery + policy
│   ├── load-balancer.ts   # eBPF load balancer + static web-server routes
│   └── hooks.ts        # Network-policy / load-balancer context layouts + default programs
├── memory/
│   └── tiered-memory.ts   # Redis-shaped global + local KV tiers
├── python/
│   ├── pyodide-runner.ts  # Python-over-local-data via Pyodide (CPython/WASM)
│   └── planner.ts      # Deterministic NL->pandas planner (LLM stand-in)
├── runtime/
│   ├── policy.ts       # Policy normalize/validate, guest code scanning, policy-as-code
│   ├── guest-exec.ts   # Shared guest compilation (inline + worker)
│   ├── guest-invoker.ts   # Timeout, retry, memory-budget enforcement
│   ├── worker/         # Worker transport: core handler, transports, worker entry
│   ├── capability-detect.ts  # Device tiering (WebGPU/WebGL2/WASM SIMD/CPU)
│   ├── backends.ts     # Compute backend selection
│   └── hardware-monitor.ts   # Utilization, thermal, energy sampling
└── types/
    ├── policy.ts       # SandboxPolicy, PolicyEditorState
    └── protocol.ts     # Host <-> sandbox message protocol (worker transport seam)
tests/                  # Vitest
```

---

## Worker mode

```ts
// In an app bundler (Vite/webpack/esbuild), point the sandbox at the worker entry.
const box = await Sandbox.create({
  runtime: 'worker',
  workerFactory: () => new Worker(new URL('@webkaya/sandbox/worker', import.meta.url), { type: 'module' }),
});
await box.run('ctx.state.n = (ctx.state.n || 0) + 1; return ctx.state.n;');
```

Guest code runs on its own thread; state crosses the boundary by structured clone (so it must be serializable). Because a real `Worker` can be terminated, worker mode enforces a hard timeout — a wedged guest is killed and the worker respawned — which the inline `Function` boundary cannot do. Without a `workerFactory`, worker mode falls back to an in-process loopback that runs the identical worker core (useful in tests and SSR) but provides no thread isolation. Worker mode does not expose the live `ctx.local`/`ctx.global` memory tiers, which require the inline realm.

## Python over local data

The wedge use case — *analyze my data without it leaving the device* — runs Python in the browser via Pyodide:

```ts
import { loadPyodideRuntime, PythonRunner, planQuestion } from '@webkaya/sandbox/python';

const runner = new PythonRunner(await loadPyodideRuntime());
await runner.loadDataframe('df', csvTextFromUserFile);          // CSV -> pandas, in-browser
const plan = planQuestion('average revenue by region', columns); // LLM stand-in (deterministic)
const result = await runner.run(plan.code);                      // executes locally; data never sent
```

`planQuestion` is a deterministic NL→pandas planner so the flow works with no API key; in production you swap it for an LLM emitting the same kind of snippet. A complete runnable demo — pick a CSV, ask in plain English, watch the "bytes of your data sent to a server" counter stay at 0 — lives in [`examples/local-data-analyst/`](examples/local-data-analyst/README.md).

## Roadmap

1. ~~**Web Worker transport**~~ — done: `runtime: 'worker'` runs guests off the main thread behind the `types/protocol.ts` contract.
2. ~~**Pyodide guest runtime**~~ — done: `@webkaya/sandbox/python` runs Python over local data; see the demo.
3. **Pyodide inside the Worker** — run the Python runtime on its own thread under the sandbox's policy and metering.
4. **Rust snapshot core** — copy-on-write state forking and I/O recording compiled to WASM (and natively for the server tier).
5. **Server overflow tier** — same SDK API, optional cloud execution for workloads beyond the browser's memory ceiling, with probe programs attaching to kernel eBPF/ubpf instead of the userspace VM.
6. **Probe toolchain** — load probes compiled from C/Rust (clang `-target bpf` ELF objects) in addition to the built-in assembler.

## Current limits

- Inline guest execution is in-process behind a token-scanned `Function` boundary — an honesty note, not a security claim. `runtime: 'worker'` with a real `Worker` adds genuine thread isolation and hard timeout enforcement; stronger sandboxing (WASM realm per guest) remains future work.
- Replay re-executes recorded code; it is reproducible for pure state-transforming guests but is not yet bit-perfect deterministic (no interception of time/randomness). Handlers that read `ctx.global` perform shared I/O, so their replay is not deterministic by construction.
- The fabric, load balancer, and memory tiers are in-process models, not real TCP/SDN/Redis; they exist to prove the programming model and to port onto the server tier unchanged.
- Snapshot state must be JSON-serializable for OPFS persistence.

## License

MIT
