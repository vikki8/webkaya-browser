# WebKaya Sandbox

**Client-side AI agent sandboxes.** Run agent-generated code in the end user's browser — next to the user's data, at zero marginal cost — with policy governance, snapshots, fork, and deterministic-leaning replay built in.

Cloud sandbox vendors run agent code in server-side microVMs, priced per second, far from the user's data. WebKaya inverts that: the browser tab is already an isolated, capable runtime. This SDK turns it into a governed execution environment for AI products — code interpreters, data-analysis agents, "chat with your files" features — where the data never leaves the device and every run is recorded, snapshottable, and replayable.

This is **not** a security product. The differentiators are **cost** (no per-second VM billing), **locality** (compute happens where the data lives), and **reproducibility** (fork/replay turns agent debugging into something like git).

Stack: **TypeScript**, **Vitest**. No runtime dependencies.

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
| **Snapshots** | Sandbox state persists to OPFS (`OpfsSnapshotStore`) with an in-memory fallback. Snapshots carry lineage (`parentSnapshotId`). |
| **Fork** | Branch a sandbox at its current state; parent and fork diverge independently. |
| **Replay** | Re-execute the recorded run sequence from initial state in a fresh sandbox — debug a run, verify reproducibility. |
| **Capabilities & backends** | `detectCapabilities()` tiers the device (WebGPU → WebGL2 → WASM SIMD → CPU); `selectComputeBackend()` picks the compute path. |
| **Hardware monitor** | Samples activity, heap, and thermal pressure (Compute Pressure API) so hosts can throttle or show a resource HUD. |

---

## Project layout

```
src/
├── index.ts            # Public API
├── sandbox/
│   ├── sandbox.ts      # Sandbox: create/run/snapshot/fork/restore/replay
│   └── snapshot-store.ts  # OPFS + in-memory snapshot persistence
├── runtime/
│   ├── policy.ts       # Policy normalize/validate, guest code scanning, policy-as-code
│   ├── guest-invoker.ts   # Timeout, retry, memory-budget enforcement
│   ├── capability-detect.ts  # Device tiering (WebGPU/WebGL2/WASM SIMD/CPU)
│   ├── backends.ts     # Compute backend selection
│   └── hardware-monitor.ts   # Utilization, thermal, energy sampling
└── types/
    ├── policy.ts       # SandboxPolicy, PolicyEditorState
    └── protocol.ts     # Host <-> sandbox message protocol (worker transport seam)
tests/                  # Vitest
```

---

## Roadmap

1. **Web Worker transport** — move guest execution off the main thread behind the existing `types/protocol.ts` contract.
2. **Pyodide guest runtime** — Python as the primary guest language for agent-generated code.
3. **Rust snapshot core** — copy-on-write state forking and I/O recording compiled to WASM (and natively for the server tier).
4. **Local file access** — guest-visible, permission-gated views over the File System Access API.
5. **Server overflow tier** — same SDK API, optional cloud execution for workloads beyond the browser's memory ceiling.

## Current limits

- v0 guest execution is in-process behind a token-scanned `Function` boundary — an honesty note, not a security claim. Worker isolation is roadmap item 1.
- Replay re-executes recorded code; it is reproducible for pure state-transforming guests but is not yet bit-perfect deterministic (no interception of time/randomness).
- Snapshot state must be JSON-serializable for OPFS persistence.

## License

MIT
