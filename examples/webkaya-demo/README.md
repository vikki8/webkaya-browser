# WebKaya demo (React + Vite)

A live visualization of the thing that makes WebKaya different: **isolated agents
coordinating through a shared blackboard.**

A map → reduce → report pipeline runs as separate agents. Each runs in its own
WebAssembly realm — no host access, and no way to call another agent. The only
thing they share is a key/value blackboard: each agent reads what earlier agents
wrote, and the orchestrator applies the writes it returns. You watch the
blackboard fill in as the phases hand off to each other.

## Run

```bash
cd examples/webkaya-demo
npm install
npm run dev      # open the printed localhost URL, click "Run pipeline"
```

`npm run build && npm run preview` for a production build.

Paste an **Anthropic API key** in the field at the top and each agent writes its
own code with Claude (with a built-in fallback). Without a key it runs the
built-in handlers.

## What it shows

- **Isolation** — every agent runs on its own **Web Worker**: a separate thread
  with no access to this page (no `window`/DOM) and no way to call another agent.
- **Coordination through memory, not I/O** — phase 2 (reduce) reads the writes
  phase 1 (map) put on the blackboard; phase 3 (report) reads phase 2's. No
  agent talks to another directly.
- **Brokered access** — an agent only ever sees the slice of the blackboard the
  orchestrator hands it in `ctx.args.read`, and can only propose writes; it never
  touches shared state directly.

The orchestration is a real SDK primitive (`IsolatedOrchestrator` from
`@webkaya/sandbox`), tested in the main suite. In Node the orchestrator defaults
to the stronger `wasm` isolation (QuickJS on WebAssembly); the browser demo
probes for a working Web Worker at run time and uses `runtime: 'worker'` when one
is available, falling back to `runtime: 'inline'` otherwise so the demo always
completes. The chip under the Run button tells you which one is active.

## How it's wired

- **React + Vite + TypeScript.** Vite aliases the SDK to its source in this repo
  (`vite.config.ts`), so the demo always matches the SDK with no separate build.
- Each agent is a sandbox whose worker entry is `src/agent.worker.ts` — a local
  module worker (inside the project root, so Vite compiles it reliably) that
  calls the SDK's `installWorkerHandler`.
- On Run, `detectRuntime()` spawns a throwaway probe agent on a worker; if it
  answers, the run uses worker isolation, otherwise it drops to inline.
