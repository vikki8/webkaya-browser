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

## What it shows

- **Isolation** — every agent is a `runtime: 'wasm'` sandbox (QuickJS on
  WebAssembly). It cannot reach `window`/`fetch`/`process` or any other agent.
- **Coordination through memory, not I/O** — phase 2 (reduce) reads the writes
  phase 1 (map) put on the blackboard; phase 3 (report) reads phase 2's. No
  agent talks to another directly.
- **Brokered access** — an agent only ever sees the slice of the blackboard the
  orchestrator hands it in `ctx.args.read`, and can only propose writes; it never
  touches shared state directly.

The orchestration is a real SDK primitive (`IsolatedOrchestrator` from
`@webkaya/sandbox`), tested in the main suite. The handlers here are plain guest
JS; each could equally be written by an LLM.

## How it's wired

- **React + Vite + TypeScript.** Vite aliases the SDK to its source in this repo
  (`vite.config.ts`), so the demo always matches the SDK with no separate build.
- **QuickJS-on-WebAssembly** runs each agent in isolation, bundled by Vite.
