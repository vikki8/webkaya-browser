# WebKaya demo (React + Vite)

A minimal, presentable demo of the client-side thesis: ask questions about a CSV,
an AI writes the analysis code, and it runs in the browser tab over your file. The
**Network** panel shows exactly what leaves the tab.

## Run

```bash
cd examples/webkaya-demo
npm install
npm run dev      # open the printed localhost URL
```

`npm run build && npm run preview` for a production build.

No API key needed — a built-in planner handles a few question shapes. Add an Anthropic
key (under "Use Claude") to have Claude write the Python; only the question is sent.

## How it's wired

- **React + Vite + TypeScript.** The WebKaya SDK is aliased to its source in this
  repo (`vite.config.ts`), so the demo always matches the SDK with no separate build.
- **Pyodide** (CPython→WASM) runs the pandas locally, loaded once from a CDN.
- **`src/privacy.ts`** wraps `fetch`/`XMLHttpRequest` before anything else loads and
  feeds the Network panel — the panel and the browser's own Network tab agree.
