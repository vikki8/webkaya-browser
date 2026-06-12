# Local Data Analyst (demo)

The wedge use case, made tangible: pick a CSV, ask a question in plain English, and
**agent-generated Python runs in your browser tab over your file** — the data never
leaves the device. Only the Pyodide runtime (~10 MB) is fetched once from a CDN.

This demonstrates the core WebKaya thesis — *zero-marginal-cost, data-never-leaves-the-device
code execution* — using the SDK's `PythonRunner` (Pyodide) and a deterministic NL→pandas
`planQuestion` planner that stands in for an LLM so the demo runs with **no API key**.

## Run it

```bash
# from the repo root
npm install
npm run build          # the demo imports from ../../dist
npx serve .            # or: python3 -m http.server
# open http://localhost:3000/examples/local-data-analyst/
```

Click **Use sample data** (or choose your own CSV), then try a question like
"average revenue by region" or "top 3 country by revenue".

### Optional: use Claude instead of the built-in planner

Paste an Anthropic API key into the key field and Claude (`claude-opus-4-8`) writes the
Python instead of the deterministic planner — and gets one shot at repairing code that
errors. The key is used directly from the browser tab against the Anthropic API (loaded
from a CDN via an import map) and is never sent anywhere else; only the *question text*
reaches Claude, never your data. Use a limited, revocable key — client-side use exposes it
to the page (`dangerouslyAllowBrowser`).

## What it shows

- **Locality**: the "bytes of your data sent to a server" counter stays at 0. The CSV is
  read with the File System Access API (with an `<input type=file>` fallback) and parsed
  into a pandas DataFrame entirely client-side.
- **Governance/replay**: every run is recorded in the session audit log — the same
  run-record/replay trail the SDK gives agent sandboxes for debugging.
- **The generated code is visible**: you see the Python the planner produced before it runs.

## From demo to production

- The Claude path above already shows the production swap: `CodeAnalyst` from
  `@webkaya/sandbox/llm` replaces `planQuestion` with a real model, runner and privacy
  invariant unchanged. On a server tier, move the key server-side and call the same API.
- Run Pyodide inside the Worker transport (`Sandbox` with `runtime: 'worker'`) so Python
  executes off the main thread under the sandbox's policy, timeout, and metering.
