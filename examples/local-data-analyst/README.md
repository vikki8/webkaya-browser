# Private Data Analyst (the killer demo)

The product thesis in one page: ask questions about a spreadsheet in plain English,
an AI writes the analysis code, and it runs **entirely in the browser tab** over the
user's file. The data never leaves the device — and a **live egress ledger** proves it
by monitoring every network request the page makes.

This is WebKaya's wedge made visceral: where a hosted code interpreter (E2B, ChatGPT
Advanced Data Analysis, etc.) uploads the user's file to a server on every question,
here that number stays at **0 bytes**, verifiably.

## Run it

```bash
# from the repo root
npm install
npm run build          # the demo imports from ../../dist
npx serve .            # or: python3 -m http.server
# open http://localhost:3000/examples/local-data-analyst/
```

Click **Use sample data** (deliberately sensitive — salaries and ID fragments) and ask
something like *"average base salary by department"*. Open your browser's Network tab
alongside the in-page ledger: the two agree.

With no API key a built-in planner handles a few question shapes. Paste an Anthropic key
to have **Claude** write the Python (and repair it if it errors) — only your *question*
is sent to Claude, never the data. Use a limited, revocable key; client-side use exposes
it to the page.

## What makes it the demo

- **Verifiable privacy, not promised.** `privacy.js` wraps `fetch`/`XMLHttpRequest`
  before anything else loads, records every outbound request, and checks each request
  body against fingerprints of the loaded dataset. The "0 bytes of your data uploaded"
  headline is an audit result, not a label.
- **Real work, locally.** A pandas analysis runs in Pyodide (CPython compiled to WASM)
  over a DataFrame built from the user's file — no backend.
- **The generated code is shown**, so the user can see exactly what ran.
- **The contrast is explicit:** how much a hosted tool would have uploaded vs. 0.

## From demo to product

- The locality and zero-egress story is the differentiator; the SDK (`@webkaya/sandbox`)
  is what a customer embeds to get this in their own app.
- For workloads beyond the browser's memory ceiling, the same SDK API has a planned
  server-overflow tier — but the default, and the pitch, is: it runs where the data lives.
