# Deployment and large datasets

## Kaggle download API (Next.js route)

The proxy at `/api/kaggle/download` streams data from Kaggle to the browser. Behavior is controlled with environment variables:

| Variable | Default | Meaning |
|----------|---------|---------|
| `BROWSER_FIRST_AI_KAGGLE_MAX_DOWNLOAD_BYTES` | *(empty)* = **unlimited** | Hard cap on bytes streamed through the server. Set to a positive number (bytes) to enforce a maximum; use `0` explicitly for unlimited. |
| `BROWSER_FIRST_AI_KAGGLE_IDLE_TIMEOUT_MS` | `300000` (5 minutes) | If no bytes arrive from Kaggle for this long, the transfer aborts. Set to `0` to disable idle timeout (not recommended on production without other safeguards). |
| `BROWSER_FIRST_AI_KAGGLE_RATE_MAX` | `6` | Max completed download requests per credential bucket per window (see below). |
| `BROWSER_FIRST_AI_KAGGLE_RATE_WINDOW_MS` | `60000` | Sliding window for the counter above. Limits are keyed by **client IP + SHA-256 of Kaggle credentials** so one user cannot exhaust another’s quota. |

The download route requires **valid Kaggle auth** (validated against Kaggle before streaming) and **same-origin** requests. For multi-instance production, replace in-memory rate limits with **Redis** (e.g. Upstash) — the current guard is per Node process.

The route sets `responseLimit: false` so Next.js does not truncate large bodies. **Serverless hosts** (e.g. Vercel functions) often impose their own **maximum duration** and **response size** limits. Multi‑gigabyte or hour‑long streams may require **self‑hosted Node** (`next start` behind your own reverse proxy), a **long‑running worker**, or **skipping the proxy** by using the Kaggle CLI locally (below).

## Browser limits

- Downloads are streamed to **Origin Private File System (OPFS)** when available so the full file is not kept in a single JavaScript `Blob` during fetch. The **ZIP file bytes** are still loaded once into memory for parsing; **fflate** inflates **one CSV/JSON entry or one image stream at a time**, which avoids JSZip-style “all entries inflated at once” peaks for mixed archives.
- ZIP archives larger than **6GB** are rejected by the in‑browser parser with a message pointing to the CLI + folder workflow.

## Recommended workflow for very large datasets (multi‑GB)

1. Install the [Kaggle API](https://github.com/Kaggle/kaggle-api) and authenticate (`kaggle.json` or `kaggle datasets download …`).
2. Download and extract on disk:

   ```bash
   kaggle datasets download -d owner/dataset
   unzip dataset.zip -d ./my_dataset
   ```

3. In **Browser-First AI Platform**, use **Open folder** (File System Access API) and select the extracted directory containing CSV/JSON or image files.

This avoids browser memory limits and long‑lived HTTP connections through your app server.

## Local development

Run `npm run dev` or `npm run start` after `npm run build`. For large downloads, prefer `next start` on a machine with enough disk and stable network; tune `BROWSER_FIRST_AI_KAGGLE_IDLE_TIMEOUT_MS` if transfers stall on slow links.
