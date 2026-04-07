# Browser-First AI Platform

**Local-first ML studio in the browser** — upload or import datasets (CSV, JSON, ZIP, Kaggle), preprocess, train classical ML and small neural nets with a Web Worker + WASM-style runtime, and export artifacts. Data stays on your machine unless you choose to use server-side API routes (e.g. Kaggle proxy).

Stack: **Next.js 14**, **React 18**, **TypeScript**, **Tailwind CSS**, **Vitest**.

---

## What to push to Git

**Commit source and project files only.** Dependencies are **not** uploaded:

| Committed | Not committed (see `.gitignore`) |
|-----------|----------------------------------|
| `src/`, `public/`, `tests/`, `docs/` | `node_modules/` |
| `package.json`, `package-lock.json` | `.next/`, `out/`, `dist/` |
| `next.config.js`, `tailwind.config.js`, `postcss.config.js`, `tsconfig.json`, `vitest.config.ts` | `.env`, `.env.local` |
| `README.md`, `.gitignore` | build caches, logs, `.vercel/` |

After cloning, everyone runs **`npm install`** (or **`npm ci`** from the lockfile) to restore libraries locally.

---

## Prerequisites

- **Node.js** 18+ (20 LTS recommended)
- **npm** 9+ (bundled with Node)

---

## Build & run

```bash
# Install dependencies (required after clone)
npm install

# Development server — http://localhost:3000
npm run dev

# Production build
npm run build

# Run production server (after build)
npm start
```

### Other scripts

```bash
npm run lint      # ESLint (Next.js)
npm run test      # Vitest unit tests
npm run test:watch
```

---

## Environment (optional)

- For **Kaggle** downloads through the app’s API routes, you normally authenticate in the UI. Server-side limits can be tuned with env vars documented in [`docs/DEPLOYMENT_AND_DATASETS.md`](docs/DEPLOYMENT_AND_DATASETS.md) (e.g. `BROWSER_FIRST_AI_KAGGLE_MAX_DOWNLOAD_BYTES`, `BROWSER_FIRST_AI_KAGGLE_IDLE_TIMEOUT_MS`).
- Do **not** commit API tokens; use `.env.local` (gitignored).

---

## Project layout (high level)

```
browser-first-ai-platform/
├── src/
│   ├── engine/           # Training worker, tensor/NN core, algorithms (k-means, RF, NN, …)
│   ├── data/             # Parsers, preprocessing, IndexedDB helpers
│   ├── pages/            # Next.js pages & API routes (dataset, Kaggle)
│   ├── ui/hooks/         # Training workflow, data worker hooks
│   └── types/            # Shared TypeScript types
├── tests/                # Vitest
├── docs/                 # Deployment & dataset notes
├── package.json
├── next.config.js
└── tailwind.config.js
```

---

## License

MIT
