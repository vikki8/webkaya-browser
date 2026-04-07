# App Router migration (optional)

The app currently uses the **Pages Router** (`src/pages/`). A ClickOps-style studio benefits from **App Router** nested layouts:

- Root layout: persistent shell (sidebar, toolbar, theme).
- Nested layouts: dataset, preprocess, training, export / inference sections.
- Server Components can render static chrome; client components hold workers and Zustand.

Migration sketch:

1. Add `src/app/layout.tsx` and `src/app/page.tsx` (move `index.tsx` content behind a client boundary).
2. Move API routes from `pages/api` to `app/api/.../route.ts` (or keep Pages API during transition — Next allows both).
3. Replace `_app` global CSS import with root layout imports.
4. Re-test COOP/COEP headers via `next.config.js` `headers()` (still applies to all routes).

This is a large refactor; track it as a dedicated milestone rather than mixing with feature work.
