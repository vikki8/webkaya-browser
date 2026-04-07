# Browser tests

Vitest browser mode runs specs in Chromium so Web Workers and WASM behave like production.

1. Install optional dev dependencies (if not already in `package.json`):

   `npm i -D @vitest/browser playwright`

2. `npx playwright install chromium`

3. `npm run test:browser`

Add specs here as `*.browser.ts` and use the Vitest browser API as needed.
