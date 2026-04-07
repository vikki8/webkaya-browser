import { defineConfig } from 'vitest/config';

/**
 * True browser tests (Workers, full WASM). Requires:
 *   npm i -D @vitest/browser playwright
 *   npx playwright install chromium
 * Then: npm run test:browser
 */
export default defineConfig({
  test: {
    include: ['tests/browser/**/*.browser.ts'],
    browser: {
      enabled: true,
      provider: 'playwright',
      instances: [{ browser: 'chromium' }],
    },
  },
});
