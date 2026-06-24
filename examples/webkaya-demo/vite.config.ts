import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { fileURLToPath, URL } from 'node:url';

// Alias the WebKaya SDK to its source in this monorepo, so the demo stays in
// sync with the SDK without a separate build step. Pyodide is loaded from a CDN
// at runtime (marked @vite-ignore in the SDK), so it is never bundled.
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: [
      { find: '@webkaya/sandbox/python', replacement: fileURLToPath(new URL('../../src/python/index.ts', import.meta.url)) },
      { find: '@webkaya/sandbox/llm', replacement: fileURLToPath(new URL('../../src/llm/index.ts', import.meta.url)) },
      { find: '@webkaya/sandbox', replacement: fileURLToPath(new URL('../../src/index.ts', import.meta.url)) },
    ],
  },
  server: { fs: { allow: ['../..'] } },
});
