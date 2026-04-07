/**
 * COOP + COEP unlock SharedArrayBuffer and high-resolution timers in cross-origin isolated contexts.
 * All subresources must be same-origin or explicitly CORP/CORS-compliant or the browser will block them.
 * App bundles (React, Framer Motion, onnxruntime-web WASM when self-hosted) are served from this origin.
 *
 * @type {import('next').NextConfig}
 */
const nextConfig = {
  async headers() {
    return [
      {
        source: '/(.*)',
        headers: [
          { key: 'Cross-Origin-Opener-Policy', value: 'same-origin' },
          { key: 'Cross-Origin-Embedder-Policy', value: 'require-corp' },
        ],
      },
    ];
  },
};

module.exports = nextConfig;
