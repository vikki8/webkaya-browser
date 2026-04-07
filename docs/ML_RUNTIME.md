# ML runtime: training vs inference

## `onnx-proto` (dependency)

This package decodes and encodes the ONNX **protobuf schema** (e.g. for export validation). It does **not** execute operators or run inference.

## `onnxruntime-web` (dependency)

Used for **real inference** in the browser: load a `.onnx` file, create an `InferenceSession`, and `run()` with input tensors.

- Entry: `src/engine/onnx-inference.ts` (`loadOnnxRuntime`, `createOnnxInferenceSession`).
- Round-trip checks in `src/engine/onnx-roundtrip.ts` may still load ORT from CDN for compatibility; prefer the npm module for new code.
- WASM files are loaded from jsDelivr by default. For offline or stricter COEP, copy `node_modules/onnxruntime-web/dist/*.wasm` into `public/onnx-wasm/` and set `ort.env.wasm.wasmPaths` to `/onnx-wasm/`.

## Training engine (`src/engine/`)

Tabular / classical models and the in-browser neural trainer are largely **pure TypeScript** (plus optional WebGPU paths where implemented). Throughput is limited compared to WASM SIMD or native runtimes.

**Near-term improvement:** delegate heavy matmul / conv blocks to `@tensorflow/tfjs` + `tfjs-backend-wasm`, or train/export to ONNX and use `onnxruntime-web` for inference-only paths.

**Long-term:** WebGPU compute shaders for large matrix cores (see product Features / roadmap).

## Headers (COOP / COEP)

`Cross-Origin-Embedder-Policy: require-corp` is required for `SharedArrayBuffer` in workers. Third-party scripts and images must be same-origin or carry `Cross-Origin-Resource-Policy` / CORS as appropriate. Next.js bundles app code from `node_modules` as same-origin.
