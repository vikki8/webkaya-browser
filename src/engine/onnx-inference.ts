/**
 * Executable ONNX inference via onnxruntime-web (WASM / WebGPU).
 * `onnx-proto` is protobuf schema only — it cannot run models (see docs/ML_RUNTIME.md).
 *
 * Call only from the browser. WASM binaries load from the CDN path below unless you
 * copy `node_modules/onnxruntime-web/dist/*.wasm` to `/public/onnx-wasm/` and set paths.
 */

const ORT_WASM_BASE = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/';

export async function loadOnnxRuntime() {
  if (typeof window === 'undefined') {
    throw new Error('ONNX Runtime Web is browser-only.');
  }
  const ort = await import('onnxruntime-web');
  ort.env.wasm.wasmPaths = ORT_WASM_BASE;
  return ort;
}

export async function createOnnxInferenceSession(
  model: ArrayBuffer | Uint8Array,
  options?: { executionProviders?: string[] }
) {
  const ort = await loadOnnxRuntime();
  const bytes = model instanceof Uint8Array ? model : new Uint8Array(model);
  return ort.InferenceSession.create(bytes, {
    executionProviders: options?.executionProviders ?? ['wasm'],
  });
}
