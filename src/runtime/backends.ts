import { Capabilities } from './capability-detect';

export type ComputeBackendKind = 'webgpu' | 'webgl2' | 'wasm-simd' | 'cpu';

export interface ComputeBackendSelection {
  kind: ComputeBackendKind;
  label: string;
  reason: string;
}

export function selectComputeBackend(
  capabilities: Capabilities | null,
  preferGpu = true
): ComputeBackendSelection {
  if (!capabilities) {
    return {
      kind: 'cpu',
      label: 'Sandbox Runtime (CPU Fallback)',
      reason: 'Capabilities unavailable. Running sandbox with CPU compute fallback.',
    };
  }

  if (capabilities.webgpu) {
    return {
      kind: 'webgpu',
      label: 'Sandbox Runtime (WebGPU)',
      reason: preferGpu
        ? 'WebGPU active: guest compute can route to GPU while host orchestration stays stable.'
        : 'WebGPU is available; sandbox can route guest kernels to GPU on demand.',
    };
  }

  if (capabilities.webgl2 && preferGpu) {
    return {
      kind: 'webgl2',
      label: 'Sandbox Runtime (WebGL2)',
      reason: 'WebGL2 available. Running sandbox with compatibility GPU fallback.',
    };
  }

  if (capabilities.wasmSimd) {
    return {
      kind: 'wasm-simd',
      label: 'Sandbox Runtime (WASM SIMD)',
      reason: 'WebGPU unavailable. Running sandbox on SIMD-optimized CPU compute path.',
    };
  }

  return {
    kind: 'cpu',
    label: 'Sandbox Runtime (CPU)',
    reason: 'No GPU acceleration available. Running sandbox with baseline CPU compute.',
  };
}
