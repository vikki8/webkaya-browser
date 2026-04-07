import { Capabilities } from '../capability-detect';

export type ComputeBackendKind = 'webgpu' | 'webgl2' | 'wasm-simd' | 'cpu';

export interface ComputeBackendSelection {
  kind: ComputeBackendKind;
  label: string;
  reason: string;
}

export function selectComputeBackend(
  capabilities: Capabilities | null,
  useMoreCompute: boolean
): ComputeBackendSelection {
  if (!capabilities) {
    return {
      kind: 'cpu',
      label: 'Hybrid Runtime (CPU Fallback)',
      reason: 'Capabilities unavailable. Running Worker + WASM orchestration with CPU compute fallback.',
    };
  }

  if (capabilities.webgpu) {
    return {
      kind: 'webgpu',
      label: 'Hybrid Runtime (WebGPU)',
      reason: useMoreCompute
        ? 'Hybrid path active: Worker + WASM orchestration with WebGPU preferred compute.'
        : 'WebGPU is available; hybrid runtime can route kernels to GPU while preserving stable worker orchestration.',
    };
  }

  if (capabilities.webgl2 && useMoreCompute) {
    return {
      kind: 'webgl2',
      label: 'Hybrid Runtime (WebGL2)',
      reason: 'WebGL2 available. Running hybrid runtime with compatibility GPU fallback.',
    };
  }

  if (capabilities.wasmSimd) {
    return {
      kind: 'wasm-simd',
      label: 'Hybrid Runtime (WASM SIMD)',
      reason: 'WebGPU unavailable. Running hybrid runtime on SIMD-optimized CPU compute path.',
    };
  }

  return {
    kind: 'cpu',
    label: 'Hybrid Runtime (CPU)',
    reason: 'No GPU acceleration available. Running hybrid runtime with baseline CPU compute fallback.',
  };
}
