import { describe, expect, it } from 'vitest';
import { selectComputeBackend } from '../src/runtime/backends';
import { Capabilities } from '../src/runtime/capability-detect';

function caps(overrides: Partial<Capabilities>): Capabilities {
  return {
    tier: 4,
    tierName: 'Baseline (CPU only)',
    webgpu: false,
    webgl2: false,
    wasmSimd: false,
    sharedArrayBuffer: false,
    offscreenCanvas: false,
    opfs: false,
    maxMemoryMB: 512,
    ...overrides,
  };
}

describe('selectComputeBackend', () => {
  it('falls back to CPU when capabilities are unknown', () => {
    expect(selectComputeBackend(null).kind).toBe('cpu');
  });

  it('prefers WebGPU when available', () => {
    expect(selectComputeBackend(caps({ webgpu: true, webgl2: true, wasmSimd: true })).kind).toBe('webgpu');
  });

  it('uses WebGL2 only when GPU is preferred', () => {
    const c = caps({ webgl2: true, wasmSimd: true });
    expect(selectComputeBackend(c, true).kind).toBe('webgl2');
    expect(selectComputeBackend(c, false).kind).toBe('wasm-simd');
  });

  it('uses WASM SIMD when no GPU path exists', () => {
    expect(selectComputeBackend(caps({ wasmSimd: true })).kind).toBe('wasm-simd');
  });

  it('falls back to plain CPU as last resort', () => {
    expect(selectComputeBackend(caps({})).kind).toBe('cpu');
  });
});
