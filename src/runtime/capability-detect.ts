export interface Capabilities {
  tier: 1 | 2 | 3 | 4;
  tierName: string;
  webgpu: boolean;
  webgl2: boolean;
  wasmSimd: boolean;
  sharedArrayBuffer: boolean;
  offscreenCanvas: boolean;
  opfs: boolean;
  maxMemoryMB: number;
  gpuName?: string;
}

export async function detectCapabilities(): Promise<Capabilities> {
  const caps: Capabilities = {
    tier: 4,
    tierName: 'Baseline (CPU only)',
    webgpu: false,
    webgl2: false,
    wasmSimd: false,
    sharedArrayBuffer: typeof SharedArrayBuffer !== 'undefined',
    offscreenCanvas: typeof OffscreenCanvas !== 'undefined',
    opfs: false,
    maxMemoryMB: 512,
  };

  try {
    if (typeof navigator !== 'undefined' && navigator.storage && typeof navigator.storage.getDirectory === 'function') {
      caps.opfs = true;
    }
  } catch { /* not available */ }

  try {
    const simdTest = WebAssembly.validate(new Uint8Array([
      0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123,
      3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11
    ]));
    caps.wasmSimd = simdTest;
  } catch { /* no SIMD */ }

  try {
    if (typeof document !== 'undefined') {
      const canvas = document.createElement('canvas');
      const gl = canvas.getContext('webgl2');
      if (gl) {
        caps.webgl2 = true;
        const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
        if (debugInfo) {
          caps.gpuName = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
        }
      }
    }
  } catch { /* no WebGL2 */ }

  try {
    if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
      const gpu = (navigator as any).gpu;
      if (gpu) {
        const adapter = await gpu.requestAdapter();
        if (adapter) {
          caps.webgpu = true;
          const info = (adapter as any).info || {};
          caps.gpuName = info.device || info.description || caps.gpuName || 'WebGPU Device';
          caps.maxMemoryMB = Math.min(
            (adapter.limits?.maxBufferSize || 256 * 1024 * 1024) / (1024 * 1024),
            4096
          );
        }
      }
    }
  } catch { /* no WebGPU */ }

  if (typeof performance !== 'undefined' && (performance as any).memory) {
    caps.maxMemoryMB = Math.floor((performance as any).memory.jsHeapSizeLimit / (1024 * 1024));
  }

  if (caps.webgpu) {
    caps.tier = 1;
    caps.tierName = 'WebGPU (Full GPU Acceleration)';
  } else if (caps.webgl2) {
    caps.tier = 2;
    caps.tierName = 'WebGL2 (Limited GPU)';
  } else if (caps.wasmSimd) {
    caps.tier = 3;
    caps.tierName = 'CPU WASM SIMD';
  } else {
    caps.tier = 4;
    caps.tierName = 'Baseline (CPU only)';
  }

  return caps;
}
