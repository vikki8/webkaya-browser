/**
 * WebGPU WGSL compute: general matrix multiply C = A @ B with A [M,K], B [K,N], C [M,N].
 * Used for neural-network training forward passes; autograd backward stays on CPU.
 */

export class WebGpuMatmulRuntime {
  private readonly device: any;
  private readonly pipeline: any;
  private readonly bindGroupLayout: any;
  lost = false;

  private constructor(device: any, pipeline: any, bindGroupLayout: any) {
    this.device = device;
    this.pipeline = pipeline;
    this.bindGroupLayout = bindGroupLayout;
  }

  static async create(onDeviceLost?: (reason: string) => void): Promise<WebGpuMatmulRuntime | null> {
    if (typeof navigator === 'undefined' || !(navigator as any).gpu) return null;
    try {
      const adapter = await (navigator as any).gpu.requestAdapter();
      if (!adapter) return null;
      const device = await adapter.requestDevice();
      const shaderStageCompute = (globalThis as any).GPUShaderStage?.COMPUTE ?? 4;
      const bufferUsage = (globalThis as any).GPUBufferUsage ?? {};
      const storageUsage = bufferUsage.STORAGE ?? 0x0080;
      const copyDstUsage = bufferUsage.COPY_DST ?? 0x0008;
      const copySrcUsage = bufferUsage.COPY_SRC ?? 0x0004;
      const uniformUsage = bufferUsage.UNIFORM ?? 0x0040;
      const mapReadUsage = bufferUsage.MAP_READ ?? 0x0001;

      const shader = device.createShaderModule({
        label: 'matmul_wgsl',
        code: `
struct Params {
  M: u32,
  K: u32,
  N: u32,
  _pad: u32,
}

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> P: Params;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  let col = gid.y;
  if (row >= P.M || col >= P.N) {
    return;
  }
  var sum = 0.0;
  for (var k: u32 = 0u; k < P.K; k = k + 1u) {
    sum = sum + A[row * P.K + k] * B[k * P.N + col];
  }
  C[row * P.N + col] = sum;
}
`,
      });
      const bindGroupLayout = device.createBindGroupLayout({
        label: 'matmul_bgl',
        entries: [
          { binding: 0, visibility: shaderStageCompute, buffer: { type: 'read-only-storage' } },
          { binding: 1, visibility: shaderStageCompute, buffer: { type: 'read-only-storage' } },
          { binding: 2, visibility: shaderStageCompute, buffer: { type: 'storage' } },
          { binding: 3, visibility: shaderStageCompute, buffer: { type: 'uniform' } },
        ],
      });
      const pipeline = device.createComputePipeline({
        label: 'matmul_pipeline',
        layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
        compute: { module: shader, entryPoint: 'main' },
      });
      const runtime = new WebGpuMatmulRuntime(device, pipeline, bindGroupLayout);
      (runtime as any)._usage = { storageUsage, copyDstUsage, copySrcUsage, uniformUsage, mapReadUsage };
      device.lost
        .then((info: any) => {
          runtime.lost = true;
          onDeviceLost?.(info?.message || 'WebGPU device lost.');
        })
        .catch(() => {
          runtime.lost = true;
          onDeviceLost?.('WebGPU device lost.');
        });
      return runtime;
    } catch {
      return null;
    }
  }

  /**
   * Returns C = A @ B. Row-major: A is M×K, B is K×N.
   */
  async matmul(a: Float32Array, b: Float32Array, M: number, K: number, N: number): Promise<Float32Array> {
    if (this.lost) throw new Error('WebGPU device unavailable after loss event.');
    if (a.length !== M * K || b.length !== K * N) {
      throw new Error(`matmul shape mismatch: A ${a.length} vs M*K ${M * K}, B ${b.length} vs K*N ${K * N}`);
    }

    const usage = (this as any)._usage as {
      storageUsage: number;
      copyDstUsage: number;
      copySrcUsage: number;
      uniformUsage: number;
      mapReadUsage: number;
    };

    const outLen = M * N;
    const outBytes = outLen * Float32Array.BYTES_PER_ELEMENT;

    const aBuf = this.device.createBuffer({
      size: a.byteLength,
      usage: usage.storageUsage | usage.copyDstUsage,
    });
    const bBuf = this.device.createBuffer({
      size: b.byteLength,
      usage: usage.storageUsage | usage.copyDstUsage,
    });
    const cBuf = this.device.createBuffer({
      size: outBytes,
      usage: usage.storageUsage | usage.copySrcUsage,
    });
    const params = new Uint32Array([M, K, N, 0]);
    const paramsBuf = this.device.createBuffer({
      size: params.byteLength,
      usage: usage.uniformUsage | usage.copyDstUsage,
    });
    const readback = this.device.createBuffer({
      size: outBytes,
      usage: usage.copyDstUsage | usage.mapReadUsage,
    });

    this.device.queue.writeBuffer(aBuf, 0, a);
    this.device.queue.writeBuffer(bBuf, 0, b);
    this.device.queue.writeBuffer(paramsBuf, 0, params);

    const bindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: aBuf } },
        { binding: 1, resource: { buffer: bBuf } },
        { binding: 2, resource: { buffer: cBuf } },
        { binding: 3, resource: { buffer: paramsBuf } },
      ],
    });

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(M / 8), Math.ceil(N / 8), 1);
    pass.end();
    encoder.copyBufferToBuffer(cBuf, 0, readback, 0, outBytes);
    this.device.queue.submit([encoder.finish()]);

    const gpuMapModeRead = (globalThis as any).GPUMapMode?.READ ?? 1;
    await readback.mapAsync(gpuMapModeRead);
    const mapped = readback.getMappedRange();
    const copy = new Float32Array(outLen);
    copy.set(new Float32Array(mapped));
    readback.unmap();

    aBuf.destroy();
    bBuf.destroy();
    cBuf.destroy();
    paramsBuf.destroy();
    readback.destroy();

    return copy;
  }
}
