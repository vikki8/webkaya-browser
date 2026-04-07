export class WebGpuLinearRuntime {
  private readonly device: any;
  private readonly pipeline: any;
  private readonly bindGroupLayout: any;
  private lost = false;

  private constructor(device: any) {
    this.device = device;
    const shaderStageCompute = (globalThis as any).GPUShaderStage?.COMPUTE ?? 4;
    const bufferUsage = (globalThis as any).GPUBufferUsage ?? {};
    const storageUsage = bufferUsage.STORAGE ?? 0x0080;
    const copyDstUsage = bufferUsage.COPY_DST ?? 0x0008;
    const copySrcUsage = bufferUsage.COPY_SRC ?? 0x0004;
    const uniformUsage = bufferUsage.UNIFORM ?? 0x0040;
    const mapReadUsage = bufferUsage.MAP_READ ?? 0x0001;

    this.bindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: shaderStageCompute, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: shaderStageCompute, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: shaderStageCompute, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: shaderStageCompute, buffer: { type: 'storage' } },
        { binding: 4, visibility: shaderStageCompute, buffer: { type: 'uniform' } },
      ],
    });
    const shader = device.createShaderModule({
      code: `
struct Params {
  rows: u32,
  features: u32,
  classes: u32,
  _pad: u32,
}

@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> O: array<f32>;
@group(0) @binding(4) var<uniform> P: Params;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  let cls = gid.y;
  if (row >= P.rows || cls >= P.classes) {
    return;
  }

  var sum = B[cls];
  for (var f: u32 = 0u; f < P.features; f = f + 1u) {
    let xVal = X[row * P.features + f];
    let wVal = W[f * P.classes + cls];
    sum = sum + xVal * wVal;
  }
  O[row * P.classes + cls] = sum;
}
      `,
    });
    this.pipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({
        bindGroupLayouts: [this.bindGroupLayout],
      }),
      compute: {
        module: shader,
        entryPoint: 'main',
      },
    });

    (this as any)._usage = {
      storageUsage,
      copyDstUsage,
      copySrcUsage,
      uniformUsage,
      mapReadUsage,
    };
  }

  static async create(
    onDeviceLost?: (reason: string) => void
  ): Promise<WebGpuLinearRuntime | null> {
    if (typeof navigator === 'undefined' || !(navigator as any).gpu) return null;
    const adapter = await (navigator as any).gpu.requestAdapter();
    if (!adapter) return null;
    const device = await adapter.requestDevice();
    const runtime = new WebGpuLinearRuntime(device);
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
  }

  async computeLogits(
    features: Float32Array,
    weights: Float32Array,
    bias: Float32Array,
    rows: number,
    featureCount: number,
    classCount: number
  ): Promise<Float32Array> {
    if (this.lost) throw new Error('WebGPU device is unavailable after loss event.');
    if (features.length !== rows * featureCount) {
      throw new Error('WebGPU features shape mismatch.');
    }
    if (weights.length !== featureCount * classCount) {
      throw new Error('WebGPU weight shape mismatch.');
    }
    if (bias.length !== classCount) {
      throw new Error('WebGPU bias shape mismatch.');
    }

    const outputLength = rows * classCount;
    const outputSize = outputLength * Float32Array.BYTES_PER_ELEMENT;
    const usage = (this as any)._usage as {
      storageUsage: number;
      copyDstUsage: number;
      copySrcUsage: number;
      uniformUsage: number;
      mapReadUsage: number;
    };
    const featuresBuffer = this.device.createBuffer({
      size: features.byteLength,
      usage: usage.storageUsage | usage.copyDstUsage,
    });
    const weightsBuffer = this.device.createBuffer({
      size: weights.byteLength,
      usage: usage.storageUsage | usage.copyDstUsage,
    });
    const biasBuffer = this.device.createBuffer({
      size: bias.byteLength,
      usage: usage.storageUsage | usage.copyDstUsage,
    });
    const outputBuffer = this.device.createBuffer({
      size: outputSize,
      usage: usage.storageUsage | usage.copySrcUsage,
    });
    const params = new Uint32Array([rows, featureCount, classCount, 0]);
    const paramsBuffer = this.device.createBuffer({
      size: params.byteLength,
      usage: usage.uniformUsage | usage.copyDstUsage,
    });
    const readbackBuffer = this.device.createBuffer({
      size: outputSize,
      usage: usage.copyDstUsage | usage.mapReadUsage,
    });

    this.device.queue.writeBuffer(featuresBuffer, 0, features);
    this.device.queue.writeBuffer(weightsBuffer, 0, weights);
    this.device.queue.writeBuffer(biasBuffer, 0, bias);
    this.device.queue.writeBuffer(paramsBuffer, 0, params);

    const bindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: featuresBuffer } },
        { binding: 1, resource: { buffer: weightsBuffer } },
        { binding: 2, resource: { buffer: biasBuffer } },
        { binding: 3, resource: { buffer: outputBuffer } },
        { binding: 4, resource: { buffer: paramsBuffer } },
      ],
    });

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(rows / 8), Math.ceil(classCount / 8), 1);
    pass.end();
    encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, outputSize);
    this.device.queue.submit([encoder.finish()]);

    const gpuMapModeRead = (globalThis as any).GPUMapMode?.READ ?? 1;
    await readbackBuffer.mapAsync(gpuMapModeRead);
    const copy = readbackBuffer.getMappedRange().slice(0);
    readbackBuffer.unmap();

    featuresBuffer.destroy();
    weightsBuffer.destroy();
    biasBuffer.destroy();
    outputBuffer.destroy();
    paramsBuffer.destroy();
    readbackBuffer.destroy();

    return new Float32Array(copy);
  }
}
