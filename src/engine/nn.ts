import {
  Tensor, relu, sigmoid, tanh_, softmax, dropout as dropoutOp,
  matmul, matmulWithWebGpu, addBias, conv2d as conv2dOp, maxpool2d as maxpool2dOp,
  avgpool2d as avgpool2dOp, flatten as flattenOp
} from './tensor';
import type { WebGpuMatmulRuntime } from './webgpu-matmul';

/** Below this many multiply-adds, CPU matmul usually wins over GPU dispatch + readback. */
const GPU_MATMUL_MIN_FLOPS = 32 * 1024;

export interface Layer {
  forward(x: Tensor, training?: boolean): Tensor;
  parameters(): Tensor[];
  name: string;
}

export class Linear implements Layer {
  weight: Tensor;
  bias: Tensor;
  name: string;

  constructor(inFeatures: number, outFeatures: number, name = 'Linear') {
    this.weight = Tensor.kaimingUniform([inFeatures, outFeatures], inFeatures, true);
    this.bias = new Tensor(new Float32Array(outFeatures).fill(0), [outFeatures], true);
    this.name = name;
  }

  forward(x: Tensor): Tensor {
    return addBias(matmul(x, this.weight), this.bias);
  }

  /**
   * WebGPU WGSL matmul for z = x @ W, then bias on CPU. Backward uses existing autograd.
   */
  async forwardAsync(x: Tensor, gpu: WebGpuMatmulRuntime | null): Promise<Tensor> {
    if (!gpu || gpu.lost) return this.forward(x);
    const [M, Kx] = x.shape;
    const [Kw, N] = this.weight.shape;
    if (Kx !== Kw) throw new Error(`Linear expected K=${Kw} but input has ${Kx} features.`);
    const flops = M * Kx * N;
    if (flops < GPU_MATMUL_MIN_FLOPS) return this.forward(x);
    try {
      const z = await matmulWithWebGpu(x, this.weight, gpu);
      return addBias(z, this.bias);
    } catch {
      return this.forward(x);
    }
  }

  parameters(): Tensor[] {
    return [this.weight, this.bias];
  }
}

export class Conv2d implements Layer {
  weight: Tensor;
  bias: Tensor;
  stride: number;
  padding: number;
  name: string;

  constructor(inChannels: number, outChannels: number, kernelSize: number, stride = 1, padding = 0, name = 'Conv2d') {
    const fanIn = inChannels * kernelSize * kernelSize;
    this.weight = Tensor.kaimingUniform([outChannels, inChannels, kernelSize, kernelSize], fanIn, true);
    this.bias = new Tensor(new Float32Array(outChannels).fill(0), [outChannels], true);
    this.stride = stride;
    this.padding = padding;
    this.name = name;
  }

  forward(x: Tensor): Tensor {
    return conv2dOp(x, this.weight, this.bias, this.stride, this.padding);
  }

  parameters(): Tensor[] {
    return [this.weight, this.bias];
  }
}

export class BatchNorm2d implements Layer {
  gamma: Tensor;
  beta: Tensor;
  runningMean: Float32Array;
  runningVar: Float32Array;
  numFeatures: number;
  momentum: number;
  eps: number;
  name: string;

  constructor(numFeatures: number, name = 'BatchNorm2d') {
    this.numFeatures = numFeatures;
    this.gamma = new Tensor(new Float32Array(numFeatures).fill(1), [numFeatures], true);
    this.beta = new Tensor(new Float32Array(numFeatures).fill(0), [numFeatures], true);
    this.runningMean = new Float32Array(numFeatures);
    this.runningVar = new Float32Array(numFeatures).fill(1);
    this.momentum = 0.1;
    this.eps = 1e-5;
    this.name = name;
  }

  forward(x: Tensor, training = true): Tensor {
    const [N, C, H, W] = x.shape;
    const spatialSize = H * W;
    const out = new Float32Array(x.size);

    for (let c = 0; c < C; c++) {
      let mean = 0, variance = 0;

      if (training) {
        for (let n = 0; n < N; n++)
          for (let hw = 0; hw < spatialSize; hw++)
            mean += x.data[n * C * spatialSize + c * spatialSize + hw];
        mean /= N * spatialSize;

        for (let n = 0; n < N; n++)
          for (let hw = 0; hw < spatialSize; hw++) {
            const diff = x.data[n * C * spatialSize + c * spatialSize + hw] - mean;
            variance += diff * diff;
          }
        variance /= N * spatialSize;

        this.runningMean[c] = (1 - this.momentum) * this.runningMean[c] + this.momentum * mean;
        this.runningVar[c] = (1 - this.momentum) * this.runningVar[c] + this.momentum * variance;
      } else {
        mean = this.runningMean[c];
        variance = this.runningVar[c];
      }

      const invStd = 1 / Math.sqrt(variance + this.eps);
      for (let n = 0; n < N; n++)
        for (let hw = 0; hw < spatialSize; hw++) {
          const idx = n * C * spatialSize + c * spatialSize + hw;
          out[idx] = this.gamma.data[c] * (x.data[idx] - mean) * invStd + this.beta.data[c];
        }
    }

    const result = new Tensor(out, [...x.shape], x.requiresGrad || this.gamma.requiresGrad);
    if (result.requiresGrad) {
      result.setBackward(() => {
        if (!result.grad) return;
        const [N2, C2, H2, W2] = x.shape;
        const sp = H2 * W2;
        const m = N2 * sp;

        for (let c = 0; c < C2; c++) {
          let mean = 0;
          for (let n = 0; n < N2; n++)
            for (let hw = 0; hw < sp; hw++)
              mean += x.data[n * C2 * sp + c * sp + hw];
          mean /= m;

          let variance = 0;
          for (let n = 0; n < N2; n++)
            for (let hw = 0; hw < sp; hw++) {
              const diff = x.data[n * C2 * sp + c * sp + hw] - mean;
              variance += diff * diff;
            }
          variance /= m;
          const invStd = 1 / Math.sqrt(variance + this.eps);

          let dGamma = 0, dBeta = 0;
          let dMean = 0, dVar = 0;

          for (let n = 0; n < N2; n++)
            for (let hw = 0; hw < sp; hw++) {
              const idx = n * C2 * sp + c * sp + hw;
              const xNorm = (x.data[idx] - mean) * invStd;
              dGamma += result.grad[idx] * xNorm;
              dBeta += result.grad[idx];
              const dxNorm = result.grad[idx] * this.gamma.data[c];
              dVar += dxNorm * (x.data[idx] - mean) * -0.5 * Math.pow(variance + this.eps, -1.5);
              dMean += dxNorm * -invStd;
            }

          if (this.gamma.grad) this.gamma.grad[c] += dGamma;
          if (this.beta.grad) this.beta.grad[c] += dBeta;

          if (x.grad) {
            for (let n = 0; n < N2; n++)
              for (let hw = 0; hw < sp; hw++) {
                const idx = n * C2 * sp + c * sp + hw;
                const dxNorm = result.grad[idx] * this.gamma.data[c];
                x.grad[idx] += dxNorm * invStd + dVar * 2 * (x.data[idx] - mean) / m + dMean / m;
              }
          }
        }
      }, [x, this.gamma, this.beta]);
    }
    return result;
  }

  parameters(): Tensor[] {
    return [this.gamma, this.beta];
  }
}

export class BatchNorm1d implements Layer {
  gamma: Tensor;
  beta: Tensor;
  runningMean: Float32Array;
  runningVar: Float32Array;
  numFeatures: number;
  momentum: number;
  eps: number;
  name: string;

  constructor(numFeatures: number, name = 'BatchNorm1d') {
    this.numFeatures = numFeatures;
    this.gamma = new Tensor(new Float32Array(numFeatures).fill(1), [numFeatures], true);
    this.beta = new Tensor(new Float32Array(numFeatures).fill(0), [numFeatures], true);
    this.runningMean = new Float32Array(numFeatures);
    this.runningVar = new Float32Array(numFeatures).fill(1);
    this.momentum = 0.1;
    this.eps = 1e-5;
    this.name = name;
  }

  forward(x: Tensor, training = true): Tensor {
    if (x.shape.length !== 2 || x.shape[1] !== this.numFeatures) {
      throw new Error(`BatchNorm1d expects [N, ${this.numFeatures}] input but got [${x.shape.join(', ')}].`);
    }
    const [N, F] = x.shape;
    const out = new Float32Array(x.size);

    for (let f = 0; f < F; f++) {
      let mean = 0;
      let variance = 0;

      if (training) {
        for (let n = 0; n < N; n++) {
          mean += x.data[n * F + f];
        }
        mean /= N;
        for (let n = 0; n < N; n++) {
          const diff = x.data[n * F + f] - mean;
          variance += diff * diff;
        }
        variance /= N;
        this.runningMean[f] = (1 - this.momentum) * this.runningMean[f] + this.momentum * mean;
        this.runningVar[f] = (1 - this.momentum) * this.runningVar[f] + this.momentum * variance;
      } else {
        mean = this.runningMean[f];
        variance = this.runningVar[f];
      }

      const invStd = 1 / Math.sqrt(variance + this.eps);
      for (let n = 0; n < N; n++) {
        const idx = n * F + f;
        out[idx] = this.gamma.data[f] * (x.data[idx] - mean) * invStd + this.beta.data[f];
      }
    }

    const result = new Tensor(out, [...x.shape], x.requiresGrad || this.gamma.requiresGrad);
    if (result.requiresGrad) {
      result.setBackward(() => {
        if (!result.grad) return;
        const m = N;

        for (let f = 0; f < F; f++) {
          let mean = 0;
          for (let n = 0; n < N; n++) mean += x.data[n * F + f];
          mean /= m;

          let variance = 0;
          for (let n = 0; n < N; n++) {
            const diff = x.data[n * F + f] - mean;
            variance += diff * diff;
          }
          variance /= m;
          const invStd = 1 / Math.sqrt(variance + this.eps);

          let dGamma = 0;
          let dBeta = 0;
          let dVar = 0;
          let dMean = 0;

          for (let n = 0; n < N; n++) {
            const idx = n * F + f;
            const xNorm = (x.data[idx] - mean) * invStd;
            dGamma += result.grad[idx] * xNorm;
            dBeta += result.grad[idx];
            const dxNorm = result.grad[idx] * this.gamma.data[f];
            dVar += dxNorm * (x.data[idx] - mean) * -0.5 * Math.pow(variance + this.eps, -1.5);
            dMean += dxNorm * -invStd;
          }

          if (this.gamma.grad) this.gamma.grad[f] += dGamma;
          if (this.beta.grad) this.beta.grad[f] += dBeta;

          if (x.grad) {
            for (let n = 0; n < N; n++) {
              const idx = n * F + f;
              const dxNorm = result.grad[idx] * this.gamma.data[f];
              x.grad[idx] += dxNorm * invStd + dVar * (2 * (x.data[idx] - mean)) / m + dMean / m;
            }
          }
        }
      }, [x, this.gamma, this.beta]);
    }
    return result;
  }

  parameters(): Tensor[] {
    return [this.gamma, this.beta];
  }
}

export class LayerNorm1d implements Layer {
  gamma: Tensor;
  beta: Tensor;
  numFeatures: number;
  eps: number;
  name: string;

  constructor(numFeatures: number, name = 'LayerNorm1d') {
    this.numFeatures = numFeatures;
    this.gamma = new Tensor(new Float32Array(numFeatures).fill(1), [numFeatures], true);
    this.beta = new Tensor(new Float32Array(numFeatures).fill(0), [numFeatures], true);
    this.eps = 1e-5;
    this.name = name;
  }

  forward(x: Tensor): Tensor {
    if (x.shape.length !== 2 || x.shape[1] !== this.numFeatures) {
      throw new Error(`LayerNorm1d expects [N, ${this.numFeatures}] input but got [${x.shape.join(', ')}].`);
    }
    const [N, F] = x.shape;
    const out = new Float32Array(x.size);
    const invStd = new Float32Array(N);
    const xHat = new Float32Array(x.size);

    for (let n = 0; n < N; n++) {
      let mean = 0;
      const rowBase = n * F;
      for (let f = 0; f < F; f++) mean += x.data[rowBase + f];
      mean /= F;

      let variance = 0;
      for (let f = 0; f < F; f++) {
        const diff = x.data[rowBase + f] - mean;
        variance += diff * diff;
      }
      variance /= F;
      invStd[n] = 1 / Math.sqrt(variance + this.eps);

      for (let f = 0; f < F; f++) {
        const idx = rowBase + f;
        const normalized = (x.data[idx] - mean) * invStd[n];
        xHat[idx] = normalized;
        out[idx] = this.gamma.data[f] * normalized + this.beta.data[f];
      }
    }

    const result = new Tensor(out, [...x.shape], x.requiresGrad || this.gamma.requiresGrad);
    if (result.requiresGrad) {
      result.setBackward(() => {
        if (!result.grad) return;
        const dGamma = new Float32Array(F);
        const dBeta = new Float32Array(F);
        for (let n = 0; n < N; n++) {
          const rowBase = n * F;
          let sumDout = 0;
          let sumDoutXHat = 0;
          for (let f = 0; f < F; f++) {
            const idx = rowBase + f;
            const gradVal = result.grad[idx];
            sumDout += gradVal * this.gamma.data[f];
            sumDoutXHat += gradVal * this.gamma.data[f] * xHat[idx];
            dGamma[f] += gradVal * xHat[idx];
            dBeta[f] += gradVal;
          }
          if (x.grad) {
            for (let f = 0; f < F; f++) {
              const idx = rowBase + f;
              const doutGamma = result.grad[idx] * this.gamma.data[f];
              x.grad[idx] += (invStd[n] / F) * (F * doutGamma - sumDout - xHat[idx] * sumDoutXHat);
            }
          }
        }

        if (this.gamma.grad) {
          for (let f = 0; f < F; f++) this.gamma.grad[f] += dGamma[f];
        }
        if (this.beta.grad) {
          for (let f = 0; f < F; f++) this.beta.grad[f] += dBeta[f];
        }
      }, [x, this.gamma, this.beta]);
    }
    return result;
  }

  parameters(): Tensor[] {
    return [this.gamma, this.beta];
  }
}

export class ReLU implements Layer {
  name = 'ReLU';
  forward(x: Tensor): Tensor { return relu(x); }
  parameters(): Tensor[] { return []; }
}

export class LeakyReLU implements Layer {
  name = 'LeakyReLU';
  negativeSlope: number;

  constructor(negativeSlope = 0.01) {
    this.negativeSlope = negativeSlope;
  }

  forward(x: Tensor): Tensor {
    const out = new Float32Array(x.size);
    for (let i = 0; i < x.size; i++) {
      const value = x.data[i];
      out[i] = value >= 0 ? value : value * this.negativeSlope;
    }
    const result = new Tensor(out, [...x.shape], x.requiresGrad);
    if (x.requiresGrad) {
      result.setBackward(() => {
        if (x.grad && result.grad) {
          for (let i = 0; i < x.size; i++) {
            const slope = x.data[i] >= 0 ? 1 : this.negativeSlope;
            x.grad[i] += result.grad[i] * slope;
          }
        }
      }, [x]);
    }
    return result;
  }

  parameters(): Tensor[] { return []; }
}

export class Sigmoid implements Layer {
  name = 'Sigmoid';
  forward(x: Tensor): Tensor { return sigmoid(x); }
  parameters(): Tensor[] { return []; }
}

export class Tanh implements Layer {
  name = 'Tanh';
  forward(x: Tensor): Tensor { return tanh_(x); }
  parameters(): Tensor[] { return []; }
}

export class Softmax implements Layer {
  name = 'Softmax';
  forward(x: Tensor): Tensor { return softmax(x); }
  parameters(): Tensor[] { return []; }
}

export class MaxPool2d implements Layer {
  kernelSize: number;
  stride: number;
  name: string;

  constructor(kernelSize: number, stride?: number, name = 'MaxPool2d') {
    this.kernelSize = kernelSize;
    this.stride = stride ?? kernelSize;
    this.name = name;
  }

  forward(x: Tensor): Tensor {
    return maxpool2dOp(x, this.kernelSize, this.stride);
  }

  parameters(): Tensor[] { return []; }
}

export class AvgPool2d implements Layer {
  kernelSize: number;
  stride: number;
  name: string;

  constructor(kernelSize: number, stride?: number, name = 'AvgPool2d') {
    this.kernelSize = kernelSize;
    this.stride = stride ?? kernelSize;
    this.name = name;
  }

  forward(x: Tensor): Tensor {
    return avgpool2dOp(x, this.kernelSize, this.stride);
  }

  parameters(): Tensor[] { return []; }
}

export class Dropout implements Layer {
  p: number;
  name: string;

  constructor(p = 0.5, name = 'Dropout') {
    this.p = p;
    this.name = name;
  }

  forward(x: Tensor, training = true): Tensor {
    return dropoutOp(x, this.p, training);
  }

  parameters(): Tensor[] { return []; }
}

export class Flatten implements Layer {
  name = 'Flatten';
  forward(x: Tensor): Tensor { return flattenOp(x); }
  parameters(): Tensor[] { return []; }
}

export class Sequential {
  layers: Layer[];

  constructor(layers: Layer[]) {
    this.layers = layers;
  }

  forward(x: Tensor, training = true): Tensor {
    let out = x;
    for (const layer of this.layers) {
      out = layer.forward(out, training);
    }
    return out;
  }

  /**
   * Runs {@link Linear} layers on the GPU (matmul) when `gpu` is non-null; other layers on CPU.
   */
  async forwardAsync(x: Tensor, training = true, gpu: WebGpuMatmulRuntime | null): Promise<Tensor> {
    let out = x;
    for (const layer of this.layers) {
      if (layer instanceof Linear && gpu && !gpu.lost) {
        out = await layer.forwardAsync(out, gpu);
      } else {
        out = layer.forward(out, training);
      }
    }
    return out;
  }

  parameters(): Tensor[] {
    return this.layers.flatMap(l => l.parameters());
  }
}
