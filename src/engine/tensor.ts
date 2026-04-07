/**
 * Minimal tensor engine with autograd for browser-based training.
 * Operates on Float32Array for WASM SIMD compatibility.
 */

import type { WebGpuMatmulRuntime } from './webgpu-matmul';

type BackwardFn = () => void;

export class Tensor {
  data: Float32Array;
  grad: Float32Array | null;
  shape: number[];
  requiresGrad: boolean;
  private _backward: BackwardFn | null = null;
  private _parents: Tensor[] = [];

  constructor(data: Float32Array | number[], shape: number[], requiresGrad = false) {
    this.data = data instanceof Float32Array ? data : new Float32Array(data);
    this.shape = shape;
    this.requiresGrad = requiresGrad;
    this.grad = requiresGrad ? new Float32Array(this.data.length).fill(0) : null;

    const expectedSize = shape.reduce((a, b) => a * b, 1);
    if (this.data.length !== expectedSize) {
      throw new Error(`Tensor data length ${this.data.length} doesn't match shape [${shape}] (expected ${expectedSize})`);
    }
  }

  get size(): number {
    return this.data.length;
  }

  get ndim(): number {
    return this.shape.length;
  }

  static zeros(shape: number[], requiresGrad = false): Tensor {
    const size = shape.reduce((a, b) => a * b, 1);
    return new Tensor(new Float32Array(size), shape, requiresGrad);
  }

  static ones(shape: number[], requiresGrad = false): Tensor {
    const size = shape.reduce((a, b) => a * b, 1);
    return new Tensor(new Float32Array(size).fill(1), shape, requiresGrad);
  }

  static rand(shape: number[], requiresGrad = false): Tensor {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float32Array(size);
    for (let i = 0; i < size; i++) data[i] = Math.random();
    return new Tensor(data, shape, requiresGrad);
  }

  static randn(shape: number[], requiresGrad = false): Tensor {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float32Array(size);
    for (let i = 0; i < size; i++) {
      // Box-Muller transform
      const u1 = Math.random() || 1e-10;
      const u2 = Math.random();
      data[i] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
    return new Tensor(data, shape, requiresGrad);
  }

  /** Kaiming (He) initialization — good default for ReLU networks */
  static kaimingUniform(shape: number[], fanIn: number, requiresGrad = true): Tensor {
    const bound = Math.sqrt(6.0 / fanIn);
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float32Array(size);
    for (let i = 0; i < size; i++) data[i] = (Math.random() * 2 - 1) * bound;
    return new Tensor(data, shape, requiresGrad);
  }

  setBackward(fn: BackwardFn, parents: Tensor[]) {
    this._backward = fn;
    this._parents = parents;
  }

  backward() {
    if (!this.grad) {
      this.grad = new Float32Array(this.data.length).fill(1);
    } else {
      this.grad.fill(1);
    }

    const topo: Tensor[] = [];
    const visited = new Set<Tensor>();

    const buildTopo = (t: Tensor) => {
      if (visited.has(t)) return;
      visited.add(t);
      for (const p of t._parents) buildTopo(p);
      topo.push(t);
    };

    buildTopo(this);
    for (let i = topo.length - 1; i >= 0; i--) {
      if (topo[i]._backward) topo[i]._backward!();
    }
  }

  zeroGrad() {
    if (this.grad) this.grad.fill(0);
  }

  detach(): Tensor {
    return new Tensor(new Float32Array(this.data), [...this.shape], false);
  }

  reshape(newShape: number[]): Tensor {
    const size = newShape.reduce((a, b) => a * b, 1);
    if (size !== this.size) throw new Error(`Cannot reshape [${this.shape}] to [${newShape}]`);
    const out = new Tensor(this.data, newShape, this.requiresGrad);
    if (this.requiresGrad) {
      out.setBackward(() => {
        if (this.grad && out.grad) {
          for (let i = 0; i < this.grad.length; i++) this.grad[i] += out.grad[i];
        }
      }, [this]);
    }
    return out;
  }

  clone(): Tensor {
    const t = new Tensor(new Float32Array(this.data), [...this.shape], this.requiresGrad);
    return t;
  }
}

// ---- Tensor Operations with Autograd ----

export function add(a: Tensor, b: Tensor): Tensor {
  if (a.size !== b.size) throw new Error(`add shape mismatch: [${a.shape}] vs [${b.shape}]`);
  const out = new Float32Array(a.size);
  for (let i = 0; i < a.size; i++) out[i] = a.data[i] + b.data[i];
  const result = new Tensor(out, [...a.shape], a.requiresGrad || b.requiresGrad);
  if (result.requiresGrad) {
    result.setBackward(() => {
      if (a.grad && result.grad) for (let i = 0; i < a.size; i++) a.grad[i] += result.grad[i];
      if (b.grad && result.grad) for (let i = 0; i < b.size; i++) b.grad[i] += result.grad[i];
    }, [a, b]);
  }
  return result;
}

/** Broadcast-add bias [outFeatures] to activations [batch, outFeatures] */
export function addBias(x: Tensor, bias: Tensor): Tensor {
  const [batch, features] = x.shape;
  if (bias.shape[0] !== features) throw new Error(`Bias size ${bias.shape[0]} != features ${features}`);
  const out = new Float32Array(x.size);
  for (let b = 0; b < batch; b++) {
    for (let f = 0; f < features; f++) {
      out[b * features + f] = x.data[b * features + f] + bias.data[f];
    }
  }
  const result = new Tensor(out, [...x.shape], x.requiresGrad || bias.requiresGrad);
  if (result.requiresGrad) {
    result.setBackward(() => {
      if (x.grad && result.grad) {
        for (let i = 0; i < x.size; i++) x.grad[i] += result.grad[i];
      }
      if (bias.grad && result.grad) {
        for (let b = 0; b < batch; b++) {
          for (let f = 0; f < features; f++) {
            bias.grad[f] += result.grad[b * features + f];
          }
        }
      }
    }, [x, bias]);
  }
  return result;
}

export function matmul(a: Tensor, b: Tensor): Tensor {
  if (a.ndim !== 2 || b.ndim !== 2) throw new Error('matmul requires 2D tensors');
  const [M, K1] = a.shape;
  const [K2, N] = b.shape;
  if (K1 !== K2) throw new Error(`matmul inner dims mismatch: ${K1} vs ${K2}`);
  const out = new Float32Array(M * N);

  for (let m = 0; m < M; m++) {
    for (let n = 0; n < N; n++) {
      let sum = 0;
      for (let k = 0; k < K1; k++) {
        sum += a.data[m * K1 + k] * b.data[k * N + n];
      }
      out[m * N + n] = sum;
    }
  }

  const result = new Tensor(out, [M, N], a.requiresGrad || b.requiresGrad);
  if (result.requiresGrad) {
    result.setBackward(() => {
      if (a.grad && result.grad) {
        for (let m = 0; m < M; m++)
          for (let k = 0; k < K1; k++) {
            let sum = 0;
            for (let n = 0; n < N; n++) sum += result.grad[m * N + n] * b.data[k * N + n];
            a.grad[m * K1 + k] += sum;
          }
      }
      if (b.grad && result.grad) {
        for (let k = 0; k < K1; k++)
          for (let n = 0; n < N; n++) {
            let sum = 0;
            for (let m = 0; m < M; m++) sum += a.data[m * K1 + k] * result.grad[m * N + n];
            b.grad[k * N + n] += sum;
          }
      }
    }, [a, b]);
  }
  return result;
}

/**
 * GPU forward for C = A @ B (WGSL compute), same CPU backward as {@link matmul}.
 * Use from the training worker when WebGPU is available; falls back via caller.
 */
export async function matmulWithWebGpu(a: Tensor, b: Tensor, gpu: WebGpuMatmulRuntime): Promise<Tensor> {
  if (a.ndim !== 2 || b.ndim !== 2) throw new Error('matmul requires 2D tensors');
  const [M, K1] = a.shape;
  const [K2, N] = b.shape;
  if (K1 !== K2) throw new Error(`matmul inner dims mismatch: ${K1} vs ${K2}`);
  const raw = await gpu.matmul(a.data, b.data, M, K1, N);
  const result = new Tensor(raw, [M, N], a.requiresGrad || b.requiresGrad);
  if (result.requiresGrad) {
    result.setBackward(() => {
      if (a.grad && result.grad) {
        for (let m = 0; m < M; m++)
          for (let k = 0; k < K1; k++) {
            let sum = 0;
            for (let n = 0; n < N; n++) sum += result.grad[m * N + n] * b.data[k * N + n];
            a.grad[m * K1 + k] += sum;
          }
      }
      if (b.grad && result.grad) {
        for (let k = 0; k < K1; k++)
          for (let n = 0; n < N; n++) {
            let sum = 0;
            for (let m = 0; m < M; m++) sum += a.data[m * K1 + k] * result.grad[m * N + n];
            b.grad[k * N + n] += sum;
          }
      }
    }, [a, b]);
  }
  return result;
}

export function relu(x: Tensor): Tensor {
  const out = new Float32Array(x.size);
  for (let i = 0; i < x.size; i++) out[i] = Math.max(0, x.data[i]);
  const result = new Tensor(out, [...x.shape], x.requiresGrad);
  if (x.requiresGrad) {
    result.setBackward(() => {
      if (x.grad && result.grad) {
        for (let i = 0; i < x.size; i++) x.grad[i] += result.grad[i] * (x.data[i] > 0 ? 1 : 0);
      }
    }, [x]);
  }
  return result;
}

export function sigmoid(x: Tensor): Tensor {
  const out = new Float32Array(x.size);
  for (let i = 0; i < x.size; i++) out[i] = 1 / (1 + Math.exp(-x.data[i]));
  const result = new Tensor(out, [...x.shape], x.requiresGrad);
  if (x.requiresGrad) {
    result.setBackward(() => {
      if (x.grad && result.grad) {
        for (let i = 0; i < x.size; i++) {
          x.grad[i] += result.grad[i] * out[i] * (1 - out[i]);
        }
      }
    }, [x]);
  }
  return result;
}

export function tanh_(x: Tensor): Tensor {
  const out = new Float32Array(x.size);
  for (let i = 0; i < x.size; i++) out[i] = Math.tanh(x.data[i]);
  const result = new Tensor(out, [...x.shape], x.requiresGrad);
  if (x.requiresGrad) {
    result.setBackward(() => {
      if (x.grad && result.grad) {
        for (let i = 0; i < x.size; i++) {
          x.grad[i] += result.grad[i] * (1 - out[i] * out[i]);
        }
      }
    }, [x]);
  }
  return result;
}

export function softmax(x: Tensor): Tensor {
  if (x.ndim !== 2) throw new Error('softmax requires 2D tensor [batch, classes]');
  const [batch, classes] = x.shape;
  const out = new Float32Array(x.size);

  for (let b = 0; b < batch; b++) {
    const offset = b * classes;
    let max = -Infinity;
    for (let c = 0; c < classes; c++) max = Math.max(max, x.data[offset + c]);
    let sum = 0;
    for (let c = 0; c < classes; c++) {
      out[offset + c] = Math.exp(x.data[offset + c] - max);
      sum += out[offset + c];
    }
    for (let c = 0; c < classes; c++) out[offset + c] /= sum;
  }

  const result = new Tensor(out, [...x.shape], x.requiresGrad);
  if (x.requiresGrad) {
    result.setBackward(() => {
      if (x.grad && result.grad) {
        for (let b = 0; b < batch; b++) {
          const offset = b * classes;
          for (let i = 0; i < classes; i++) {
            for (let j = 0; j < classes; j++) {
              const dSoftmax = i === j
                ? out[offset + i] * (1 - out[offset + i])
                : -out[offset + i] * out[offset + j];
              x.grad[offset + i] += result.grad[offset + j] * dSoftmax;
            }
          }
        }
      }
    }, [x]);
  }
  return result;
}

export function dropout(x: Tensor, p: number, training: boolean): Tensor {
  if (!training || p === 0) return x;
  const scale = 1 / (1 - p);
  const out = new Float32Array(x.size);
  const mask = new Float32Array(x.size);
  for (let i = 0; i < x.size; i++) {
    mask[i] = Math.random() > p ? scale : 0;
    out[i] = x.data[i] * mask[i];
  }
  const result = new Tensor(out, [...x.shape], x.requiresGrad);
  if (x.requiresGrad) {
    result.setBackward(() => {
      if (x.grad && result.grad) {
        for (let i = 0; i < x.size; i++) x.grad[i] += result.grad[i] * mask[i];
      }
    }, [x]);
  }
  return result;
}

/** Conv2d forward: input [N,C,H,W], weight [outC,inC,kH,kW] */
export function conv2d(
  input: Tensor, weight: Tensor, bias: Tensor | null,
  stride: number, padding: number
): Tensor {
  const [N, C, H, W] = input.shape;
  const [outC, inC, kH, kW] = weight.shape;
  if (C !== inC) throw new Error(`Conv2d channel mismatch: input ${C} vs kernel ${inC}`);

  const outH = Math.floor((H + 2 * padding - kH) / stride) + 1;
  const outW = Math.floor((W + 2 * padding - kW) / stride) + 1;
  const out = new Float32Array(N * outC * outH * outW);

  for (let n = 0; n < N; n++) {
    for (let oc = 0; oc < outC; oc++) {
      for (let oh = 0; oh < outH; oh++) {
        for (let ow = 0; ow < outW; ow++) {
          let sum = bias ? bias.data[oc] : 0;
          for (let ic = 0; ic < inC; ic++) {
            for (let kh = 0; kh < kH; kh++) {
              for (let kw = 0; kw < kW; kw++) {
                const ih = oh * stride - padding + kh;
                const iw = ow * stride - padding + kw;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                  const inputIdx = n * C * H * W + ic * H * W + ih * W + iw;
                  const weightIdx = oc * inC * kH * kW + ic * kH * kW + kh * kW + kw;
                  sum += input.data[inputIdx] * weight.data[weightIdx];
                }
              }
            }
          }
          out[n * outC * outH * outW + oc * outH * outW + oh * outW + ow] = sum;
        }
      }
    }
  }

  const result = new Tensor(out, [N, outC, outH, outW], input.requiresGrad || weight.requiresGrad);
  if (result.requiresGrad) {
    result.setBackward(() => {
      if (input.grad && result.grad) {
        for (let n = 0; n < N; n++)
          for (let oc = 0; oc < outC; oc++)
            for (let oh = 0; oh < outH; oh++)
              for (let ow = 0; ow < outW; ow++) {
                const gradOut = result.grad[n * outC * outH * outW + oc * outH * outW + oh * outW + ow];
                for (let ic = 0; ic < inC; ic++)
                  for (let kh = 0; kh < kH; kh++)
                    for (let kw = 0; kw < kW; kw++) {
                      const ih = oh * stride - padding + kh;
                      const iw = ow * stride - padding + kw;
                      if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                        input.grad[n * C * H * W + ic * H * W + ih * W + iw] +=
                          gradOut * weight.data[oc * inC * kH * kW + ic * kH * kW + kh * kW + kw];
                      }
                    }
              }
      }
      if (weight.grad && result.grad) {
        for (let n = 0; n < N; n++)
          for (let oc = 0; oc < outC; oc++)
            for (let oh = 0; oh < outH; oh++)
              for (let ow = 0; ow < outW; ow++) {
                const gradOut = result.grad[n * outC * outH * outW + oc * outH * outW + oh * outW + ow];
                for (let ic = 0; ic < inC; ic++)
                  for (let kh = 0; kh < kH; kh++)
                    for (let kw = 0; kw < kW; kw++) {
                      const ih = oh * stride - padding + kh;
                      const iw = ow * stride - padding + kw;
                      if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                        weight.grad[oc * inC * kH * kW + ic * kH * kW + kh * kW + kw] +=
                          gradOut * input.data[n * C * H * W + ic * H * W + ih * W + iw];
                      }
                    }
              }
      }
      if (bias && bias.grad && result.grad) {
        for (let n = 0; n < N; n++)
          for (let oc = 0; oc < outC; oc++)
            for (let oh = 0; oh < outH; oh++)
              for (let ow = 0; ow < outW; ow++) {
                bias.grad[oc] += result.grad[n * outC * outH * outW + oc * outH * outW + oh * outW + ow];
              }
      }
    }, [input, weight, ...(bias ? [bias] : [])]);
  }
  return result;
}

export function maxpool2d(input: Tensor, kernelSize: number, stride: number): Tensor {
  const [N, C, H, W] = input.shape;
  const outH = Math.floor((H - kernelSize) / stride) + 1;
  const outW = Math.floor((W - kernelSize) / stride) + 1;
  const out = new Float32Array(N * C * outH * outW);
  const indices = new Int32Array(N * C * outH * outW);

  for (let n = 0; n < N; n++)
    for (let c = 0; c < C; c++)
      for (let oh = 0; oh < outH; oh++)
        for (let ow = 0; ow < outW; ow++) {
          let maxVal = -Infinity;
          let maxIdx = 0;
          for (let kh = 0; kh < kernelSize; kh++)
            for (let kw = 0; kw < kernelSize; kw++) {
              const ih = oh * stride + kh;
              const iw = ow * stride + kw;
              const idx = n * C * H * W + c * H * W + ih * W + iw;
              if (input.data[idx] > maxVal) {
                maxVal = input.data[idx];
                maxIdx = idx;
              }
            }
          const outIdx = n * C * outH * outW + c * outH * outW + oh * outW + ow;
          out[outIdx] = maxVal;
          indices[outIdx] = maxIdx;
        }

  const result = new Tensor(out, [N, C, outH, outW], input.requiresGrad);
  if (input.requiresGrad) {
    result.setBackward(() => {
      if (input.grad && result.grad) {
        for (let i = 0; i < indices.length; i++) {
          input.grad[indices[i]] += result.grad[i];
        }
      }
    }, [input]);
  }
  return result;
}

export function avgpool2d(input: Tensor, kernelSize: number, stride: number): Tensor {
  const [N, C, H, W] = input.shape;
  const outH = Math.floor((H - kernelSize) / stride) + 1;
  const outW = Math.floor((W - kernelSize) / stride) + 1;
  const out = new Float32Array(N * C * outH * outW);
  const count = kernelSize * kernelSize;

  for (let n = 0; n < N; n++)
    for (let c = 0; c < C; c++)
      for (let oh = 0; oh < outH; oh++)
        for (let ow = 0; ow < outW; ow++) {
          let sum = 0;
          for (let kh = 0; kh < kernelSize; kh++)
            for (let kw = 0; kw < kernelSize; kw++) {
              sum += input.data[n * C * H * W + c * H * W + (oh * stride + kh) * W + (ow * stride + kw)];
            }
          out[n * C * outH * outW + c * outH * outW + oh * outW + ow] = sum / count;
        }

  const result = new Tensor(out, [N, C, outH, outW], input.requiresGrad);
  if (input.requiresGrad) {
    result.setBackward(() => {
      if (input.grad && result.grad) {
        for (let n = 0; n < N; n++)
          for (let c = 0; c < C; c++)
            for (let oh = 0; oh < outH; oh++)
              for (let ow = 0; ow < outW; ow++) {
                const g = result.grad[n * C * outH * outW + c * outH * outW + oh * outW + ow] / count;
                for (let kh = 0; kh < kernelSize; kh++)
                  for (let kw = 0; kw < kernelSize; kw++) {
                    input.grad![n * C * H * W + c * H * W + (oh * stride + kh) * W + (ow * stride + kw)] += g;
                  }
              }
      }
    }, [input]);
  }
  return result;
}

/** Flatten all dims after batch: [N, C, H, W] → [N, C*H*W] */
export function flatten(x: Tensor): Tensor {
  const batch = x.shape[0];
  const features = x.data.length / batch;
  return x.reshape([batch, features]);
}

/** Cross-entropy loss (expects logits, not softmax output) */
export function crossEntropyLoss(logits: Tensor, targets: Int32Array): Tensor {
  const [batch, classes] = logits.shape;
  const probs = new Float32Array(logits.size);
  const loss = new Float32Array(1);

  for (let b = 0; b < batch; b++) {
    const offset = b * classes;
    let max = -Infinity;
    for (let c = 0; c < classes; c++) max = Math.max(max, logits.data[offset + c]);
    let sum = 0;
    for (let c = 0; c < classes; c++) {
      probs[offset + c] = Math.exp(logits.data[offset + c] - max);
      sum += probs[offset + c];
    }
    for (let c = 0; c < classes; c++) probs[offset + c] /= sum;
    loss[0] -= Math.log(probs[offset + targets[b]] + 1e-10);
  }
  loss[0] /= batch;

  const result = new Tensor(loss, [1], logits.requiresGrad);
  if (logits.requiresGrad) {
    result.setBackward(() => {
      if (logits.grad && result.grad) {
        for (let b = 0; b < batch; b++) {
          const offset = b * classes;
          for (let c = 0; c < classes; c++) {
            logits.grad[offset + c] += (probs[offset + c] - (c === targets[b] ? 1 : 0)) / batch;
          }
        }
      }
    }, [logits]);
  }
  return result;
}

export function mseLoss(predictions: Tensor, targets: Tensor): Tensor {
  if (predictions.size !== targets.size) throw new Error('MSE shape mismatch');
  const loss = new Float32Array(1);
  for (let i = 0; i < predictions.size; i++) {
    const diff = predictions.data[i] - targets.data[i];
    loss[0] += diff * diff;
  }
  loss[0] /= predictions.size;

  const result = new Tensor(loss, [1], predictions.requiresGrad);
  if (predictions.requiresGrad) {
    result.setBackward(() => {
      if (predictions.grad && result.grad) {
        const scale = (2.0 / predictions.size) * result.grad[0];
        for (let i = 0; i < predictions.size; i++) {
          predictions.grad[i] += scale * (predictions.data[i] - targets.data[i]);
        }
      }
    }, [predictions]);
  }
  return result;
}
