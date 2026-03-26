import { Tensor } from './tensor';

export interface Optimizer {
  step(): void;
  zeroGrad(): void;
}

export class SGD implements Optimizer {
  params: Tensor[];
  lr: number;

  constructor(params: Tensor[], lr: number) {
    this.params = params;
    this.lr = lr;
  }

  step() {
    for (const p of this.params) {
      if (p.grad) {
        for (let i = 0; i < p.data.length; i++) {
          p.data[i] -= this.lr * p.grad[i];
        }
      }
    }
  }

  zeroGrad() {
    for (const p of this.params) p.zeroGrad();
  }
}

export class Adam implements Optimizer {
  params: Tensor[];
  lr: number;
  beta1: number;
  beta2: number;
  eps: number;
  t: number;
  m: Float32Array[];
  v: Float32Array[];

  constructor(params: Tensor[], lr = 0.001, beta1 = 0.9, beta2 = 0.999, eps = 1e-8) {
    this.params = params;
    this.lr = lr;
    this.beta1 = beta1;
    this.beta2 = beta2;
    this.eps = eps;
    this.t = 0;
    this.m = params.map(p => new Float32Array(p.data.length));
    this.v = params.map(p => new Float32Array(p.data.length));
  }

  step() {
    this.t++;
    const bc1 = 1 - Math.pow(this.beta1, this.t);
    const bc2 = 1 - Math.pow(this.beta2, this.t);

    for (let pi = 0; pi < this.params.length; pi++) {
      const p = this.params[pi];
      if (!p.grad) continue;
      const m = this.m[pi];
      const v = this.v[pi];

      for (let i = 0; i < p.data.length; i++) {
        m[i] = this.beta1 * m[i] + (1 - this.beta1) * p.grad[i];
        v[i] = this.beta2 * v[i] + (1 - this.beta2) * p.grad[i] * p.grad[i];
        const mHat = m[i] / bc1;
        const vHat = v[i] / bc2;
        p.data[i] -= this.lr * mHat / (Math.sqrt(vHat) + this.eps);
      }
    }
  }

  zeroGrad() {
    for (const p of this.params) p.zeroGrad();
  }
}
