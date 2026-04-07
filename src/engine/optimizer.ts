import { Tensor } from './tensor';

export interface Optimizer {
  step(): void;
  zeroGrad(): void;
  setLearningRate(lr: number): void;
}

export class SGD implements Optimizer {
  params: Tensor[];
  lr: number;
  momentum: number;
  velocity: Float32Array[];

  constructor(params: Tensor[], lr: number, momentum = 0) {
    this.params = params;
    this.lr = lr;
    this.momentum = Math.max(0, Math.min(0.9999, momentum));
    this.velocity = params.map((p) => new Float32Array(p.data.length));
  }

  step() {
    for (let pi = 0; pi < this.params.length; pi++) {
      const p = this.params[pi];
      if (p.grad) {
        const v = this.velocity[pi];
        for (let i = 0; i < p.data.length; i++) {
          if (this.momentum > 0) {
            v[i] = this.momentum * v[i] + p.grad[i];
            p.data[i] -= this.lr * v[i];
          } else {
            p.data[i] -= this.lr * p.grad[i];
          }
        }
      }
    }
  }

  zeroGrad() {
    for (const p of this.params) p.zeroGrad();
  }

  setLearningRate(lr: number) {
    this.lr = Math.max(1e-8, lr);
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

  setLearningRate(lr: number) {
    this.lr = Math.max(1e-8, lr);
  }
}

export class AdamW implements Optimizer {
  params: Tensor[];
  lr: number;
  beta1: number;
  beta2: number;
  eps: number;
  weightDecay: number;
  t: number;
  m: Float32Array[];
  v: Float32Array[];

  constructor(params: Tensor[], lr = 0.001, beta1 = 0.9, beta2 = 0.999, weightDecay = 0.01, eps = 1e-8) {
    this.params = params;
    this.lr = lr;
    this.beta1 = beta1;
    this.beta2 = beta2;
    this.weightDecay = Math.max(0, weightDecay);
    this.eps = eps;
    this.t = 0;
    this.m = params.map((p) => new Float32Array(p.data.length));
    this.v = params.map((p) => new Float32Array(p.data.length));
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
        const g = p.grad[i];
        m[i] = this.beta1 * m[i] + (1 - this.beta1) * g;
        v[i] = this.beta2 * v[i] + (1 - this.beta2) * g * g;
        const mHat = m[i] / bc1;
        const vHat = v[i] / bc2;
        const adamStep = mHat / (Math.sqrt(vHat) + this.eps);
        const decayStep = this.weightDecay * p.data[i];
        p.data[i] -= this.lr * (adamStep + decayStep);
      }
    }
  }

  zeroGrad() {
    for (const p of this.params) p.zeroGrad();
  }

  setLearningRate(lr: number) {
    this.lr = Math.max(1e-8, lr);
  }
}

export class Adamax implements Optimizer {
  params: Tensor[];
  lr: number;
  beta1: number;
  beta2: number;
  eps: number;
  t: number;
  m: Float32Array[];
  u: Float32Array[];

  constructor(params: Tensor[], lr = 0.002, beta1 = 0.9, beta2 = 0.999, eps = 1e-8) {
    this.params = params;
    this.lr = lr;
    this.beta1 = beta1;
    this.beta2 = beta2;
    this.eps = eps;
    this.t = 0;
    this.m = params.map((p) => new Float32Array(p.data.length));
    this.u = params.map((p) => new Float32Array(p.data.length));
  }

  step() {
    this.t++;
    const lrT = this.lr / (1 - Math.pow(this.beta1, this.t));
    for (let pi = 0; pi < this.params.length; pi++) {
      const p = this.params[pi];
      if (!p.grad) continue;
      const m = this.m[pi];
      const u = this.u[pi];
      for (let i = 0; i < p.data.length; i++) {
        const g = p.grad[i];
        m[i] = this.beta1 * m[i] + (1 - this.beta1) * g;
        u[i] = Math.max(this.beta2 * u[i], Math.abs(g));
        p.data[i] -= (lrT * m[i]) / (u[i] + this.eps);
      }
    }
  }

  zeroGrad() {
    for (const p of this.params) p.zeroGrad();
  }

  setLearningRate(lr: number) {
    this.lr = Math.max(1e-8, lr);
  }
}
