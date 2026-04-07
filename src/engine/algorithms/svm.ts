export type SvmKernel = 'linear' | 'rbf' | 'poly';

export interface SvmTrainOptions {
  epochs: number;
  learningRate: number;
  regularization: number;
  kernel: SvmKernel;
  batchSize?: number;
  shuffle?: boolean;
  earlyStoppingPatience?: number;
  optimizer?: 'adamw' | 'sgd_momentum' | 'adam' | 'adamax';
  momentum?: number;
  beta1?: number;
  beta2?: number;
  scheduler?: 'constant' | 'linear_decay' | 'cosine_annealing' | 'step_lr';
  warmupSteps?: number;
  stepSize?: number;
  schedulerGamma?: number;
  gradientClipping?: number;
  weightDecay?: number;
  shouldStop?: () => boolean;
}

export interface SvmEpochStat {
  epoch: number;
  loss: number;
  accuracy: number;
}

export interface SvmModel {
  featureCount: number;
  classCount: number;
  kernel: SvmKernel;
  weights: number[][];
  bias: number[];
  lossCurve: SvmEpochStat[];
}

function mapKernelFeature(value: number, kernel: SvmKernel): number {
  if (kernel === 'poly') return value * value;
  if (kernel === 'rbf') return Math.exp(-0.5 * value * value);
  return value;
}

interface VectorOptimizerState {
  type: NonNullable<SvmTrainOptions['optimizer']>;
  momentum: number;
  beta1: number;
  beta2: number;
  eps: number;
  weightDecay: number;
  step: number;
  velocity: Float32Array;
  m: Float32Array;
  v: Float32Array;
  u: Float32Array;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function maybeShuffleIndices(count: number, shuffle: boolean): number[] {
  const indices = Array.from({ length: count }, (_, idx) => idx);
  if (!shuffle) return indices;
  for (let i = indices.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }
  return indices;
}

function scheduledLearningRate(baseLearningRate: number, epoch: number, totalEpochs: number, options: SvmTrainOptions): number {
  const base = Math.max(1e-6, baseLearningRate);
  const warmup = Math.max(0, Math.floor(options.warmupSteps ?? 0));
  if (warmup > 0 && epoch < warmup) return base * ((epoch + 1) / warmup);
  const scheduler = options.scheduler ?? 'constant';
  if (scheduler === 'linear_decay') {
    const denom = Math.max(1, totalEpochs - 1);
    return Math.max(1e-7, base * (1 - epoch / denom));
  }
  if (scheduler === 'cosine_annealing') {
    const denom = Math.max(1, totalEpochs - 1);
    return Math.max(1e-7, base * (0.5 * (1 + Math.cos((Math.PI * epoch) / denom))));
  }
  if (scheduler === 'step_lr') {
    const stepSize = Math.max(1, Math.floor(options.stepSize ?? 10));
    const gamma = clamp(options.schedulerGamma ?? 0.5, 0.01, 0.99);
    return Math.max(1e-7, base * Math.pow(gamma, Math.floor(epoch / stepSize)));
  }
  return base;
}

function createOptimizerState(length: number, options: SvmTrainOptions): VectorOptimizerState {
  return {
    type: options.optimizer ?? 'adamw',
    momentum: clamp(options.momentum ?? 0.9, 0, 0.9999),
    beta1: clamp(options.beta1 ?? 0.9, 0, 0.9999),
    beta2: clamp(options.beta2 ?? 0.999, 0, 0.99999),
    eps: 1e-8,
    weightDecay: Math.max(0, options.weightDecay ?? options.regularization ?? 0),
    step: 0,
    velocity: new Float32Array(length),
    m: new Float32Array(length),
    v: new Float32Array(length),
    u: new Float32Array(length),
  };
}

function clipGradient(grad: Float32Array, clipValue: number): void {
  if (clipValue <= 0) return;
  const maxAbs = Math.abs(clipValue);
  for (let i = 0; i < grad.length; i++) {
    if (grad[i] > maxAbs) grad[i] = maxAbs;
    else if (grad[i] < -maxAbs) grad[i] = -maxAbs;
  }
}

function applyVectorOptimizer(
  params: Float32Array,
  grad: Float32Array,
  learningRate: number,
  state: VectorOptimizerState,
  applyCoupledDecay: boolean
): void {
  state.step += 1;
  const lr = Math.max(1e-8, learningRate);
  const beta1 = state.beta1;
  const beta2 = state.beta2;
  const eps = state.eps;
  const bc1 = 1 - Math.pow(beta1, state.step);
  const bc2 = 1 - Math.pow(beta2, state.step);

  if (state.type === 'sgd_momentum') {
    for (let i = 0; i < params.length; i++) {
      const decayGrad = applyCoupledDecay ? state.weightDecay * params[i] : 0;
      const g = grad[i] + decayGrad;
      state.velocity[i] = state.momentum * state.velocity[i] + g;
      params[i] -= lr * state.velocity[i];
    }
    return;
  }

  if (state.type === 'adam') {
    for (let i = 0; i < params.length; i++) {
      const decayGrad = applyCoupledDecay ? state.weightDecay * params[i] : 0;
      const g = grad[i] + decayGrad;
      state.m[i] = beta1 * state.m[i] + (1 - beta1) * g;
      state.v[i] = beta2 * state.v[i] + (1 - beta2) * g * g;
      const mHat = state.m[i] / bc1;
      const vHat = state.v[i] / bc2;
      params[i] -= lr * mHat / (Math.sqrt(vHat) + eps);
    }
    return;
  }

  if (state.type === 'adamax') {
    const lrT = lr / bc1;
    for (let i = 0; i < params.length; i++) {
      const decayGrad = applyCoupledDecay ? state.weightDecay * params[i] : 0;
      const g = grad[i] + decayGrad;
      state.m[i] = beta1 * state.m[i] + (1 - beta1) * g;
      state.u[i] = Math.max(beta2 * state.u[i], Math.abs(g));
      params[i] -= (lrT * state.m[i]) / (state.u[i] + eps);
    }
    return;
  }

  for (let i = 0; i < params.length; i++) {
    const g = grad[i];
    state.m[i] = beta1 * state.m[i] + (1 - beta1) * g;
    state.v[i] = beta2 * state.v[i] + (1 - beta2) * g * g;
    const mHat = state.m[i] / bc1;
    const vHat = state.v[i] / bc2;
    params[i] -= lr * (mHat / (Math.sqrt(vHat) + eps) + state.weightDecay * params[i]);
  }
}

function mapKernelVector(input: number[], kernel: SvmKernel): number[] {
  if (kernel === 'linear') return input;
  return input.map((value) => mapKernelFeature(value, kernel));
}

function argMax(values: number[]): number {
  let bestIdx = 0;
  let bestVal = -Infinity;
  for (let i = 0; i < values.length; i++) {
    if (values[i] > bestVal) {
      bestVal = values[i];
      bestIdx = i;
    }
  }
  return bestIdx;
}

function scoreClass(weights: number[], bias: number, sample: number[]): number {
  let score = bias;
  for (let i = 0; i < sample.length; i++) {
    score += weights[i] * sample[i];
  }
  return score;
}

export function predictSvmClassifier(model: SvmModel, samples: number[][]): number[] {
  return samples.map((sample) => {
    const mapped = mapKernelVector(sample, model.kernel);
    const scores = new Array(model.classCount).fill(0);
    for (let c = 0; c < model.classCount; c++) {
      scores[c] = scoreClass(model.weights[c], model.bias[c], mapped);
    }
    return argMax(scores);
  });
}

export function svmFeatureImportance(model: SvmModel): number[] {
  const importance = new Array(model.featureCount).fill(0);
  for (let f = 0; f < model.featureCount; f++) {
    let sum = 0;
    for (let c = 0; c < model.classCount; c++) {
      sum += Math.abs(model.weights[c][f] ?? 0);
    }
    importance[f] = sum / model.classCount;
  }
  return importance;
}

export function trainSvmClassifier(
  features: number[][],
  labels: number[],
  classCount: number,
  options: SvmTrainOptions,
  onEpoch?: (stat: SvmEpochStat) => void
): SvmModel {
  if (!features.length) throw new Error('Cannot train SVM on empty dataset.');
  const featureCount = features[0].length;
  const epochs = Math.max(1, Math.floor(options.epochs || 1));
  const learningRate = Math.max(1e-6, options.learningRate || 0.001);
  const regularization = Math.max(1e-7, options.regularization || 0.0001);
  const kernel = options.kernel ?? 'linear';
  const batchSize = clamp(Math.floor(options.batchSize ?? features.length), 1, features.length);
  const shuffle = options.shuffle !== false;
  const gradientClipping = Math.max(0, options.gradientClipping ?? 0);
  const patience = Math.max(0, Math.floor(options.earlyStoppingPatience ?? 0));

  const mappedFeatures = features.map((sample) => mapKernelVector(sample, kernel));
  const model: SvmModel = {
    featureCount,
    classCount,
    kernel,
    weights: Array.from({ length: classCount }, () => new Array(featureCount).fill(0)),
    bias: new Array(classCount).fill(0),
    lossCurve: [],
  };

  const sampleCount = mappedFeatures.length;
  const flatWeights = new Float32Array(featureCount * classCount);
  const bias = new Float32Array(classCount);
  const weightState = createOptimizerState(flatWeights.length, { ...options, weightDecay: regularization });
  const biasState = createOptimizerState(classCount, { ...options, weightDecay: 0 });
  let bestLoss = Number.POSITIVE_INFINITY;
  let staleEpochs = 0;

  for (let epoch = 0; epoch < epochs; epoch++) {
    if (options.shouldStop?.()) break;
    let marginLoss = 0;
    let correct = 0;
    const indices = maybeShuffleIndices(sampleCount, shuffle);
    const lrEpoch = scheduledLearningRate(learningRate, epoch, epochs, options);

    for (let batchStart = 0; batchStart < sampleCount; batchStart += batchSize) {
      if (options.shouldStop?.()) break;
      const batchEnd = Math.min(sampleCount, batchStart + batchSize);
      const count = Math.max(1, batchEnd - batchStart);
      const gradW = new Float32Array(flatWeights.length);
      const gradB = new Float32Array(classCount);

      for (let ptr = batchStart; ptr < batchEnd; ptr++) {
        const idx = indices[ptr];
        const x = mappedFeatures[idx];
        const y = labels[idx] ?? 0;

        const scores = new Array(classCount).fill(0);
        for (let c = 0; c < classCount; c++) {
          let score = bias[c];
          for (let f = 0; f < featureCount; f++) {
            score += flatWeights[c * featureCount + f] * x[f];
          }
          scores[c] = score;
        }
        if (argMax(scores) === y) correct++;

        for (let c = 0; c < classCount; c++) {
          const target = c === y ? 1 : -1;
          const margin = target * scores[c];
          const violates = margin < 1;
          if (violates) marginLoss += 1 - margin;
          const gradMultiplier = violates ? -target : 0;
          for (let f = 0; f < featureCount; f++) {
            gradW[c * featureCount + f] += gradMultiplier * x[f];
          }
          gradB[c] += gradMultiplier;
        }
      }

      const invBatch = 1 / count;
      for (let i = 0; i < gradW.length; i++) gradW[i] *= invBatch;
      for (let i = 0; i < gradB.length; i++) gradB[i] *= invBatch;
      clipGradient(gradW, gradientClipping);
      clipGradient(gradB, gradientClipping);

      const coupledDecay = weightState.type !== 'adamw';
      applyVectorOptimizer(flatWeights, gradW, lrEpoch, weightState, coupledDecay);
      applyVectorOptimizer(bias, gradB, lrEpoch, biasState, false);
    }

    for (let c = 0; c < classCount; c++) {
      for (let f = 0; f < featureCount; f++) {
        model.weights[c][f] = flatWeights[c * featureCount + f];
      }
      model.bias[c] = bias[c];
    }

    const stat: SvmEpochStat = {
      epoch: epoch + 1,
      loss: marginLoss / Math.max(1, sampleCount * classCount),
      accuracy: correct / Math.max(1, sampleCount),
    };
    model.lossCurve.push(stat);
    onEpoch?.(stat);

    if (patience > 0) {
      if (stat.loss + 1e-9 < bestLoss) {
        bestLoss = stat.loss;
        staleEpochs = 0;
      } else {
        staleEpochs += 1;
        if (staleEpochs >= patience) break;
      }
    }
  }

  return model;
}
