export interface LinearRegressionTrainOptions {
  epochs: number;
  learningRate: number;
  l2: number;
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
}

export interface LinearRegressionEpochStat {
  epoch: number;
  loss: number;
  accuracy: number;
}

export interface LinearRegressionClassifierModel {
  featureCount: number;
  classCount: number;
  weights: number[];
  bias: number[];
  lossCurve: number[];
}

export interface LinearRegressionRegressorModel {
  featureCount: number;
  weights: number[];
  bias: number;
  lossCurve: number[];
}

function argMax(values: Float32Array): number {
  let bestIndex = 0;
  let bestValue = -Infinity;
  for (let i = 0; i < values.length; i++) {
    if (values[i] > bestValue) {
      bestIndex = i;
      bestValue = values[i];
    }
  }
  return bestIndex;
}

interface VectorOptimizerState {
  type: NonNullable<LinearRegressionTrainOptions['optimizer']>;
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

function scheduledLearningRate(
  baseLearningRate: number,
  epoch: number,
  totalEpochs: number,
  options: LinearRegressionTrainOptions
): number {
  const base = Math.max(1e-6, baseLearningRate);
  const warmup = Math.max(0, Math.floor(options.warmupSteps ?? 0));
  if (warmup > 0 && epoch < warmup) {
    return base * ((epoch + 1) / warmup);
  }
  const scheduler = options.scheduler ?? 'constant';
  if (scheduler === 'linear_decay') {
    const denom = Math.max(1, totalEpochs - 1);
    return Math.max(1e-7, base * (1 - epoch / denom));
  }
  if (scheduler === 'cosine_annealing') {
    const denom = Math.max(1, totalEpochs - 1);
    const cosine = 0.5 * (1 + Math.cos((Math.PI * epoch) / denom));
    return Math.max(1e-7, base * cosine);
  }
  if (scheduler === 'step_lr') {
    const stepSize = Math.max(1, Math.floor(options.stepSize ?? 10));
    const gamma = clamp(options.schedulerGamma ?? 0.5, 0.01, 0.99);
    return Math.max(1e-7, base * Math.pow(gamma, Math.floor(epoch / stepSize)));
  }
  return base;
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

function createOptimizerState(length: number, options: LinearRegressionTrainOptions): VectorOptimizerState {
  return {
    type: options.optimizer ?? 'adamw',
    momentum: clamp(options.momentum ?? 0.9, 0, 0.9999),
    beta1: clamp(options.beta1 ?? 0.9, 0, 0.9999),
    beta2: clamp(options.beta2 ?? 0.999, 0, 0.99999),
    eps: 1e-8,
    weightDecay: Math.max(0, options.weightDecay ?? options.l2 ?? 0),
    step: 0,
    velocity: new Float32Array(length),
    m: new Float32Array(length),
    v: new Float32Array(length),
    u: new Float32Array(length),
  };
}

function clipGradientArray(grad: Float32Array, clipValue: number): void {
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

  // AdamW
  for (let i = 0; i < params.length; i++) {
    const g = grad[i];
    state.m[i] = beta1 * state.m[i] + (1 - beta1) * g;
    state.v[i] = beta2 * state.v[i] + (1 - beta2) * g * g;
    const mHat = state.m[i] / bc1;
    const vHat = state.v[i] / bc2;
    const adamStep = mHat / (Math.sqrt(vHat) + eps);
    params[i] -= lr * (adamStep + state.weightDecay * params[i]);
  }
}

export function predictLinearRegressionClassifier(
  model: LinearRegressionClassifierModel,
  features: number[][]
): number[] {
  const { featureCount, classCount } = model;
  return features.map((sample) => {
    const logits = new Float32Array(classCount);
    for (let classIdx = 0; classIdx < classCount; classIdx++) {
      let sum = model.bias[classIdx] ?? 0;
      for (let featureIdx = 0; featureIdx < featureCount; featureIdx++) {
        sum += sample[featureIdx] * (model.weights[featureIdx * classCount + classIdx] ?? 0);
      }
      logits[classIdx] = sum;
    }
    return argMax(logits);
  });
}

export function linearRegressionFeatureImportance(model: LinearRegressionClassifierModel): number[] {
  const { featureCount, classCount, weights } = model;
  const scores = new Array(featureCount).fill(0);
  for (let featureIdx = 0; featureIdx < featureCount; featureIdx++) {
    let sum = 0;
    for (let classIdx = 0; classIdx < classCount; classIdx++) {
      sum += Math.abs(weights[featureIdx * classCount + classIdx] ?? 0);
    }
    scores[featureIdx] = sum;
  }
  return scores;
}

export function linearRegressorFeatureImportance(model: LinearRegressionRegressorModel): number[] {
  return model.weights.map((value) => Math.abs(value));
}

export function predictLinearRegressionRegressor(
  model: LinearRegressionRegressorModel,
  features: number[][]
): number[] {
  return features.map((sample) => {
    let value = model.bias;
    for (let i = 0; i < model.featureCount; i++) {
      value += (sample[i] ?? 0) * (model.weights[i] ?? 0);
    }
    return value;
  });
}

export function trainLinearRegressionRegressor(
  features: number[][],
  targets: number[],
  options: LinearRegressionTrainOptions,
  hooks?: {
    shouldStop?: () => boolean;
    onEpoch?: (stats: LinearRegressionEpochStat) => void;
  }
): LinearRegressionRegressorModel {
  if (!features.length) throw new Error('Cannot train linear regression regressor on empty dataset.');
  if (features.length !== targets.length) throw new Error('Regression features and targets length mismatch.');

  const featureCount = features[0].length;
  const epochs = Math.max(1, Math.floor(options.epochs));
  const learningRate = Math.max(1e-6, options.learningRate);
  const l2 = Math.max(0, options.weightDecay ?? options.l2);
  const sampleCount = features.length;
  const batchSize = clamp(Math.floor(options.batchSize ?? sampleCount), 1, sampleCount);
  const shuffle = options.shuffle !== false;
  const patience = Math.max(0, Math.floor(options.earlyStoppingPatience ?? 0));
  const gradientClip = Math.max(0, options.gradientClipping ?? 0);

  const weights = new Float32Array(featureCount);
  const bias = new Float32Array(1);
  const lossCurve: number[] = [];
  const weightOptimizer = createOptimizerState(featureCount, { ...options, weightDecay: l2 });
  const biasOptimizer = createOptimizerState(1, { ...options, weightDecay: 0 });
  let bestLoss = Number.POSITIVE_INFINITY;
  let staleEpochs = 0;

  for (let epoch = 0; epoch < epochs; epoch++) {
    if (hooks?.shouldStop?.()) break;
    let lossSum = 0;
    const indices = maybeShuffleIndices(sampleCount, shuffle);
    const lrEpoch = scheduledLearningRate(learningRate, epoch, epochs, options);

    for (let batchStart = 0; batchStart < sampleCount; batchStart += batchSize) {
      if (hooks?.shouldStop?.()) break;
      const batchEnd = Math.min(sampleCount, batchStart + batchSize);
      const batchCount = Math.max(1, batchEnd - batchStart);
      const gradW = new Float32Array(featureCount);
      const gradB = new Float32Array(1);

      for (let ptr = batchStart; ptr < batchEnd; ptr++) {
        const rowIdx = indices[ptr];
        const sample = features[rowIdx];
        const target = targets[rowIdx] ?? 0;
        let prediction = bias[0];
        for (let f = 0; f < featureCount; f++) {
          prediction += (sample[f] ?? 0) * weights[f];
        }
        const error = prediction - target;
        lossSum += error * error;
        gradB[0] += 2 * error;
        for (let f = 0; f < featureCount; f++) {
          gradW[f] += 2 * error * (sample[f] ?? 0);
        }
      }

      const invBatch = 1 / batchCount;
      for (let f = 0; f < featureCount; f++) {
        gradW[f] *= invBatch;
      }
      gradB[0] *= invBatch;
      clipGradientArray(gradW, gradientClip);
      clipGradientArray(gradB, gradientClip);
      const coupledDecay = weightOptimizer.type !== 'adamw';
      applyVectorOptimizer(weights, gradW, lrEpoch, weightOptimizer, coupledDecay);
      applyVectorOptimizer(bias, gradB, lrEpoch, biasOptimizer, false);
    }
    const mse = lossSum / sampleCount;
    const rmse = Math.sqrt(mse);
    lossCurve.push(mse);
    hooks?.onEpoch?.({
      epoch,
      loss: mse,
      accuracy: 1 / (1 + rmse),
    });

    if (patience > 0) {
      if (mse + 1e-9 < bestLoss) {
        bestLoss = mse;
        staleEpochs = 0;
      } else {
        staleEpochs += 1;
        if (staleEpochs >= patience) break;
      }
    }
  }

  return {
    featureCount,
    weights: Array.from(weights),
    bias: bias[0],
    lossCurve,
  };
}

export function trainLinearRegressionClassifier(
  features: number[][],
  labels: number[],
  classCount: number,
  options: LinearRegressionTrainOptions,
  hooks?: {
    shouldStop?: () => boolean;
    onEpoch?: (stats: LinearRegressionEpochStat) => void;
  }
): LinearRegressionClassifierModel {
  if (!features.length) throw new Error('Cannot train linear regression on empty dataset.');
  const featureCount = features[0].length;
  const epochs = Math.max(1, Math.floor(options.epochs));
  const learningRate = Math.max(1e-6, options.learningRate);
  const l2 = Math.max(0, options.weightDecay ?? options.l2);
  const sampleCount = features.length;
  const batchSize = clamp(Math.floor(options.batchSize ?? sampleCount), 1, sampleCount);
  const shuffle = options.shuffle !== false;
  const patience = Math.max(0, Math.floor(options.earlyStoppingPatience ?? 0));
  const gradientClip = Math.max(0, options.gradientClipping ?? 0);

  const weights = new Float32Array(featureCount * classCount);
  const bias = new Float32Array(classCount);
  const logits = new Float32Array(classCount);
  const targets = new Float32Array(classCount);
  const lossCurve: number[] = [];
  const weightOptimizer = createOptimizerState(featureCount * classCount, { ...options, weightDecay: l2 });
  const biasOptimizer = createOptimizerState(classCount, { ...options, weightDecay: 0 });
  let bestLoss = Number.POSITIVE_INFINITY;
  let staleEpochs = 0;

  for (let epoch = 0; epoch < epochs; epoch++) {
    if (hooks?.shouldStop?.()) break;
    let lossSum = 0;
    let correct = 0;
    const indices = maybeShuffleIndices(sampleCount, shuffle);
    const lrEpoch = scheduledLearningRate(learningRate, epoch, epochs, options);

    for (let batchStart = 0; batchStart < sampleCount; batchStart += batchSize) {
      if (hooks?.shouldStop?.()) break;
      const batchEnd = Math.min(sampleCount, batchStart + batchSize);
      const batchCount = Math.max(1, batchEnd - batchStart);
      const gradW = new Float32Array(featureCount * classCount);
      const gradB = new Float32Array(classCount);

      for (let ptr = batchStart; ptr < batchEnd; ptr++) {
        const sampleIdx = indices[ptr];
        const sample = features[sampleIdx];
        const label = labels[sampleIdx] ?? 0;

        for (let classIdx = 0; classIdx < classCount; classIdx++) {
          let value = bias[classIdx];
          for (let featureIdx = 0; featureIdx < featureCount; featureIdx++) {
            value += sample[featureIdx] * weights[featureIdx * classCount + classIdx];
          }
          logits[classIdx] = value;
          targets[classIdx] = classIdx === label ? 1 : 0;
        }

        if (argMax(logits) === label) correct++;

        for (let classIdx = 0; classIdx < classCount; classIdx++) {
          const error = logits[classIdx] - targets[classIdx];
          lossSum += error * error;
          gradB[classIdx] += (2 * error) / classCount;
          for (let featureIdx = 0; featureIdx < featureCount; featureIdx++) {
            gradW[featureIdx * classCount + classIdx] += (2 * error * sample[featureIdx]) / classCount;
          }
        }
      }

      const invBatch = 1 / batchCount;
      for (let i = 0; i < gradW.length; i++) gradW[i] *= invBatch;
      for (let c = 0; c < gradB.length; c++) gradB[c] *= invBatch;
      clipGradientArray(gradW, gradientClip);
      clipGradientArray(gradB, gradientClip);

      const coupledDecay = weightOptimizer.type !== 'adamw';
      applyVectorOptimizer(weights, gradW, lrEpoch, weightOptimizer, coupledDecay);
      applyVectorOptimizer(bias, gradB, lrEpoch, biasOptimizer, false);
    }

    const loss = lossSum / sampleCount;
    const accuracy = correct / sampleCount;
    lossCurve.push(loss);
    hooks?.onEpoch?.({
      epoch,
      loss,
      accuracy,
    });

    if (patience > 0) {
      if (loss + 1e-9 < bestLoss) {
        bestLoss = loss;
        staleEpochs = 0;
      } else {
        staleEpochs += 1;
        if (staleEpochs >= patience) break;
      }
    }
  }

  return {
    featureCount,
    classCount,
    weights: Array.from(weights),
    bias: Array.from(bias),
    lossCurve,
  };
}

