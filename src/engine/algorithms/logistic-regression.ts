export interface LogisticRegressionTrainOptions {
  epochs: number;
  learningRate: number;
  l2?: number;
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

export interface LogisticRegressionEpochStat {
  epoch: number;
  loss: number;
  accuracy: number;
}

export interface LogisticRegressionModel {
  featureCount: number;
  classCount: number;
  weights: number[][];
  bias: number[];
  lossCurve: LogisticRegressionEpochStat[];
}

function argMax(values: number[]): number {
  let bestIndex = 0;
  let bestValue = -Infinity;
  for (let i = 0; i < values.length; i++) {
    if (values[i] > bestValue) {
      bestValue = values[i];
      bestIndex = i;
    }
  }
  return bestIndex;
}

interface VectorOptimizerState {
  type: NonNullable<LogisticRegressionTrainOptions['optimizer']>;
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

function scheduledLearningRate(
  baseLearningRate: number,
  epoch: number,
  totalEpochs: number,
  options: LogisticRegressionTrainOptions
): number {
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

function createOptimizerState(length: number, options: LogisticRegressionTrainOptions): VectorOptimizerState {
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

function softmax(logits: number[]): number[] {
  let max = -Infinity;
  for (const value of logits) max = Math.max(max, value);
  const exps = logits.map((value) => Math.exp(value - max));
  const sum = exps.reduce((acc, value) => acc + value, 0) || 1;
  return exps.map((value) => value / sum);
}

function computeLogits(model: LogisticRegressionModel, sample: number[]): number[] {
  const logits = new Array(model.classCount).fill(0);
  for (let c = 0; c < model.classCount; c++) {
    let score = model.bias[c];
    for (let f = 0; f < model.featureCount; f++) score += model.weights[c][f] * sample[f];
    logits[c] = score;
  }
  return logits;
}

export function predictLogisticRegressionClassifier(
  model: LogisticRegressionModel,
  features: number[][]
): number[] {
  return features.map((sample) => {
    const logits = computeLogits(model, sample);
    return argMax(logits);
  });
}

export function logisticRegressionFeatureImportance(model: LogisticRegressionModel): number[] {
  const scores = new Array(model.featureCount).fill(0);
  for (let f = 0; f < model.featureCount; f++) {
    let sum = 0;
    for (let c = 0; c < model.classCount; c++) sum += Math.abs(model.weights[c][f]);
    scores[f] = sum / model.classCount;
  }
  return scores;
}

export function trainLogisticRegressionClassifier(
  features: number[][],
  labels: number[],
  classCount: number,
  options: LogisticRegressionTrainOptions,
  onEpoch?: (stat: LogisticRegressionEpochStat) => void
): LogisticRegressionModel {
  if (!features.length) throw new Error('Cannot train Logistic Regression on empty dataset.');
  const featureCount = features[0].length;
  const model: LogisticRegressionModel = {
    featureCount,
    classCount,
    weights: Array.from({ length: classCount }, () =>
      Array.from({ length: featureCount }, () => (Math.random() - 0.5) * 0.01)
    ),
    bias: new Array(classCount).fill(0),
    lossCurve: [],
  };

  const totalSamples = features.length;
  const epochs = Math.max(1, options.epochs);
  const learningRate = Math.max(1e-6, options.learningRate);
  const l2 = Math.max(0, options.weightDecay ?? options.l2 ?? 0);
  const batchSize = clamp(Math.floor(options.batchSize ?? totalSamples), 1, totalSamples);
  const shuffle = options.shuffle !== false;
  const gradientClipping = Math.max(0, options.gradientClipping ?? 0);
  const patience = Math.max(0, Math.floor(options.earlyStoppingPatience ?? 0));
  const weightState = createOptimizerState(classCount * featureCount, { ...options, weightDecay: l2 });
  const biasState = createOptimizerState(classCount, { ...options, weightDecay: 0 });
  const flatWeights = new Float32Array(classCount * featureCount);
  for (let c = 0; c < classCount; c++) {
    for (let f = 0; f < featureCount; f++) {
      flatWeights[c * featureCount + f] = model.weights[c][f] ?? 0;
    }
  }
  const bias = Float32Array.from(model.bias);
  let bestLoss = Number.POSITIVE_INFINITY;
  let staleEpochs = 0;

  for (let epoch = 0; epoch < epochs; epoch++) {
    if (options.shouldStop?.()) break;
    let loss = 0;
    let correct = 0;
    const indices = maybeShuffleIndices(totalSamples, shuffle);
    const lrEpoch = scheduledLearningRate(learningRate, epoch, epochs, options);

    for (let batchStart = 0; batchStart < totalSamples; batchStart += batchSize) {
      if (options.shouldStop?.()) break;
      const batchEnd = Math.min(totalSamples, batchStart + batchSize);
      const count = Math.max(1, batchEnd - batchStart);
      const gradW = new Float32Array(classCount * featureCount);
      const gradB = new Float32Array(classCount);

      for (let ptr = batchStart; ptr < batchEnd; ptr++) {
        const i = indices[ptr];
        const sample = features[i];
        const target = labels[i];
        const logits = new Array(classCount).fill(0);
        for (let c = 0; c < classCount; c++) {
          let score = bias[c];
          for (let f = 0; f < featureCount; f++) {
            score += flatWeights[c * featureCount + f] * sample[f];
          }
          logits[c] = score;
        }
        const probs = softmax(logits);
        const prediction = argMax(probs);
        if (prediction === target) correct++;
        loss += -Math.log((probs[target] ?? 1e-10) + 1e-10);

        for (let c = 0; c < classCount; c++) {
          const error = probs[c] - (c === target ? 1 : 0);
          gradB[c] += error;
          for (let f = 0; f < featureCount; f++) {
            gradW[c * featureCount + f] += error * sample[f];
          }
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
    if (options.shouldStop?.()) break;

    for (let c = 0; c < classCount; c++) {
      for (let f = 0; f < featureCount; f++) {
        model.weights[c][f] = flatWeights[c * featureCount + f];
      }
      model.bias[c] = bias[c];
    }

    const stat: LogisticRegressionEpochStat = {
      epoch: epoch + 1,
      loss: loss / totalSamples,
      accuracy: correct / totalSamples,
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
