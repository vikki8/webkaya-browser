export type RecurrentVariant = 'rnn' | 'lstm' | 'gru';

export interface RecurrentTrainOptions {
  variant: RecurrentVariant;
  epochs: number;
  learningRate: number;
  hiddenSize: number;
  inputSize?: number;
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

export interface RecurrentEpochStat {
  epoch: number;
  loss: number;
  accuracy: number;
}

export interface RecurrentClassifierModel {
  variant: RecurrentVariant;
  featureCount: number;
  classCount: number;
  inputSize: number;
  sequenceLength: number;
  hiddenSize: number;
  params: Record<string, number[]>;
  outputWeights: number[];
  outputBias: number[];
  lossCurve: RecurrentEpochStat[];
  featureImportance: number[];
}

type NumericDict = Record<string, Float32Array>;

interface VectorOptimizerState {
  type: NonNullable<RecurrentTrainOptions['optimizer']>;
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
  options: RecurrentTrainOptions
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

function createOptimizerState(length: number, options: RecurrentTrainOptions): VectorOptimizerState {
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

function argMax(values: number[]): number {
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

function sigmoid(value: number): number {
  return 1 / (1 + Math.exp(-value));
}

function softmax(logits: number[]): number[] {
  let max = -Infinity;
  for (const value of logits) max = Math.max(max, value);
  const exps = logits.map((value) => Math.exp(value - max));
  const sum = exps.reduce((acc, value) => acc + value, 0) || 1;
  return exps.map((value) => value / sum);
}

function initMatrix(rows: number, cols: number, scale = 0.08): Float32Array {
  const values = new Float32Array(rows * cols);
  for (let i = 0; i < values.length; i++) {
    values[i] = (Math.random() - 0.5) * scale;
  }
  return values;
}

function initVector(size: number, value = 0): Float32Array {
  return new Float32Array(size).fill(value);
}

function initializeParams(variant: RecurrentVariant, inputSize: number, hiddenSize: number): NumericDict {
  if (variant === 'rnn') {
    return {
      wx: initMatrix(inputSize, hiddenSize, 0.1),
      wh: initMatrix(hiddenSize, hiddenSize, 0.06),
      b: initVector(hiddenSize, 0),
    };
  }
  if (variant === 'lstm') {
    return {
      wi: initMatrix(inputSize, hiddenSize, 0.08),
      wf: initMatrix(inputSize, hiddenSize, 0.08),
      wo: initMatrix(inputSize, hiddenSize, 0.08),
      wg: initMatrix(inputSize, hiddenSize, 0.08),
      ui: initMatrix(hiddenSize, hiddenSize, 0.05),
      uf: initMatrix(hiddenSize, hiddenSize, 0.05),
      uo: initMatrix(hiddenSize, hiddenSize, 0.05),
      ug: initMatrix(hiddenSize, hiddenSize, 0.05),
      bi: initVector(hiddenSize, 0),
      bf: initVector(hiddenSize, 0.5),
      bo: initVector(hiddenSize, 0),
      bg: initVector(hiddenSize, 0),
    };
  }
  return {
    wz: initMatrix(inputSize, hiddenSize, 0.08),
    wr: initMatrix(inputSize, hiddenSize, 0.08),
    wn: initMatrix(inputSize, hiddenSize, 0.08),
    uz: initMatrix(hiddenSize, hiddenSize, 0.05),
    ur: initMatrix(hiddenSize, hiddenSize, 0.05),
    un: initMatrix(hiddenSize, hiddenSize, 0.05),
    bz: initVector(hiddenSize, 0),
    br: initVector(hiddenSize, 0),
    bn: initVector(hiddenSize, 0),
  };
}

function sampleValue(sample: number[], step: number, inputIndex: number, inputSize: number): number {
  const idx = step * inputSize + inputIndex;
  return sample[idx] ?? 0;
}

function projectInputAt(
  sample: number[],
  step: number,
  inputSize: number,
  hiddenSize: number,
  hiddenIndex: number,
  matrix: Float32Array
): number {
  let sum = 0;
  for (let inputIndex = 0; inputIndex < inputSize; inputIndex++) {
    sum += sampleValue(sample, step, inputIndex, inputSize) * matrix[inputIndex * hiddenSize + hiddenIndex];
  }
  return sum;
}

function projectHidden(
  prevHidden: Float32Array,
  hiddenSize: number,
  hiddenIndex: number,
  matrix: Float32Array
): number {
  let sum = 0;
  for (let h = 0; h < hiddenSize; h++) {
    sum += prevHidden[h] * matrix[h * hiddenSize + hiddenIndex];
  }
  return sum;
}

function computeEmbedding(
  sample: number[],
  variant: RecurrentVariant,
  params: NumericDict,
  inputSize: number,
  sequenceLength: number,
  hiddenSize: number
): Float32Array {
  const hidden = new Float32Array(hiddenSize);
  const cell = variant === 'lstm' ? new Float32Array(hiddenSize) : null;
  const prevHidden = new Float32Array(hiddenSize);
  const prevCell = variant === 'lstm' ? new Float32Array(hiddenSize) : null;

  for (let step = 0; step < sequenceLength; step++) {
    prevHidden.set(hidden);
    if (prevCell && cell) prevCell.set(cell);

    if (variant === 'rnn') {
      for (let h = 0; h < hiddenSize; h++) {
        const inputTerm = projectInputAt(sample, step, inputSize, hiddenSize, h, params.wx);
        const hiddenTerm = projectHidden(prevHidden, hiddenSize, h, params.wh);
        hidden[h] = Math.tanh(params.b[h] + inputTerm + hiddenTerm);
      }
      continue;
    }

    if (variant === 'lstm') {
      for (let h = 0; h < hiddenSize; h++) {
        const iGate = sigmoid(
          params.bi[h] +
            projectInputAt(sample, step, inputSize, hiddenSize, h, params.wi) +
            projectHidden(prevHidden, hiddenSize, h, params.ui)
        );
        const fGate = sigmoid(
          params.bf[h] +
            projectInputAt(sample, step, inputSize, hiddenSize, h, params.wf) +
            projectHidden(prevHidden, hiddenSize, h, params.uf)
        );
        const oGate = sigmoid(
          params.bo[h] +
            projectInputAt(sample, step, inputSize, hiddenSize, h, params.wo) +
            projectHidden(prevHidden, hiddenSize, h, params.uo)
        );
        const gGate = Math.tanh(
          params.bg[h] +
            projectInputAt(sample, step, inputSize, hiddenSize, h, params.wg) +
            projectHidden(prevHidden, hiddenSize, h, params.ug)
        );
        const nextCell = fGate * (prevCell?.[h] ?? 0) + iGate * gGate;
        if (cell) cell[h] = nextCell;
        hidden[h] = oGate * Math.tanh(nextCell);
      }
      continue;
    }

    for (let h = 0; h < hiddenSize; h++) {
      const zGate = sigmoid(
        params.bz[h] +
          projectInputAt(sample, step, inputSize, hiddenSize, h, params.wz) +
          projectHidden(prevHidden, hiddenSize, h, params.uz)
      );
      const rGate = sigmoid(
        params.br[h] +
          projectInputAt(sample, step, inputSize, hiddenSize, h, params.wr) +
          projectHidden(prevHidden, hiddenSize, h, params.ur)
      );
      let candidateHidden = 0;
      for (let j = 0; j < hiddenSize; j++) {
        candidateHidden += (rGate * prevHidden[j]) * params.un[j * hiddenSize + h];
      }
      const nGate = Math.tanh(
        params.bn[h] + projectInputAt(sample, step, inputSize, hiddenSize, h, params.wn) + candidateHidden
      );
      hidden[h] = (1 - zGate) * nGate + zGate * prevHidden[h];
    }
  }

  return hidden;
}

function buildFeatureImportance(
  variant: RecurrentVariant,
  params: NumericDict,
  outputWeights: Float32Array,
  featureCount: number,
  inputSize: number,
  hiddenSize: number,
  classCount: number
): number[] {
  const outputInfluence = new Float32Array(hiddenSize);
  for (let h = 0; h < hiddenSize; h++) {
    let sum = 0;
    for (let c = 0; c < classCount; c++) {
      sum += Math.abs(outputWeights[h * classCount + c] ?? 0);
    }
    outputInfluence[h] = sum / Math.max(1, classCount);
  }

  let sources: Float32Array[] = [];
  if (variant === 'rnn') {
    sources = [params.wx];
  } else if (variant === 'lstm') {
    sources = [params.wi, params.wf, params.wo, params.wg];
  } else {
    sources = [params.wz, params.wr, params.wn];
  }

  const slotScores = new Float32Array(inputSize);
  for (let slot = 0; slot < inputSize; slot++) {
    let slotSum = 0;
    for (let h = 0; h < hiddenSize; h++) {
      let gateWeight = 0;
      for (const source of sources) {
        gateWeight += Math.abs(source[slot * hiddenSize + h] ?? 0);
      }
      gateWeight /= Math.max(1, sources.length);
      slotSum += gateWeight * outputInfluence[h];
    }
    slotScores[slot] = slotSum / Math.max(1, hiddenSize);
  }

  const featureScores = new Array<number>(featureCount).fill(0);
  for (let featureIndex = 0; featureIndex < featureCount; featureIndex++) {
    featureScores[featureIndex] = slotScores[featureIndex % inputSize] ?? 0;
  }

  const maxScore = featureScores.reduce((max, value) => Math.max(max, value), 0);
  if (maxScore > 0) {
    for (let i = 0; i < featureScores.length; i++) {
      featureScores[i] /= maxScore;
    }
  }
  return featureScores;
}

function toArrayDict(params: NumericDict): Record<string, number[]> {
  return Object.fromEntries(Object.entries(params).map(([key, value]) => [key, Array.from(value)]));
}

function fromArrayDict(params: Record<string, number[]>): NumericDict {
  const entries = Object.entries(params).map(([key, value]) => [key, Float32Array.from(value)] as const);
  return Object.fromEntries(entries);
}

function computeLogits(hidden: Float32Array, weights: Float32Array, bias: Float32Array, classCount: number): number[] {
  const logits = new Array(classCount).fill(0);
  for (let c = 0; c < classCount; c++) {
    let value = bias[c] ?? 0;
    for (let h = 0; h < hidden.length; h++) {
      value += hidden[h] * (weights[h * classCount + c] ?? 0);
    }
    logits[c] = value;
  }
  return logits;
}

export function trainRecurrentClassifier(
  features: number[][],
  labels: number[],
  classCount: number,
  options: RecurrentTrainOptions,
  onEpoch?: (stat: RecurrentEpochStat) => void
): RecurrentClassifierModel {
  if (!features.length) throw new Error(`Cannot train ${options.variant.toUpperCase()} on empty dataset.`);
  const featureCount = features[0].length;
  const inputSize = clamp(
    Math.floor(options.inputSize ?? Math.min(16, Math.max(4, featureCount))),
    2,
    Math.max(2, Math.min(64, featureCount))
  );
  const hiddenSize = clamp(Math.floor(options.hiddenSize || 48), 8, 256);
  const sequenceLength = Math.max(1, Math.ceil(featureCount / inputSize));
  const epochs = Math.max(1, Math.floor(options.epochs));
  const learningRate = Math.max(1e-6, options.learningRate);
  const l2 = Math.max(0, options.weightDecay ?? options.l2 ?? 0);
  const sampleCount = features.length;
  const batchSize = clamp(Math.floor(options.batchSize ?? sampleCount), 1, sampleCount);
  const shuffle = options.shuffle !== false;
  const gradientClipping = Math.max(0, options.gradientClipping ?? 0);
  const patience = Math.max(0, Math.floor(options.earlyStoppingPatience ?? 0));

  const params = initializeParams(options.variant, inputSize, hiddenSize);
  const embeddingMatrix = features.map((sample) =>
    computeEmbedding(sample, options.variant, params, inputSize, sequenceLength, hiddenSize)
  );
  const outputWeights = initMatrix(hiddenSize, classCount, 0.12);
  const outputBias = initVector(classCount, 0);
  const lossCurve: RecurrentEpochStat[] = [];
  const weightState = createOptimizerState(outputWeights.length, { ...options, weightDecay: l2 });
  const biasState = createOptimizerState(classCount, { ...options, weightDecay: 0 });
  let bestLoss = Number.POSITIVE_INFINITY;
  let staleEpochs = 0;

  for (let epoch = 0; epoch < epochs; epoch++) {
    if (options.shouldStop?.()) break;
    let loss = 0;
    let correct = 0;
    const indices = maybeShuffleIndices(sampleCount, shuffle);
    const lrEpoch = scheduledLearningRate(learningRate, epoch, epochs, options);

    for (let batchStart = 0; batchStart < sampleCount; batchStart += batchSize) {
      if (options.shouldStop?.()) break;
      const batchEnd = Math.min(sampleCount, batchStart + batchSize);
      const count = Math.max(1, batchEnd - batchStart);
      const gradW = new Float32Array(outputWeights.length);
      const gradB = new Float32Array(classCount);

      for (let ptr = batchStart; ptr < batchEnd; ptr++) {
        const row = indices[ptr];
        const hidden = embeddingMatrix[row];
        const target = labels[row] ?? 0;
        const logits = computeLogits(hidden, outputWeights, outputBias, classCount);
        const probs = softmax(logits);
        const predicted = argMax(probs);
        if (predicted === target) correct++;
        loss += -Math.log((probs[target] ?? 1e-10) + 1e-10);

        for (let c = 0; c < classCount; c++) {
          const error = probs[c] - (c === target ? 1 : 0);
          gradB[c] += error;
          for (let h = 0; h < hiddenSize; h++) {
            gradW[h * classCount + c] += error * hidden[h];
          }
        }
      }

      const invBatch = 1 / count;
      for (let i = 0; i < gradW.length; i++) gradW[i] *= invBatch;
      for (let c = 0; c < gradB.length; c++) gradB[c] *= invBatch;
      clipGradient(gradW, gradientClipping);
      clipGradient(gradB, gradientClipping);
      const coupledDecay = weightState.type !== 'adamw';
      applyVectorOptimizer(outputWeights, gradW, lrEpoch, weightState, coupledDecay);
      applyVectorOptimizer(outputBias, gradB, lrEpoch, biasState, false);
    }

    const epochStat: RecurrentEpochStat = {
      epoch: epoch + 1,
      loss: loss / Math.max(1, sampleCount),
      accuracy: correct / Math.max(1, sampleCount),
    };
    lossCurve.push(epochStat);
    onEpoch?.(epochStat);

    if (patience > 0) {
      if (epochStat.loss + 1e-9 < bestLoss) {
        bestLoss = epochStat.loss;
        staleEpochs = 0;
      } else {
        staleEpochs += 1;
        if (staleEpochs >= patience) break;
      }
    }
  }

  return {
    variant: options.variant,
    featureCount,
    classCount,
    inputSize,
    sequenceLength,
    hiddenSize,
    params: toArrayDict(params),
    outputWeights: Array.from(outputWeights),
    outputBias: Array.from(outputBias),
    lossCurve,
    featureImportance: buildFeatureImportance(
      options.variant,
      params,
      outputWeights,
      featureCount,
      inputSize,
      hiddenSize,
      classCount
    ),
  };
}

export function predictRecurrentClassifier(
  model: RecurrentClassifierModel,
  features: number[][]
): number[] {
  const params = fromArrayDict(model.params);
  const weights = Float32Array.from(model.outputWeights);
  const bias = Float32Array.from(model.outputBias);
  return features.map((sample) => {
    const hidden = computeEmbedding(
      sample,
      model.variant,
      params,
      model.inputSize,
      model.sequenceLength,
      model.hiddenSize
    );
    const logits = computeLogits(hidden, weights, bias, model.classCount);
    return argMax(logits);
  });
}

export function recurrentFeatureImportance(model: RecurrentClassifierModel): number[] {
  return model.featureImportance;
}

