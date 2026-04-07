import { Adam, AdamW, Adamax, SGD } from './optimizer';
import {
  BatchNorm1d,
  Dropout,
  LayerNorm1d,
  LeakyReLU,
  Linear,
  ReLU,
  Sequential,
  Sigmoid,
  Softmax,
  Tanh,
} from './nn';
import { Tensor, crossEntropyLoss } from './tensor';
import { selectComputeBackend } from './backends';
import { computeClassificationMetrics, computeRegressionMetrics, rankFeatureImportance } from './metrics';
import {
  DbscanModel,
  KMeansModel,
  KnnModel,
  LinearRegressionClassifierModel,
  LinearRegressionRegressorModel,
  RecurrentClassifierModel,
  dbscanFeatureImportance,
  knnFeatureImportance,
  kmeansFeatureImportance,
  linearRegressorFeatureImportance,
  linearRegressionFeatureImportance,
  logisticRegressionFeatureImportance,
  recurrentFeatureImportance,
  predictDbscanClassifierAsync,
  predictKnnClassifier,
  predictKMeansClassifierAsync,
  predictRecurrentClassifier,
  predictRandomForestAlgorithm,
  predictLogisticRegressionClassifier,
  predictLinearRegressionClassifier,
  predictLinearRegressionRegressor,
  predictSvmClassifier,
  LogisticRegressionModel,
  RandomForestModel,
  SvmModel,
  trainDbscanClassifier,
  trainKnnClassifier,
  trainKMeansClassifier,
  trainRecurrentClassifier,
  trainRandomForestAlgorithm,
  trainDecisionTree,
  trainLogisticRegressionClassifier,
  trainLinearRegressionClassifier,
  trainLinearRegressionRegressor,
  trainSvmClassifier,
  svmFeatureImportance,
} from './algorithms';
import { exportModelArtifact } from './exporter';
import { WebGpuLinearRuntime } from './webgpu-linear';
import { WebGpuMatmulRuntime } from './webgpu-matmul';
import { WasmFunctionRuntime } from './wasm-function-runtime';
import {
  normalizeWasmEditorState,
  resolveWasmInvocationPolicy,
} from './wasm-function-editor';
import {
  DatasetTransferPayload,
  MainToTrainingWorkerMessage,
  ModelChoice,
  ModelMetrics,
  ResolvedModel,
  TrainedModelArtifact,
  TrainingPreferences,
  WasmFunctionInvocationPolicy,
  WorkerToTrainingMainMessage,
} from '../types/training-workflow';
import { ProcessedDataset } from '../types/data';
import { Capabilities } from './capability-detect';

interface NnCheckpoint {
  kind: 'nn';
  runId: string;
  epoch: number;
  params: number[][];
}

interface RfCheckpoint {
  kind: 'rf';
  runId: string;
  model: RandomForestModel;
}

type WorkerCheckpoint = NnCheckpoint | RfCheckpoint;

interface TrainContext {
  runId: string;
  dataset: ProcessedDataset;
  datasetTransfer: DatasetTransferPayload | null;
  modelChoice: ModelChoice;
  resolvedModel: ResolvedModel;
  preferences: TrainingPreferences;
  wasmPolicy: WasmFunctionInvocationPolicy;
  capabilities: Capabilities | null;
  backend: ReturnType<typeof selectComputeBackend>;
}

let context: TrainContext | null = null;
let shouldStop = false;
let nnModel: Sequential | null = null;
let rfModel: RandomForestModel | null = null;
let linearClassifierModel: LinearRegressionClassifierModel | null = null;
let linearRegressorModel: LinearRegressionRegressorModel | null = null;
let logisticModel: LogisticRegressionModel | null = null;
let svmModel: SvmModel | null = null;
let knnModel: KnnModel | null = null;
let recurrentModel: RecurrentClassifierModel | null = null;
let kmeansModel: KMeansModel | null = null;
let dbscanModel: DbscanModel | null = null;
let latestMetrics: ModelMetrics | null = null;
let latestArtifact: TrainedModelArtifact | null = null;
let resumed = false;
let pendingCheckpoint: WorkerCheckpoint | null = null;
let opfsDisabledReason: string | null = null;
let opfsFailureLogged = false;

function post(message: WorkerToTrainingMainMessage) {
  (self as any).postMessage(message);
}

function checkpointFileName(runId: string) {
  return `browser-first-ai-checkpoint-${runId}.json`;
}

function messageFromError(error: unknown): string {
  if (error instanceof Error) return error.message;
  if (typeof error === 'string') return error;
  return 'Unknown error';
}

function hydrateDatasetFromTransfer(dataset: ProcessedDataset, transfer: DatasetTransferPayload | null): void {
  if (!transfer) return;
  if (dataset.features.length && dataset.labels.length) return;
  if (!transfer.rowCount || !transfer.featureCount) return;

  const rowCount = transfer.rowCount;
  const featureCount = transfer.featureCount;
  const featuresView = new Float32Array(transfer.featuresBuffer);
  const labelsView = new Int32Array(transfer.labelsBuffer);

  if (featuresView.length !== rowCount * featureCount) {
    throw new Error('Dataset transfer features buffer shape mismatch.');
  }
  if (labelsView.length !== rowCount) {
    throw new Error('Dataset transfer labels buffer shape mismatch.');
  }

  const features = new Array<number[]>(rowCount);
  for (let row = 0; row < rowCount; row++) {
    const values = new Array<number>(featureCount);
    const base = row * featureCount;
    for (let col = 0; col < featureCount; col++) {
      values[col] = featuresView[base + col];
    }
    features[row] = values;
  }

  dataset.features = features;
  dataset.labels = Array.from(labelsView);
  if (transfer.regressionTargetsBuffer) {
    dataset.regressionTargets = Array.from(new Float32Array(transfer.regressionTargetsBuffer));
  }
}

function normalizePreferences(preferences: TrainingPreferences): TrainingPreferences {
  const wasmEditor = normalizeWasmEditorState(preferences.runtime?.wasmEditor);
  const neuralNetwork = preferences.neuralNetwork ?? {
    hiddenLayers: 2,
    neuronsPerLayer: 128,
    activation: 'relu',
    useBatchNorm: false,
    useLayerNorm: false,
    dropoutRate: 0.1,
    gradientClipping: 1,
    optimizer: 'adamw',
    weightDecay: 0.0001,
  };
  const algorithm = preferences.algorithm ?? {
    knnNeighbors: 7,
    knnDistanceMetric: 'euclidean',
    svmKernel: 'rbf',
    kmeansClusters: 8,
    dbscanEpsilon: 0.8,
    dbscanMinSamples: 6,
  };
  const rawOptimizer = preferences.optimizer ?? neuralNetwork.optimizer ?? 'adamw';
  const normalizedOptimizer =
    rawOptimizer === 'adam' || rawOptimizer === 'adamax' || rawOptimizer === 'sgd_momentum' ? rawOptimizer : 'adamw';
  const rawScheduler = preferences.lrScheduler ?? 'constant';
  const normalizedScheduler =
    rawScheduler === 'linear_decay' || rawScheduler === 'cosine_annealing' || rawScheduler === 'step_lr'
      ? rawScheduler
      : 'constant';
  return {
    speedVsAccuracy: clamp(preferences.speedVsAccuracy, 0, 100),
    useMoreCompute: preferences.useMoreCompute,
    optimizeForSmallerModel: preferences.optimizeForSmallerModel,
    epochs: clamp(Math.round(preferences.epochs || 1), 1, 800),
    learningRate: Math.max(1e-6, preferences.learningRate || 0.001),
    batchSize: clamp(Math.round(preferences.batchSize || 32), 1, 4096),
    shuffleEachEpoch: preferences.shuffleEachEpoch !== false,
    earlyStoppingPatience: clamp(Math.round(preferences.earlyStoppingPatience ?? 6), 0, 200),
    optimizer: normalizedOptimizer,
    weightDecay: clamp(
      typeof preferences.weightDecay === 'number' ? preferences.weightDecay : neuralNetwork.weightDecay || 0,
      0,
      1
    ),
    momentum: clamp(preferences.momentum ?? 0.9, 0, 0.9999),
    beta1: clamp(preferences.beta1 ?? 0.9, 0, 0.9999),
    beta2: clamp(preferences.beta2 ?? 0.999, 0, 0.99999),
    lrScheduler: normalizedScheduler,
    warmupSteps: clamp(Math.round(preferences.warmupSteps ?? 0), 0, 1000),
    schedulerStepSize: clamp(Math.round(preferences.schedulerStepSize ?? 10), 1, 1000),
    schedulerGamma: clamp(preferences.schedulerGamma ?? 0.5, 0.01, 0.99),
    neuralNetwork: {
      hiddenLayers: clamp(Math.round(neuralNetwork.hiddenLayers || 2), 1, 8),
      neuronsPerLayer: clamp(Math.round(neuralNetwork.neuronsPerLayer || 128), 8, 2048),
      activation:
        neuralNetwork.activation === 'tanh' ||
        neuralNetwork.activation === 'sigmoid' ||
        neuralNetwork.activation === 'leaky_relu' ||
        neuralNetwork.activation === 'softmax'
          ? neuralNetwork.activation
          : 'relu',
      useBatchNorm: Boolean(neuralNetwork.useBatchNorm),
      useLayerNorm: Boolean(neuralNetwork.useLayerNorm),
      dropoutRate: clamp(neuralNetwork.dropoutRate || 0, 0, 0.9),
      gradientClipping: clamp(neuralNetwork.gradientClipping ?? 1, 0, 100),
      optimizer: normalizedOptimizer,
      weightDecay: clamp(
        typeof preferences.weightDecay === 'number' ? preferences.weightDecay : neuralNetwork.weightDecay || 0,
        0,
        1
      ),
    },
    algorithm: {
      knnNeighbors: clamp(Math.round(algorithm.knnNeighbors || 7), 1, 200),
      knnDistanceMetric:
        algorithm.knnDistanceMetric === 'manhattan' || algorithm.knnDistanceMetric === 'cosine'
          ? algorithm.knnDistanceMetric
          : 'euclidean',
      svmKernel: algorithm.svmKernel === 'linear' || algorithm.svmKernel === 'poly' ? algorithm.svmKernel : 'rbf',
      kmeansClusters: clamp(Math.round(algorithm.kmeansClusters || 8), 2, 128),
      dbscanEpsilon: clamp(algorithm.dbscanEpsilon || 0.8, 0.01, 10),
      dbscanMinSamples: clamp(Math.round(algorithm.dbscanMinSamples || 6), 2, 128),
    },
    runtime: {
      pipeline: 'hybrid_worker_wasm_webgpu',
      wasmEditor,
    },
  };
}

async function getOpfsRoot(): Promise<any | null> {
  if (opfsDisabledReason) return null;
  try {
    if (typeof self !== 'undefined' && !self.isSecureContext) {
      opfsDisabledReason = 'OPFS disabled: secure context required.';
      return null;
    }
    if (!navigator.storage || typeof (navigator.storage as any).getDirectory !== 'function') {
      opfsDisabledReason = 'OPFS disabled: navigator.storage.getDirectory unavailable.';
      return null;
    }
    return await (navigator.storage as any).getDirectory();
  } catch (error) {
    opfsDisabledReason = `OPFS disabled: ${messageFromError(error)}`;
    if (!opfsFailureLogged) {
      opfsFailureLogged = true;
      post({
        type: 'log',
        payload: {
          message: `${opfsDisabledReason} Training continues without persistent checkpointing.`,
        },
      });
    }
    return null;
  }
}

async function loadCheckpoint(runId: string): Promise<WorkerCheckpoint | null> {
  try {
    const root = await getOpfsRoot();
    if (!root) return null;
    const handle = await root.getFileHandle(checkpointFileName(runId), { create: false });
    const file = await handle.getFile();
    const text = await file.text();
    return JSON.parse(text) as WorkerCheckpoint;
  } catch {
    return null;
  }
}

async function saveCheckpoint(checkpoint: WorkerCheckpoint): Promise<void> {
  try {
    const root = await getOpfsRoot();
    if (!root) return;
    const handle = await root.getFileHandle(checkpointFileName(checkpoint.runId), { create: true });
    const writable = await handle.createWritable();
    await writable.write(JSON.stringify(checkpoint));
    await writable.close();
  } catch (error) {
    opfsDisabledReason = `OPFS checkpoint write failed: ${messageFromError(error)}`;
    if (!opfsFailureLogged) {
      opfsFailureLogged = true;
      post({
        type: 'log',
        payload: {
          message: `${opfsDisabledReason} Training will keep running without OPFS checkpoint writes.`,
        },
      });
    }
  }
}

async function clearCheckpoint(runId: string): Promise<void> {
  try {
    const root = await getOpfsRoot();
    if (!root) return;
    await root.removeEntry(checkpointFileName(runId));
  } catch {
    // ignore
  }
}

function chooseModel(
  choice: ModelChoice,
  dataset: ProcessedDataset,
  prefs: TrainingPreferences
): { resolvedModel: ResolvedModel; adaptationNote: string | null } {
  if (
    choice === 'random_forest' ||
    choice === 'neural_network' ||
    choice === 'decision_tree' ||
    choice === 'linear_regression' ||
    choice === 'logistic_regression' ||
    choice === 'svm' ||
    choice === 'knn' ||
    choice === 'kmeans' ||
    choice === 'dbscan' ||
    choice === 'rnn' ||
    choice === 'lstm' ||
    choice === 'gru'
  ) {
    return { resolvedModel: choice, adaptationNote: null };
  }

  if (choice === 'cnn') {
    return {
      resolvedModel: 'neural_network',
      adaptationNote: `Algorithm "${choice}" maps to Neural Network execution in the current browser trainer.`,
    };
  }

  const rows = dataset.features.length;
  const featureCount = dataset.featureNames.length;
  if (!prefs.useMoreCompute && rows < 100_000 && featureCount < 300) {
    return { resolvedModel: 'random_forest', adaptationNote: null };
  }
  return { resolvedModel: 'neural_network', adaptationNote: null };
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function splitIndices(indices: number[], shardCount: number): number[][] {
  const count = Math.max(1, shardCount);
  if (count === 1 || indices.length <= count) return [indices];
  const shards: number[][] = [];
  const shardSize = Math.ceil(indices.length / count);
  for (let i = 0; i < indices.length; i += shardSize) {
    shards.push(indices.slice(i, i + shardSize));
  }
  return shards;
}

function estimateBatchMemoryMB(sampleCount: number, featureCount: number, classCount: number): number {
  const bytes = sampleCount * (featureCount * 4 + classCount * 4 + 32);
  return bytes / (1024 * 1024);
}

function scheduledLearningRate(
  prefs: TrainingPreferences,
  stepIndex: number,
  totalSteps: number
): number {
  const base = Math.max(1e-6, prefs.learningRate);
  const warmup = Math.max(0, prefs.warmupSteps);
  if (warmup > 0 && stepIndex < warmup) {
    return base * ((stepIndex + 1) / warmup);
  }
  const scheduler = prefs.lrScheduler;
  if (scheduler === 'linear_decay') {
    const denom = Math.max(1, totalSteps - 1);
    return Math.max(1e-7, base * (1 - stepIndex / denom));
  }
  if (scheduler === 'cosine_annealing') {
    const denom = Math.max(1, totalSteps - 1);
    const cosine = 0.5 * (1 + Math.cos((Math.PI * stepIndex) / denom));
    return Math.max(1e-7, base * cosine);
  }
  if (scheduler === 'step_lr') {
    const stepSize = Math.max(1, prefs.schedulerStepSize);
    const factor = Math.pow(prefs.schedulerGamma, Math.floor(stepIndex / stepSize));
    return Math.max(1e-7, base * factor);
  }
  return base;
}

function activationLayer(name: TrainingPreferences['neuralNetwork']['activation']) {
  if (name === 'leaky_relu') return new LeakyReLU(0.01);
  if (name === 'tanh') return new Tanh();
  if (name === 'sigmoid') return new Sigmoid();
  if (name === 'softmax') return new Softmax();
  return new ReLU();
}

function buildNeuralNet(inputSize: number, classes: number, prefs: TrainingPreferences): Sequential {
  const complexity = prefs.speedVsAccuracy / 100;
  const cfg = prefs.neuralNetwork;
  const recommendedHidden = clamp(
    Math.floor(inputSize * (0.8 + complexity * 2.2)),
    16,
    prefs.useMoreCompute ? 768 : 384
  );
  const hiddenLayers = clamp(cfg.hiddenLayers, 1, 8);
  const hiddenUnits = clamp(cfg.neuronsPerLayer || recommendedHidden, 8, prefs.useMoreCompute ? 2048 : 1024);
  const dropoutRate = clamp(cfg.dropoutRate, 0, 0.8);

  const layers: Sequential['layers'] = [];
  let inFeatures = inputSize;
  for (let i = 0; i < hiddenLayers; i++) {
    layers.push(new Linear(inFeatures, hiddenUnits, `dense_${i + 1}`));
    if (cfg.useBatchNorm) layers.push(new BatchNorm1d(hiddenUnits, `bn_${i + 1}`));
    if (cfg.useLayerNorm) layers.push(new LayerNorm1d(hiddenUnits, `ln_${i + 1}`));
    layers.push(activationLayer(cfg.activation));
    if (dropoutRate > 0) layers.push(new Dropout(dropoutRate, `dropout_${i + 1}`));
    inFeatures = hiddenUnits;
  }
  layers.push(new Linear(inFeatures, classes, 'dense_out'));
  return new Sequential(layers);
}

function argmaxRow(data: Float32Array, row: number, classes: number): number {
  let best = 0;
  let bestValue = -Infinity;
  const offset = row * classes;
  for (let c = 0; c < classes; c++) {
    const value = data[offset + c];
    if (value > bestValue) {
      best = c;
      bestValue = value;
    }
  }
  return best;
}

function computeLinearLogitsCpu(
  flatFeatures: Float32Array,
  weights: Float32Array,
  bias: Float32Array,
  rowCount: number,
  featureCount: number,
  classCount: number
): Float32Array {
  const logits = new Float32Array(rowCount * classCount);
  for (let row = 0; row < rowCount; row++) {
    const rowBase = row * featureCount;
    const outBase = row * classCount;
    for (let cls = 0; cls < classCount; cls++) {
      let sum = bias[cls] ?? 0;
      for (let f = 0; f < featureCount; f++) {
        sum += flatFeatures[rowBase + f] * weights[f * classCount + cls];
      }
      logits[outBase + cls] = sum;
    }
  }
  return logits;
}

async function computeLinearLogits(
  gpuRuntime: WebGpuLinearRuntime | null,
  flatFeatures: Float32Array,
  weights: Float32Array,
  bias: Float32Array,
  rowCount: number,
  featureCount: number,
  classCount: number
): Promise<Float32Array> {
  if (!gpuRuntime) {
    return computeLinearLogitsCpu(flatFeatures, weights, bias, rowCount, featureCount, classCount);
  }
  try {
    return await gpuRuntime.computeLogits(flatFeatures, weights, bias, rowCount, featureCount, classCount);
  } catch {
    return computeLinearLogitsCpu(flatFeatures, weights, bias, rowCount, featureCount, classCount);
  }
}

function batchFromIndices(dataset: ProcessedDataset, indices: number[]): { inputs: Tensor; targets: Int32Array } {
  const featureCount = dataset.featureNames.length;
  const inputData = new Float32Array(indices.length * featureCount);
  const targets = new Int32Array(indices.length);
  for (let i = 0; i < indices.length; i++) {
    const rowIndex = indices[i];
    inputData.set(dataset.features[rowIndex], i * featureCount);
    targets[i] = dataset.labels[rowIndex];
  }
  return {
    inputs: new Tensor(inputData, [indices.length, featureCount]),
    targets,
  };
}

async function evaluateNeuralNet(
  model: Sequential,
  dataset: ProcessedDataset,
  gpu: WebGpuMatmulRuntime | null
): Promise<{ predictions: number[]; loss: number; accuracy: number }> {
  const featureCount = dataset.featureNames.length;
  const classes = dataset.labelNames.length;
  const inputsData = new Float32Array(dataset.features.length * featureCount);
  for (let i = 0; i < dataset.features.length; i++) {
    inputsData.set(dataset.features[i], i * featureCount);
  }
  const inputs = new Tensor(inputsData, [dataset.features.length, featureCount]);
  const logits = await model.forwardAsync(inputs, false, gpu);
  const loss = crossEntropyLoss(logits, Int32Array.from(dataset.labels)).data[0];
  const predictions = dataset.features.map((_, row) => argmaxRow(logits.data, row, classes));
  let correct = 0;
  for (let i = 0; i < predictions.length; i++) if (predictions[i] === dataset.labels[i]) correct++;
  return {
    predictions,
    loss,
    accuracy: predictions.length ? correct / predictions.length : 0,
  };
}

function firstLayerImportance(model: Sequential, featureNames: string[]): number[] {
  const firstLayer = model.layers.find((layer) => layer instanceof Linear) as Linear | undefined;
  if (!firstLayer) return new Array(featureNames.length).fill(0);
  const [inputSize, outputSize] = firstLayer.weight.shape;
  const scores = new Array(inputSize).fill(0);
  for (let i = 0; i < inputSize; i++) {
    let sum = 0;
    for (let j = 0; j < outputSize; j++) {
      sum += Math.abs(firstLayer.weight.data[i * outputSize + j]);
    }
    scores[i] = sum;
  }
  return scores;
}

function serializeNeuralNetParams(model: Sequential): number[][] {
  return model.parameters().map((param) => Array.from(param.data));
}

function serializeDenseLayers(model: Sequential): Array<{
  inputSize: number;
  outputSize: number;
  weights: number[];
  bias: number[];
}> {
  return model.layers
    .filter((layer): layer is Linear => layer instanceof Linear)
    .map((layer) => ({
      inputSize: layer.weight.shape[0],
      outputSize: layer.weight.shape[1],
      weights: Array.from(layer.weight.data),
      bias: Array.from(layer.bias.data),
    }));
}

function serializeArchitecture(model: Sequential): string[] {
  return model.layers.map((layer) => layer.name || layer.constructor.name);
}

function serializeBatchNormLayers(model: Sequential): Array<{
  featureCount: number;
  gamma: number[];
  beta: number[];
  runningMean: number[];
  runningVar: number[];
  epsilon: number;
}> {
  return model.layers
    .filter((layer): layer is BatchNorm1d => layer instanceof BatchNorm1d)
    .map((layer) => ({
      featureCount: layer.numFeatures,
      gamma: Array.from(layer.gamma.data),
      beta: Array.from(layer.beta.data),
      runningMean: Array.from(layer.runningMean),
      runningVar: Array.from(layer.runningVar),
      epsilon: layer.eps,
    }));
}

function restoreNeuralNetParams(model: Sequential, params: number[][]): void {
  const modelParams = model.parameters();
  if (modelParams.length !== params.length) throw new Error('Checkpoint parameter mismatch.');
  for (let i = 0; i < modelParams.length; i++) {
    const incoming = params[i];
    if (incoming.length !== modelParams[i].data.length) throw new Error('Checkpoint tensor shape mismatch.');
    modelParams[i].data.set(incoming);
  }
}

function clipGradients(model: Sequential, clipValue: number): void {
  if (clipValue <= 0) return;
  const maxAbs = Math.abs(clipValue);
  for (const param of model.parameters()) {
    if (!param.grad) continue;
    for (let i = 0; i < param.grad.length; i++) {
      if (param.grad[i] > maxAbs) param.grad[i] = maxAbs;
      else if (param.grad[i] < -maxAbs) param.grad[i] = -maxAbs;
    }
  }
}

function applyWeightDecay(model: Sequential, weightDecay: number): void {
  if (weightDecay <= 0) return;
  for (const param of model.parameters()) {
    if (!param.grad) continue;
    for (let i = 0; i < param.grad.length; i++) {
      param.grad[i] += weightDecay * param.data[i];
    }
  }
}

function buildOptimizer(model: Sequential, prefs: TrainingPreferences): {
  optimizer: Adam | Adamax | AdamW | SGD;
  decoupledWeightDecay: boolean;
} {
  const params = model.parameters();
  const lr = Math.max(1e-6, prefs.learningRate);
  if (prefs.optimizer === 'sgd_momentum') {
    return {
      optimizer: new SGD(params, lr, prefs.momentum),
      decoupledWeightDecay: false,
    };
  }
  if (prefs.optimizer === 'adam') {
    return {
      optimizer: new Adam(params, lr, prefs.beta1, prefs.beta2),
      decoupledWeightDecay: false,
    };
  }
  if (prefs.optimizer === 'adamax') {
    return {
      optimizer: new Adamax(params, lr, prefs.beta1, prefs.beta2),
      decoupledWeightDecay: false,
    };
  }
  return {
    optimizer: new AdamW(params, lr, prefs.beta1, prefs.beta2, prefs.weightDecay),
    decoupledWeightDecay: true,
  };
}

interface FlatOptimizerState {
  type: TrainingPreferences['optimizer'];
  momentum: number;
  beta1: number;
  beta2: number;
  weightDecay: number;
  eps: number;
  step: number;
  velocity: Float32Array;
  m: Float32Array;
  v: Float32Array;
  u: Float32Array;
}

function createFlatOptimizerState(length: number, prefs: TrainingPreferences, weightDecay: number): FlatOptimizerState {
  return {
    type: prefs.optimizer,
    momentum: prefs.momentum,
    beta1: prefs.beta1,
    beta2: prefs.beta2,
    weightDecay: Math.max(0, weightDecay),
    eps: 1e-8,
    step: 0,
    velocity: new Float32Array(length),
    m: new Float32Array(length),
    v: new Float32Array(length),
    u: new Float32Array(length),
  };
}

function clipGradientVector(gradient: Float32Array, clipValue: number): void {
  if (clipValue <= 0) return;
  const maxAbs = Math.abs(clipValue);
  for (let i = 0; i < gradient.length; i++) {
    if (gradient[i] > maxAbs) gradient[i] = maxAbs;
    else if (gradient[i] < -maxAbs) gradient[i] = -maxAbs;
  }
}

function applyFlatOptimizerStep(
  params: Float32Array,
  gradient: Float32Array,
  learningRate: number,
  state: FlatOptimizerState,
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
      const g = gradient[i] + decayGrad;
      state.velocity[i] = state.momentum * state.velocity[i] + g;
      params[i] -= lr * state.velocity[i];
    }
    return;
  }

  if (state.type === 'adam') {
    for (let i = 0; i < params.length; i++) {
      const decayGrad = applyCoupledDecay ? state.weightDecay * params[i] : 0;
      const g = gradient[i] + decayGrad;
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
      const g = gradient[i] + decayGrad;
      state.m[i] = beta1 * state.m[i] + (1 - beta1) * g;
      state.u[i] = Math.max(beta2 * state.u[i], Math.abs(g));
      params[i] -= (lrT * state.m[i]) / (state.u[i] + eps);
    }
    return;
  }

  for (let i = 0; i < params.length; i++) {
    const g = gradient[i];
    state.m[i] = beta1 * state.m[i] + (1 - beta1) * g;
    state.v[i] = beta2 * state.v[i] + (1 - beta2) * g * g;
    const mHat = state.m[i] / bc1;
    const vHat = state.v[i] / bc2;
    params[i] -= lr * (mHat / (Math.sqrt(vHat) + eps) + state.weightDecay * params[i]);
  }
}

function makeRuntime(policy: WasmFunctionInvocationPolicy): WasmFunctionRuntime {
  const runtime = new WasmFunctionRuntime(
    policy,
    (message) => post({ type: 'log', payload: { message } }),
    (message) => post({ type: 'log', payload: { message } })
  );
  if (!runtime.isWasmSupported()) {
    post({
      type: 'log',
      payload: {
        message:
          'WASM SIMD probe failed in this environment. Hybrid runtime stays active with compatibility compute fallback.',
      },
    });
  }
  return runtime;
}

interface OpfsBatchCache {
  totalBatches: number;
  getBatch: (batchIndex: number) => Promise<{ inputs: Tensor; targets: Int32Array }>;
}

async function prepareOpfsBatchCache(
  runId: string,
  dataset: ProcessedDataset,
  batchSize: number
): Promise<OpfsBatchCache | null> {
  const minRowsForCache = 20_000;
  if (dataset.features.length < minRowsForCache) return null;
  const root = await getOpfsRoot();
  if (!root) return null;

  const totalBatches = Math.ceil(dataset.features.length / batchSize);
  const featureCount = dataset.featureNames.length;
  const directory = await root.getDirectoryHandle(`browser-first-ai-batch-cache-${runId}`, { create: true });
  const metaHandle = await directory.getFileHandle('meta.json', { create: true });

  let shouldRewrite = true;
  try {
    const metaFile = await metaHandle.getFile();
    const metaJson = JSON.parse(await metaFile.text());
    shouldRewrite =
      metaJson?.rows !== dataset.features.length ||
      metaJson?.features !== featureCount ||
      metaJson?.batchSize !== batchSize;
  } catch {
    shouldRewrite = true;
  }

  if (shouldRewrite) {
    for (let batch = 0; batch < totalBatches; batch++) {
      const start = batch * batchSize;
      const end = Math.min(start + batchSize, dataset.features.length);
      const payload = {
        features: dataset.features.slice(start, end),
        labels: dataset.labels.slice(start, end),
      };
      const handle = await directory.getFileHandle(`batch-${batch}.json`, { create: true });
      const writable = await handle.createWritable();
      await writable.write(JSON.stringify(payload));
      await writable.close();
    }
    const metaWriter = await metaHandle.createWritable();
    await metaWriter.write(
      JSON.stringify({
        rows: dataset.features.length,
        features: featureCount,
        batchSize,
      })
    );
    await metaWriter.close();
    post({
      type: 'log',
      payload: {
        message: `OPFS batch cache prepared (${totalBatches} batches).`,
      },
    });
  } else {
    post({
      type: 'log',
      payload: {
        message: `Using existing OPFS batch cache (${totalBatches} batches).`,
      },
    });
  }

  const prefetchWindow = 3;
  const pending = new Map<number, Promise<{ inputs: Tensor; targets: Int32Array }>>();

  const loadBatch = async (batchIndex: number): Promise<{ inputs: Tensor; targets: Int32Array }> => {
    const handle = await directory.getFileHandle(`batch-${batchIndex}.json`, { create: false });
    const file = await handle.getFile();
    const text = await file.text();
    const payload = JSON.parse(text) as { features: number[][]; labels: number[] };
    const rows = payload.features.length;
    const inputData = new Float32Array(rows * featureCount);
    for (let row = 0; row < rows; row++) {
      inputData.set(payload.features[row], row * featureCount);
    }
    return {
      inputs: new Tensor(inputData, [rows, featureCount]),
      targets: Int32Array.from(payload.labels),
    };
  };

  const schedule = (batchIndex: number) => {
    if (batchIndex < 0 || batchIndex >= totalBatches || pending.has(batchIndex)) return;
    pending.set(batchIndex, loadBatch(batchIndex));
  };

  return {
    totalBatches,
    getBatch: async (batchIndex: number) => {
      for (let i = batchIndex; i < Math.min(totalBatches, batchIndex + prefetchWindow); i++) {
        schedule(i);
      }
      const current = pending.get(batchIndex) ?? loadBatch(batchIndex);
      pending.set(batchIndex, current);
      const result = await current;
      pending.delete(batchIndex);
      return result;
    },
  };
}

async function trainNeuralNetwork(
  resumeCheckpoint: NnCheckpoint | null,
  runtime: WasmFunctionRuntime
): Promise<void> {
  if (!context) throw new Error('Worker not initialized.');
  const dataset = context.dataset;
  const classes = dataset.labelNames.length;
  nnModel = buildNeuralNet(dataset.featureNames.length, classes, context.preferences);
  const { optimizer, decoupledWeightDecay } = buildOptimizer(nnModel, context.preferences);

  let startEpoch = 0;
  if (resumeCheckpoint?.params?.length) {
    restoreNeuralNetParams(nnModel, resumeCheckpoint.params);
    startEpoch = resumeCheckpoint.epoch;
    resumed = true;
    post({ type: 'log', payload: { message: `Restored checkpoint from epoch ${startEpoch}.` } });
  }

  const epochs = Math.max(1, context.preferences.epochs);
  const batchSize = Math.max(1, context.preferences.batchSize);
  const wasmPolicy = context.wasmPolicy;
  const shardCount = Math.max(1, wasmPolicy.shardCount);
  const checkpointEveryNEpochs = Math.max(1, wasmPolicy.checkpointEveryNEpochs);
  const opfsBatchCache = await prepareOpfsBatchCache(context.runId, dataset, batchSize);
  const totalBatches = opfsBatchCache?.totalBatches ?? Math.ceil(dataset.features.length / batchSize);
  if (opfsBatchCache) {
    post({
      type: 'log',
      payload: {
        message: 'OPFS batch prefetch enabled for large dataset training.',
      },
    });
  }
  const totalSteps = (epochs - startEpoch) * totalBatches;
  let globalStep = 0;
  const patience = context.preferences.earlyStoppingPatience;
  let bestLoss = Number.POSITIVE_INFINITY;
  let staleEpochs = 0;

  post({ type: 'status', payload: { phase: 'training_model', message: 'Training neural network...' } });

  let nnGpuMatmul: WebGpuMatmulRuntime | null = null;
  if (context.backend.kind === 'webgpu') {
    nnGpuMatmul = await WebGpuMatmulRuntime.create((reason) => {
      nnGpuMatmul = null;
      post({
        type: 'log',
        payload: { message: `WebGPU matmul device lost: ${reason}. Using CPU matmul for remaining steps.` },
      });
    });
    if (nnGpuMatmul) {
      post({
        type: 'log',
        payload: { message: 'WebGPU WGSL compute matmul enabled for linear layer forwards (backward stays on CPU).' },
      });
    }
  }

  for (let epoch = startEpoch; epoch < epochs; epoch++) {
    if (shouldStop) return;
    optimizer.setLearningRate(scheduledLearningRate(context.preferences, epoch, epochs));
    let epochLossTotal = 0;
    let epochCorrectTotal = 0;
    let epochSampleTotal = 0;

    const indices = opfsBatchCache ? [] : Array.from({ length: dataset.features.length }, (_, idx) => idx);
    if (!opfsBatchCache && context.preferences.shuffleEachEpoch) {
      for (let i = indices.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
      }
    }

    for (let batch = 0; batch < totalBatches; batch++) {
      if (shouldStop) return;
      // Yield each batch so stop messages are handled quickly.
      await new Promise((resolve) => setTimeout(resolve, 0));
      if (shouldStop) return;
      let totalLoss = 0;
      let totalCorrect = 0;
      let totalSamples = 0;

      const runShard = async (payload: { inputs: Tensor; targets: Int32Array }) => {
        const { inputs, targets } = payload;
        optimizer.zeroGrad();
        const logits = await nnModel!.forwardAsync(inputs, true, nnGpuMatmul);
        const loss = crossEntropyLoss(logits, targets);
        if (!Number.isFinite(loss.data[0])) {
          throw new Error('NaN/Inf detected during training. Try lower learning rate.');
        }
        loss.backward();
        const clipValue =
          context!.preferences.neuralNetwork.gradientClipping > 0
            ? context!.preferences.neuralNetwork.gradientClipping
            : context!.wasmPolicy.gradientClipValue;
        clipGradients(nnModel!, clipValue);
        if (!decoupledWeightDecay) {
          applyWeightDecay(nnModel!, context!.preferences.weightDecay);
        }
        optimizer.step();

        let correct = 0;
        for (let row = 0; row < targets.length; row++) {
          if (argmaxRow(logits.data, row, classes) === targets[row]) correct++;
        }
        return {
          loss: loss.data[0],
          correct,
          count: targets.length,
        };
      };

      if (opfsBatchCache) {
        const batchPayload = await opfsBatchCache.getBatch(batch);
        const estimatedMemory = estimateBatchMemoryMB(batchPayload.targets.length, dataset.featureNames.length, classes);
        const batchResult = await runtime.invoke(wasmPolicy.functionName, estimatedMemory, () => runShard(batchPayload));
        if (shouldStop) return;
        totalLoss += batchResult.loss * batchResult.count;
        totalCorrect += batchResult.correct;
        totalSamples += batchResult.count;
      } else {
        const start = batch * batchSize;
        const batchIndices = indices.slice(start, Math.min(start + batchSize, indices.length));
        const shards = splitIndices(batchIndices, shardCount);
        for (const shard of shards) {
          if (shouldStop) return;
          const payload = batchFromIndices(dataset, shard);
          const estimatedMemory = estimateBatchMemoryMB(shard.length, dataset.featureNames.length, classes);
          const shardResult = await runtime.invoke(wasmPolicy.functionName, estimatedMemory, () => runShard(payload));
          if (shouldStop) return;
          totalLoss += shardResult.loss * shardResult.count;
          totalCorrect += shardResult.correct;
          totalSamples += shardResult.count;
        }
      }

      epochLossTotal += totalLoss;
      epochCorrectTotal += totalCorrect;
      epochSampleTotal += totalSamples;

      globalStep += 1;
      post({
        type: 'progress',
        payload: {
          step: globalStep,
          epoch,
          loss: totalSamples ? totalLoss / totalSamples : 0,
          accuracy: totalSamples ? totalCorrect / totalSamples : 0,
          percent: totalSteps ? Math.floor((globalStep / totalSteps) * 100) : 0,
        },
      });
    }

    const epochLoss = epochSampleTotal ? epochLossTotal / epochSampleTotal : 0;
    const epochAccuracy = epochSampleTotal ? epochCorrectTotal / epochSampleTotal : 0;
    post({
      type: 'log',
      payload: {
        message:
          `Epoch ${epoch + 1}/${epochs} - loss: ${epochLoss.toFixed(6)} - ` +
          `accuracy: ${(epochAccuracy * 100).toFixed(2)}% - ` +
          `lr: ${scheduledLearningRate(context.preferences, epoch, epochs).toExponential(2)}`,
      },
    });

    if ((epoch + 1) % checkpointEveryNEpochs === 0) {
      await saveCheckpoint({
        kind: 'nn',
        runId: context.runId,
        epoch: epoch + 1,
        params: serializeNeuralNetParams(nnModel),
      });
    }

    if (patience > 0) {
      if (epochLoss + 1e-9 < bestLoss) {
        bestLoss = epochLoss;
        staleEpochs = 0;
      } else {
        staleEpochs += 1;
        if (staleEpochs >= patience) {
          post({
            type: 'log',
            payload: {
              message: `Early stopping triggered after ${patience} stale epochs.`,
            },
          });
          break;
        }
      }
    }
  }

  post({ type: 'status', payload: { phase: 'optimizing_parameters', message: 'Evaluating model quality...' } });
  const evaluation = await evaluateNeuralNet(nnModel, dataset, nnGpuMatmul);
  const metricsBase = computeClassificationMetrics(dataset.labels, evaluation.predictions, dataset.labelNames.length);
  const importanceScores = firstLayerImportance(nnModel, dataset.featureNames);
  latestMetrics = {
    kind: 'classification',
    ...metricsBase,
    featureImportance: rankFeatureImportance(dataset.featureNames, importanceScores).slice(0, 10),
  };
  latestArtifact = {
    modelType: 'neural_network',
    backend: context.backend.label,
    trainedAt: new Date().toISOString(),
    featureNames: dataset.featureNames,
    labelNames: dataset.labelNames,
    modelData: {
      layers: serializeNeuralNetParams(nnModel),
      linearLayers: serializeDenseLayers(nnModel),
      batchNormLayers: serializeBatchNormLayers(nnModel),
      architecture: serializeArchitecture(nnModel),
      activation: context.preferences.neuralNetwork.activation,
      hiddenLayers: context.preferences.neuralNetwork.hiddenLayers,
      neuronsPerLayer: context.preferences.neuralNetwork.neuronsPerLayer,
      useBatchNorm: context.preferences.neuralNetwork.useBatchNorm,
      useLayerNorm: context.preferences.neuralNetwork.useLayerNorm,
      dropoutRate: context.preferences.neuralNetwork.dropoutRate,
      optimizer: context.preferences.optimizer,
      weightDecay: context.preferences.weightDecay,
      gradientClipping: context.preferences.neuralNetwork.gradientClipping,
      lrScheduler: context.preferences.lrScheduler,
      warmupSteps: context.preferences.warmupSteps,
      finalLoss: evaluation.loss,
    },
  };
}

async function trainRandomForestModel(
  resumeCheckpoint: RfCheckpoint | null,
  runtime: WasmFunctionRuntime
): Promise<void> {
  if (!context) throw new Error('Worker not initialized.');
  const dataset = context.dataset;

  post({ type: 'status', payload: { phase: 'training_model', message: 'Training random forest...' } });

  if (resumeCheckpoint?.model) {
    rfModel = resumeCheckpoint.model;
    resumed = true;
    post({ type: 'log', payload: { message: 'Restored Random Forest checkpoint.' } });
  } else {
    let lastLoggedPercent = -1;
    const trees = clamp(
      Math.floor((context.preferences.speedVsAccuracy / 100) * 120) + (context.preferences.useMoreCompute ? 60 : 30),
      20,
      220
    );
    const maxDepth = clamp(Math.floor((context.preferences.speedVsAccuracy / 100) * 12) + 4, 3, 20);
    const runForest = () =>
      trainRandomForestAlgorithm(
        dataset.features,
        dataset.labels,
        {
          trees,
          maxDepth,
          minSamplesSplit: 4,
          featureSampleRate: 0.6,
        },
        (progress) => {
          if (shouldStop) {
            throw new Error('Training stopped by user.');
          }
          const percent = Math.floor(progress * 100);
          post({
            type: 'progress',
            payload: {
              step: Math.floor(progress * trees),
              epoch: 0,
              loss: 1 - progress,
              accuracy: progress,
              percent,
            },
          });
          if (percent >= lastLoggedPercent + 10 || percent === 100) {
            lastLoggedPercent = percent;
            post({
              type: 'log',
              payload: {
                message: `RandomForest progress ${percent}% (${Math.floor(progress * trees)}/${trees} trees).`,
              },
            });
          }
        }
      );
    try {
      rfModel = await runtime.invoke(
        context.wasmPolicy.functionName,
        estimateBatchMemoryMB(dataset.features.length, dataset.featureNames.length, dataset.labelNames.length),
        runForest
      );
    } catch (error) {
      if (shouldStop) return;
      throw error;
    }
    if (shouldStop) return;
    await saveCheckpoint({ kind: 'rf', runId: context.runId, model: rfModel });
  }

  if (!rfModel) throw new Error('Random Forest model was not created.');

  post({ type: 'status', payload: { phase: 'optimizing_parameters', message: 'Computing validation metrics...' } });
  const predictions = predictRandomForestAlgorithm(rfModel, dataset.features);
  const metricsBase = computeClassificationMetrics(dataset.labels, predictions, dataset.labelNames.length);
  post({
    type: 'log',
    payload: {
      message:
        `Epoch 1/1 - loss: ${(1 - metricsBase.accuracy).toFixed(6)} - ` +
        `accuracy: ${(metricsBase.accuracy * 100).toFixed(2)}%`,
    },
  });
  latestMetrics = {
    kind: 'classification',
    ...metricsBase,
    featureImportance: rankFeatureImportance(dataset.featureNames, rfModel.featureImportance).slice(0, 10),
  };
  latestArtifact = {
    modelType: 'random_forest',
    backend: context.backend.label,
    trainedAt: new Date().toISOString(),
    featureNames: dataset.featureNames,
    labelNames: dataset.labelNames,
    modelData: {
      trees: rfModel.trees,
      classes: rfModel.classes,
      featureImportance: rfModel.featureImportance,
    },
  };
}

async function trainDecisionTreeModel(
  resumeCheckpoint: RfCheckpoint | null,
  runtime: WasmFunctionRuntime
): Promise<void> {
  if (!context) throw new Error('Worker not initialized.');
  const dataset = context.dataset;

  post({ type: 'status', payload: { phase: 'training_model', message: 'Training decision tree...' } });

  if (resumeCheckpoint?.model?.trees?.length === 1) {
    rfModel = resumeCheckpoint.model;
    resumed = true;
    post({ type: 'log', payload: { message: 'Restored Decision Tree checkpoint.' } });
  } else {
    let lastLoggedPercent = -1;
    const maxDepth = clamp(Math.floor((context.preferences.speedVsAccuracy / 100) * 12) + 4, 2, 32);
    const runTree = () =>
      trainDecisionTree(
        dataset.features,
        dataset.labels,
        {
          maxDepth,
          minSamplesSplit: 2,
        },
        (progress) => {
          if (shouldStop) {
            throw new Error('Training stopped by user.');
          }
          const percent = Math.floor(progress * 100);
          post({
            type: 'progress',
            payload: {
              step: Math.floor(progress * 100),
              epoch: 0,
              loss: 1 - progress,
              accuracy: progress,
              percent,
            },
          });
          if (percent >= lastLoggedPercent + 10 || percent === 100) {
            lastLoggedPercent = percent;
            post({
              type: 'log',
              payload: {
                message: `DecisionTree progress ${percent}% (max_depth=${maxDepth}).`,
              },
            });
          }
        }
      );
    try {
      rfModel = await runtime.invoke(
        context.wasmPolicy.functionName,
        estimateBatchMemoryMB(dataset.features.length, dataset.featureNames.length, dataset.labelNames.length),
        runTree
      );
    } catch (error) {
      if (shouldStop) return;
      throw error;
    }
    if (shouldStop) return;
    await saveCheckpoint({ kind: 'rf', runId: context.runId, model: rfModel });
  }

  if (!rfModel) throw new Error('Decision Tree model was not created.');

  post({ type: 'status', payload: { phase: 'optimizing_parameters', message: 'Computing validation metrics...' } });
  const predictions = predictRandomForestAlgorithm(rfModel, dataset.features);
  const metricsBase = computeClassificationMetrics(dataset.labels, predictions, dataset.labelNames.length);
  post({
    type: 'log',
    payload: {
      message:
        `Epoch 1/1 - loss: ${(1 - metricsBase.accuracy).toFixed(6)} - ` +
        `accuracy: ${(metricsBase.accuracy * 100).toFixed(2)}%`,
    },
  });
  latestMetrics = {
    kind: 'classification',
    ...metricsBase,
    featureImportance: rankFeatureImportance(dataset.featureNames, rfModel.featureImportance).slice(0, 10),
  };
  latestArtifact = {
    modelType: 'decision_tree',
    backend: context.backend.label,
    trainedAt: new Date().toISOString(),
    featureNames: dataset.featureNames,
    labelNames: dataset.labelNames,
    modelData: {
      trees: rfModel.trees,
      classes: rfModel.classes,
      featureImportance: rfModel.featureImportance,
    },
  };
}

async function trainLinearRegressionModel(runtime: WasmFunctionRuntime): Promise<void> {
  if (!context) throw new Error('Worker not initialized.');
  const dataset = context.dataset;

  post({ type: 'status', payload: { phase: 'training_model', message: 'Training linear regression...' } });

  const epochs = Math.max(4, context.preferences.epochs);

  if (dataset.problemType === 'regression' && dataset.regressionTargets?.length === dataset.features.length) {
    const runRegressor = () =>
      trainLinearRegressionRegressor(
        dataset.features,
        dataset.regressionTargets!,
        {
          epochs,
          learningRate: context!.preferences.learningRate,
          l2: context!.preferences.weightDecay,
          weightDecay: context!.preferences.weightDecay,
          batchSize: context!.preferences.batchSize,
          shuffle: context!.preferences.shuffleEachEpoch,
          earlyStoppingPatience: context!.preferences.earlyStoppingPatience,
          optimizer: context!.preferences.optimizer,
          momentum: context!.preferences.momentum,
          beta1: context!.preferences.beta1,
          beta2: context!.preferences.beta2,
          scheduler: context!.preferences.lrScheduler,
          warmupSteps: context!.preferences.warmupSteps,
          stepSize: context!.preferences.schedulerStepSize,
          schedulerGamma: context!.preferences.schedulerGamma,
          gradientClipping: context!.preferences.neuralNetwork.gradientClipping,
        },
        {
          shouldStop: () => shouldStop,
          onEpoch: (stats) => {
            const percent = Math.floor(((stats.epoch + 1) / epochs) * 100);
            post({
              type: 'progress',
              payload: {
                step: stats.epoch + 1,
                epoch: stats.epoch,
                loss: stats.loss,
                accuracy: stats.accuracy,
                percent,
              },
            });
            post({
              type: 'log',
              payload: {
                message:
                  `Epoch ${stats.epoch + 1}/${epochs} - mse: ${stats.loss.toFixed(6)} - ` +
                  `score: ${stats.accuracy.toFixed(4)} - lr: ${scheduledLearningRate(context!.preferences, stats.epoch, epochs).toExponential(2)}`,
              },
            });
          },
        }
      );

    try {
      linearRegressorModel = await runtime.invoke(
        context.wasmPolicy.functionName,
        estimateBatchMemoryMB(dataset.features.length, dataset.featureNames.length, 1),
        runRegressor
      );
    } catch (error) {
      if (shouldStop) return;
      throw error;
    }
    if (shouldStop) return;
    if (!linearRegressorModel) throw new Error('Linear Regression regressor was not created.');

    post({ type: 'status', payload: { phase: 'optimizing_parameters', message: 'Computing regression metrics...' } });
    const predictions = predictLinearRegressionRegressor(linearRegressorModel, dataset.features);
    const metricsBase = computeRegressionMetrics(dataset.regressionTargets, predictions);
    const importanceScores = linearRegressorFeatureImportance(linearRegressorModel);
    latestMetrics = {
      kind: 'regression',
      ...metricsBase,
      featureImportance: rankFeatureImportance(dataset.featureNames, importanceScores).slice(0, 10),
    };
    latestArtifact = {
      modelType: 'linear_regression',
      backend: context.backend.label,
      trainedAt: new Date().toISOString(),
      featureNames: dataset.featureNames,
      labelNames: dataset.labelNames,
      modelData: {
        weights: linearRegressorModel.weights,
        bias: [linearRegressorModel.bias],
        featureCount: linearRegressorModel.featureCount,
        classCount: 1,
        lossCurve: linearRegressorModel.lossCurve,
        mode: 'regressor',
        algorithm: 'linear_regression_regressor',
      },
    };
    return;
  }

  const classCount = Math.max(2, dataset.labelNames.length);
  const runClassifier = () =>
    trainLinearRegressionClassifier(
      dataset.features,
      dataset.labels,
      classCount,
      {
        epochs,
        learningRate: context!.preferences.learningRate,
        l2: context!.preferences.weightDecay,
        weightDecay: context!.preferences.weightDecay,
        batchSize: context!.preferences.batchSize,
        shuffle: context!.preferences.shuffleEachEpoch,
        earlyStoppingPatience: context!.preferences.earlyStoppingPatience,
        optimizer: context!.preferences.optimizer,
        momentum: context!.preferences.momentum,
        beta1: context!.preferences.beta1,
        beta2: context!.preferences.beta2,
        scheduler: context!.preferences.lrScheduler,
        warmupSteps: context!.preferences.warmupSteps,
        stepSize: context!.preferences.schedulerStepSize,
        schedulerGamma: context!.preferences.schedulerGamma,
        gradientClipping: context!.preferences.neuralNetwork.gradientClipping,
      },
      {
        shouldStop: () => shouldStop,
        onEpoch: (stats) => {
          const percent = Math.floor(((stats.epoch + 1) / epochs) * 100);
          post({
            type: 'progress',
            payload: {
              step: stats.epoch + 1,
              epoch: stats.epoch,
              loss: stats.loss,
              accuracy: stats.accuracy,
              percent,
            },
          });
          post({
            type: 'log',
            payload: {
              message:
                `Epoch ${stats.epoch + 1}/${epochs} - loss: ${stats.loss.toFixed(6)} - ` +
                  `accuracy: ${(stats.accuracy * 100).toFixed(2)}% - ` +
                  `lr: ${scheduledLearningRate(context!.preferences, stats.epoch, epochs).toExponential(2)}`,
            },
          });
        },
      }
    );

  try {
    linearClassifierModel = await runtime.invoke(
      context.wasmPolicy.functionName,
      estimateBatchMemoryMB(dataset.features.length, dataset.featureNames.length, classCount),
      runClassifier
    );
  } catch (error) {
    if (shouldStop) return;
    throw error;
  }
  if (shouldStop) return;
  if (!linearClassifierModel) throw new Error('Linear Regression model was not created.');

  post({ type: 'status', payload: { phase: 'optimizing_parameters', message: 'Computing validation metrics...' } });
  const predictions = predictLinearRegressionClassifier(linearClassifierModel, dataset.features);
  const metricsBase = computeClassificationMetrics(dataset.labels, predictions, dataset.labelNames.length);
  const importanceScores = linearRegressionFeatureImportance(linearClassifierModel);
  latestMetrics = {
    kind: 'classification',
    ...metricsBase,
    featureImportance: rankFeatureImportance(dataset.featureNames, importanceScores).slice(0, 10),
  };
  latestArtifact = {
    modelType: 'linear_regression',
    backend: context.backend.label,
    trainedAt: new Date().toISOString(),
    featureNames: dataset.featureNames,
    labelNames: dataset.labelNames,
    modelData: {
      weights: linearClassifierModel.weights,
      bias: linearClassifierModel.bias,
      featureCount: linearClassifierModel.featureCount,
      classCount: linearClassifierModel.classCount,
      lossCurve: linearClassifierModel.lossCurve,
      mode: 'classifier',
      algorithm: 'linear_regression_classifier',
    },
  };
}

async function trainLogisticRegressionModel(
  runtime: WasmFunctionRuntime,
  gpuRuntime: WebGpuLinearRuntime | null
): Promise<void> {
  if (!context) throw new Error('Worker not initialized.');
  const dataset = context.dataset;

  post({ type: 'status', payload: { phase: 'training_model', message: 'Training logistic regression...' } });

  const epochs = Math.max(4, context.preferences.epochs);
  const classCount = Math.max(2, dataset.labelNames.length);

  if (gpuRuntime) {
    const featureCount = dataset.featureNames.length;
    const sampleCount = dataset.features.length;
    const weights = new Float32Array(featureCount * classCount);
    const bias = new Float32Array(classCount);
    for (let i = 0; i < weights.length; i++) {
      weights[i] = (Math.random() - 0.5) * 0.01;
    }
    const batchSize = Math.max(1, Math.min(sampleCount, context.preferences.batchSize));
    const l2 = context.preferences.weightDecay;
    const gradientClip = context.preferences.neuralNetwork.gradientClipping;
    const patience = context.preferences.earlyStoppingPatience;
    const weightState = createFlatOptimizerState(weights.length, context.preferences, l2);
    const biasState = createFlatOptimizerState(bias.length, context.preferences, 0);
    const lossCurve: Array<{ epoch: number; loss: number; accuracy: number }> = [];
    let bestLoss = Number.POSITIVE_INFINITY;
    let staleEpochs = 0;

    for (let epoch = 0; epoch < epochs; epoch++) {
      if (shouldStop) return;
      const indices = Array.from({ length: sampleCount }, (_, i) => i);
      if (context.preferences.shuffleEachEpoch) {
        for (let i = indices.length - 1; i > 0; i--) {
          const j = Math.floor(Math.random() * (i + 1));
          [indices[i], indices[j]] = [indices[j], indices[i]];
        }
      }
      const lrEpoch = scheduledLearningRate(context.preferences, epoch, epochs);
      let loss = 0;
      let correct = 0;

      for (let start = 0; start < sampleCount; start += batchSize) {
        if (shouldStop) return;
        const end = Math.min(sampleCount, start + batchSize);
        const count = Math.max(1, end - start);
        const batchFeatures = new Float32Array(count * featureCount);
        const batchTargets = new Int32Array(count);
        for (let i = 0; i < count; i++) {
          const rowIndex = indices[start + i];
          batchFeatures.set(dataset.features[rowIndex], i * featureCount);
          batchTargets[i] = dataset.labels[rowIndex] ?? 0;
        }

        const logits = await computeLinearLogits(
          gpuRuntime,
          batchFeatures,
          weights,
          bias,
          count,
          featureCount,
          classCount
        );
        const gradW = new Float32Array(weights.length);
        const gradB = new Float32Array(classCount);

        for (let row = 0; row < count; row++) {
          const base = row * classCount;
          const target = batchTargets[row] ?? 0;
          let maxLogit = -Infinity;
          for (let c = 0; c < classCount; c++) maxLogit = Math.max(maxLogit, logits[base + c]);
          let expSum = 0;
          const probs = new Array(classCount);
          for (let c = 0; c < classCount; c++) {
            const value = Math.exp(logits[base + c] - maxLogit);
            probs[c] = value;
            expSum += value;
          }
          for (let c = 0; c < classCount; c++) probs[c] /= expSum || 1;
          let predicted = 0;
          let best = -Infinity;
          for (let c = 0; c < classCount; c++) {
            if (probs[c] > best) {
              best = probs[c];
              predicted = c;
            }
          }
          if (predicted === target) correct++;
          loss += -Math.log((probs[target] ?? 1e-10) + 1e-10);

          for (let c = 0; c < classCount; c++) {
            const error = probs[c] - (c === target ? 1 : 0);
            gradB[c] += error;
            const rowBase = row * featureCount;
            for (let f = 0; f < featureCount; f++) {
              gradW[f * classCount + c] += error * batchFeatures[rowBase + f];
            }
          }
        }

        const invBatch = 1 / count;
        for (let i = 0; i < gradW.length; i++) gradW[i] *= invBatch;
        for (let c = 0; c < gradB.length; c++) gradB[c] *= invBatch;
        clipGradientVector(gradW, gradientClip);
        clipGradientVector(gradB, gradientClip);
        const coupledDecay = weightState.type !== 'adamw';
        applyFlatOptimizerStep(weights, gradW, lrEpoch, weightState, coupledDecay);
        applyFlatOptimizerStep(bias, gradB, lrEpoch, biasState, false);
      }

      const epochStats = {
        epoch: epoch + 1,
        loss: loss / Math.max(1, sampleCount),
        accuracy: correct / Math.max(1, sampleCount),
      };
      lossCurve.push(epochStats);
      post({
        type: 'progress',
        payload: {
          step: epoch + 1,
          epoch,
          loss: epochStats.loss,
          accuracy: epochStats.accuracy,
          percent: Math.floor(((epoch + 1) / epochs) * 100),
        },
      });
      post({
        type: 'log',
        payload: {
          message:
            `Epoch ${epoch + 1}/${epochs} - loss: ${epochStats.loss.toFixed(6)} - ` +
            `accuracy: ${(epochStats.accuracy * 100).toFixed(2)}% - lr: ${lrEpoch.toExponential(2)} (WebGPU)`,
        },
      });

      if (patience > 0) {
        if (epochStats.loss + 1e-9 < bestLoss) {
          bestLoss = epochStats.loss;
          staleEpochs = 0;
        } else {
          staleEpochs += 1;
          if (staleEpochs >= patience) {
            post({ type: 'log', payload: { message: `Early stopping triggered after ${patience} stale epochs.` } });
            break;
          }
        }
      }
    }

    if (shouldStop) return;
    logisticModel = {
      featureCount,
      classCount,
      weights: Array.from({ length: classCount }, (_, c) =>
        Array.from({ length: featureCount }, (_, f) => weights[f * classCount + c] ?? 0)
      ),
      bias: Array.from(bias),
      lossCurve,
    };
  } else {
    const runLogistic = () =>
      trainLogisticRegressionClassifier(
        dataset.features,
        dataset.labels,
        classCount,
        {
          epochs,
          learningRate: context!.preferences.learningRate,
          l2: context!.preferences.weightDecay,
          weightDecay: context!.preferences.weightDecay,
          batchSize: context!.preferences.batchSize,
          shuffle: context!.preferences.shuffleEachEpoch,
          earlyStoppingPatience: context!.preferences.earlyStoppingPatience,
          optimizer: context!.preferences.optimizer,
          momentum: context!.preferences.momentum,
          beta1: context!.preferences.beta1,
          beta2: context!.preferences.beta2,
          scheduler: context!.preferences.lrScheduler,
          warmupSteps: context!.preferences.warmupSteps,
          stepSize: context!.preferences.schedulerStepSize,
          schedulerGamma: context!.preferences.schedulerGamma,
          gradientClipping: context!.preferences.neuralNetwork.gradientClipping,
          shouldStop: () => shouldStop,
        },
        (stats) => {
          const percent = Math.floor((stats.epoch / epochs) * 100);
          post({
            type: 'progress',
            payload: {
              step: stats.epoch,
              epoch: stats.epoch - 1,
              loss: stats.loss,
              accuracy: stats.accuracy,
              percent,
            },
          });
          post({
            type: 'log',
            payload: {
              message:
                `Epoch ${stats.epoch}/${epochs} - loss: ${stats.loss.toFixed(6)} - ` +
                  `accuracy: ${(stats.accuracy * 100).toFixed(2)}% - ` +
                  `lr: ${scheduledLearningRate(context!.preferences, stats.epoch - 1, epochs).toExponential(2)}`,
            },
          });
        }
      );

    try {
      logisticModel = await runtime.invoke(
        context.wasmPolicy.functionName,
        estimateBatchMemoryMB(dataset.features.length, dataset.featureNames.length, classCount),
        runLogistic
      );
    } catch (error) {
      if (shouldStop) return;
      throw error;
    }
  }
  if (shouldStop) return;
  if (!logisticModel) throw new Error('Logistic Regression model was not created.');

  post({ type: 'status', payload: { phase: 'optimizing_parameters', message: 'Computing validation metrics...' } });
  const predictions = predictLogisticRegressionClassifier(logisticModel, dataset.features);
  const metricsBase = computeClassificationMetrics(dataset.labels, predictions, dataset.labelNames.length);
  const importanceScores = logisticRegressionFeatureImportance(logisticModel);
  latestMetrics = {
    kind: 'classification',
    ...metricsBase,
    featureImportance: rankFeatureImportance(dataset.featureNames, importanceScores).slice(0, 10),
  };
  latestArtifact = {
    modelType: 'logistic_regression',
    backend: context.backend.label,
    trainedAt: new Date().toISOString(),
    featureNames: dataset.featureNames,
    labelNames: dataset.labelNames,
    modelData: {
      weights: logisticModel.weights.flat(),
      bias: logisticModel.bias,
      featureCount: logisticModel.featureCount,
      classCount: logisticModel.classCount,
      lossCurve: logisticModel.lossCurve,
      algorithm: 'logistic_regression_classifier',
    },
  };
}

async function trainSvmModel(
  runtime: WasmFunctionRuntime,
  gpuRuntime: WebGpuLinearRuntime | null
): Promise<void> {
  if (!context) throw new Error('Worker not initialized.');
  const dataset = context.dataset;

  post({ type: 'status', payload: { phase: 'training_model', message: 'Training SVM...' } });

  const epochs = Math.max(4, context.preferences.epochs);
  const classCount = Math.max(2, dataset.labelNames.length);
  if (gpuRuntime && context.preferences.algorithm.svmKernel === 'linear') {
    const featureCount = dataset.featureNames.length;
    const sampleCount = dataset.features.length;
    const weights = new Float32Array(featureCount * classCount);
    const bias = new Float32Array(classCount);
    const regularization = context.preferences.weightDecay;
    const batchSize = Math.max(1, Math.min(sampleCount, context.preferences.batchSize));
    const gradientClip = context.preferences.neuralNetwork.gradientClipping;
    const patience = context.preferences.earlyStoppingPatience;
    const weightState = createFlatOptimizerState(weights.length, context.preferences, regularization);
    const biasState = createFlatOptimizerState(bias.length, context.preferences, 0);
    const lossCurve: Array<{ epoch: number; loss: number; accuracy: number }> = [];
    let bestLoss = Number.POSITIVE_INFINITY;
    let staleEpochs = 0;

    for (let epoch = 0; epoch < epochs; epoch++) {
      if (shouldStop) return;
      const indices = Array.from({ length: sampleCount }, (_, i) => i);
      if (context.preferences.shuffleEachEpoch) {
        for (let i = indices.length - 1; i > 0; i--) {
          const j = Math.floor(Math.random() * (i + 1));
          [indices[i], indices[j]] = [indices[j], indices[i]];
        }
      }
      const lrEpoch = scheduledLearningRate(context.preferences, epoch, epochs);
      let marginLoss = 0;
      let correct = 0;

      for (let start = 0; start < sampleCount; start += batchSize) {
        if (shouldStop) return;
        const end = Math.min(sampleCount, start + batchSize);
        const count = Math.max(1, end - start);
        const batchFeatures = new Float32Array(count * featureCount);
        const batchTargets = new Int32Array(count);
        for (let i = 0; i < count; i++) {
          const rowIndex = indices[start + i];
          batchFeatures.set(dataset.features[rowIndex], i * featureCount);
          batchTargets[i] = dataset.labels[rowIndex] ?? 0;
        }
        const scores = await computeLinearLogits(
          gpuRuntime,
          batchFeatures,
          weights,
          bias,
          count,
          featureCount,
          classCount
        );
        const gradW = new Float32Array(weights.length);
        const gradB = new Float32Array(classCount);

        for (let row = 0; row < count; row++) {
          const outBase = row * classCount;
          let pred = 0;
          let best = -Infinity;
          for (let c = 0; c < classCount; c++) {
            const value = scores[outBase + c];
            if (value > best) {
              best = value;
              pred = c;
            }
          }
          if (pred === (batchTargets[row] ?? 0)) correct++;

          for (let c = 0; c < classCount; c++) {
            const target = c === (batchTargets[row] ?? 0) ? 1 : -1;
            const margin = target * scores[outBase + c];
            const violates = margin < 1;
            if (violates) {
              marginLoss += 1 - margin;
            }
            const gradMultiplier = violates ? -target : 0;
            const rowBase = row * featureCount;
            for (let f = 0; f < featureCount; f++) {
              gradW[f * classCount + c] += gradMultiplier * batchFeatures[rowBase + f];
            }
            gradB[c] += gradMultiplier;
          }
        }

        const invBatch = 1 / count;
        for (let i = 0; i < gradW.length; i++) gradW[i] *= invBatch;
        for (let c = 0; c < gradB.length; c++) gradB[c] *= invBatch;
        clipGradientVector(gradW, gradientClip);
        clipGradientVector(gradB, gradientClip);
        const coupledDecay = weightState.type !== 'adamw';
        applyFlatOptimizerStep(weights, gradW, lrEpoch, weightState, coupledDecay);
        applyFlatOptimizerStep(bias, gradB, lrEpoch, biasState, false);
      }

      const stats = {
        epoch: epoch + 1,
        loss: marginLoss / Math.max(1, sampleCount * classCount),
        accuracy: correct / Math.max(1, sampleCount),
      };
      lossCurve.push(stats);
      post({
        type: 'progress',
        payload: {
          step: stats.epoch,
          epoch: epoch,
          loss: stats.loss,
          accuracy: stats.accuracy,
          percent: Math.floor((stats.epoch / epochs) * 100),
        },
      });
      post({
        type: 'log',
        payload: {
          message:
            `Epoch ${stats.epoch}/${epochs} - hinge_loss: ${stats.loss.toFixed(6)} - ` +
            `accuracy: ${(stats.accuracy * 100).toFixed(2)}% - lr: ${lrEpoch.toExponential(2)} (WebGPU)`,
        },
      });

      if (patience > 0) {
        if (stats.loss + 1e-9 < bestLoss) {
          bestLoss = stats.loss;
          staleEpochs = 0;
        } else {
          staleEpochs += 1;
          if (staleEpochs >= patience) {
            post({ type: 'log', payload: { message: `Early stopping triggered after ${patience} stale epochs.` } });
            break;
          }
        }
      }
    }

    svmModel = {
      featureCount,
      classCount,
      kernel: 'linear',
      weights: Array.from({ length: classCount }, (_, c) =>
        Array.from({ length: featureCount }, (_, f) => weights[f * classCount + c] ?? 0)
      ),
      bias: Array.from(bias),
      lossCurve,
    };
  } else {
    const runSvm = () =>
      trainSvmClassifier(
        dataset.features,
        dataset.labels,
        classCount,
        {
          epochs,
          learningRate: context!.preferences.learningRate,
          regularization: context!.preferences.weightDecay,
          weightDecay: context!.preferences.weightDecay,
          batchSize: context!.preferences.batchSize,
          shuffle: context!.preferences.shuffleEachEpoch,
          earlyStoppingPatience: context!.preferences.earlyStoppingPatience,
          optimizer: context!.preferences.optimizer,
          momentum: context!.preferences.momentum,
          beta1: context!.preferences.beta1,
          beta2: context!.preferences.beta2,
          scheduler: context!.preferences.lrScheduler,
          warmupSteps: context!.preferences.warmupSteps,
          stepSize: context!.preferences.schedulerStepSize,
          schedulerGamma: context!.preferences.schedulerGamma,
          gradientClipping: context!.preferences.neuralNetwork.gradientClipping,
          kernel: context!.preferences.algorithm.svmKernel,
          shouldStop: () => shouldStop,
        },
        (stats) => {
          post({
            type: 'progress',
            payload: {
              step: stats.epoch,
              epoch: stats.epoch - 1,
              loss: stats.loss,
              accuracy: stats.accuracy,
              percent: Math.floor((stats.epoch / epochs) * 100),
            },
          });
          post({
            type: 'log',
            payload: {
              message:
                `Epoch ${stats.epoch}/${epochs} - hinge_loss: ${stats.loss.toFixed(6)} - ` +
                  `accuracy: ${(stats.accuracy * 100).toFixed(2)}% - ` +
                  `lr: ${scheduledLearningRate(context!.preferences, stats.epoch - 1, epochs).toExponential(2)}`,
            },
          });
        }
      );

    try {
      svmModel = await runtime.invoke(
        context.wasmPolicy.functionName,
        estimateBatchMemoryMB(dataset.features.length, dataset.featureNames.length, classCount),
        runSvm
      );
    } catch (error) {
      if (shouldStop) return;
      throw error;
    }
  }
  if (shouldStop) return;
  if (!svmModel) throw new Error('SVM model was not created.');

  post({ type: 'status', payload: { phase: 'optimizing_parameters', message: 'Computing validation metrics...' } });
  const predictions = predictSvmClassifier(svmModel, dataset.features);
  const metricsBase = computeClassificationMetrics(dataset.labels, predictions, dataset.labelNames.length);
  const importanceScores = svmFeatureImportance(svmModel);
  latestMetrics = {
    kind: 'classification',
    ...metricsBase,
    featureImportance: rankFeatureImportance(dataset.featureNames, importanceScores).slice(0, 10),
  };
  latestArtifact = {
    modelType: 'svm',
    backend: context.backend.label,
    trainedAt: new Date().toISOString(),
    featureNames: dataset.featureNames,
    labelNames: dataset.labelNames,
    modelData: {
      weights: svmModel.weights.flat(),
      bias: svmModel.bias,
      featureCount: svmModel.featureCount,
      classCount: svmModel.classCount,
      kernel: svmModel.kernel,
      lossCurve: svmModel.lossCurve,
      algorithm: 'svm_classifier',
    },
  };
}

async function trainKnnModel(runtime: WasmFunctionRuntime): Promise<void> {
  if (!context) throw new Error('Worker not initialized.');
  const dataset = context.dataset;
  const classCount = Math.max(2, dataset.labelNames.length);

  post({ type: 'status', payload: { phase: 'training_model', message: 'Building KNN index...' } });

  const runKnn = () =>
    trainKnnClassifier(dataset.features, dataset.labels, classCount, {
      neighbors: context!.preferences.algorithm.knnNeighbors,
      distanceMetric: context!.preferences.algorithm.knnDistanceMetric,
    });

  try {
    knnModel = await runtime.invoke(
      context.wasmPolicy.functionName,
      estimateBatchMemoryMB(dataset.features.length, dataset.featureNames.length, classCount),
      runKnn
    );
  } catch (error) {
    if (shouldStop) return;
    throw error;
  }
  if (shouldStop) return;
  if (!knnModel) throw new Error('KNN model was not created.');

  post({
    type: 'progress',
    payload: {
      step: 1,
      epoch: 0,
      loss: 0,
      accuracy: 1,
      percent: 100,
    },
  });
  post({
    type: 'log',
    payload: {
      message: `KNN index ready (k=${knnModel.k}, metric=${knnModel.distanceMetric}).`,
    },
  });

  post({ type: 'status', payload: { phase: 'optimizing_parameters', message: 'Computing validation metrics...' } });
  const predictions = predictKnnClassifier(knnModel, dataset.features);
  const metricsBase = computeClassificationMetrics(dataset.labels, predictions, dataset.labelNames.length);
  const importanceScores = knnFeatureImportance(knnModel);
  latestMetrics = {
    kind: 'classification',
    ...metricsBase,
    featureImportance: rankFeatureImportance(dataset.featureNames, importanceScores).slice(0, 10),
  };
  latestArtifact = {
    modelType: 'knn',
    backend: context.backend.label,
    trainedAt: new Date().toISOString(),
    featureNames: dataset.featureNames,
    labelNames: dataset.labelNames,
    modelData: {
      k: knnModel.k,
      distanceMetric: knnModel.distanceMetric,
      featureCount: knnModel.featureCount,
      classCount: knnModel.classCount,
      trainFeatures: knnModel.trainFeatures,
      trainLabels: knnModel.trainLabels,
      algorithm: 'knn_classifier',
    },
  };
}

async function trainRecurrentModel(runtime: WasmFunctionRuntime): Promise<void> {
  if (!context) throw new Error('Worker not initialized.');
  if (context.resolvedModel !== 'rnn' && context.resolvedModel !== 'lstm' && context.resolvedModel !== 'gru') {
    throw new Error('Recurrent trainer invoked for non-recurrent model.');
  }
  const dataset = context.dataset;
  const classCount = Math.max(2, dataset.labelNames.length);
  const variant = context.resolvedModel;
  const epochs = Math.max(4, context.preferences.epochs);
  const featureCount = dataset.featureNames.length;

  post({
    type: 'status',
    payload: {
      phase: 'training_model',
      message: `Training ${variant.toUpperCase()} sequence model...`,
    },
  });

  const runRecurrent = () =>
    trainRecurrentClassifier(
      dataset.features,
      dataset.labels,
      classCount,
      {
        variant,
        epochs,
        learningRate: context!.preferences.learningRate,
        hiddenSize: clamp(Math.round(context!.preferences.neuralNetwork.neuronsPerLayer || 64), 16, 256),
        inputSize: clamp(Math.round(Math.sqrt(Math.max(4, featureCount))), 2, Math.min(64, Math.max(2, featureCount))),
        l2: context!.preferences.weightDecay,
        weightDecay: context!.preferences.weightDecay,
        batchSize: context!.preferences.batchSize,
        shuffle: context!.preferences.shuffleEachEpoch,
        earlyStoppingPatience: context!.preferences.earlyStoppingPatience,
        optimizer: context!.preferences.optimizer,
        momentum: context!.preferences.momentum,
        beta1: context!.preferences.beta1,
        beta2: context!.preferences.beta2,
        scheduler: context!.preferences.lrScheduler,
        warmupSteps: context!.preferences.warmupSteps,
        stepSize: context!.preferences.schedulerStepSize,
        schedulerGamma: context!.preferences.schedulerGamma,
        gradientClipping: context!.preferences.neuralNetwork.gradientClipping,
        shouldStop: () => shouldStop,
      },
      (stats) => {
        const percent = Math.floor((stats.epoch / epochs) * 100);
        post({
          type: 'progress',
          payload: {
            step: stats.epoch,
            epoch: stats.epoch - 1,
            loss: stats.loss,
            accuracy: stats.accuracy,
            percent,
          },
        });
        post({
          type: 'log',
          payload: {
            message:
              `Epoch ${stats.epoch}/${epochs} - loss: ${stats.loss.toFixed(6)} - ` +
              `accuracy: ${(stats.accuracy * 100).toFixed(2)}% - ` +
              `lr: ${scheduledLearningRate(context!.preferences, stats.epoch - 1, epochs).toExponential(2)} (${variant.toUpperCase()})`,
          },
        });
      }
    );

  try {
    recurrentModel = await runtime.invoke(
      context.wasmPolicy.functionName,
      estimateBatchMemoryMB(dataset.features.length, dataset.featureNames.length, classCount),
      runRecurrent
    );
  } catch (error) {
    if (shouldStop) return;
    throw error;
  }
  if (shouldStop) return;
  if (!recurrentModel) throw new Error(`${variant.toUpperCase()} model was not created.`);

  post({ type: 'status', payload: { phase: 'optimizing_parameters', message: 'Computing validation metrics...' } });
  const predictions = predictRecurrentClassifier(recurrentModel, dataset.features);
  const metricsBase = computeClassificationMetrics(dataset.labels, predictions, dataset.labelNames.length);
  const importanceScores = recurrentFeatureImportance(recurrentModel);
  latestMetrics = {
    kind: 'classification',
    ...metricsBase,
    featureImportance: rankFeatureImportance(dataset.featureNames, importanceScores).slice(0, 10),
  };
  latestArtifact = {
    modelType: variant,
    backend: context.backend.label,
    trainedAt: new Date().toISOString(),
    featureNames: dataset.featureNames,
    labelNames: dataset.labelNames,
    modelData: {
      variant: recurrentModel.variant,
      featureCount: recurrentModel.featureCount,
      classCount: recurrentModel.classCount,
      inputSize: recurrentModel.inputSize,
      sequenceLength: recurrentModel.sequenceLength,
      hiddenSize: recurrentModel.hiddenSize,
      params: recurrentModel.params,
      outputWeights: recurrentModel.outputWeights,
      outputBias: recurrentModel.outputBias,
      lossCurve: recurrentModel.lossCurve,
      featureImportance: recurrentModel.featureImportance,
      algorithm: `${variant}_classifier`,
    },
  };
}

async function trainKMeansModel(runtime: WasmFunctionRuntime): Promise<void> {
  if (!context) throw new Error('Worker not initialized.');
  const dataset = context.dataset;
  const classCount = Math.max(2, dataset.labelNames.length);
  const maxIterations = Math.max(8, Math.min(300, context.preferences.epochs * 4));

  post({ type: 'status', payload: { phase: 'training_model', message: 'Training K-Means clustering...' } });
  post({ type: 'log', payload: { message: 'K-Means uses unlabeled clusters, then maps clusters to labels for evaluation metrics.' } });

  const runKMeans = () =>
    trainKMeansClassifier(
      dataset.features,
      dataset.labels,
      classCount,
      {
        clusters: context!.preferences.algorithm.kmeansClusters,
        maxIterations,
        tolerance: context!.preferences.optimizeForSmallerModel ? 1e-3 : 1e-4,
        shouldStop: () => shouldStop,
      },
      (stats) => {
        const percent = Math.floor((stats.iteration / maxIterations) * 100);
        post({
          type: 'progress',
          payload: {
            step: stats.iteration,
            epoch: stats.iteration - 1,
            loss: stats.inertia,
            accuracy: 1 - Math.min(1, stats.movedFraction),
            percent,
          },
        });
        post({
          type: 'log',
          payload: {
            message:
              `Iter ${stats.iteration}/${maxIterations} - inertia: ${stats.inertia.toFixed(6)} - ` +
              `moved: ${(stats.movedFraction * 100).toFixed(2)}%`,
          },
        });
      }
    );

  const kmeansMB = estimateBatchMemoryMB(dataset.features.length, dataset.featureNames.length, classCount);
  if (kmeansMB > context.wasmPolicy.memoryBudgetMB) {
    throw new Error(
      `K-Means needs ~${kmeansMB.toFixed(2)}MB (estimated), above WASM memory budget ${context.wasmPolicy.memoryBudgetMB}MB.`
    );
  }

  try {
    kmeansModel = runKMeans();
  } catch (error) {
    if (shouldStop) return;
    throw error;
  }
  if (shouldStop) return;
  if (!kmeansModel) throw new Error('K-Means model was not created.');

  post({
    type: 'log',
    payload: {
      message: `K-Means finished after ${kmeansModel.lossCurve.length} iteration(s) (max ${maxIterations}).`,
    },
  });

  post({ type: 'status', payload: { phase: 'optimizing_parameters', message: 'Computing validation metrics...' } });
  post({
    type: 'log',
    payload: {
      message: `Scoring ${dataset.features.length.toLocaleString()} training samples (cluster → label mapping)...`,
    },
  });

  let lastPredPct = -1;
  const predictions = await predictKMeansClassifierAsync(kmeansModel, dataset.features, {
    onProgress: (done, total) => {
      if (total < 5000) return;
      const pct = Math.floor((done / total) * 100);
      if (done === total || pct >= lastPredPct + 10) {
        lastPredPct = pct;
        post({
          type: 'log',
          payload: { message: `Prediction progress ${done.toLocaleString()}/${total.toLocaleString()} (${pct}%)` },
        });
      }
    },
  });
  const metricsBase = computeClassificationMetrics(dataset.labels, predictions, dataset.labelNames.length);
  const importanceScores = kmeansFeatureImportance(kmeansModel);
  latestMetrics = {
    kind: 'classification',
    ...metricsBase,
    featureImportance: rankFeatureImportance(dataset.featureNames, importanceScores).slice(0, 10),
  };
  latestArtifact = {
    modelType: 'kmeans',
    backend: context.backend.label,
    trainedAt: new Date().toISOString(),
    featureNames: dataset.featureNames,
    labelNames: dataset.labelNames,
    modelData: {
      featureCount: kmeansModel.featureCount,
      clusterCount: kmeansModel.clusterCount,
      centroids: kmeansModel.centroids,
      assignments: kmeansModel.assignments.length <= 100_000 ? kmeansModel.assignments : [],
      assignmentsOmitted: kmeansModel.assignments.length > 100_000,
      clusterToLabel: kmeansModel.clusterToLabel,
      classCount: kmeansModel.classCount,
      fallbackLabel: kmeansModel.fallbackLabel,
      lossCurve: kmeansModel.lossCurve,
      algorithm: 'kmeans_classifier',
    },
  };
}

async function trainDbscanModel(runtime: WasmFunctionRuntime): Promise<void> {
  if (!context) throw new Error('Worker not initialized.');
  const dataset = context.dataset;
  const classCount = Math.max(2, dataset.labelNames.length);
  const totalPoints = Math.max(1, dataset.features.length);
  let lastReportedPercent = -1;
  let lastLoggedPercent = -1;

  post({ type: 'status', payload: { phase: 'training_model', message: 'Training DBSCAN clustering...' } });
  post({
    type: 'log',
    payload: {
      message:
        `DBSCAN params - eps=${context.preferences.algorithm.dbscanEpsilon.toFixed(3)}, ` +
        `min_samples=${context.preferences.algorithm.dbscanMinSamples}`,
    },
  });

  const runDbscan = () =>
    trainDbscanClassifier(
      dataset.features,
      dataset.labels,
      classCount,
      {
        epsilon: context!.preferences.algorithm.dbscanEpsilon,
        minSamples: context!.preferences.algorithm.dbscanMinSamples,
        shouldStop: () => shouldStop,
      },
      (stats) => {
        const percent = Math.floor((stats.processedPoints / totalPoints) * 100);
        if (percent !== lastReportedPercent) {
          lastReportedPercent = percent;
          post({
            type: 'progress',
            payload: {
              step: stats.processedPoints,
              epoch: stats.clusterCount,
              loss: stats.noisePoints / Math.max(1, stats.processedPoints),
              accuracy: stats.clusterCount / Math.max(1, classCount * 2),
              percent,
            },
          });
        }
        if (percent >= lastLoggedPercent + 10 || percent === 100) {
          lastLoggedPercent = percent;
          post({
            type: 'log',
            payload: {
              message:
                `DBSCAN progress ${percent}% - clusters: ${stats.clusterCount}, ` +
                `noise: ${stats.noisePoints}`,
            },
          });
        }
      }
    );

  const dbscanMB = estimateBatchMemoryMB(dataset.features.length, dataset.featureNames.length, classCount);
  if (dbscanMB > context.wasmPolicy.memoryBudgetMB) {
    throw new Error(
      `DBSCAN needs ~${dbscanMB.toFixed(2)}MB (estimated), above WASM memory budget ${context.wasmPolicy.memoryBudgetMB}MB.`
    );
  }

  try {
    dbscanModel = runDbscan();
  } catch (error) {
    if (shouldStop) return;
    throw error;
  }
  if (shouldStop) return;
  if (!dbscanModel) throw new Error('DBSCAN model was not created.');

  post({ type: 'status', payload: { phase: 'optimizing_parameters', message: 'Computing validation metrics...' } });
  post({
    type: 'log',
    payload: {
      message: `Scoring ${dataset.features.length.toLocaleString()} samples with DBSCAN core points...`,
    },
  });

  let lastDbPct = -1;
  const predictions = await predictDbscanClassifierAsync(dbscanModel, dataset.features, {
    onProgress: (done, total) => {
      if (total < 5000) return;
      const pct = Math.floor((done / total) * 100);
      if (done === total || pct >= lastDbPct + 10) {
        lastDbPct = pct;
        post({
          type: 'log',
          payload: { message: `Prediction progress ${done.toLocaleString()}/${total.toLocaleString()} (${pct}%)` },
        });
      }
    },
  });
  const metricsBase = computeClassificationMetrics(dataset.labels, predictions, dataset.labelNames.length);
  const importanceScores = dbscanFeatureImportance(dbscanModel);
  latestMetrics = {
    kind: 'classification',
    ...metricsBase,
    featureImportance: rankFeatureImportance(dataset.featureNames, importanceScores).slice(0, 10),
  };
  latestArtifact = {
    modelType: 'dbscan',
    backend: context.backend.label,
    trainedAt: new Date().toISOString(),
    featureNames: dataset.featureNames,
    labelNames: dataset.labelNames,
    modelData: {
      featureCount: dbscanModel.featureCount,
      epsilon: dbscanModel.epsilon,
      minSamples: dbscanModel.minSamples,
      clusterCount: dbscanModel.clusterCount,
      noiseCount: dbscanModel.noiseCount,
      trainFeatures: dbscanModel.trainFeatures,
      clusterLabels: dbscanModel.clusterLabels,
      corePointIndices: dbscanModel.corePointIndices,
      clusterToLabel: dbscanModel.clusterToLabel,
      classCount: dbscanModel.classCount,
      fallbackLabel: dbscanModel.fallbackLabel,
      lossCurve: dbscanModel.lossCurve,
      algorithm: 'dbscan_classifier',
    },
  };
}

async function startTraining(): Promise<void> {
  if (!context) throw new Error('Worker not initialized.');
  hydrateDatasetFromTransfer(context.dataset, context.datasetTransfer);
  shouldStop = false;
  linearClassifierModel = null;
  linearRegressorModel = null;
  logisticModel = null;
  svmModel = null;
  knnModel = null;
  recurrentModel = null;
  kmeansModel = null;
  dbscanModel = null;
  latestArtifact = null;
  latestMetrics = null;
  const runtime = makeRuntime(context.wasmPolicy);
  post({
    type: 'log',
    payload: {
      message:
        `Hybrid runtime active ("${context.wasmPolicy.functionName}") ` +
        `timeout=${context.wasmPolicy.invocationTimeoutMs}ms ` +
        `retries=${context.wasmPolicy.retryCount} ` +
        `memoryBudget=${context.wasmPolicy.memoryBudgetMB}MB`,
    },
  });
  post({
    type: 'log',
    payload: {
      message:
        `Loop config - epochs=${context.preferences.epochs}, batch_size=${context.preferences.batchSize}, ` +
        `shuffle=${context.preferences.shuffleEachEpoch ? 'on' : 'off'}, ` +
        `early_stop_patience=${context.preferences.earlyStoppingPatience}`,
    },
  });
  post({
    type: 'log',
    payload: {
      message:
        `Optimizer config - type=${context.preferences.optimizer}, lr=${context.preferences.learningRate}, ` +
        `scheduler=${context.preferences.lrScheduler}, warmup=${context.preferences.warmupSteps}, ` +
        `weight_decay=${context.preferences.weightDecay}`,
    },
  });

  let gpuLinearRuntime: WebGpuLinearRuntime | null = null;
  if (context.backend.kind === 'webgpu') {
    gpuLinearRuntime = await WebGpuLinearRuntime.create((reason) => {
      shouldStop = true;
      post({
        type: 'status',
        payload: {
          phase: 'error',
          message: `WebGPU device lost: ${reason}`,
        },
      });
      post({
        type: 'log',
        payload: {
          message: `WebGPU device lost (${reason}). Training run stopped safely.`,
        },
      });
    });
    post({
      type: 'log',
      payload: {
        message: gpuLinearRuntime
          ? 'WebGPU linear training backend initialized for eligible models.'
          : 'WebGPU backend unavailable at runtime. Falling back to CPU linear kernels.',
      },
    });
  }

  const checkpoint = pendingCheckpoint ?? (await loadCheckpoint(context.runId));
  pendingCheckpoint = null;
  const nnCheckpoint = checkpoint?.kind === 'nn' ? checkpoint : null;
  const rfCheckpoint = checkpoint?.kind === 'rf' ? checkpoint : null;

  post({ type: 'status', payload: { phase: 'cleaning_data', message: 'Validating and preparing dataset...' } });
  await new Promise((resolve) => setTimeout(resolve, 150));

  if (context.resolvedModel === 'neural_network') {
    await trainNeuralNetwork(nnCheckpoint, runtime);
  } else if (context.resolvedModel === 'decision_tree') {
    await trainDecisionTreeModel(rfCheckpoint, runtime);
  } else if (context.resolvedModel === 'logistic_regression') {
    await trainLogisticRegressionModel(runtime, gpuLinearRuntime);
  } else if (context.resolvedModel === 'linear_regression') {
    await trainLinearRegressionModel(runtime);
  } else if (context.resolvedModel === 'svm') {
    await trainSvmModel(runtime, gpuLinearRuntime);
  } else if (context.resolvedModel === 'knn') {
    await trainKnnModel(runtime);
  } else if (context.resolvedModel === 'rnn' || context.resolvedModel === 'lstm' || context.resolvedModel === 'gru') {
    await trainRecurrentModel(runtime);
  } else if (context.resolvedModel === 'kmeans') {
    await trainKMeansModel(runtime);
  } else if (context.resolvedModel === 'dbscan') {
    await trainDbscanModel(runtime);
  } else {
    await trainRandomForestModel(rfCheckpoint, runtime);
  }

  if (shouldStop) return;
  if (!latestMetrics || !latestArtifact) throw new Error('Training finished without producing results.');

  post({ type: 'status', payload: { phase: 'completed', message: 'Model training completed.' } });
  post({
    type: 'training_complete',
    payload: {
      metrics: latestMetrics,
      artifact: latestArtifact,
    },
  });
}

async function exportModel(format: 'pth' | 'tensor' | 'onnx' | 'kaya'): Promise<void> {
  if (!context || !latestArtifact || !latestMetrics) {
    throw new Error('Train a model before exporting.');
  }
  const { blob, filename } = await exportModelArtifact(format, {
    runId: context.runId,
    artifact: latestArtifact,
    metrics: latestMetrics,
    dataset: context.dataset,
    preferences: context.preferences,
  });
  post({ type: 'export_ready', payload: { format, filename, blob } });
}

self.onmessage = async (event: MessageEvent<MainToTrainingWorkerMessage>) => {
  const message = event.data;
  try {
    switch (message.type) {
      case 'init': {
        resumed = false;
        const { runId, dataset, datasetTransfer, modelChoice, preferences, capabilities } = message.payload;
        opfsDisabledReason = null;
        opfsFailureLogged = false;
        const normalizedPreferences = normalizePreferences(preferences);
        const wasmPolicy = resolveWasmInvocationPolicy(normalizedPreferences.runtime.wasmEditor);
        const backend = selectComputeBackend(capabilities, normalizedPreferences.useMoreCompute);
        const modelResolution = chooseModel(modelChoice, dataset, normalizedPreferences);
        const resolvedModel = modelResolution.resolvedModel;
        pendingCheckpoint = await loadCheckpoint(runId);

        context = {
          runId,
          dataset,
          datasetTransfer: datasetTransfer ?? null,
          modelChoice,
          resolvedModel,
          preferences: normalizedPreferences,
          wasmPolicy,
          capabilities,
          backend,
        };

        post({
          type: 'status',
          payload: {
            phase: 'idle',
            message: `Hybrid backend selected: ${backend.label}. ${backend.reason}`,
          },
        });
        post({
          type: 'ready',
          payload: {
            resumed: Boolean(pendingCheckpoint),
            resolvedModel,
            backend: backend.label,
          },
        });
        if (modelResolution.adaptationNote) {
          post({
            type: 'log',
            payload: {
              message: modelResolution.adaptationNote,
            },
          });
        }
        break;
      }
      case 'start':
        await startTraining();
        break;
      case 'stop':
        shouldStop = true;
        post({ type: 'status', payload: { phase: 'idle', message: 'Training stopped by user.' } });
        break;
      case 'export':
        await exportModel(message.payload.format);
        break;
      case 'clear_checkpoint':
        await clearCheckpoint(message.payload.runId);
        if (context?.runId === message.payload.runId) {
          pendingCheckpoint = null;
        }
        post({ type: 'log', payload: { message: 'Checkpoint cleared.' } });
        break;
    }
  } catch (error: any) {
    post({ type: 'error', payload: { message: error?.message ?? 'Training worker error.' } });
  }
};
