import { exportModelArtifact } from './exporter';
import { ProcessedDataset } from '../types/data';
import { ModelMetrics, TrainedModelArtifact, TrainingPreferences } from '../types/training-workflow';

export interface OnnxRoundTripCheckResult {
  ok: boolean;
  localPrediction: number;
  onnxPrediction: number;
  localLabel: string;
  onnxLabel: string;
  outputNames: string[];
  detail: string;
}

function argMax(values: number[]): number {
  let idx = 0;
  let best = -Infinity;
  for (let i = 0; i < values.length; i++) {
    if (values[i] > best) {
      best = values[i];
      idx = i;
    }
  }
  return idx;
}

function applyActivation(values: number[], activation: unknown): number[] {
  if (activation === 'tanh') {
    return values.map((value) => Math.tanh(value));
  }
  if (activation === 'sigmoid') {
    return values.map((value) => 1 / (1 + Math.exp(-value)));
  }
  return values.map((value) => Math.max(0, value));
}

function denseForward(input: number[], weights: number[], bias: number[], outputSize: number): number[] {
  const output = new Array(outputSize).fill(0);
  for (let o = 0; o < outputSize; o++) {
    let sum = bias[o] ?? 0;
    for (let i = 0; i < input.length; i++) {
      sum += input[i] * (weights[i * outputSize + o] ?? 0);
    }
    output[o] = sum;
  }
  return output;
}

function applyBatchNorm(
  input: number[],
  layer: {
    gamma: number[];
    beta: number[];
    runningMean: number[];
    runningVar: number[];
    epsilon?: number;
  }
): number[] {
  const eps = layer.epsilon ?? 1e-5;
  return input.map((value, index) => {
    const mean = layer.runningMean[index] ?? 0;
    const variance = layer.runningVar[index] ?? 1;
    const gamma = layer.gamma[index] ?? 1;
    const beta = layer.beta[index] ?? 0;
    return gamma * ((value - mean) / Math.sqrt(variance + eps)) + beta;
  });
}

type ForestNode = {
  prediction: number;
  featureIndex: number | null;
  threshold: number;
  left: ForestNode | null;
  right: ForestNode | null;
};

declare global {
  interface Window {
    ort?: any;
  }
}

const ORT_SCRIPT_URL = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js';
const ORT_WASM_BASE_URL = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
let ortLoadPromise: Promise<any> | null = null;

function loadOrtGlobal(): Promise<any> {
  if (typeof window === 'undefined') {
    throw new Error('ONNX Runtime round-trip check is only available in browser context.');
  }
  if (window.ort) return Promise.resolve(window.ort);
  if (ortLoadPromise) return ortLoadPromise;

  ortLoadPromise = new Promise((resolve, reject) => {
    const existing = document.querySelector('script[data-ort-runtime="true"]') as HTMLScriptElement | null;
    if (existing) {
      existing.addEventListener('load', () => resolve(window.ort), { once: true });
      existing.addEventListener('error', () => reject(new Error('Failed to load ONNX Runtime script.')), {
        once: true,
      });
      return;
    }

    const script = document.createElement('script');
    script.src = ORT_SCRIPT_URL;
    script.async = true;
    script.dataset.ortRuntime = 'true';
    script.onload = () => resolve(window.ort);
    script.onerror = () => reject(new Error('Failed to load ONNX Runtime script from CDN.'));
    document.head.appendChild(script);
  });

  return ortLoadPromise;
}

function predictForestTree(node: ForestNode, sample: number[]): number {
  if (node.featureIndex === null || !node.left || !node.right) return node.prediction;
  if (sample[node.featureIndex] <= node.threshold) return predictForestTree(node.left, sample);
  return predictForestTree(node.right, sample);
}

function predictLocal(artifact: TrainedModelArtifact, sample: number[]): number {
  if (artifact.modelType === 'random_forest' || artifact.modelType === 'decision_tree') {
    const trees = (artifact.modelData.trees as ForestNode[]) ?? [];
    const votes = new Map<number, number>();
    for (const tree of trees) {
      const prediction = predictForestTree(tree, sample);
      votes.set(prediction, (votes.get(prediction) ?? 0) + 1);
    }
    return Array.from(votes.entries()).sort((a, b) => b[1] - a[1])[0]?.[0] ?? 0;
  }

  if (
    artifact.modelType === 'linear_regression' ||
    artifact.modelType === 'logistic_regression' ||
    artifact.modelType === 'svm'
  ) {
    const weights = (artifact.modelData.weights as number[]) ?? [];
    const bias = (artifact.modelData.bias as number[]) ?? [];
    const mode = String(artifact.modelData.mode ?? 'classifier');
    if (artifact.modelType === 'linear_regression' && mode === 'regressor') {
      if (weights.length !== sample.length || !bias.length) return 0;
      let value = bias[0] ?? 0;
      for (let featureIdx = 0; featureIdx < sample.length; featureIdx++) {
        value += sample[featureIdx] * (weights[featureIdx] ?? 0);
      }
      return value;
    }
    const classCount = bias.length;
    if (!classCount || weights.length !== sample.length * classCount) return 0;
    const logits = new Array(classCount).fill(0);
    for (let classIdx = 0; classIdx < classCount; classIdx++) {
      let sum = bias[classIdx] ?? 0;
      for (let featureIdx = 0; featureIdx < sample.length; featureIdx++) {
        sum += sample[featureIdx] * (weights[featureIdx * classCount + classIdx] ?? 0);
      }
      logits[classIdx] = sum;
    }
    return argMax(logits);
  }

  const linearLayers = artifact.modelData.linearLayers as
    | Array<{ inputSize: number; outputSize: number; weights: number[]; bias: number[] }>
    | undefined;
  if (Array.isArray(linearLayers) && linearLayers.length > 0) {
    let current = [...sample];
    const activation = artifact.modelData.activation;
    const batchNormLayers = artifact.modelData.batchNormLayers as
      | Array<{
          gamma: number[];
          beta: number[];
          runningMean: number[];
          runningVar: number[];
          epsilon?: number;
        }>
      | undefined;
    for (let i = 0; i < linearLayers.length; i++) {
      const layer = linearLayers[i];
      if (layer.inputSize !== current.length || layer.weights.length !== layer.inputSize * layer.outputSize) return 0;
      current = denseForward(current, layer.weights, layer.bias, layer.outputSize);
      if (i < linearLayers.length - 1) {
        if (Array.isArray(batchNormLayers) && batchNormLayers[i]) {
          current = applyBatchNorm(current, batchNormLayers[i]);
        }
        current = applyActivation(current, activation);
      }
    }
    return argMax(current);
  }

  const layers = (artifact.modelData.layers as number[][]) ?? [];
  if (layers.length < 4) return 0;
  const [w1, b1, w2, b2] = layers;
  const hiddenSize = b1.length;
  const outputSize = b2.length;

  const hidden = new Array(hiddenSize).fill(0);
  for (let h = 0; h < hiddenSize; h++) {
    let sum = b1[h];
    for (let f = 0; f < sample.length; f++) sum += sample[f] * w1[f * hiddenSize + h];
    hidden[h] = Math.max(0, sum);
  }

  const output = new Array(outputSize).fill(0);
  for (let o = 0; o < outputSize; o++) {
    let sum = b2[o];
    for (let h = 0; h < hiddenSize; h++) sum += hidden[h] * w2[h * outputSize + o];
    output[o] = sum;
  }
  return argMax(output);
}

function tensorDataToNumberArray(data: unknown): number[] {
  if (ArrayBuffer.isView(data)) return Array.from(data as any);
  if (Array.isArray(data)) return data.map((value) => Number(value));
  return [];
}

function resolveOnnxPrediction(
  artifact: TrainedModelArtifact,
  outputs: Record<string, any>,
  labelCount: number
): number {
  if (artifact.modelType === 'random_forest' || artifact.modelType === 'decision_tree') {
    const scoreOutput = outputs.Z || outputs[Object.keys(outputs).find((name) => name.toLowerCase().includes('z')) || ''];
    if (scoreOutput?.data) {
      const scores = tensorDataToNumberArray(scoreOutput.data).slice(0, labelCount);
      if (scores.length === labelCount) return argMax(scores);
    }
    const labelOutput = outputs.Y || outputs[Object.keys(outputs)[0]];
    if (labelOutput?.data) {
      const value = labelOutput.data[0];
      return Number(typeof value === 'bigint' ? Number(value) : value) || 0;
    }
    return 0;
  }

  if (artifact.modelType === 'linear_regression' && String(artifact.modelData.mode ?? 'classifier') === 'regressor') {
    const output = outputs.Y || outputs[Object.keys(outputs)[0]];
    const values = output?.data ? tensorDataToNumberArray(output.data) : [];
    return values[0] ?? 0;
  }

  const output = outputs.Y || outputs[Object.keys(outputs)[0]];
  const values = output?.data ? tensorDataToNumberArray(output.data).slice(0, labelCount) : [];
  if (!values.length) return 0;
  return argMax(values);
}

export async function runOnnxRoundTripCheck(params: {
  runId: string;
  artifact: TrainedModelArtifact;
  dataset: ProcessedDataset;
  preferences: TrainingPreferences;
  metrics: ModelMetrics;
}): Promise<OnnxRoundTripCheckResult> {
  const { runId, artifact, dataset, preferences, metrics } = params;
  if (!dataset.features.length) throw new Error('Dataset has no rows for ONNX sanity check.');
  if (!dataset.featureNames.length) throw new Error('Dataset has no feature columns for ONNX sanity check.');

  const sample = dataset.features[0];
  const localPrediction = predictLocal(artifact, sample);

  const exported = await exportModelArtifact('onnx', {
    runId,
    artifact,
    dataset,
    preferences,
    metrics,
  });
  const onnxBytes = new Uint8Array(await exported.blob.arrayBuffer());

  const ort = await loadOrtGlobal();
  if (!ort) {
    throw new Error('ONNX Runtime did not initialize after script load.');
  }
  if (ort?.env?.wasm) {
    ort.env.wasm.wasmPaths = ORT_WASM_BASE_URL;
  }

  let session: any;
  try {
    session = await ort.InferenceSession.create(onnxBytes, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
  } catch (error: any) {
    throw new Error(
      `ONNX Runtime initialization failed. Ensure browser can load onnxruntime-web WASM assets. ${error?.message ?? ''}`
    );
  }

  const inputName = session.inputNames[0] || 'X';
  const inputTensor = new ort.Tensor('float32', Float32Array.from(sample), [1, sample.length]);
  const outputs = await session.run({ [inputName]: inputTensor });
  const onnxPrediction = resolveOnnxPrediction(artifact, outputs as any, dataset.labelNames.length);
  const isRegressionArtifact =
    artifact.modelType === 'linear_regression' && String(artifact.modelData.mode ?? 'classifier') === 'regressor';

  const localLabel = isRegressionArtifact
    ? localPrediction.toFixed(6)
    : dataset.labelNames[localPrediction] ?? `class_${localPrediction}`;
  const onnxLabel = isRegressionArtifact
    ? onnxPrediction.toFixed(6)
    : dataset.labelNames[onnxPrediction] ?? `class_${onnxPrediction}`;
  const ok = isRegressionArtifact
    ? Math.abs(localPrediction - onnxPrediction) <= 1e-3
    : localPrediction === onnxPrediction;

  return {
    ok,
    localPrediction,
    onnxPrediction,
    localLabel,
    onnxLabel,
    outputNames: Object.keys(outputs),
    detail: ok
      ? isRegressionArtifact
        ? `ONNX output matches local model on sanity sample (value: ${onnxLabel}).`
        : `ONNX output matches local model on sanity sample (class: ${onnxLabel}).`
      : `Mismatch detected (local=${localLabel}, onnx=${onnxLabel}).`,
  };
}
