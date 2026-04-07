import React, { DragEvent, useEffect, useMemo, useRef, useState } from 'react';
import Head from 'next/head';
import { Inter, JetBrains_Mono } from 'next/font/google';
import { AnimatePresence, motion } from 'framer-motion';
import {
  clearPendingRunState,
  DatasetSplitChoice,
  loadPendingRunState,
  loadPreferencesFromIndexedDb,
  savePendingRunState,
  savePreferencesToIndexedDb,
  saveTrainingHistoryEntry,
} from '../data/indexeddb';
import { streamKaggleDownloadToFile } from '../data/kaggle-download-client';
import { DatasetParseProgress, parseDatasetBlob, parseDatasetDirectoryHandle, parseDatasetFile } from '../data/parsers';
import { analyzeDataset, buildColumnProfiles, getAlgorithmCatalogSuggestions } from '../data/insights';
import { computeClassificationMetrics, computeRegressionMetrics } from '../engine/metrics';
import { RecurrentClassifierModel, predictRecurrentClassifier } from '../engine/algorithms/recurrent';
import { EnergyImpactLevel, HardwareMonitor, ThermalPressureState } from '../engine/hardware-monitor';
import { generateExecutableFromTemplate, normalizeTemplateConfig } from '../engine/wasm-function-editor';
import { useCapabilities } from '../ui/hooks/useCapabilities';
import { useWorkflowTraining } from '../ui/hooks/useWorkflowTraining';
import {
  AlgorithmId,
  AlgorithmSuggestion,
  AutoInsights,
  DataColumnProfile,
  DatasetModality,
  DataRow,
  KaggleDatasetResult,
  ParsedDataset,
  PreprocessingConfig,
  ProcessedDataset,
} from '../types/data';
import {
  DatasetTransferPayload,
  ModelMetrics,
  ModelChoice,
  TrainedModelArtifact,
  TrainingPreferences,
  WasmFunctionTemplateConfig,
} from '../types/training-workflow';

type WorkflowStep = 'dataset' | 'preprocess' | 'model' | 'setup' | 'training' | 'results' | 'export';
type PredictionFilter = 'all' | 'correct' | 'incorrect';
type TaskMode = 'classification' | 'regression' | 'unsupervised';
type StudioTab = 'studio' | 'training' | 'export' | 'inference';
type DataSourceMode = 'local' | 'kaggle' | 'huggingface';
type MethodMode = 'supervised' | 'unsupervised';
type EvalStrategyMode = 'split' | 'upload_file';

const inter = Inter({ subsets: ['latin'], variable: '--font-inter' });
const jetbrains = JetBrains_Mono({ subsets: ['latin'], variable: '--font-jetbrains' });

interface PredictionSample {
  rowIndex: number;
  rawRow: DataRow | null;
  actualLabel: string;
  predictedLabel: string;
  actualIndex?: number;
  predictedIndex?: number;
  correct: boolean;
  actualValue?: number;
  predictedValue?: number;
  absoluteError?: number;
}

interface EvaluationSnapshot {
  splitChoice: DatasetSplitChoice;
  trainCount: number;
  testCount: number;
  trainMetrics: ModelMetrics | null;
  testMetrics: ModelMetrics | null;
  fitSignal: 'balanced' | 'overfitting' | 'underfitting' | 'unknown';
  fitMessage: string;
  samples: PredictionSample[];
}

const PIXEL_PREVIEW_CACHE = new WeakMap<object, string | null>();

const STEPS: Array<{ id: WorkflowStep; title: string; icon: string }> = [
  { id: 'dataset', title: 'Add Your Data', icon: '1' },
  { id: 'preprocess', title: 'Understand Your Data', icon: '2' },
  { id: 'model', title: 'Choose Your Model', icon: '3' },
  { id: 'setup', title: 'Training Preferences', icon: '4' },
  { id: 'training', title: 'Training in Progress', icon: '5' },
  { id: 'results', title: 'Your Model is Ready', icon: '6' },
  { id: 'export', title: 'Export / Inference', icon: '7' },
];

const DEFAULT_PREPROCESS: PreprocessingConfig = {
  fixMissingValues: true,
  encodeCategories: true,
  normalizeData: true,
  augmentImageData: false,
  imageAugmentationFactor: 1,
  imageAugmentationNoise: 0.06,
  droppedColumns: [],
  targetColumn: null,
};

const DEFAULT_WASM_TEMPLATE: WasmFunctionTemplateConfig = normalizeTemplateConfig({
  functionName: 'train_batch',
  invocationTimeoutMs: 10000,
  retryCount: 1,
  memoryBudgetMB: 512,
  shardCount: 1,
  checkpointEveryNEpochs: 1,
  gradientClipValue: 0,
  coldStartMs: 50,
});

const DEFAULT_PREFERENCES: TrainingPreferences = {
  speedVsAccuracy: 60,
  useMoreCompute: false,
  optimizeForSmallerModel: false,
  epochs: 16,
  learningRate: 0.001,
  batchSize: 32,
  shuffleEachEpoch: true,
  earlyStoppingPatience: 6,
  optimizer: 'adamw',
  weightDecay: 0.0001,
  momentum: 0.9,
  beta1: 0.9,
  beta2: 0.999,
  lrScheduler: 'cosine_annealing',
  warmupSteps: 2,
  schedulerStepSize: 10,
  schedulerGamma: 0.5,
  algorithm: {
    knnNeighbors: 7,
    knnDistanceMetric: 'euclidean',
    svmKernel: 'rbf',
    kmeansClusters: 8,
    dbscanEpsilon: 0.8,
    dbscanMinSamples: 6,
  },
  neuralNetwork: {
    hiddenLayers: 2,
    neuronsPerLayer: 128,
    activation: 'relu',
    useBatchNorm: false,
    useLayerNorm: false,
    dropoutRate: 0.1,
    gradientClipping: 1,
    optimizer: 'adamw',
    weightDecay: 0.0001,
  },
  runtime: {
    pipeline: 'hybrid_worker_wasm_webgpu',
    wasmEditor: {
      advancedMode: 'template',
      templateConfig: DEFAULT_WASM_TEMPLATE,
      executableCode: '',
    },
  },
};

type AlgorithmGroup = 'Tree-based' | 'Linear Model' | 'Kernel/Neighbors' | 'Deep Learning' | 'Clustering';
type LevelLabel = 'High' | 'Medium' | 'Low';

const ALGORITHM_UI_META: Record<AlgorithmId, { group: AlgorithmGroup; interpretability: LevelLabel; memory: LevelLabel }> = {
  decision_tree: { group: 'Tree-based', interpretability: 'High', memory: 'Low' },
  random_forest: { group: 'Tree-based', interpretability: 'Medium', memory: 'Medium' },
  logistic_regression: { group: 'Linear Model', interpretability: 'High', memory: 'Low' },
  svm: { group: 'Kernel/Neighbors', interpretability: 'Medium', memory: 'Medium' },
  knn: { group: 'Kernel/Neighbors', interpretability: 'Medium', memory: 'High' },
  neural_network: { group: 'Deep Learning', interpretability: 'Low', memory: 'High' },
  linear_regression: { group: 'Linear Model', interpretability: 'High', memory: 'Low' },
  kmeans: { group: 'Clustering', interpretability: 'Medium', memory: 'Medium' },
  dbscan: { group: 'Clustering', interpretability: 'Medium', memory: 'High' },
  cnn: { group: 'Deep Learning', interpretability: 'Low', memory: 'High' },
  rnn: { group: 'Deep Learning', interpretability: 'Low', memory: 'High' },
  lstm: { group: 'Deep Learning', interpretability: 'Low', memory: 'High' },
  gru: { group: 'Deep Learning', interpretability: 'Low', memory: 'Medium' },
};

function resolveTaskMode(insights: AutoInsights | null): TaskMode {
  if (!insights) return 'classification';
  if (insights.learningMode === 'unsupervised') return 'unsupervised';
  return insights.taskType === 'regression' || insights.problemType === 'regression' ? 'regression' : 'classification';
}

function displayAlgorithmLabel(id: AlgorithmId, taskMode: TaskMode): string {
  if (id === 'random_forest') {
    if (taskMode === 'classification') return 'Random Forest Classifier';
    if (taskMode === 'regression') return 'Random Forest Regressor';
  }
  if (id === 'decision_tree') {
    if (taskMode === 'classification') return 'Decision Tree Classifier';
    if (taskMode === 'regression') return 'Decision Tree Regressor';
  }
  if (id === 'linear_regression') {
    return taskMode === 'classification' ? 'Linear Classifier' : 'Linear Regression';
  }
  if (id === 'svm') {
    return taskMode === 'regression' ? 'SVM Regressor' : 'SVM Classifier';
  }
  if (id === 'knn') {
    return taskMode === 'regression' ? 'KNN Regressor' : 'KNN Classifier';
  }
  if (id === 'logistic_regression') {
    return 'Logistic Regression (Classifier)';
  }
  if (id === 'cnn') return 'CNN';
  if (id === 'rnn') return 'RNN';
  if (id === 'lstm') return 'LSTM';
  if (id === 'gru') return 'GRU';
  if (id === 'kmeans') return 'K-Means';
  if (id === 'dbscan') return 'DBSCAN';
  return id.replace(/_/g, ' ').replace(/\b\w/g, (char) => char.toUpperCase());
}

function algorithmCompatibility(
  id: AlgorithmId,
  taskMode: TaskMode,
  modality: DatasetModality
): { compatible: boolean; reason: string } {
  if (taskMode === 'unsupervised') {
    if (id === 'kmeans' || id === 'dbscan') {
      return { compatible: true, reason: 'Compatible with unlabeled clustering workflows.' };
    }
    return { compatible: false, reason: 'Requires supervised target-based training.' };
  }

  if (id === 'kmeans' || id === 'dbscan') {
    return { compatible: false, reason: 'Unsupervised algorithm; use when no target column is selected.' };
  }

  if (taskMode === 'classification') {
    if (id === 'cnn' && modality !== 'image') {
      return { compatible: false, reason: 'CNN requires image modality.' };
    }
    return { compatible: true, reason: 'Compatible with classification targets.' };
  }

  if (taskMode === 'regression') {
    if (id === 'logistic_regression') {
      return { compatible: false, reason: 'Logistic regression is classification-only.' };
    }
    if (id === 'svm' || id === 'knn') {
      return { compatible: false, reason: 'Regressor variant is planned; classifier variant is currently implemented.' };
    }
    if (id === 'cnn' && modality !== 'image') {
      return { compatible: false, reason: 'CNN requires image modality.' };
    }
    return { compatible: true, reason: 'Compatible with regression targets.' };
  }

  return { compatible: false, reason: 'Not compatible with detected task.' };
}

function taskModeDescription(taskMode: TaskMode): string {
  if (taskMode === 'classification') {
    return 'Task locked to Classification. Switch target type in Step 2 to move to regression metrics.';
  }
  if (taskMode === 'regression') {
    return 'Task locked to Regression. Switching to classification changes metrics from RMSE/MAE/R² to Accuracy/F1.';
  }
  return 'Task locked to Unsupervised mode. Compatible clustering models are enabled.';
}

function isNnFamilyChoice(choice: ModelChoice): boolean {
  return (
    choice === 'neural_network' ||
    choice === 'cnn'
  );
}

function usesNeuralTuning(choice: ModelChoice): boolean {
  return (
    isNnFamilyChoice(choice) ||
    choice === 'rnn' ||
    choice === 'lstm' ||
    choice === 'gru'
  );
}

function applyRecommendedPreferencePreset(
  base: TrainingPreferences,
  algorithm: AlgorithmId,
  insights: AutoInsights
): TrainingPreferences {
  if (isNnFamilyChoice(algorithm)) {
    const large = insights.datasetSize === 'large';
    return {
      ...base,
      speedVsAccuracy: large ? 82 : 72,
      useMoreCompute: true,
      optimizeForSmallerModel: !large,
      epochs: large ? 32 : 20,
      learningRate: large ? 0.0007 : 0.001,
      batchSize: large ? 64 : 32,
      optimizer: 'adamw',
      weightDecay: large ? 0.0002 : 0.0001,
      lrScheduler: 'cosine_annealing',
      warmupSteps: 3,
      earlyStoppingPatience: large ? 8 : 6,
      neuralNetwork: {
        ...base.neuralNetwork,
        hiddenLayers: large ? 3 : 2,
        neuronsPerLayer: large ? 256 : 128,
        activation: large ? 'relu' : 'tanh',
        useBatchNorm: large,
        useLayerNorm: false,
        dropoutRate: large ? 0.2 : 0.1,
        gradientClipping: large ? 1.5 : 1,
        optimizer: 'adamw',
        weightDecay: large ? 0.0002 : 0.0001,
      },
    };
  }

  if (algorithm === 'rnn' || algorithm === 'lstm' || algorithm === 'gru') {
    const large = insights.datasetSize === 'large';
    return {
      ...base,
      speedVsAccuracy: large ? 78 : 68,
      useMoreCompute: true,
      optimizeForSmallerModel: !large,
      epochs: large ? 28 : 18,
      learningRate: large ? 0.0008 : 0.001,
      batchSize: large ? 64 : 32,
      optimizer: 'adam',
      weightDecay: 0.0003,
      lrScheduler: 'linear_decay',
      warmupSteps: 2,
      earlyStoppingPatience: large ? 7 : 5,
      neuralNetwork: {
        ...base.neuralNetwork,
        hiddenLayers: 2,
        neuronsPerLayer: large ? 192 : 96,
        activation: 'tanh',
        useBatchNorm: false,
        useLayerNorm: true,
        dropoutRate: large ? 0.2 : 0.1,
        gradientClipping: 1,
        optimizer: 'adam',
        weightDecay: 0.0003,
      },
    };
  }

  if (algorithm === 'kmeans' || algorithm === 'dbscan') {
    return {
      ...base,
      speedVsAccuracy: 52,
      useMoreCompute: false,
      optimizeForSmallerModel: true,
      epochs: 20,
      learningRate: 0.001,
      batchSize: 32,
      optimizer: 'adam',
      weightDecay: 0.0001,
      lrScheduler: 'constant',
      warmupSteps: 0,
      earlyStoppingPatience: 6,
      algorithm: {
        ...base.algorithm,
        kmeansClusters: insights.datasetSize === 'small' ? 6 : insights.datasetSize === 'medium' ? 8 : 12,
        dbscanEpsilon: insights.modality === 'image' ? 0.9 : 0.8,
        dbscanMinSamples: insights.datasetSize === 'small' ? 4 : 6,
      },
    };
  }

  if (
    algorithm === 'svm' ||
    algorithm === 'knn' ||
    algorithm === 'linear_regression' ||
    algorithm === 'logistic_regression'
  ) {
    return {
      ...base,
      speedVsAccuracy: 46,
      useMoreCompute: false,
      optimizeForSmallerModel: true,
      epochs: 12,
      learningRate: 0.001,
      batchSize: 24,
      optimizer: 'adamax',
      weightDecay: 0.0005,
      lrScheduler: 'step_lr',
      schedulerStepSize: 6,
      schedulerGamma: 0.6,
      warmupSteps: 1,
      earlyStoppingPatience: 5,
      neuralNetwork: {
        ...base.neuralNetwork,
        hiddenLayers: 1,
        neuronsPerLayer: 64,
        activation: 'relu',
        useBatchNorm: false,
        useLayerNorm: false,
        dropoutRate: 0.05,
        gradientClipping: 0.8,
        optimizer: 'adamax',
        weightDecay: 0.0005,
      },
    };
  }

  return {
    ...base,
    speedVsAccuracy: insights.datasetSize === 'small' ? 52 : 62,
    useMoreCompute: insights.datasetSize === 'large',
    optimizeForSmallerModel: insights.datasetSize !== 'large',
    epochs: 16,
    learningRate: 0.001,
    batchSize: 32,
    optimizer: 'adamw',
    weightDecay: 0.0001,
    lrScheduler: 'cosine_annealing',
    warmupSteps: 2,
    earlyStoppingPatience: 6,
  };
}

function isSelectableSuggestion(suggestion: AlgorithmSuggestion): boolean {
  return suggestion.runtimeSupport === 'native' || suggestion.runtimeSupport === 'mapped';
}

function bytesToHuman(value?: number): string {
  if (!value || value <= 0) return 'N/A';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let v = value;
  let unit = 0;
  while (v >= 1024 && unit < units.length - 1) {
    v /= 1024;
    unit++;
  }
  return `${v.toFixed(unit >= 2 ? 2 : 0)} ${units[unit]}`;
}

function shouldDisableNormalizationForLargeDataset(dataset: ParsedDataset): boolean {
  const cellCount = dataset.rows.length * Math.max(1, dataset.columns.length);
  return dataset.inferredFormat === 'images_zip' || cellCount >= 2_500_000;
}

function describeDatasetParseProgress(progress: DatasetParseProgress): string {
  const totalText = progress.total > 0 ? progress.total.toLocaleString() : '?';
  const processedText = progress.processed.toLocaleString();
  switch (progress.stage) {
    case 'loading_archive':
      return progress.message || 'Loading ZIP archive...';
    case 'discovering_files':
      return progress.message || `Scanning files in ZIP (${processedText}/${totalText})`;
    case 'extracting_images':
      return progress.currentFile
        ? `Extracting ${progress.currentFile} (${processedText}/${totalText})`
        : `Extracting image files (${processedText}/${totalText})`;
    case 'processing_images':
      return progress.message || `Processing images (${processedText}/${totalText})`;
    case 'completed':
      return progress.message || `Import complete (${processedText}/${totalText})`;
    default:
      return progress.message || 'Importing dataset...';
  }
}

const RUN_VERSION_STORAGE_KEY = 'browser-first-ai.run_versions.v1';

function runBaseForDataset(
  dataset: ProcessedDataset,
  modelChoice: ModelChoice,
  splitChoice?: DatasetSplitChoice
): string {
  const splitKey = splitChoice ?? 'full';
  const base = `${dataset.targetColumn}_${dataset.featureNames.length}_${dataset.features.length}_${modelChoice}_${splitKey}`;
  return `wk_${base.replace(/[^a-z0-9_]+/gi, '_').toLowerCase()}`;
}

function runIdForDataset(
  dataset: ProcessedDataset,
  modelChoice: ModelChoice,
  version: number,
  splitChoice?: DatasetSplitChoice
): string {
  const normalizedVersion = Math.max(1, Math.floor(version) || 1);
  return `${runBaseForDataset(dataset, modelChoice, splitChoice)}_v${normalizedVersion}`;
}

function parseRunBaseFromRunId(runId: string): string {
  const match = runId.match(/^(.*)_v\d+$/);
  return match ? match[1] : runId;
}

function readRunVersionMap(): Record<string, number> {
  if (typeof window === 'undefined') return {};
  try {
    const raw = window.localStorage.getItem(RUN_VERSION_STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw) as Record<string, number>;
    if (!parsed || typeof parsed !== 'object') return {};
    return parsed;
  } catch {
    return {};
  }
}

function writeRunVersionMap(map: Record<string, number>) {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(RUN_VERSION_STORAGE_KEY, JSON.stringify(map));
  } catch {
    // ignore storage errors
  }
}

function getStoredRunVersion(runBase: string): { version: number; hasExisting: boolean } {
  const map = readRunVersionMap();
  const existing = map[runBase];
  const hasExisting = Number.isInteger(existing) && existing >= 1;
  return {
    version: hasExisting ? existing : 1,
    hasExisting,
  };
}

function setStoredRunVersion(runBase: string, version: number) {
  const map = readRunVersionMap();
  map[runBase] = Math.max(1, Math.floor(version) || 1);
  writeRunVersionMap(map);
}

function clearStoredRunVersion(runId: string) {
  const runBase = parseRunBaseFromRunId(runId);
  const map = readRunVersionMap();
  if (!(runBase in map)) return;
  delete map[runBase];
  writeRunVersionMap(map);
}

function mergeStoredPreferences(
  base: TrainingPreferences,
  stored: Partial<TrainingPreferences>
): TrainingPreferences {
  return {
    ...base,
    ...stored,
    neuralNetwork: {
      ...base.neuralNetwork,
      ...(stored.neuralNetwork ?? {}),
    },
    algorithm: {
      ...base.algorithm,
      ...(stored.algorithm ?? {}),
    },
    runtime: {
      ...base.runtime,
      ...(stored.runtime ?? {}),
      wasmEditor: {
        ...base.runtime.wasmEditor,
        ...(stored.runtime?.wasmEditor ?? {}),
        templateConfig: normalizeTemplateConfig({
          ...base.runtime.wasmEditor.templateConfig,
          ...(stored.runtime?.wasmEditor?.templateConfig ?? {}),
        }),
      },
    },
  };
}

function createDatasetTransferPayload(
  dataset: ProcessedDataset,
  caps: { sharedArrayBuffer: boolean } | null
): DatasetTransferPayload | null {
  if (!caps?.sharedArrayBuffer || typeof SharedArrayBuffer === 'undefined') return null;
  const rowCount = dataset.features.length;
  const featureCount = dataset.featureNames.length;
  if (!rowCount || !featureCount) return null;

  const featuresBuffer = new SharedArrayBuffer(Float32Array.BYTES_PER_ELEMENT * rowCount * featureCount);
  const featuresView = new Float32Array(featuresBuffer);
  for (let row = 0; row < rowCount; row++) {
    const featureRow = dataset.features[row];
    for (let col = 0; col < featureCount; col++) {
      featuresView[row * featureCount + col] = featureRow[col] ?? 0;
    }
  }

  const labelsBuffer = new SharedArrayBuffer(Int32Array.BYTES_PER_ELEMENT * rowCount);
  const labelsView = new Int32Array(labelsBuffer);
  for (let i = 0; i < rowCount; i++) labelsView[i] = dataset.labels[i] ?? 0;

  let regressionTargetsBuffer: SharedArrayBuffer | undefined;
  if (dataset.problemType === 'regression' && dataset.regressionTargets?.length === rowCount) {
    regressionTargetsBuffer = new SharedArrayBuffer(Float32Array.BYTES_PER_ELEMENT * rowCount);
    const targetsView = new Float32Array(regressionTargetsBuffer);
    for (let i = 0; i < rowCount; i++) targetsView[i] = dataset.regressionTargets[i] ?? 0;
  }

  return {
    rowCount,
    featureCount,
    featuresBuffer,
    labelsBuffer,
    regressionTargetsBuffer,
  };
}

function datasetPayloadWithoutDenseMatrices(dataset: ProcessedDataset): ProcessedDataset {
  return {
    ...dataset,
    features: [],
    labels: [],
    regressionTargets: [],
  };
}

function splitRatioForChoice(choice: DatasetSplitChoice): number {
  return choice === '90_10' ? 0.9 : 0.8;
}

function splitLabelForChoice(choice: DatasetSplitChoice): string {
  return choice === '90_10' ? '90% / 10%' : '80% / 20%';
}

function hashStringToSeed(input: string): number {
  let hash = 2166136261;
  for (let i = 0; i < input.length; i++) {
    hash ^= input.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function seededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6D2B79F5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffledIndices(size: number, seed: number): number[] {
  const indices = Array.from({ length: size }, (_, idx) => idx);
  const rand = seededRandom(seed || 1);
  for (let i = indices.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }
  return indices;
}

function subsetProcessedDataset(
  source: ProcessedDataset,
  indices: number[],
  rawRows: DataRow[] | null
): { dataset: ProcessedDataset; rows: DataRow[] } {
  const rows = rawRows ? indices.map((idx) => rawRows[idx]).filter(Boolean) : [];
  return {
    dataset: {
      ...source,
      features: indices.map((idx) => source.features[idx]),
      labels: indices.map((idx) => source.labels[idx]),
      regressionTargets: source.regressionTargets ? indices.map((idx) => source.regressionTargets![idx]) : undefined,
      sampleRows: rows.slice(0, 50),
    },
    rows,
  };
}

function createTrainTestSplit(
  dataset: ProcessedDataset,
  rawRows: DataRow[] | null,
  choice: DatasetSplitChoice,
  seedKey: string
): {
  train: { dataset: ProcessedDataset; rows: DataRow[] };
  test: { dataset: ProcessedDataset; rows: DataRow[] };
} {
  const total = dataset.features.length;
  if (total < 2) {
    throw new Error('Need at least 2 rows to create a train/test split.');
  }
  const trainRatio = splitRatioForChoice(choice);
  const seed = hashStringToSeed(`${seedKey}_${dataset.targetColumn}_${dataset.featureNames.length}_${total}`);
  const shuffled = shuffledIndices(total, seed);
  const proposedTrainCount = Math.max(1, Math.floor(total * trainRatio));
  const trainCount = Math.min(total - 1, proposedTrainCount);

  const trainIndices = shuffled.slice(0, trainCount);
  const testIndices = shuffled.slice(trainCount);
  return {
    train: subsetProcessedDataset(dataset, trainIndices, rawRows),
    test: subsetProcessedDataset(dataset, testIndices, rawRows),
  };
}

function detectFitSignal(train: ModelMetrics | null, test: ModelMetrics | null): {
  signal: 'balanced' | 'overfitting' | 'underfitting' | 'unknown';
  message: string;
} {
  if (!train || !test) {
    return { signal: 'unknown', message: 'Train/test metrics are not available yet.' };
  }

  if (train.kind === 'classification' && test.kind === 'classification') {
    const trainAcc = train.accuracy ?? 0;
    const testAcc = test.accuracy ?? 0;
    const gap = trainAcc - testAcc;
    if (gap > 0.12 && trainAcc >= 0.8) {
      return {
        signal: 'overfitting',
        message: 'Train accuracy is significantly above test accuracy. This points to potential overfitting.',
      };
    }
    if (trainAcc < 0.65 && testAcc < 0.65) {
      return {
        signal: 'underfitting',
        message: 'Both train and test accuracy are low. Model likely underfits current feature setup.',
      };
    }
    return {
      signal: 'balanced',
      message: 'Train and test classification metrics are reasonably aligned.',
    };
  }

  if (train.kind === 'regression' && test.kind === 'regression') {
    const trainRmse = train.rmse ?? 0;
    const testRmse = test.rmse ?? 0;
    const trainR2 = train.r2 ?? 0;
    const testR2 = test.r2 ?? 0;
    if (trainRmse > 0 && testRmse / trainRmse > 1.35) {
      return {
        signal: 'overfitting',
        message: 'Test RMSE is much higher than train RMSE. This points to potential overfitting.',
      };
    }
    if (trainR2 < 0.4 && testR2 < 0.4) {
      return {
        signal: 'underfitting',
        message: 'Both train and test R² are low. Model likely underfits current feature setup.',
      };
    }
    return {
      signal: 'balanced',
      message: 'Train and test regression metrics are reasonably aligned.',
    };
  }

  return {
    signal: 'unknown',
    message: 'Train and test metric kinds are not aligned for fit analysis.',
  };
}

function toColorByte(value: unknown, fallback: number): number {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return fallback;
  if (numeric >= 0 && numeric <= 1) return Math.max(0, Math.min(255, Math.round(numeric * 255)));
  return Math.max(0, Math.min(255, Math.round(numeric)));
}

function imagePlaceholderStyle(row: DataRow | null): React.CSSProperties {
  if (!row) {
    return {
      background: 'linear-gradient(135deg, var(--bg-tertiary), var(--bg-hover))',
    };
  }
  const r = toColorByte(row.mean_r, 120);
  const g = toColorByte(row.mean_g, 120);
  const b = toColorByte(row.mean_b, 120);
  return {
    background: `linear-gradient(135deg, rgba(${r}, ${g}, ${b}, 0.95), rgba(${Math.max(
      0,
      r - 50
    )}, ${Math.max(0, g - 50)}, ${Math.max(0, b - 50)}, 0.95))`,
  };
}

function matrixHeatStyle(value: number, maxValue: number): React.CSSProperties {
  const normalized = maxValue > 0 ? value / maxValue : 0;
  const alpha = 0.08 + normalized * 0.42;
  return {
    background: `rgba(37, 99, 235, ${alpha.toFixed(3)})`,
  };
}

function normalizeLabelForDisplay(raw: unknown, fallbackIndex: number): string {
  if (typeof raw === 'number' && Number.isFinite(raw)) {
    return Number.isInteger(raw) ? String(raw) : raw.toFixed(4);
  }
  const text = String(raw ?? '').trim();
  if (!text) return `Class ${fallbackIndex}`;

  // Replace one-hot/vector-like labels with a clean class name.
  if (/^\[\s*-?\d+(\.\d+)?(\s*,\s*-?\d+(\.\d+)?)*\s*\]$/.test(text)) {
    const values = text
      .slice(1, -1)
      .split(',')
      .map((item) => Number(item.trim()))
      .filter((item) => Number.isFinite(item));
    if (values.length) {
      let bestIdx = 0;
      let bestVal = -Infinity;
      for (let i = 0; i < values.length; i++) {
        if (values[i] > bestVal) {
          bestVal = values[i];
          bestIdx = i;
        }
      }
      if (bestVal > 0) return `Class ${bestIdx}`;
    }
    return `Class ${fallbackIndex}`;
  }

  return text.length > 60 ? `${text.slice(0, 57)}...` : text;
}

function labelNameAt(labelNames: string[] | undefined, index: number): string {
  return normalizeLabelForDisplay(labelNames?.[index], index);
}

function matrixCellStyle(
  value: number,
  maxValue: number,
  rowIndex?: number,
  colIndex?: number
): React.CSSProperties {
  const normalized = maxValue > 0 ? value / maxValue : 0;
  const alpha = 0.08 + normalized * 0.58;
  const diagonal = typeof rowIndex === 'number' && typeof colIndex === 'number' && rowIndex === colIndex;
  return {
    background: diagonal
      ? `rgba(16, 185, 129, ${alpha.toFixed(3)})`
      : `rgba(248, 113, 113, ${(0.05 + normalized * 0.42).toFixed(3)})`,
  };
}

function thermalStateLabel(state: ThermalPressureState): string {
  if (state === 'nominal') return 'Optimal';
  if (state === 'fair') return 'Elevated';
  if (state === 'serious') return 'High';
  if (state === 'critical') return 'Critical';
  return 'Unknown';
}

function thermalStateToPercent(state: ThermalPressureState): number {
  if (state === 'nominal') return 18;
  if (state === 'fair') return 42;
  if (state === 'serious') return 74;
  if (state === 'critical') return 96;
  return 0;
}

function energyImpactLabel(level: EnergyImpactLevel): string {
  if (level === 'idle') return 'Idle';
  if (level === 'low') return 'Low';
  if (level === 'moderate') return 'Moderate';
  if (level === 'high') return 'High';
  return 'Heavy';
}

function energyImpactToPercent(level: EnergyImpactLevel): number {
  if (level === 'idle') return 0;
  if (level === 'low') return 25;
  if (level === 'moderate') return 52;
  if (level === 'high') return 76;
  return 94;
}

function isPixelColumnName(name: string): boolean {
  const lower = name.toLowerCase();
  return /^pixel_?\d+$/i.test(lower) || /^x_?\d+$/i.test(lower) || /^\d+$/.test(lower);
}

function renderPixelRowPreview(row: DataRow): string | null {
  if (typeof document === 'undefined') return null;
  if (typeof row.image_preview === 'string' && row.image_preview.startsWith('data:image/')) {
    return row.image_preview;
  }
  const cached = PIXEL_PREVIEW_CACHE.get(row as object);
  if (cached !== undefined) return cached;

  const pixelEntries = Object.entries(row)
    .filter(([name, value]) => isPixelColumnName(name) && typeof value === 'number' && Number.isFinite(value))
    .sort((a, b) => Number(a[0].replace(/\D+/g, '') || 0) - Number(b[0].replace(/\D+/g, '') || 0));

  if (pixelEntries.length < 64) {
    PIXEL_PREVIEW_CACHE.set(row as object, null);
    return null;
  }

  const side = Math.round(Math.sqrt(pixelEntries.length));
  if (side * side !== pixelEntries.length || side > 64) {
    PIXEL_PREVIEW_CACHE.set(row as object, null);
    return null;
  }

  const values = pixelEntries.map(([, value]) => Number(value));
  const max = values.reduce((acc, value) => Math.max(acc, value), -Infinity);
  const min = values.reduce((acc, value) => Math.min(acc, value), Infinity);
  const span = Math.max(1e-9, max - min);
  const useUnitScale = max <= 1.5 && min >= -0.1;

  try {
    const canvas = document.createElement('canvas');
    canvas.width = side;
    canvas.height = side;
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      PIXEL_PREVIEW_CACHE.set(row as object, null);
      return null;
    }
    const image = ctx.createImageData(side, side);
    for (let i = 0; i < values.length; i++) {
      const gray = useUnitScale
        ? Math.max(0, Math.min(255, Math.round(values[i] * 255)))
        : Math.max(0, Math.min(255, Math.round(((values[i] - min) / span) * 255)));
      const offset = i * 4;
      image.data[offset] = gray;
      image.data[offset + 1] = gray;
      image.data[offset + 2] = gray;
      image.data[offset + 3] = 255;
    }
    ctx.putImageData(image, 0, 0);
    const dataUrl = canvas.toDataURL('image/png');
    PIXEL_PREVIEW_CACHE.set(row as object, dataUrl);
    return dataUrl;
  } catch {
    PIXEL_PREVIEW_CACHE.set(row as object, null);
    return null;
  }
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

function mapKernelFeature(value: number, kernel: string): number {
  if (kernel === 'poly') return value * value;
  if (kernel === 'rbf') return Math.exp(-0.5 * value * value);
  return value;
}

function mapKernelVector(sample: number[], kernel: string): number[] {
  if (kernel === 'linear') return sample;
  return sample.map((value) => mapKernelFeature(value, kernel));
}

function distanceForMetric(metric: string, a: number[], b: number[]): number {
  if (metric === 'manhattan') {
    let sum = 0;
    for (let i = 0; i < a.length; i++) sum += Math.abs((a[i] ?? 0) - (b[i] ?? 0));
    return sum;
  }
  if (metric === 'cosine') {
    let dot = 0;
    let normA = 0;
    let normB = 0;
    for (let i = 0; i < a.length; i++) {
      const av = a[i] ?? 0;
      const bv = b[i] ?? 0;
      dot += av * bv;
      normA += av * av;
      normB += bv * bv;
    }
    if (normA === 0 || normB === 0) return 1;
    return 1 - dot / (Math.sqrt(normA) * Math.sqrt(normB));
  }
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const delta = (a[i] ?? 0) - (b[i] ?? 0);
    sum += delta * delta;
  }
  return Math.sqrt(sum);
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

function predictForestTree(node: ForestNode, sample: number[]): number {
  if (node.featureIndex === null || !node.left || !node.right) return node.prediction;
  if (sample[node.featureIndex] <= node.threshold) return predictForestTree(node.left, sample);
  return predictForestTree(node.right, sample);
}

function predictFromArtifact(artifact: TrainedModelArtifact, sample: number[]): number {
  if (artifact.modelType === 'random_forest' || artifact.modelType === 'decision_tree') {
    const trees = (artifact.modelData.trees as ForestNode[]) ?? [];
    if (!trees.length) return 0;
    const mode = String(artifact.modelData.mode ?? 'classifier');
    if (mode === 'regressor') {
      let total = 0;
      let count = 0;
      for (const tree of trees) {
        const prediction = predictForestTree(tree, sample);
        if (Number.isFinite(prediction)) {
          total += prediction;
          count += 1;
        }
      }
      return count ? total / count : 0;
    }
    const votes = new Map<number, number>();
    for (const tree of trees) {
      const prediction = predictForestTree(tree, sample);
      votes.set(prediction, (votes.get(prediction) ?? 0) + 1);
    }
    return Array.from(votes.entries()).sort((a, b) => b[1] - a[1])[0]?.[0] ?? 0;
  }

  if (artifact.modelType === 'svm') {
    const weights = (artifact.modelData.weights as number[]) ?? [];
    const bias = (artifact.modelData.bias as number[]) ?? [];
    const kernel = String(artifact.modelData.kernel ?? 'linear');
    const classCount = bias.length;
    if (!classCount || weights.length !== sample.length * classCount) return 0;
    const mapped = mapKernelVector(sample, kernel);
    const scores = new Array(classCount).fill(0);
    for (let classIdx = 0; classIdx < classCount; classIdx++) {
      let sum = bias[classIdx] ?? 0;
      for (let featureIdx = 0; featureIdx < mapped.length; featureIdx++) {
        sum += mapped[featureIdx] * (weights[classIdx * mapped.length + featureIdx] ?? 0);
      }
      scores[classIdx] = sum;
    }
    return argMax(scores);
  }

  if (artifact.modelType === 'knn') {
    const trainFeatures = (artifact.modelData.trainFeatures as number[][]) ?? [];
    const trainLabels = (artifact.modelData.trainLabels as number[]) ?? [];
    const k = Math.max(1, Number(artifact.modelData.k ?? 7));
    const metric = String(artifact.modelData.distanceMetric ?? 'euclidean');
    const classCount = Math.max(2, Number(artifact.modelData.classCount ?? 2));
    if (!trainFeatures.length || !trainLabels.length) return 0;

    const ranked = trainFeatures
      .map((row, idx) => ({ idx, distance: distanceForMetric(metric, sample, row) }))
      .sort((a, b) => a.distance - b.distance);
    const votes = new Array(classCount).fill(0);
    for (let i = 0; i < Math.min(k, ranked.length); i++) {
      const label = trainLabels[ranked[i].idx] ?? 0;
      const weight = ranked[i].distance === 0 ? 1 : 1 / ranked[i].distance;
      votes[label] += weight;
    }
    return argMax(votes);
  }

  if (artifact.modelType === 'kmeans') {
    const centroids = (artifact.modelData.centroids as number[][]) ?? [];
    const clusterToLabel = (artifact.modelData.clusterToLabel as number[]) ?? [];
    const fallbackLabel = Number(artifact.modelData.fallbackLabel ?? 0);
    if (!centroids.length) return fallbackLabel;
    let bestCluster = 0;
    let bestDistance = Infinity;
    for (let cluster = 0; cluster < centroids.length; cluster++) {
      let distance = 0;
      const centroid = centroids[cluster];
      for (let featureIdx = 0; featureIdx < sample.length; featureIdx++) {
        const diff = (sample[featureIdx] ?? 0) - (centroid?.[featureIdx] ?? 0);
        distance += diff * diff;
      }
      if (distance < bestDistance) {
        bestDistance = distance;
        bestCluster = cluster;
      }
    }
    return clusterToLabel[bestCluster] ?? fallbackLabel;
  }

  if (artifact.modelType === 'dbscan') {
    const trainFeatures = (artifact.modelData.trainFeatures as number[][]) ?? [];
    const corePointIndices = (artifact.modelData.corePointIndices as number[]) ?? [];
    const clusterLabels = (artifact.modelData.clusterLabels as number[]) ?? [];
    const clusterToLabel = (artifact.modelData.clusterToLabel as number[]) ?? [];
    const fallbackLabel = Number(artifact.modelData.fallbackLabel ?? 0);
    const epsilon = Math.max(1e-6, Number(artifact.modelData.epsilon ?? 0.8));
    const epsilonSquared = epsilon * epsilon;
    if (!trainFeatures.length || !corePointIndices.length) return fallbackLabel;
    let bestCore = -1;
    let bestDistance = Infinity;
    for (const coreIndex of corePointIndices) {
      const candidate = trainFeatures[coreIndex];
      if (!candidate) continue;
      let distance = 0;
      for (let featureIdx = 0; featureIdx < sample.length; featureIdx++) {
        const diff = (sample[featureIdx] ?? 0) - (candidate[featureIdx] ?? 0);
        distance += diff * diff;
      }
      if (distance <= epsilonSquared && distance < bestDistance) {
        bestDistance = distance;
        bestCore = coreIndex;
      }
    }
    if (bestCore < 0) return fallbackLabel;
    const cluster = clusterLabels[bestCore] ?? -1;
    if (cluster < 0) return fallbackLabel;
    return clusterToLabel[cluster] ?? fallbackLabel;
  }

  if (artifact.modelType === 'rnn' || artifact.modelType === 'lstm' || artifact.modelType === 'gru') {
    const modelData = artifact.modelData as Partial<RecurrentClassifierModel>;
    const variant = artifact.modelType;
    if (
      !modelData.params ||
      !Array.isArray(modelData.outputWeights) ||
      !Array.isArray(modelData.outputBias) ||
      typeof modelData.inputSize !== 'number' ||
      typeof modelData.sequenceLength !== 'number' ||
      typeof modelData.hiddenSize !== 'number' ||
      typeof modelData.classCount !== 'number' ||
      typeof modelData.featureCount !== 'number'
    ) {
      return 0;
    }
    const recurrentModel: RecurrentClassifierModel = {
      variant,
      params: modelData.params as Record<string, number[]>,
      outputWeights: modelData.outputWeights as number[],
      outputBias: modelData.outputBias as number[],
      inputSize: modelData.inputSize,
      sequenceLength: modelData.sequenceLength,
      hiddenSize: modelData.hiddenSize,
      classCount: modelData.classCount,
      featureCount: modelData.featureCount,
      lossCurve: Array.isArray(modelData.lossCurve) ? modelData.lossCurve : [],
      featureImportance: Array.isArray(modelData.featureImportance) ? modelData.featureImportance : [],
    };
    return predictRecurrentClassifier(recurrentModel, [sample])[0] ?? 0;
  }

  if (artifact.modelType === 'linear_regression' || artifact.modelType === 'logistic_regression') {
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

function parseInferenceInput(raw: string, featureNames: string[]): number[] {
  const trimmed = raw.trim();
  if (!trimmed) throw new Error('Enter values first.');
  if (trimmed.startsWith('{')) {
    const asObject = JSON.parse(trimmed) as Record<string, unknown>;
    return featureNames.map((name) => Number(asObject[name] ?? 0) || 0);
  }
  const values = trimmed.split(',').map((v) => Number(v.trim()));
  if (values.length !== featureNames.length) {
    throw new Error(`Expected ${featureNames.length} values but received ${values.length}.`);
  }
  if (values.some((v) => !Number.isFinite(v))) {
    throw new Error('All inference input values must be numbers.');
  }
  return values;
}

function Stepper({
  step,
  onSelectStep,
}: {
  step: WorkflowStep;
  onSelectStep: (nextStep: WorkflowStep) => void;
}) {
  const navLabels: Record<WorkflowStep, string> = {
    dataset: 'Dataset',
    preprocess: 'Preprocess',
    model: 'Models',
    setup: 'Setup',
    training: 'Training',
    results: 'Results',
    export: 'Inference',
  };
  return (
    <div className="wk-stepper">
      {STEPS.map((item) => {
        const status = item.id === step ? 'current' : 'upcoming';
        return (
          <button
            key={item.id}
            className={`wk-step-item wk-${status}`}
            onClick={() => onSelectStep(item.id)}
            title={`Open "${item.title}"`}
          >
            <div className="wk-step-label">{navLabels[item.id]}</div>
          </button>
        );
      })}
    </div>
  );
}

function StepPlaceholder({
  title,
  message,
  actionLabel,
  onAction,
}: {
  title: string;
  message: string;
  actionLabel?: string;
  onAction?: () => void;
}) {
  return (
    <div className="wk-coming-soon">
      <h4>{title}</h4>
      <p>{message}</p>
      {actionLabel && onAction && (
        <button className="btn btn-primary" style={{ marginTop: 12 }} onClick={onAction}>
          {actionLabel}
        </button>
      )}
    </div>
  );
}

function formatDisplayNumber(value: number): string {
  if (!Number.isFinite(value)) return String(value);
  if (Number.isInteger(value)) return value.toLocaleString();
  return Number(value.toFixed(4)).toString();
}

function metricFixed4(value: number | null | undefined): string {
  const numeric = Number(value ?? 0);
  if (!Number.isFinite(numeric)) return '0.0000';
  return numeric.toFixed(4);
}

function formatCellDisplay(value: unknown): { text: string; numeric: boolean } {
  if (typeof value === 'number') {
    return { text: formatDisplayNumber(value), numeric: true };
  }
  if (typeof value === 'boolean') {
    return { text: value ? 'true' : 'false', numeric: false };
  }
  if (value === null || value === undefined) {
    return { text: '', numeric: false };
  }
  return { text: String(value), numeric: false };
}

function DataTable({
  rows,
  columns,
  targetColumn,
  onHeaderClick,
  className,
}: {
  rows: DataRow[];
  columns: string[];
  targetColumn?: string | null;
  onHeaderClick?: (column: string) => void;
  className?: string;
}) {
  return (
    <div className={`overflow-auto rounded-xl border border-white/10 bg-[#0D131E] ${className ?? ''}`}>
      <table className="min-w-full border-collapse text-xs">
        <thead>
          <tr>
            {columns.map((column) => (
              <th
                key={column}
                className={`sticky top-0 border-b border-white/10 px-3 py-2 text-left text-[11px] font-semibold uppercase tracking-[0.08em] ${
                  targetColumn === column ? 'bg-emerald-500/18 text-emerald-300' : 'bg-[#121B2A] text-slate-400'
                }`}
                onClick={() => onHeaderClick?.(column)}
                title={onHeaderClick ? 'Click to set as target column' : undefined}
              >
                {column}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, rowIndex) => (
            <tr key={`r_${rowIndex}`} className="border-b border-white/5 hover:bg-white/5">
              {columns.map((column) => {
                const cell = formatCellDisplay(row[column]);
                return (
                  <td
                    key={`${rowIndex}_${column}`}
                    className={`px-3 py-2 text-slate-200 ${cell.numeric ? 'font-mono text-[12px]' : 'text-[12px]'}`}
                  >
                    {cell.text}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Curve({ points }: { points: Array<{ loss: number; accuracy: number }> }) {
  if (points.length < 2) {
    return (
      <div className="flex h-[240px] items-center justify-center rounded-2xl border border-white/10 bg-[#0F1521] text-sm text-slate-400">
        Waiting for training metrics...
      </div>
    );
  }
  const width = 760;
  const height = 240;
  const padLeft = 46;
  const padRight = 54;
  const padTop = 20;
  const padBottom = 28;
  const plotWidth = width - padLeft - padRight;
  const plotHeight = height - padTop - padBottom;
  const accuracyTicks = [0, 0.25, 0.5, 0.75, 1];
  const losses = points.map((p) => p.loss);
  const minRawLoss = Math.min(...losses);
  const maxRawLoss = Math.max(...losses);
  const lossPadding = Math.max(1e-6, (maxRawLoss - minRawLoss) * 0.08);
  const minLoss = Math.max(0, minRawLoss - lossPadding);
  const maxLoss = maxRawLoss + lossPadding;
  const lossRange = maxLoss - minLoss || 1;

  const xAt = (index: number) => padLeft + (index / (points.length - 1)) * plotWidth;
  const yForAccuracy = (value: number) => padTop + (1 - Math.max(0, Math.min(1, value))) * plotHeight;
  const yForLoss = (value: number) => padTop + (1 - (value - minLoss) / lossRange) * plotHeight;

  const line = (mapper: (point: { loss: number; accuracy: number }) => number) =>
    points
      .map((point, index) => {
        const x = xAt(index);
        const y = mapper(point);
        return `${x},${y}`;
      })
      .join(' ');

  const lossLine = line((point) => yForLoss(point.loss));
  const accuracyLine = line((point) => yForAccuracy(point.accuracy));

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="h-[240px] w-full rounded-2xl border border-white/10 bg-[#0F1521]">
      {accuracyTicks.map((tick) => {
        const y = yForAccuracy(tick);
        const lossTick = maxLoss - tick * (maxLoss - minLoss);
        return (
          <g key={`tick_${tick}`}>
            <line x1={padLeft} y1={y} x2={width - padRight} y2={y} stroke="rgba(255,255,255,0.12)" strokeDasharray="4 4" />
            <text x={6} y={y + 4} fill="#94A3B8" fontSize="10" className="font-mono">
              {(tick * 100).toFixed(0)}%
            </text>
            <text x={width - padRight + 6} y={y + 4} fill="#94A3B8" fontSize="10" className="font-mono">
              {lossTick.toFixed(3)}
            </text>
          </g>
        );
      })}
      <line x1={padLeft} y1={padTop} x2={padLeft} y2={height - padBottom} stroke="rgba(255,255,255,0.24)" />
      <line x1={width - padRight} y1={padTop} x2={width - padRight} y2={height - padBottom} stroke="rgba(255,255,255,0.24)" />
      <line x1={padLeft} y1={height - padBottom} x2={width - padRight} y2={height - padBottom} stroke="rgba(255,255,255,0.24)" />

      <polyline points={accuracyLine} fill="none" stroke="#22D3EE" strokeWidth="2.2" />
      <polyline points={lossLine} fill="none" stroke="#10B981" strokeWidth="2.2" />

      <text x={padLeft} y={14} fill="#22D3EE" fontSize="11" fontWeight="600">
        Accuracy (left axis)
      </text>
      <text x={width - padRight - 108} y={14} fill="#10B981" fontSize="11" fontWeight="600">
        Loss (right axis)
      </text>
    </svg>
  );
}

function LineChartCard({
  title,
  values,
  color,
  valueFormatter,
}: {
  title: string;
  values: number[];
  color: string;
  valueFormatter: (value: number) => string;
}) {
  if (values.length < 2) {
    return (
      <div className="wk-panel-card">
        <div className="mb-2 text-sm font-semibold text-slate-200">{title}</div>
        <div className="flex h-[180px] items-center justify-center rounded-xl border border-white/10 bg-[#0D1320] text-sm text-slate-400">
          Waiting for data...
        </div>
      </div>
    );
  }

  const width = 460;
  const height = 220;
  const pad = 20;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = Math.max(1e-6, max - min);
  const points = values
    .map((value, index) => {
      const x = pad + (index / (values.length - 1)) * (width - pad * 2);
      const y = height - pad - ((value - min) / range) * (height - pad * 2);
      return `${x},${y}`;
    })
    .join(' ');

  return (
    <div className="wk-panel-card">
      <div className="mb-2 flex items-center justify-between">
        <div className="text-sm font-semibold text-slate-200">{title}</div>
        <div className="wk-number text-emerald-300">{valueFormatter(values[values.length - 1] ?? 0)}</div>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} className="h-[180px] w-full rounded-xl border border-white/10 bg-[#0D1320]">
        <line x1={pad} y1={pad} x2={pad} y2={height - pad} stroke="rgba(255,255,255,0.2)" />
        <line x1={pad} y1={height - pad} x2={width - pad} y2={height - pad} stroke="rgba(255,255,255,0.2)" />
        <polyline points={points} fill="none" stroke={color} strokeWidth="2.4" />
      </svg>
    </div>
  );
}

export default function Studio() {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const evalFileInputRef = useRef<HTMLInputElement | null>(null);
  const inferenceFileInputRef = useRef<HTMLInputElement | null>(null);
  const resumePromptedRunBasesRef = useRef<Set<string>>(new Set());
  const pendingRestoreAttemptedRef = useRef(false);
  const [step, setStep] = useState<WorkflowStep>('dataset');
  const [studioTab, setStudioTab] = useState<StudioTab>('studio');
  const [activeSource, setActiveSource] = useState<'upload' | 'kaggle' | 'huggingface' | 'external'>('upload');
  const [sourceMode, setSourceMode] = useState<DataSourceMode>('local');
  const [huggingFaceToken, setHuggingFaceToken] = useState('');
  const [huggingFaceDatasetId, setHuggingFaceDatasetId] = useState('');
  const [datasetViewerOpen, setDatasetViewerOpen] = useState(false);
  const [datasetViewerMode, setDatasetViewerMode] = useState<'table' | 'images'>('table');
  const [datasetViewerPage, setDatasetViewerPage] = useState(0);
  const [modelMethod, setModelMethod] = useState<MethodMode>('supervised');
  const [evalStrategy, setEvalStrategy] = useState<EvalStrategyMode>('split');
  const [uploadedEvalFileName, setUploadedEvalFileName] = useState<string | null>(null);
  const [uploadedEvalDataset, setUploadedEvalDataset] = useState<ParsedDataset | null>(null);
  const [dataset, setDataset] = useState<ParsedDataset | null>(null);
  const [columns, setColumns] = useState<string[]>([]);
  const [previewRows, setPreviewRows] = useState<DataRow[]>([]);
  const [profiles, setProfiles] = useState<DataColumnProfile[]>([]);
  const [insights, setInsights] = useState<AutoInsights | null>(null);
  const [preprocessConfig, setPreprocessConfig] = useState<PreprocessingConfig>(DEFAULT_PREPROCESS);
  const [processedDataset, setProcessedDataset] = useState<ProcessedDataset | null>(null);
  const [preprocessBusy, setPreprocessBusy] = useState(false);
  const [preprocessProgress, setPreprocessProgress] = useState<{ percent: number; message: string } | null>(null);
  const [preprocessError, setPreprocessError] = useState<string | null>(null);
  const [preprocessNotice, setPreprocessNotice] = useState<string | null>(null);
  const [showAdvancedPreprocess, setShowAdvancedPreprocess] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [datasetLoadError, setDatasetLoadError] = useState<string | null>(null);
  const [datasetLoadBusy, setDatasetLoadBusy] = useState(false);
  const [datasetLoadProgress, setDatasetLoadProgress] = useState<string | null>(null);
  const [modelChoice, setModelChoice] = useState<ModelChoice>('recommended');
  const [modelChoiceTouched, setModelChoiceTouched] = useState(false);
  const [showAdvancedModelConfig, setShowAdvancedModelConfig] = useState(false);
  const [treeDepthHint, setTreeDepthHint] = useState(12);
  const [preferences, setPreferences] = useState<TrainingPreferences>(DEFAULT_PREFERENCES);
  const [showAdvancedTraining, setShowAdvancedTraining] = useState(false);
  const [runId, setRunId] = useState('');
  const [splitChoice, setSplitChoice] = useState<DatasetSplitChoice>('80_20');
  const [splitUsed, setSplitUsed] = useState<DatasetSplitChoice | null>(null);
  const [splitCounts, setSplitCounts] = useState<{ train: number; test: number } | null>(null);
  const [heldOutDataset, setHeldOutDataset] = useState<ProcessedDataset | null>(null);
  const [heldOutRows, setHeldOutRows] = useState<DataRow[]>([]);
  const [predictionFilter, setPredictionFilter] = useState<PredictionFilter>('all');
  const [selectedConfusionCell, setSelectedConfusionCell] = useState<{ actual: number; predicted: number } | null>(null);
  const [evaluationMatrixExpanded, setEvaluationMatrixExpanded] = useState(false);
  const [evaluationExportStatus, setEvaluationExportStatus] = useState<string | null>(null);
  const [inferenceInput, setInferenceInput] = useState('');
  const [inferenceResult, setInferenceResult] = useState<string | null>(null);
  const [inferenceError, setInferenceError] = useState<string | null>(null);
  const [liveInferenceTestIndex, setLiveInferenceTestIndex] = useState(0);
  const [hardwareMetrics, setHardwareMetrics] = useState<{
    utilizationPercent: number;
    memoryMB: number;
    thermalState: ThermalPressureState;
    energyImpact: EnergyImpactLevel;
    updatedAt: number;
  }>({
    utilizationPercent: 0,
    memoryMB: 0,
    thermalState: 'unknown',
    energyImpact: 'idle',
    updatedAt: 0,
  });
  const hardwareSnapshotRef = useRef<{
    trainingActive: boolean;
    progressPercent: number;
    curveStep: number;
    curveLength: number;
    datasetRows: number;
    featureCount: number;
  }>({
    trainingActive: false,
    progressPercent: 0,
    curveStep: 0,
    curveLength: 0,
    datasetRows: 0,
    featureCount: 0,
  });

  // Kaggle state
  const [kaggleAuthMode, setKaggleAuthMode] = useState<'oauth_token' | 'legacy_key'>('oauth_token');
  const [kaggleApiToken, setKaggleApiToken] = useState('');
  const [kaggleConnected, setKaggleConnected] = useState(false);
  const [kaggleConnectionInfo, setKaggleConnectionInfo] = useState<string | null>(null);
  const [kaggleUsername, setKaggleUsername] = useState('');
  const [kaggleKey, setKaggleKey] = useState('');
  const [kaggleQuery, setKaggleQuery] = useState('');
  const [kaggleResults, setKaggleResults] = useState<KaggleDatasetResult[]>([]);
  const [selectedKaggleRef, setSelectedKaggleRef] = useState('');
  const [kaggleFiles, setKaggleFiles] = useState<Array<{ name: string; totalBytes?: number }>>([]);
  const [selectedKaggleFile, setSelectedKaggleFile] = useState('');
  const [kaggleLoading, setKaggleLoading] = useState(false);
  const [kaggleError, setKaggleError] = useState<string | null>(null);
  const [kaggleInlineStatus, setKaggleInlineStatus] = useState<string | null>(null);
  const [kaggleUseCompleteMessage, setKaggleUseCompleteMessage] = useState<string | null>(null);
  const [workflowColumnHover, setWorkflowColumnHover] = useState<1 | 2 | 3 | 4 | null>(null);
  const [hyperAdvancedOpen, setHyperAdvancedOpen] = useState(false);

  const { caps } = useCapabilities();
  const training = useWorkflowTraining();

  const currentStepIndex = STEPS.findIndex((item) => item.id === step);
  const latestMetric = training.state.curve[training.state.curve.length - 1];
  const kaggleAuthPayload = useMemo(() => {
    if (kaggleAuthMode === 'oauth_token') {
      return { apiToken: kaggleApiToken.trim() };
    }
    return {
      username: kaggleUsername.trim(),
      apiKey: kaggleKey.trim(),
    };
  }, [kaggleAuthMode, kaggleApiToken, kaggleUsername, kaggleKey]);
  const selectedKaggleMeta = useMemo(
    () => kaggleResults.find((r) => r.ref === selectedKaggleRef),
    [kaggleResults, selectedKaggleRef]
  );
  const markKaggleAuthInvalid = (message: string) => {
    if (kaggleAuthMode !== 'oauth_token') return;
    if (!/invalid|unauthorized|forbidden|authentication|token/i.test(message)) return;
    setKaggleConnected(false);
    setKaggleConnectionInfo('Kaggle OAuth token is invalid or expired. Please reconnect.');
  };

  const metricCards = useMemo(() => {
    if (!training.state.metrics) return [];
    if (training.state.metrics.kind === 'regression') {
      return [
        { label: 'RMSE', value: metricFixed4(training.state.metrics.rmse ?? 0) },
        { label: 'MAE', value: metricFixed4(training.state.metrics.mae ?? 0) },
        { label: 'R²', value: metricFixed4(training.state.metrics.r2 ?? 0) },
      ];
    }
    return [
      { label: 'Accuracy', value: `${metricFixed4((training.state.metrics.accuracy ?? 0) * 100)}%` },
      { label: 'Precision', value: `${metricFixed4((training.state.metrics.precision ?? 0) * 100)}%` },
      { label: 'Recall', value: `${metricFixed4((training.state.metrics.recall ?? 0) * 100)}%` },
      { label: 'F1 Score', value: `${metricFixed4((training.state.metrics.f1 ?? 0) * 100)}%` },
    ];
  }, [training.state.metrics]);
  const trainingConfusionMax = useMemo(() => {
    const matrix = training.state.metrics?.confusionMatrix ?? [];
    const values = matrix.flat();
    if (!values.length) return 1;
    return Math.max(1, ...values);
  }, [training.state.metrics?.confusionMatrix]);
  const evaluationSnapshot = useMemo<EvaluationSnapshot | null>(() => {
    if (!training.state.artifact || !heldOutDataset || !splitUsed || !splitCounts) return null;
    if (!heldOutDataset.features.length) return null;
    const evaluationKind = training.state.metrics?.kind ?? heldOutDataset.problemType;

    if (evaluationKind === 'classification') {
      const classCount = Math.max(1, heldOutDataset.labelNames.length);
      const normalizeClassIndex = (value: number): number => {
        const safe = Number.isFinite(value) ? Math.floor(value) : 0;
        return Math.max(0, Math.min(classCount - 1, safe));
      };
      const actual = heldOutDataset.labels.map((value) => normalizeClassIndex(value));
      const predicted = heldOutDataset.features.map((featureRow) => {
        const prediction = predictFromArtifact(training.state.artifact as TrainedModelArtifact, featureRow);
        return normalizeClassIndex(prediction);
      });
      const metricsBase = computeClassificationMetrics(
        actual,
        predicted,
        classCount
      );
      const testMetrics: ModelMetrics = {
        kind: 'classification',
        ...metricsBase,
        featureImportance: training.state.metrics?.featureImportance ?? [],
      };
      const fit = detectFitSignal(training.state.metrics, testMetrics);
      const samples: PredictionSample[] = predicted.map((predictedIndex, rowIndex) => {
        const actualIndex = actual[rowIndex] ?? 0;
        return {
          rowIndex,
          rawRow: heldOutRows[rowIndex] ?? null,
          actualLabel: labelNameAt(heldOutDataset.labelNames, actualIndex),
          predictedLabel: labelNameAt(heldOutDataset.labelNames, predictedIndex),
          actualIndex,
          predictedIndex,
          correct: actualIndex === predictedIndex,
        };
      });
      return {
        splitChoice: splitUsed,
        trainCount: splitCounts.train,
        testCount: splitCounts.test,
        trainMetrics: training.state.metrics,
        testMetrics,
        fitSignal: fit.signal,
        fitMessage: fit.message,
        samples,
      };
    }

    const targets =
      heldOutDataset.regressionTargets && heldOutDataset.regressionTargets.length === heldOutDataset.features.length
        ? heldOutDataset.regressionTargets
        : heldOutDataset.labels.map((value) => Number(value) || 0);
    const predictedValues = heldOutDataset.features.map((featureRow) =>
      Number(predictFromArtifact(training.state.artifact as TrainedModelArtifact, featureRow)) || 0
    );
    const metricsBase = computeRegressionMetrics(targets, predictedValues);
    const testMetrics: ModelMetrics = {
      kind: 'regression',
      ...metricsBase,
      featureImportance: training.state.metrics?.featureImportance ?? [],
    };
    const fit = detectFitSignal(training.state.metrics, testMetrics);
    const samples: PredictionSample[] = predictedValues.map((predictedValue, rowIndex) => {
      const actualValue = targets[rowIndex] ?? 0;
      const absError = Math.abs(actualValue - predictedValue);
      return {
        rowIndex,
        rawRow: heldOutRows[rowIndex] ?? null,
        actualLabel: actualValue.toFixed(4),
        predictedLabel: predictedValue.toFixed(4),
        correct: absError <= Math.max(0.05, Math.abs(actualValue) * 0.1),
        actualValue,
        predictedValue,
        absoluteError: absError,
      };
    });
    return {
      splitChoice: splitUsed,
      trainCount: splitCounts.train,
      testCount: splitCounts.test,
      trainMetrics: training.state.metrics,
      testMetrics,
      fitSignal: fit.signal,
      fitMessage: fit.message,
      samples,
    };
  }, [heldOutDataset, heldOutRows, splitCounts, splitUsed, training.state.artifact, training.state.metrics]);
  const isImageEvaluation =
    insights?.modality === 'image' ||
    modelChoice === 'cnn' ||
    heldOutRows.some((row) => typeof row?.image_name === 'string');
  const filteredEvaluationSamples = useMemo(() => {
    if (!evaluationSnapshot) return [];
    const base =
      predictionFilter === 'correct'
        ? evaluationSnapshot.samples.filter((sample) => sample.correct)
        : predictionFilter === 'incorrect'
          ? evaluationSnapshot.samples.filter((sample) => !sample.correct)
          : evaluationSnapshot.samples;
    if (isImageEvaluation) {
      return [...base].sort((a, b) => {
        const aHasPreview = typeof a.rawRow?.image_preview === 'string' ? 1 : 0;
        const bHasPreview = typeof b.rawRow?.image_preview === 'string' ? 1 : 0;
        if (aHasPreview !== bHasPreview) return bHasPreview - aHasPreview;
        return a.rowIndex - b.rowIndex;
      });
    }
    if (evaluationSnapshot.testMetrics?.kind === 'regression') {
      return [...base].sort((a, b) => (b.absoluteError ?? 0) - (a.absoluteError ?? 0));
    }
    return base;
  }, [evaluationSnapshot, isImageEvaluation, predictionFilter]);
  const evaluationMiniMatrix = useMemo(() => {
    if (!evaluationSnapshot || evaluationSnapshot.testMetrics?.kind !== 'classification') return [];
    const matrix = evaluationSnapshot.testMetrics.confusionMatrix ?? [];
    const size = Math.min(5, matrix.length);
    return Array.from({ length: size }, (_, rowIdx) =>
      Array.from({ length: size }, (_, colIdx) => matrix[rowIdx]?.[colIdx] ?? 0)
    );
  }, [evaluationSnapshot]);
  const evaluationMatrixHasOverflow = useMemo(() => {
    if (!evaluationSnapshot || evaluationSnapshot.testMetrics?.kind !== 'classification') return false;
    const matrix = evaluationSnapshot.testMetrics.confusionMatrix ?? [];
    if (matrix.length > 5) return true;
    return matrix.some((row) => row.length > 5);
  }, [evaluationSnapshot]);
  const confusionCellSamples = useMemo(() => {
    if (!evaluationSnapshot || !selectedConfusionCell || evaluationSnapshot.testMetrics?.kind !== 'classification') return [];
    return evaluationSnapshot.samples.filter((sample) => {
      return sample.actualIndex === selectedConfusionCell.actual && sample.predictedIndex === selectedConfusionCell.predicted;
    });
  }, [evaluationSnapshot, selectedConfusionCell]);
  const regressionScatterPoints = useMemo(() => {
    if (!evaluationSnapshot || evaluationSnapshot.testMetrics?.kind !== 'regression') return [];
    return evaluationSnapshot.samples
      .filter(
        (sample): sample is PredictionSample & { actualValue: number; predictedValue: number } =>
          typeof sample.actualValue === 'number' && typeof sample.predictedValue === 'number'
      )
      .slice(0, 300);
  }, [evaluationSnapshot]);
  const evaluationConfusionMax = useMemo(() => {
    const matrix = evaluationSnapshot?.testMetrics?.kind === 'classification'
      ? evaluationSnapshot.testMetrics.confusionMatrix ?? []
      : [];
    const values = matrix.flat();
    if (!values.length) return 1;
    return Math.max(1, ...values);
  }, [evaluationSnapshot?.testMetrics]);
  const evaluationPrecisionRecallPoints = useMemo(() => {
    if (!evaluationSnapshot || evaluationSnapshot.testMetrics?.kind !== 'classification') return [] as Array<{ recall: number; precision: number }>;
    const matrix = evaluationSnapshot.testMetrics.confusionMatrix ?? [];
    if (!matrix.length) return [] as Array<{ recall: number; precision: number }>;
    const points: Array<{ recall: number; precision: number }> = [];
    for (let cls = 0; cls < matrix.length; cls++) {
      const tp = matrix[cls]?.[cls] ?? 0;
      let fp = 0;
      let fn = 0;
      for (let i = 0; i < matrix.length; i++) {
        if (i !== cls) {
          fp += matrix[i]?.[cls] ?? 0;
          fn += matrix[cls]?.[i] ?? 0;
        }
      }
      const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
      const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
      points.push({ recall, precision });
    }
    points.sort((a, b) => a.recall - b.recall);
    return [{ recall: 0, precision: 1 }, ...points, { recall: 1, precision: 0 }];
  }, [evaluationSnapshot]);
  const suggestedAlgorithms = useMemo(() => insights?.suggestedAlgorithms ?? [], [insights?.suggestedAlgorithms]);
  const allAlgorithms = useMemo(() => getAlgorithmCatalogSuggestions(), []);
  const suggestedReasonById = useMemo(
    () => new Map(suggestedAlgorithms.map((algorithm) => [algorithm.id, algorithm.reason])),
    [suggestedAlgorithms]
  );
  const taskMode = resolveTaskMode(insights);
  const recommendedAlgorithm = insights?.recommendedAlgorithm ?? null;
  const recommendedSuggestion =
    recommendedAlgorithm ? allAlgorithms.find((algorithm) => algorithm.id === recommendedAlgorithm) ?? null : null;
  const compatibilityById = useMemo(
    () =>
      new Map(
        allAlgorithms.map((algorithm) => [
          algorithm.id,
          algorithmCompatibility(algorithm.id, taskMode, insights?.modality ?? 'tabular'),
        ])
      ),
    [allAlgorithms, insights?.modality, taskMode]
  );
  const compatibleAlgorithms = useMemo(
    () => allAlgorithms.filter((algorithm) => compatibilityById.get(algorithm.id)?.compatible),
    [allAlgorithms, compatibilityById]
  );
  const incompatibleAlgorithms = useMemo(
    () => allAlgorithms.filter((algorithm) => !compatibilityById.get(algorithm.id)?.compatible),
    [allAlgorithms, compatibilityById]
  );
  const selectableAlgorithmSet = useMemo(
    () =>
      new Set(
        allAlgorithms
          .filter((algorithm) => isSelectableSuggestion(algorithm) && compatibilityById.get(algorithm.id)?.compatible)
          .map((algorithm) => algorithm.id)
      ),
    [allAlgorithms, compatibilityById]
  );
  const effectiveModelChoice =
    modelChoice === 'recommended' && recommendedAlgorithm ? recommendedAlgorithm : modelChoice;
  const selectedAlgorithmInfo = allAlgorithms.find((algorithm) => algorithm.id === effectiveModelChoice) ?? null;
  const resolvedTrainingChoice = useMemo<ModelChoice>(() => {
    if (selectedAlgorithmInfo) {
      return selectedAlgorithmInfo.id;
    }
    if (effectiveModelChoice && effectiveModelChoice !== 'recommended' && effectiveModelChoice !== 'auto') {
      return effectiveModelChoice;
    }
    if (recommendedAlgorithm) {
      return recommendedAlgorithm;
    }
    if (insights?.suggestedAlgorithms?.length) {
      return insights.suggestedAlgorithms[0].id;
    }
    return 'random_forest';
  }, [selectedAlgorithmInfo, effectiveModelChoice, recommendedAlgorithm, insights?.suggestedAlgorithms]);
  const recommendationSignature = insights
    ? `${insights.recommendedAlgorithm ?? 'none'}|${insights.datasetSize}|${insights.modality}|${insights.taskType}|${insights.rowCount}`
    : 'none';
  const wasmEditor = preferences.runtime.wasmEditor;
  const wasmTemplate = wasmEditor.templateConfig;
  const datasetLooksImage = useMemo(
    () =>
      (insights?.modality === 'image') ||
      Boolean(
        dataset?.rows.some(
          (row) => typeof row?.image_preview === 'string' || typeof row?.image_name === 'string' || typeof row?.image_path === 'string'
        )
      ),
    [dataset?.rows, insights?.modality]
  );
  const detectedTaskLabel = useMemo(() => {
    if (!insights) return 'Classification';
    if (insights.learningMode === 'unsupervised') return 'Clustering';
    const mode = resolveTaskMode(insights);
    return mode === 'regression' ? 'Regression' : 'Classification';
  }, [insights]);
  const methodTaskLabel = modelMethod === 'unsupervised' ? 'Clustering' : detectedTaskLabel;
  const viewerRowsPerPage = 250;
  const viewerPageCount = dataset ? Math.max(1, Math.ceil(dataset.rows.length / viewerRowsPerPage)) : 1;
  const viewerRows = dataset
    ? dataset.rows.slice(datasetViewerPage * viewerRowsPerPage, (datasetViewerPage + 1) * viewerRowsPerPage)
    : [];
  const trainingLossValues = training.state.curve.map((point) => point.loss);
  const trainingAccuracyValues = training.state.curve.map((point) => point.accuracy);
  const memoryCapacityForBar =
    caps?.maxMemoryMB && Number.isFinite(caps.maxMemoryMB) && caps.maxMemoryMB > 0
      ? caps.maxMemoryMB
      : Math.max(1, hardwareMetrics.memoryMB);
  const memoryUsagePercent = Math.min(
    100,
    Math.max(0, (hardwareMetrics.memoryMB / Math.max(1, memoryCapacityForBar)) * 100)
  );
  const studioModelOptions = useMemo(() => {
    if (modelMethod === 'unsupervised') {
      return allAlgorithms.filter((algorithm) => algorithm.id === 'kmeans' || algorithm.id === 'dbscan');
    }
    const targetMode = methodTaskLabel === 'Regression' ? 'regression' : 'classification';
    const candidates = allAlgorithms.filter((algorithm) =>
      algorithmCompatibility(algorithm.id, targetMode as TaskMode, insights?.modality ?? 'tabular').compatible
    );
    if (!candidates.length) return allAlgorithms;
    const recommended = recommendedAlgorithm
      ? candidates.find((candidate) => candidate.id === recommendedAlgorithm) ?? null
      : null;
    const ordered = recommended
      ? [recommended, ...candidates.filter((candidate) => candidate.id !== recommended.id)]
      : candidates;
    return ordered;
  }, [allAlgorithms, methodTaskLabel, modelMethod, insights?.modality, recommendedAlgorithm]);
  const column1Valid = Boolean(dataset) && Boolean(preprocessConfig.targetColumn) && Boolean(processedDataset);
  const column2Valid = Boolean(resolvedTrainingChoice);
  const column3Valid = preferences.epochs > 0 && preferences.batchSize > 0;
  const updateNeuralSettings = (patch: Partial<TrainingPreferences['neuralNetwork']>) => {
    setPreferences((prev) => ({
      ...prev,
      neuralNetwork: {
        ...prev.neuralNetwork,
        ...patch,
      },
    }));
  };
  const insightColumns = useMemo(() => {
    if (!columns.length) return columns;
    const target = preprocessConfig.targetColumn;
    const filtered = columns.filter(
      (column) => !preprocessConfig.droppedColumns.includes(column) || (target !== null && column === target)
    );
    return filtered.length ? filtered : columns;
  }, [columns, preprocessConfig.droppedColumns, preprocessConfig.targetColumn]);

  useEffect(() => {
    if (typeof document === 'undefined') return;
    document.documentElement.setAttribute('data-theme', 'dark');
  }, []);

  useEffect(() => {
    if (sourceMode === 'local') setActiveSource('upload');
    if (sourceMode === 'kaggle') setActiveSource('kaggle');
    if (sourceMode === 'huggingface') setActiveSource('huggingface');
  }, [sourceMode]);

  useEffect(() => {
    if (modelMethod === 'unsupervised') {
      setModelChoice((prev) => (prev === 'dbscan' ? 'dbscan' : 'kmeans'));
      setModelChoiceTouched(true);
      return;
    }
    if (modelChoice === 'kmeans' || modelChoice === 'dbscan') {
      setModelChoice(studioModelOptions[0]?.id ?? 'random_forest');
      setModelChoiceTouched(true);
    }
  }, [modelChoice, modelMethod, studioModelOptions]);

  useEffect(() => {
    setDatasetViewerPage(0);
    if (datasetViewerOpen) {
      setDatasetViewerMode(datasetLooksImage ? 'images' : 'table');
    }
  }, [datasetViewerOpen, dataset?.source.name, datasetLooksImage]);

  useEffect(() => {
    if (studioTab !== 'inference') {
      setEvaluationMatrixExpanded(false);
      setSelectedConfusionCell(null);
    }
  }, [studioTab]);

  useEffect(() => {
    const latestStep = training.state.curve[training.state.curve.length - 1]?.step ?? 0;
    hardwareSnapshotRef.current = {
      trainingActive:
        training.state.phase === 'cleaning_data' ||
        training.state.phase === 'training_model' ||
        training.state.phase === 'optimizing_parameters',
      progressPercent: training.state.progressPercent,
      curveStep: latestStep,
      curveLength: training.state.curve.length,
      datasetRows: processedDataset?.features.length ?? 0,
      featureCount: processedDataset?.featureNames.length ?? 0,
    };
  }, [processedDataset?.featureNames.length, processedDataset?.features.length, training.state.curve, training.state.phase, training.state.progressPercent]);

  useEffect(() => {
    const monitor = new HardwareMonitor((metrics) => {
      setHardwareMetrics(metrics);
    }, 800);
    monitor.start(() => hardwareSnapshotRef.current);
    return () => {
      monitor.stop();
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const stored = await loadPreferencesFromIndexedDb();
        if (!cancelled && stored) {
          setPreferences((prev) => mergeStoredPreferences(prev, stored));
        }
      } catch {
        // ignore IndexedDB read failures
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      void savePreferencesToIndexedDb(preferences).catch(() => {
        // ignore IndexedDB write failures
      });
    }, 250);
    return () => window.clearTimeout(timer);
  }, [preferences]);

  useEffect(() => {
    if (pendingRestoreAttemptedRef.current) return;
    pendingRestoreAttemptedRef.current = true;

    let cancelled = false;
    (async () => {
      try {
        const pending = await loadPendingRunState();
        if (cancelled || !pending) return;

        const shouldResume = window.confirm(
          `Resume incomplete training run "${pending.runId}" from ${new Date(pending.createdAt).toLocaleString()}?`
        );
        if (!shouldResume) {
          await clearPendingRunState();
          return;
        }

        const restoredPreferences = mergeStoredPreferences(DEFAULT_PREFERENCES, pending.preferences);
        const trainingDataset = pending.trainingDataset ?? pending.processedDataset;
        const restoredSplit = pending.splitChoice ?? '80_20';
        const restoredHeldOutDataset = pending.heldOutDataset ?? null;
        const restoredHeldOutRows = pending.heldOutRows ?? [];
        const { insights: restoredInsights } = analyzeDataset(pending.dataset.rows, pending.dataset.columns, {
          inferredFormat: pending.dataset.inferredFormat,
          selectedTarget: pending.processedDataset.targetColumn,
        });

        setDataset(pending.dataset);
        setColumns(pending.dataset.columns);
        setPreviewRows(pending.dataset.rows.slice(0, 50));
        setProfiles(buildColumnProfiles(pending.dataset.rows, pending.dataset.columns));
        setInsights(restoredInsights);
        setPreprocessConfig({
          ...DEFAULT_PREPROCESS,
          ...(pending.processedDataset.preprocessing ?? {}),
        });
        setProcessedDataset(pending.processedDataset);
        setSplitChoice(restoredSplit);
        setSplitUsed(restoredSplit);
        setSplitCounts(
          restoredHeldOutDataset
            ? {
                train: trainingDataset.features.length,
                test: restoredHeldOutDataset.features.length,
              }
            : null
        );
        setHeldOutDataset(restoredHeldOutDataset);
        setHeldOutRows(restoredHeldOutRows);
        setPreferences(restoredPreferences);
        setModelChoice(pending.modelChoice);
        setModelChoiceTouched(true);
        setRunId(pending.runId);
        setStep('training');

        const datasetTransfer = createDatasetTransferPayload(trainingDataset, caps ?? null);
        const datasetForWorker = datasetTransfer ? datasetPayloadWithoutDenseMatrices(trainingDataset) : trainingDataset;
        await training.initialize(
          pending.runId,
          datasetForWorker,
          datasetTransfer,
          pending.modelChoice,
          restoredPreferences,
          caps ?? null
        );
        training.start();
      } catch (error: any) {
        if (!cancelled) {
          setPreprocessError(error?.message ?? 'Could not restore pending run.');
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [caps, training]);

  useEffect(() => {
    if (training.state.phase !== 'completed' || !training.state.metrics || !processedDataset || !runId) return;
    void saveTrainingHistoryEntry({
      runId,
      modelChoice: resolvedTrainingChoice,
      resolvedModel: training.state.resolvedModel,
      targetColumn: processedDataset.targetColumn,
      problemType: processedDataset.problemType,
      datasetRows: splitCounts?.train ?? processedDataset.features.length,
      metrics: training.state.metrics,
      createdAt: Date.now(),
    }).catch(() => {
      // ignore IndexedDB write failures
    });
    void clearPendingRunState().catch(() => {
      // ignore IndexedDB write failures
    });
  }, [training.state.phase, training.state.metrics, training.state.resolvedModel, processedDataset, runId, resolvedTrainingChoice, splitCounts?.train]);

  useEffect(() => {
    if (!dataset || !insightColumns.length) return;
    const { insights: nextInsights } = analyzeDataset(dataset.rows, insightColumns, {
      inferredFormat: dataset.inferredFormat,
      selectedTarget: preprocessConfig.targetColumn,
    });
    setInsights(nextInsights);
  }, [dataset, insightColumns, preprocessConfig.targetColumn]);

  useEffect(() => {
    if (!insights) return;
    const resolvedRecommended =
      recommendedAlgorithm && selectableAlgorithmSet.has(recommendedAlgorithm)
        ? recommendedAlgorithm
        : insights.suggestedAlgorithms.find((algorithm) => selectableAlgorithmSet.has(algorithm.id))?.id ??
          Array.from(selectableAlgorithmSet.values())[0] ??
          null;
    if (!resolvedRecommended) return;

    const currentChoice =
      modelChoice === 'recommended' && recommendedAlgorithm ? recommendedAlgorithm : modelChoice;
    const currentIsAlgorithm = typeof currentChoice === 'string' && currentChoice !== 'auto' && currentChoice !== 'recommended';
    const currentValid = currentIsAlgorithm ? selectableAlgorithmSet.has(currentChoice as AlgorithmId) : true;

    if (!modelChoiceTouched || !currentValid) {
      setModelChoice(resolvedRecommended);
      setPreferences((prev) => applyRecommendedPreferencePreset(prev, resolvedRecommended, insights));
      setModelChoiceTouched(false);
    }
  }, [insights, recommendationSignature, recommendedAlgorithm, modelChoice, modelChoiceTouched, selectableAlgorithmSet]);

  const updateWasmTemplate = (patch: Partial<WasmFunctionTemplateConfig>) => {
    setPreferences((prev) => {
      const nextTemplate = normalizeTemplateConfig({
        ...prev.runtime.wasmEditor.templateConfig,
        ...patch,
      });
      return {
        ...prev,
        runtime: {
          ...prev.runtime,
          wasmEditor: {
            ...prev.runtime.wasmEditor,
            templateConfig: nextTemplate,
          },
        },
      };
    });
  };

  const switchToEjectMode = () => {
    setPreferences((prev) => {
      const normalizedTemplate = normalizeTemplateConfig(prev.runtime.wasmEditor.templateConfig);
      const existingCode = prev.runtime.wasmEditor.executableCode.trim();
      return {
        ...prev,
        runtime: {
          ...prev.runtime,
          wasmEditor: {
            ...prev.runtime.wasmEditor,
            advancedMode: 'eject',
            templateConfig: normalizedTemplate,
            executableCode: existingCode || generateExecutableFromTemplate(normalizedTemplate),
          },
        },
      };
    });
  };

  const switchToTemplateMode = () => {
    setPreferences((prev) => ({
      ...prev,
      runtime: {
        ...prev.runtime,
        wasmEditor: {
          ...prev.runtime.wasmEditor,
          advancedMode: 'template',
        },
      },
    }));
  };

  const resetExecutableFromTemplate = () => {
    setPreferences((prev) => {
      const normalizedTemplate = normalizeTemplateConfig(prev.runtime.wasmEditor.templateConfig);
      return {
        ...prev,
        runtime: {
          ...prev.runtime,
          wasmEditor: {
            ...prev.runtime.wasmEditor,
            templateConfig: normalizedTemplate,
            executableCode: generateExecutableFromTemplate(normalizedTemplate),
          },
        },
      };
    });
  };

  useEffect(() => {
    if (training.state.phase === 'completed' && training.state.metrics && step === 'training') {
      setStep('results');
    }
  }, [training.state.phase, training.state.metrics, step]);

  const applyParsedDataset = (parsed: ParsedDataset) => {
    const { insights: detectedInsights } = analyzeDataset(parsed.rows, parsed.columns, {
      inferredFormat: parsed.inferredFormat,
    });
    const disableNormalizationByDefault = shouldDisableNormalizationForLargeDataset(parsed);
    setDataset(parsed);
    setColumns(parsed.columns);
    setPreviewRows(parsed.rows.slice(0, 50));
    setProfiles(buildColumnProfiles(parsed.rows, parsed.columns));
    setInsights(detectedInsights);
    setModelChoice('recommended');
    setModelChoiceTouched(false);
    setShowAdvancedModelConfig(false);
    resumePromptedRunBasesRef.current.clear();
    setRunId('');
    setPreprocessConfig((prev) => ({
      ...DEFAULT_PREPROCESS,
      targetColumn: detectedInsights.detectedTarget ?? prev.targetColumn,
      normalizeData: disableNormalizationByDefault ? false : DEFAULT_PREPROCESS.normalizeData,
    }));
    setPreprocessNotice(
      disableNormalizationByDefault
        ? 'Large/high-dimensional dataset detected. Normalization is disabled by default for faster server preprocessing.'
        : null
    );
    setProcessedDataset(null);
    setSplitChoice('80_20');
    setEvalStrategy('split');
    setUploadedEvalFileName(null);
    setUploadedEvalDataset(null);
    setDatasetLoadError(null);
    setDatasetLoadBusy(false);
    setDatasetLoadProgress(null);
    setKaggleInlineStatus(null);
    setKaggleUseCompleteMessage(null);
    setPreprocessError(null);
    setInferenceInput('');
    setInferenceResult(null);
    setInferenceError(null);
    setSplitUsed(null);
    setSplitCounts(null);
    setHeldOutDataset(null);
    setHeldOutRows([]);
    setEvaluationExportStatus(null);
    setPredictionFilter('all');
    setSelectedConfusionCell(null);
    setLiveInferenceTestIndex(0);
    void clearPendingRunState().catch(() => {
      // ignore IndexedDB write failures
    });
    setStep('preprocess');
  };

  const handleUpload = async (file: File) => {
    if (datasetLoadBusy || kaggleLoading) return;
    try {
      setDatasetLoadBusy(true);
      setDatasetLoadProgress('Preparing dataset import...');
      setDatasetLoadError(null);
      const parsed = await parseDatasetFile(file, {
        onProgress: (progress) => setDatasetLoadProgress(describeDatasetParseProgress(progress)),
      });
      applyParsedDataset(parsed);
    } catch (error: any) {
      setDatasetLoadError(error?.message ?? 'Could not parse the uploaded file.');
      setDatasetLoadBusy(false);
      setDatasetLoadProgress(null);
    }
  };

  const handleEvalFileUpload = async (file: File) => {
    if (datasetLoadBusy || kaggleLoading) return;
    try {
      setDatasetLoadBusy(true);
      setDatasetLoadError(null);
      setDatasetLoadProgress('Parsing evaluation dataset...');
      const parsed = await parseDatasetFile(file, {
        onProgress: (progress) => setDatasetLoadProgress(describeDatasetParseProgress(progress)),
      });
      setUploadedEvalFileName(file.name);
      setUploadedEvalDataset(parsed);
      setPreprocessNotice(
        `Evaluation file "${file.name}" loaded (${parsed.rows.length.toLocaleString()} rows). It will be used as held-out data.`
      );
      setDatasetLoadBusy(false);
      setDatasetLoadProgress(null);
    } catch (error: any) {
      setDatasetLoadError(error?.message ?? 'Could not parse evaluation file.');
      setDatasetLoadBusy(false);
      setDatasetLoadProgress(null);
    }
  };

  const onDrop = async (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragActive(false);
    if (datasetLoadBusy || kaggleLoading) return;
    const file = event.dataTransfer.files?.[0];
    if (file) await handleUpload(file);
  };

  const handleConnectDirectory = async () => {
    if (datasetLoadBusy || kaggleLoading) return;
    if (typeof window === 'undefined' || typeof (window as any).showDirectoryPicker !== 'function') {
      setDatasetLoadError('File System Access API is unavailable in this browser. Use Upload or Kaggle import.');
      return;
    }
    try {
      setDatasetLoadBusy(true);
      setDatasetLoadProgress('Opening local directory...');
      setDatasetLoadError(null);
      const directoryHandle = await (window as any).showDirectoryPicker();
      const parsed = await parseDatasetDirectoryHandle(directoryHandle, {
        onProgress: (progress) => setDatasetLoadProgress(describeDatasetParseProgress(progress)),
      });
      applyParsedDataset(parsed);
    } catch (error: any) {
      if (error?.name === 'AbortError') {
        setDatasetLoadBusy(false);
        setDatasetLoadProgress(null);
        return;
      }
      setDatasetLoadError(error?.message ?? 'Could not read data from selected directory.');
      setDatasetLoadBusy(false);
      setDatasetLoadProgress(null);
    }
  };

  const runPreprocess = async (advance: boolean) => {
    if (!dataset) return;
    const rowsForRequest = advance ? dataset.rows : dataset.rows.slice(0, Math.min(dataset.rows.length, 3000));
    const modeLabel = advance ? 'preprocessing' : 'preview';
    let localProgress = 4;
    setPreprocessProgress({
      percent: localProgress,
      message: advance
        ? 'Sending dataset to server preprocessing engine...'
        : `Running server preview on ${rowsForRequest.length.toLocaleString()} sampled rows...`,
    });
    const timer = window.setInterval(() => {
      localProgress = Math.min(94, localProgress + (advance ? 4 : 6));
      setPreprocessProgress({
        percent: localProgress,
        message: `Server ${modeLabel} in progress...`,
      });
    }, 1200);

    try {
      setPreprocessBusy(true);
      setPreprocessError(null);
      const response = await fetch('/api/dataset/preprocess', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          rows: rowsForRequest,
          columns,
          config: preprocessConfig,
          previewOnly: !advance,
        }),
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || payload.error || 'Preprocessing failed on server.');
      }

      setInsights(payload.insights);
      setPreviewRows(payload.previewRows);
      setProcessedDataset(payload.processed as ProcessedDataset);
      if (advance && payload.processed.features.length === 0) {
        throw new Error('Preprocessing did not produce training features. Please retry.');
      }
      setPreprocessProgress({
        percent: 100,
        message: advance ? 'Server preprocessing complete.' : 'Server preview complete.',
      });
      if (advance) setStep('model');
    } catch (error: any) {
      setPreprocessError(error?.message ?? 'Preprocessing failed.');
    } finally {
      window.clearInterval(timer);
      setPreprocessBusy(false);
      window.setTimeout(() => setPreprocessProgress(null), 800);
    }
  };

  const openKaggleOAuthStart = async () => {
    try {
      const response = await fetch('/api/kaggle/oauth/start');
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || 'Could not start Kaggle OAuth flow.');
      if (payload.authorizeUrl) {
        window.open(payload.authorizeUrl, '_blank', 'noopener,noreferrer');
      }
      setKaggleConnectionInfo(payload.message || null);
    } catch (error: any) {
      setKaggleError(error?.message ?? 'Could not open Kaggle OAuth start.');
    }
  };

  const handleKaggleOAuthConnect = async () => {
    if (!kaggleApiToken.trim()) {
      setKaggleError('Enter Kaggle OAuth token first.');
      setKaggleConnectionInfo(null);
      return;
    }
    setKaggleLoading(true);
    setKaggleError(null);
    setKaggleConnectionInfo('Validating Kaggle OAuth token...');
    try {
      const response = await fetch('/api/kaggle/oauth/connect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ apiToken: kaggleApiToken.trim() }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || payload.error || 'OAuth connect failed.');
      setKaggleConnected(true);
      setKaggleConnectionInfo(payload.message || `Connected via ${payload.tokenType || 'OAuth token'}.`);
    } catch (error: any) {
      setKaggleConnected(false);
      const message = error?.message ?? 'OAuth connect failed.';
      setKaggleError(message);
      setKaggleConnectionInfo(`Connection failed: ${message}`);
    } finally {
      setKaggleLoading(false);
    }
  };

  const handleKaggleSearch = async () => {
    if (!kaggleQuery.trim()) return;
    if (kaggleAuthMode === 'oauth_token' && (!kaggleApiToken.trim() || !kaggleConnected)) {
      setKaggleError('Connect OAuth token before searching Kaggle.');
      return;
    }
    if (kaggleAuthMode === 'legacy_key' && (!kaggleUsername.trim() || !kaggleKey.trim())) {
      setKaggleError('Enter Kaggle username and API key before searching.');
      return;
    }
    setKaggleLoading(true);
    setKaggleError(null);
    setKaggleInlineStatus(null);
    setKaggleUseCompleteMessage(null);
    try {
      const response = await fetch('/api/kaggle/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...kaggleAuthPayload,
          query: kaggleQuery.trim(),
        }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || payload.detail || 'Kaggle search failed.');
      setKaggleResults(payload.results ?? []);
    } catch (error: any) {
      const message = error?.message ?? 'Kaggle search failed.';
      markKaggleAuthInvalid(message);
      setKaggleError(message);
    } finally {
      setKaggleLoading(false);
    }
  };

  const handleKaggleRefSelect = async (ref: string) => {
    setSelectedKaggleRef(ref);
    setSelectedKaggleFile('');
    setKaggleFiles([]);
    setKaggleInlineStatus(null);
    setKaggleUseCompleteMessage(null);
    if (kaggleAuthMode === 'oauth_token' && (!kaggleApiToken.trim() || !kaggleConnected)) return;
    if (kaggleAuthMode === 'legacy_key' && (!kaggleUsername.trim() || !kaggleKey.trim())) return;
    try {
      setKaggleLoading(true);
      const response = await fetch('/api/kaggle/files', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...kaggleAuthPayload,
          datasetRef: ref,
        }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || payload.detail || 'Failed to load files.');
      setKaggleFiles(payload.files ?? []);
      const firstTabular = (payload.files ?? []).find((f: any) =>
        String(f.name).toLowerCase().endsWith('.csv') || String(f.name).toLowerCase().endsWith('.json')
      );
      if (firstTabular?.name) setSelectedKaggleFile(firstTabular.name);
    } catch (error: any) {
      const message = error?.message ?? 'Failed to load dataset files.';
      markKaggleAuthInvalid(message);
      setKaggleError(message);
    } finally {
      setKaggleLoading(false);
    }
  };

  const handleKaggleUseDataset = async () => {
    if (!selectedKaggleRef) return;
    if (kaggleAuthMode === 'oauth_token' && (!kaggleApiToken.trim() || !kaggleConnected)) return;
    if (kaggleAuthMode === 'legacy_key' && (!kaggleUsername.trim() || !kaggleKey.trim())) return;
    try {
      setKaggleLoading(true);
      setDatasetLoadBusy(true);
      setKaggleUseCompleteMessage(null);
      setKaggleInlineStatus('Downloading dataset from Kaggle...');
      setDatasetLoadProgress(null);
      setKaggleError(null);
      const response = await fetch('/api/kaggle/download', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...kaggleAuthPayload,
          datasetRef: selectedKaggleRef,
          fileName: selectedKaggleFile || undefined,
        }),
      });

      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.error || payload.detail || 'Kaggle download failed.');
      }
      const filename = response.headers.get('x-browser-first-ai-filename') || `${selectedKaggleRef.split('/')[1]}.zip`;
      const sizeHeader = response.headers.get('x-browser-first-ai-content-length') || response.headers.get('content-length');
      const expectedBytes = sizeHeader ? Number.parseInt(sizeHeader, 10) : NaN;
      if (Number.isFinite(expectedBytes) && expectedBytes > 512 * 1024 * 1024) {
        setKaggleInlineStatus(
          `Large download (${bytesToHuman(expectedBytes)}). For multi-GB files, using the Kaggle CLI and "Open folder" is often more reliable.`
        );
      }
      const { file: downloadedFile, removeTemp } = await streamKaggleDownloadToFile(response, filename, {
        onProgress: ({ bytesReceived, totalBytes }) => {
          const part =
            totalBytes != null && totalBytes > 0
              ? `${bytesToHuman(bytesReceived)} / ${bytesToHuman(totalBytes)}`
              : bytesToHuman(bytesReceived);
          setKaggleInlineStatus(`Downloading from Kaggle… ${part}`);
        },
      });
      let lastParseMessage = '';
      try {
        const parsed = await parseDatasetBlob(downloadedFile, filename, 'kaggle', {
          onProgress: (progress) => {
            setKaggleInlineStatus(describeDatasetParseProgress(progress));
            if (progress.stage === 'completed') lastParseMessage = progress.message;
          },
        });
        applyParsedDataset(parsed);
        setKaggleUseCompleteMessage(
          lastParseMessage ||
            (parsed.inferredFormat === 'images_zip'
              ? `Finished extracting image features for ${parsed.rows.length.toLocaleString()} images.`
              : `Loaded ${parsed.rows.length.toLocaleString()} rows.`)
        );
        setKaggleInlineStatus(null);
      } finally {
        await removeTemp();
      }
      setDatasetLoadBusy(false);
    } catch (error: any) {
      const message = error?.message ?? 'Kaggle import failed.';
      markKaggleAuthInvalid(message);
      setKaggleError(message);
      setDatasetLoadBusy(false);
      setKaggleInlineStatus(null);
      setDatasetLoadProgress(null);
    } finally {
      setKaggleLoading(false);
    }
  };

  const startTraining = async () => {
    if (!processedDataset || processedDataset.features.length === 0 || !dataset) {
      setPreprocessError('Please preprocess the dataset before training.');
      return;
    }
    const trainingChoice = resolvedTrainingChoice;
    let trainingDataset = processedDataset;
    let testingDataset: ProcessedDataset;
    let testingRows: DataRow[] = [];
    if (evalStrategy === 'upload_file') {
      if (!uploadedEvalDataset) {
        setPreprocessError('Upload an evaluation file before starting training with upload-eval strategy.');
        return;
      }
      setDatasetLoadProgress('Preprocessing uploaded evaluation file...');
      const evalResponse = await fetch('/api/dataset/preprocess', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          rows: uploadedEvalDataset.rows,
          columns: uploadedEvalDataset.columns,
          config: preprocessConfig,
          previewOnly: false,
        }),
      });
      const evalPayload = await evalResponse.json().catch(() => ({}));
      if (!evalResponse.ok || !evalPayload?.processed) {
        setDatasetLoadProgress(null);
        setPreprocessError(evalPayload?.detail || evalPayload?.error || 'Could not preprocess uploaded evaluation file.');
        return;
      }
      testingDataset = evalPayload.processed as ProcessedDataset;
      testingRows = uploadedEvalDataset.rows;
      setDatasetLoadProgress(null);
    } else {
      const split = createTrainTestSplit(
        processedDataset,
        dataset.rows,
        splitChoice,
        `${dataset.source.name}_${trainingChoice}_${processedDataset.targetColumn}`
      );
      trainingDataset = split.train.dataset;
      testingDataset = split.test.dataset;
      testingRows = split.test.rows;
    }
    setSplitUsed(splitChoice);
    setSplitCounts({
      train: trainingDataset.features.length,
      test: testingDataset.features.length,
    });
    setHeldOutDataset(testingDataset);
    setHeldOutRows(testingRows);
    setPredictionFilter('all');
    setSelectedConfusionCell(null);
    setEvaluationMatrixExpanded(false);
    setEvaluationExportStatus(null);
    setLiveInferenceTestIndex(0);

    const runBase = runBaseForDataset(trainingDataset, trainingChoice, splitChoice);
    const stored = getStoredRunVersion(runBase);
    let runVersion = stored.version;
    const shouldPromptResume = stored.hasExisting && !resumePromptedRunBasesRef.current.has(runBase);
    if (shouldPromptResume) {
      const resumeExisting = window.confirm(
        'A previous checkpointed run was found for this dataset/model. Select OK to resume it, or Cancel to start a fresh run.'
      );
      if (!resumeExisting) {
        runVersion += 1;
      }
      resumePromptedRunBasesRef.current.add(runBase);
    }
    setStoredRunVersion(runBase, runVersion);

    setStep('training');
    setStudioTab('training');
    try {
      setPreprocessError(null);
      const computedRunId = runIdForDataset(trainingDataset, trainingChoice, runVersion, splitChoice);
      const datasetTransfer = createDatasetTransferPayload(trainingDataset, caps ?? null);
      const datasetForWorker = datasetTransfer ? datasetPayloadWithoutDenseMatrices(trainingDataset) : trainingDataset;
      setRunId(computedRunId);
      setInferenceResult(null);
      setInferenceError(null);
      await savePendingRunState({
        runId: computedRunId,
        dataset,
        processedDataset,
        splitChoice,
        trainingDataset,
        heldOutDataset: testingDataset,
        heldOutRows: testingRows.slice(0, 1000),
        modelChoice: trainingChoice,
        preferences,
        createdAt: Date.now(),
      });
      await training.initialize(computedRunId, datasetForWorker, datasetTransfer, trainingChoice, preferences, caps ?? null);
      training.start();
    } catch (error: any) {
      setStep('setup');
      setPreprocessError(error?.message ?? 'Could not start training.');
      setDatasetLoadProgress(null);
    }
  };

  const onRunInference = () => {
    if (!training.state.artifact || !processedDataset) return;
    try {
      const sample = parseInferenceInput(inferenceInput, processedDataset.featureNames);
      const prediction = predictFromArtifact(training.state.artifact, sample);
      setInferenceError(null);
      if (
        training.state.artifact.modelType === 'linear_regression' &&
        String(training.state.artifact.modelData.mode ?? 'classifier') === 'regressor'
      ) {
        setInferenceResult(`Predicted value: ${prediction.toFixed(6)}`);
      } else {
        const className = labelNameAt(processedDataset.labelNames, Math.max(0, Math.floor(Number(prediction))));
        setInferenceResult(`Predicted class: ${className}`);
      }
    } catch (error: any) {
      setInferenceResult(null);
      setInferenceError(error?.message ?? 'Inference failed.');
    }
  };

  const onRunLiveInferenceFromTestSample = () => {
    if (!training.state.artifact || !heldOutDataset || !heldOutDataset.features.length) return;
    const safeIndex = Math.min(
      Math.max(0, Math.floor(liveInferenceTestIndex)),
      Math.max(0, heldOutDataset.features.length - 1)
    );
    try {
      const featureRow = heldOutDataset.features[safeIndex];
      const prediction = predictFromArtifact(training.state.artifact, featureRow);
      const inferenceKind = training.state.metrics?.kind ?? heldOutDataset.problemType;
      if (inferenceKind === 'regression') {
        const actualValue =
          heldOutDataset.regressionTargets?.[safeIndex] ?? Number(heldOutDataset.labels[safeIndex] ?? 0);
        setInferenceResult(
          `Test sample #${safeIndex + 1}: predicted=${Number(prediction).toFixed(6)}, actual=${actualValue.toFixed(6)}`
        );
      } else {
        const actualIndex = heldOutDataset.labels[safeIndex] ?? 0;
        const predictedIndex = Math.max(0, Math.floor(Number(prediction)));
        const actualLabel = labelNameAt(heldOutDataset.labelNames, actualIndex);
        const predictedLabel = labelNameAt(heldOutDataset.labelNames, predictedIndex);
        setInferenceResult(
          `Test sample #${safeIndex + 1}: predicted=${predictedLabel}, actual=${actualLabel} ${actualIndex === predictedIndex ? '✅' : '❌'}`
        );
      }
      setInferenceError(null);
    } catch (error: any) {
      setInferenceResult(null);
      setInferenceError(error?.message ?? 'Inference failed.');
    }
  };

  const onUploadInferenceSample = async (file: File) => {
    if (!processedDataset || !training.state.artifact) return;
    try {
      const text = await file.text();
      const parsed = JSON.parse(text);
      const sample = Array.isArray(parsed)
        ? parsed.map((value) => Number(value) || 0)
        : processedDataset.featureNames.map((name) => Number((parsed as Record<string, unknown>)[name] ?? 0) || 0);
      if (sample.length !== processedDataset.featureNames.length) {
        throw new Error(`Expected ${processedDataset.featureNames.length} features but received ${sample.length}.`);
      }
      setInferenceInput(JSON.stringify(parsed, null, 2));
      const prediction = predictFromArtifact(training.state.artifact, sample);
      if (
        training.state.artifact.modelType === 'linear_regression' &&
        String(training.state.artifact.modelData.mode ?? 'classifier') === 'regressor'
      ) {
        setInferenceResult(`Uploaded sample prediction: ${prediction.toFixed(6)}`);
      } else {
        const className = labelNameAt(processedDataset.labelNames, Math.max(0, Math.floor(Number(prediction))));
        setInferenceResult(`Uploaded sample prediction: ${className}`);
      }
      setInferenceError(null);
    } catch (error: any) {
      setInferenceResult(null);
      setInferenceError(error?.message ?? 'Could not run inference on uploaded sample.');
    }
  };

  const exportEvaluationMetrics = (format: 'json' | 'csv') => {
    if (!evaluationSnapshot || !runId) return;
    const payload = {
      runId,
      split: splitLabelForChoice(evaluationSnapshot.splitChoice),
      trainCount: evaluationSnapshot.trainCount,
      testCount: evaluationSnapshot.testCount,
      fitSignal: evaluationSnapshot.fitSignal,
      fitMessage: evaluationSnapshot.fitMessage,
      trainMetrics: evaluationSnapshot.trainMetrics,
      testMetrics: evaluationSnapshot.testMetrics,
    };
    if (format === 'json') {
      const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement('a');
      anchor.href = url;
      anchor.download = `${runId}_evaluation_metrics.json`;
      anchor.click();
      URL.revokeObjectURL(url);
      setEvaluationExportStatus('Evaluation metrics exported as JSON.');
      return;
    }

    const rows = [
      ['run_id', runId],
      ['split', splitLabelForChoice(evaluationSnapshot.splitChoice)],
      ['train_count', String(evaluationSnapshot.trainCount)],
      ['test_count', String(evaluationSnapshot.testCount)],
      ['fit_signal', evaluationSnapshot.fitSignal],
      ['fit_message', evaluationSnapshot.fitMessage],
      ['train_kind', evaluationSnapshot.trainMetrics?.kind ?? 'unknown'],
      ['test_kind', evaluationSnapshot.testMetrics?.kind ?? 'unknown'],
      ['train_accuracy', String(evaluationSnapshot.trainMetrics?.accuracy ?? '')],
      ['test_accuracy', String(evaluationSnapshot.testMetrics?.accuracy ?? '')],
      ['train_precision', String(evaluationSnapshot.trainMetrics?.precision ?? '')],
      ['test_precision', String(evaluationSnapshot.testMetrics?.precision ?? '')],
      ['train_recall', String(evaluationSnapshot.trainMetrics?.recall ?? '')],
      ['test_recall', String(evaluationSnapshot.testMetrics?.recall ?? '')],
      ['train_f1', String(evaluationSnapshot.trainMetrics?.f1 ?? '')],
      ['test_f1', String(evaluationSnapshot.testMetrics?.f1 ?? '')],
      ['train_rmse', String(evaluationSnapshot.trainMetrics?.rmse ?? '')],
      ['test_rmse', String(evaluationSnapshot.testMetrics?.rmse ?? '')],
      ['train_mae', String(evaluationSnapshot.trainMetrics?.mae ?? '')],
      ['test_mae', String(evaluationSnapshot.testMetrics?.mae ?? '')],
      ['train_r2', String(evaluationSnapshot.trainMetrics?.r2 ?? '')],
      ['test_r2', String(evaluationSnapshot.testMetrics?.r2 ?? '')],
    ];
    const csv = rows.map((row) => row.map((value) => `"${String(value).replace(/"/g, '""')}"`).join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = `${runId}_evaluation_metrics.csv`;
    anchor.click();
    URL.revokeObjectURL(url);
    setEvaluationExportStatus('Evaluation metrics exported as CSV.');
  };

  const exportEvaluationPredictions = () => {
    if (!evaluationSnapshot || !runId) return;
    const header =
      evaluationSnapshot.testMetrics?.kind === 'regression'
        ? ['row_index', 'actual', 'predicted', 'absolute_error', 'correct']
        : ['row_index', 'actual_label', 'predicted_label', 'correct'];
    const rows = filteredEvaluationSamples.map((sample) =>
      evaluationSnapshot.testMetrics?.kind === 'regression'
        ? [
            sample.rowIndex,
            sample.actualValue?.toFixed(6) ?? '',
            sample.predictedValue?.toFixed(6) ?? '',
            sample.absoluteError?.toFixed(6) ?? '',
            sample.correct ? 'true' : 'false',
          ]
        : [sample.rowIndex, sample.actualLabel, sample.predictedLabel, sample.correct ? 'true' : 'false']
    );
    const csv = [header, ...rows]
      .map((row) => row.map((value) => `"${String(value).replace(/"/g, '""')}"`).join(','))
      .join('\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = `${runId}_predictions.csv`;
    anchor.click();
    URL.revokeObjectURL(url);
    setEvaluationExportStatus('Prediction subset exported as CSV.');
  };

  const copyRunIdToClipboard = () => {
    if (!runId) return;
    if (!navigator.clipboard?.writeText) return;
    void navigator.clipboard.writeText(runId).catch(() => {
      // ignore clipboard failures
    });
  };

  const handleStartNewRun = () => {
    training.stop();
    if (runId) {
      training.clearCheckpoint(runId);
      clearStoredRunVersion(runId);
    }
    void clearPendingRunState().catch(() => {
      // ignore IndexedDB write failures
    });
    setRunId('');
    setInferenceResult(null);
    setInferenceError(null);
    setInferenceInput('');
    setSelectedConfusionCell(null);
    setEvaluationMatrixExpanded(false);
    setPredictionFilter('all');
    setEvaluationExportStatus(null);
    setLiveInferenceTestIndex(0);
    setStudioTab('studio');
    setStep(dataset ? 'model' : 'dataset');
  };
  const selectedStudioModel = modelMethod === 'unsupervised'
    ? (modelChoice === 'dbscan' ? 'dbscan' : 'kmeans')
    : (resolvedTrainingChoice as AlgorithmId);
  const trainingActive =
    training.state.phase === 'cleaning_data' ||
    training.state.phase === 'training_model' ||
    training.state.phase === 'optimizing_parameters';

  return (
    <>
      <Head>
        <title>Browser-First AI Platform</title>
        <meta
          name="description"
          content="Upload or import datasets, preprocess, train models locally, and export/deploy from browser."
        />
      </Head>

      <main className={`${inter.variable} ${jetbrains.variable} mx-auto min-h-screen max-w-[1600px] bg-[#0B0E14] px-6 pb-10 pt-6 text-slate-100`}>
        <header className="relative mb-7 flex min-h-[64px] items-center justify-center">
          <div className="absolute left-[20%] top-1/2 -translate-y-1/2">
            <div className="text-lg font-bold tracking-tight text-emerald-400">Browser-First AI Platform</div>
            <div className="text-xs uppercase tracking-[0.16em] text-slate-500">Local-first ML</div>
          </div>

          <nav className="wk-nav-pill">
            {(['studio', 'training', 'export', 'inference'] as StudioTab[]).map((tab) => (
              <button
                key={tab}
                className={`wk-nav-button ${studioTab === tab ? 'wk-nav-button-active' : ''}`}
                onClick={() => setStudioTab(tab)}
              >
                {tab[0].toUpperCase() + tab.slice(1)}
              </button>
            ))}
          </nav>

          <div className="absolute right-0 flex items-center gap-2 text-sm">
            <button
              className="rounded-full border border-emerald-400/40 bg-emerald-500/15 px-3 py-1 text-emerald-300 transition hover:bg-emerald-500/25"
              onClick={handleStartNewRun}
            >
              New Run
            </button>
            {runId && (
              <button
                className="rounded-full border border-white/10 bg-white/5 px-3 py-1 font-mono text-slate-200 transition hover:border-emerald-400/50"
                onClick={copyRunIdToClipboard}
              >
                Run {runId.slice(0, 10)}...
              </button>
            )}
          </div>
        </header>

        {studioTab === 'studio' && (
          <section className="animate-fade-slide">
            <div className="mb-5 rounded-2xl border border-white/10 bg-gradient-to-r from-[#131b29] via-[#111723] to-[#0f1521] px-6 py-5 shadow-[0_20px_40px_rgba(0,0,0,0.35)]">
              <div className="flex flex-wrap items-start justify-between gap-4">
                <div>
                  <div className="text-[22px] font-semibold tracking-tight">Unified Studio Dashboard</div>
                  <div className="mt-1 text-base text-slate-400">
                    High-density local-first workflow inspired by Unsloth-style studio surfaces.
                  </div>
                </div>
                <div className="grid min-w-[360px] flex-1 gap-2 sm:grid-cols-2 lg:grid-cols-4">
                  <div className="rounded-xl border border-white/10 bg-[#0e1420] px-3 py-2">
                    <div className="wk-card-label">Dataset Rows</div>
                    <div className="wk-number text-[15px] text-slate-100">{dataset?.rows.length.toLocaleString() ?? '0'}</div>
                  </div>
                  <div className="rounded-xl border border-white/10 bg-[#0e1420] px-3 py-2">
                    <div className="wk-card-label">Task</div>
                    <div className="wk-number text-[15px] text-emerald-300">{methodTaskLabel}</div>
                  </div>
                  <div className="rounded-xl border border-white/10 bg-[#0e1420] px-3 py-2">
                    <div className="wk-card-label">Model</div>
                    <div className="wk-number text-[15px] text-slate-100">{displayAlgorithmLabel(selectedStudioModel, methodTaskLabel === 'Regression' ? 'regression' : methodTaskLabel === 'Clustering' ? 'unsupervised' : 'classification')}</div>
                  </div>
                  <div className="rounded-xl border border-white/10 bg-[#0e1420] px-3 py-2">
                    <div className="wk-card-label">Runtime</div>
                    <div className="wk-number text-[15px] text-slate-100">{training.state.backend || 'Hybrid local runtime'}</div>
                  </div>
                </div>
              </div>
            </div>

            <div className="grid gap-4 xl:grid-cols-4">
              <motion.div
                animate={{ opacity: workflowColumnHover !== null && workflowColumnHover !== 1 ? 0.42 : 1 }}
                transition={{ duration: 0.35, ease: [0.4, 0, 0.2, 1] }}
                className={`wk-chain-col ${column1Valid ? 'wk-chain-col-active' : ''} flex flex-col`}
                onMouseEnter={() => setWorkflowColumnHover(1)}
                onMouseLeave={() => setWorkflowColumnHover(null)}
              >
                <div className="mb-4 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="flex h-8 w-8 items-center justify-center rounded-full border border-emerald-400/40 bg-emerald-500/15 text-sm font-bold text-emerald-300">
                      1
                    </span>
                    <h2 className="text-lg font-semibold tracking-tight">Data &amp; Preprocessing</h2>
                  </div>
                  <span className={`wk-number text-xs ${column1Valid ? 'text-emerald-300' : 'text-slate-400'}`}>
                    {column1Valid ? 'VALID' : 'PENDING'}
                  </span>
                </div>

                <div className="mb-3 grid grid-cols-3 gap-2 text-xs">
                  <button
                    className={`rounded-lg border px-2 py-2 ${
                      sourceMode === 'local'
                        ? 'border-emerald-400/60 bg-emerald-500/20 text-emerald-300'
                        : 'border-white/10 bg-[#0E1420] text-slate-300'
                    }`}
                    onClick={() => setSourceMode('local')}
                  >
                    Local
                  </button>
                  <button
                    className={`rounded-lg border px-2 py-2 ${
                      sourceMode === 'kaggle'
                        ? 'border-emerald-400/60 bg-emerald-500/20 text-emerald-300'
                        : 'border-white/10 bg-[#0E1420] text-slate-300'
                    }`}
                    onClick={() => setSourceMode('kaggle')}
                  >
                    Kaggle
                  </button>
                  <button
                    className={`rounded-lg border px-2 py-2 ${
                      sourceMode === 'huggingface'
                        ? 'border-emerald-400/60 bg-emerald-500/20 text-emerald-300'
                        : 'border-white/10 bg-[#0E1420] text-slate-300'
                    }`}
                    onClick={() => setSourceMode('huggingface')}
                  >
                    HF
                  </button>
                </div>

                {sourceMode === 'local' && (
                  <div className="mb-4 space-y-2">
                    <div
                      className={`cursor-pointer rounded-xl border border-dashed p-4 text-center text-sm transition ${
                        dragActive ? 'border-emerald-400 bg-emerald-500/10' : 'border-white/20 bg-[#0E1420]'
                      }`}
                      onDragOver={(event) => {
                        event.preventDefault();
                        if (datasetLoadBusy || kaggleLoading) return;
                        setDragActive(true);
                      }}
                      onDragLeave={() => setDragActive(false)}
                      onDrop={onDrop}
                      onClick={() => fileInputRef.current?.click()}
                    >
                      <div>Drop CSV / JSON / ZIP</div>
                      <div className="mt-1 text-xs text-slate-500">or click to upload dataset</div>
                    </div>
                    <div className="grid grid-cols-2 gap-2">
                      <button
                        className="rounded-xl border border-emerald-400/50 bg-emerald-500/20 px-3 py-2 text-sm font-semibold text-emerald-300"
                        onClick={() => fileInputRef.current?.click()}
                        disabled={datasetLoadBusy || kaggleLoading}
                      >
                        {datasetLoadBusy ? 'Uploading...' : 'Upload'}
                      </button>
                      <button
                        className="rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-sm text-slate-200"
                        onClick={() => void handleConnectDirectory()}
                        disabled={datasetLoadBusy || kaggleLoading}
                      >
                        Directory
                      </button>
                    </div>
                  </div>
                )}

                {sourceMode === 'kaggle' && (
                  <div className="mb-4 space-y-2">
                    <input
                      className="wk-input"
                      type="password"
                      placeholder="Kaggle API token"
                      value={kaggleApiToken}
                      onChange={(event) => {
                        setKaggleApiToken(event.target.value);
                        setKaggleConnected(false);
                        setKaggleConnectionInfo(null);
                        setKaggleError(null);
                      }}
                    />
                    <div className="grid grid-cols-2 gap-2">
                      <button className="rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-xs" onClick={() => void openKaggleOAuthStart()}>
                        OAuth
                      </button>
                      <button
                        className="rounded-xl border border-emerald-400/60 bg-emerald-500/20 px-3 py-2 text-xs font-semibold text-emerald-300"
                        onClick={() => void handleKaggleOAuthConnect()}
                        disabled={!kaggleApiToken.trim() || kaggleLoading}
                      >
                        Connect
                      </button>
                    </div>
                    {(kaggleConnectionInfo || kaggleError) && (
                      <div className={`rounded-xl border px-3 py-2 text-xs ${
                        kaggleError
                          ? 'border-rose-400/40 bg-rose-500/10 text-rose-200'
                          : kaggleConnected
                            ? 'border-emerald-400/40 bg-emerald-500/10 text-emerald-200'
                            : 'border-white/10 bg-white/5 text-slate-300'
                      }`}>
                        {kaggleError ?? kaggleConnectionInfo}
                      </div>
                    )}
                    <div className="grid grid-cols-[1fr_auto] gap-2">
                      <input
                        className="wk-input"
                        value={kaggleQuery}
                        onChange={(event) => setKaggleQuery(event.target.value)}
                        placeholder="Search Kaggle datasets"
                      />
                      <button className="rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-xs" onClick={() => void handleKaggleSearch()}>
                        Search
                      </button>
                    </div>
                    {kaggleResults.length > 0 && (
                      <div className="max-h-48 space-y-1 overflow-auto rounded-xl border border-white/10 bg-[#0E1420] p-2">
                        {kaggleResults.slice(0, 15).map((result) => (
                          <button
                            key={result.ref}
                            className={`w-full rounded-lg border px-2 py-2 text-left text-xs transition-colors ${
                              selectedKaggleRef === result.ref
                                ? 'border-emerald-400/60 bg-emerald-500/10 text-emerald-300'
                                : 'border-white/10 bg-white/5 text-slate-300'
                            }`}
                            onClick={() => void handleKaggleRefSelect(result.ref)}
                          >
                            <div className="truncate font-semibold">{result.title || result.ref}</div>
                            {result.subtitle ? (
                              <div className="mt-0.5 line-clamp-2 text-[10px] leading-snug text-slate-500">{result.subtitle}</div>
                            ) : null}
                            <div className="mt-0.5 font-mono text-[10px] text-slate-500">
                              {bytesToHuman(result.totalBytes)} · {(result.downloadCount ?? 0).toLocaleString()} dl
                              {result.owner ? ` · ${result.owner}` : ''}
                            </div>
                          </button>
                        ))}
                      </div>
                    )}
                    {selectedKaggleRef && (
                      <div className="space-y-2">
                        {selectedKaggleMeta && (
                          <div className="rounded-xl border border-white/10 bg-[#0B121C] px-3 py-2 text-xs text-slate-300">
                            <div className="font-semibold text-emerald-200">{selectedKaggleMeta.title}</div>
                            {selectedKaggleMeta.subtitle ? (
                              <p className="mt-1 text-[11px] leading-relaxed text-slate-400">{selectedKaggleMeta.subtitle}</p>
                            ) : null}
                            <div className="mt-1 flex flex-wrap gap-x-2 gap-y-0.5 font-mono text-[10px] text-slate-500">
                              <span>{bytesToHuman(selectedKaggleMeta.totalBytes)}</span>
                              <span>{(selectedKaggleMeta.downloadCount ?? 0).toLocaleString()} downloads</span>
                              {selectedKaggleMeta.lastUpdated ? <span>Updated {selectedKaggleMeta.lastUpdated}</span> : null}
                            </div>
                          </div>
                        )}
                        <label className="wk-card-label mb-1 block">File (optional)</label>
                        <select
                          className="wk-input mb-1"
                          value={selectedKaggleFile}
                          onChange={(event) => setSelectedKaggleFile(event.target.value)}
                        >
                          <option value="">Entire dataset bundle</option>
                          {kaggleFiles.map((file) => (
                            <option key={file.name} value={file.name}>
                              {file.name} ({bytesToHuman(file.totalBytes)})
                            </option>
                          ))}
                        </select>
                        <button
                          type="button"
                          className="w-full rounded-xl border border-emerald-400/60 bg-emerald-500/20 px-3 py-2 text-xs font-semibold text-emerald-300 transition hover:bg-emerald-500/30 disabled:opacity-50"
                          onClick={() => void handleKaggleUseDataset()}
                          disabled={kaggleLoading}
                        >
                          {kaggleLoading ? 'Importing…' : 'Use dataset'}
                        </button>
                        {kaggleInlineStatus && <p className="text-xs text-slate-400">{kaggleInlineStatus}</p>}
                        {kaggleUseCompleteMessage && !kaggleInlineStatus && (
                          <p className="text-xs text-emerald-300">{kaggleUseCompleteMessage}</p>
                        )}
                      </div>
                    )}
                  </div>
                )}

                {sourceMode === 'huggingface' && (
                  <div className="mb-4 space-y-2">
                    <input
                      className="wk-input"
                      type="password"
                      placeholder="HuggingFace token"
                      value={huggingFaceToken}
                      onChange={(event) => setHuggingFaceToken(event.target.value)}
                    />
                    <input
                      className="wk-input"
                      placeholder="Dataset ID"
                      value={huggingFaceDatasetId}
                      onChange={(event) => setHuggingFaceDatasetId(event.target.value)}
                    />
                    <div className="rounded-xl border border-white/10 bg-[#0E1420] px-3 py-2 text-xs text-slate-400">
                      Token wiring is ready; backend connector can be bound next.
                    </div>
                  </div>
                )}

                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".csv,.json,.zip"
                  className="hidden"
                  onChange={(event) => {
                    const file = event.target.files?.[0];
                    if (file) void handleUpload(file);
                    event.currentTarget.value = '';
                  }}
                />

                <div className="wk-panel-card mb-3">
                  <div className="wk-card-label mb-2">Target Label</div>
                  <select
                    className="wk-input"
                    value={preprocessConfig.targetColumn ?? ''}
                    onChange={(event) =>
                      setPreprocessConfig((prev) => ({
                        ...prev,
                        targetColumn: event.target.value || null,
                      }))
                    }
                  >
                    <option value="">Select target label</option>
                    {columns.map((column) => (
                      <option key={column} value={column}>
                        {column}
                      </option>
                    ))}
                  </select>
                  <div className="mt-2 text-xs text-slate-400">
                    Detected task: <span className="font-semibold text-emerald-300">{detectedTaskLabel}</span>
                  </div>
                </div>

                <div className="wk-panel-card mb-3">
                  <div className="wk-card-label mb-2">Preprocessing</div>
                  <div className="space-y-2 text-xs">
                    <label className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        checked={preprocessConfig.fixMissingValues}
                        onChange={(event) => setPreprocessConfig((prev) => ({ ...prev, fixMissingValues: event.target.checked }))}
                      />
                      Missing Values
                    </label>
                    <label className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        checked={preprocessConfig.normalizeData}
                        onChange={(event) => setPreprocessConfig((prev) => ({ ...prev, normalizeData: event.target.checked }))}
                      />
                      Normalization
                    </label>
                    <label className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        checked={preprocessConfig.encodeCategories}
                        onChange={(event) => setPreprocessConfig((prev) => ({ ...prev, encodeCategories: event.target.checked }))}
                      />
                      Encoding
                    </label>
                    {datasetLooksImage && (
                      <>
                        <label className="flex items-center gap-2">
                          <input
                            type="checkbox"
                            checked={preprocessConfig.augmentImageData}
                            onChange={(event) =>
                              setPreprocessConfig((prev) => ({
                                ...prev,
                                augmentImageData: event.target.checked,
                                imageAugmentationFactor: event.target.checked ? 2 : 1,
                              }))
                            }
                          />
                          Rotation / Flip / Zoom / Brightness
                        </label>
                        {preprocessConfig.augmentImageData && (
                          <div className="grid grid-cols-2 gap-2">
                            <input
                              className="wk-input wk-number"
                              type="number"
                              min={1}
                              max={3}
                              value={preprocessConfig.imageAugmentationFactor}
                              onChange={(event) =>
                                setPreprocessConfig((prev) => ({
                                  ...prev,
                                  imageAugmentationFactor: Math.max(1, Math.min(3, Number(event.target.value) || 1)) as 1 | 2 | 3,
                                }))
                              }
                            />
                            <input
                              className="wk-input wk-number"
                              type="number"
                              min={0}
                              max={0.5}
                              step={0.01}
                              value={preprocessConfig.imageAugmentationNoise}
                              onChange={(event) =>
                                setPreprocessConfig((prev) => ({
                                  ...prev,
                                  imageAugmentationNoise: Math.max(0, Math.min(0.5, Number(event.target.value) || 0)),
                                }))
                              }
                            />
                          </div>
                        )}
                      </>
                    )}
                  </div>
                  {processedDataset && (
                    <div className="mt-3 border-t border-white/10 pt-3 text-xs text-slate-300">
                      <div>Missing before/after: {processedDataset.stats.beforeMissing} → {processedDataset.stats.afterMissing}</div>
                      <div>Encoded: {processedDataset.stats.encodedColumns.length}, Normalized: {processedDataset.stats.normalizedColumns.length}</div>
                    </div>
                  )}
                </div>

                <div className="wk-panel-card mb-3">
                  <div className="wk-card-label mb-2">Eval Dataset</div>
                  <div className="mb-2 flex gap-2">
                    <button
                      className={`flex-1 rounded-lg border px-2 py-1 text-xs ${evalStrategy === 'split' ? 'border-emerald-400/60 bg-emerald-500/20 text-emerald-300' : 'border-white/10 bg-white/5 text-slate-300'}`}
                      onClick={() => setEvalStrategy('split')}
                    >
                      Split
                    </button>
                    <button
                      className={`flex-1 rounded-lg border px-2 py-1 text-xs ${evalStrategy === 'upload_file' ? 'border-emerald-400/60 bg-emerald-500/20 text-emerald-300' : 'border-white/10 bg-white/5 text-slate-300'}`}
                      onClick={() => setEvalStrategy('upload_file')}
                    >
                      Upload Eval
                    </button>
                  </div>
                  {evalStrategy === 'split' ? (
                    <div className="grid grid-cols-2 gap-2">
                      <button
                        className={`rounded-lg border px-3 py-2 text-xs ${splitChoice === '80_20' ? 'border-emerald-400/60 bg-emerald-500/20 text-emerald-300' : 'border-white/10 bg-white/5 text-slate-300'}`}
                        onClick={() => setSplitChoice('80_20')}
                      >
                        80 / 20
                      </button>
                      <button
                        className={`rounded-lg border px-3 py-2 text-xs ${splitChoice === '90_10' ? 'border-emerald-400/60 bg-emerald-500/20 text-emerald-300' : 'border-white/10 bg-white/5 text-slate-300'}`}
                        onClick={() => setSplitChoice('90_10')}
                      >
                        90 / 10
                      </button>
                    </div>
                  ) : (
                    <div className="space-y-2">
                      <button
                        className="w-full rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-xs"
                        onClick={() => evalFileInputRef.current?.click()}
                      >
                        {uploadedEvalFileName ?? 'Upload Evaluation File'}
                      </button>
                      <input
                        ref={evalFileInputRef}
                        type="file"
                        accept=".csv,.json,.zip"
                        className="hidden"
                        onChange={(event) => {
                          const file = event.target.files?.[0];
                          if (file) void handleEvalFileUpload(file);
                          event.currentTarget.value = '';
                        }}
                      />
                      {uploadedEvalDataset && (
                        <p className="text-xs text-slate-400">Rows: {uploadedEvalDataset.rows.length.toLocaleString()}</p>
                      )}
                    </div>
                  )}
                </div>

                <div className="wk-panel-card mb-3">
                  <div className="wk-card-label mb-2">Dataset Metadata</div>
                  <div className="space-y-1 text-sm text-slate-300">
                    <div className="flex justify-between"><span>Rows</span><span className="wk-number">{dataset?.rows.length.toLocaleString() ?? '0'}</span></div>
                    <div className="flex justify-between"><span>Columns</span><span className="wk-number">{columns.length}</span></div>
                    <div className="flex justify-between"><span>Detected Task</span><span className="wk-number text-emerald-300">{methodTaskLabel}</span></div>
                  </div>
                </div>

                <div className="mt-auto rounded-xl border border-white/10 bg-[#0E1420] p-3">
                  <div className="mb-2 text-xs uppercase tracking-[0.08em] text-slate-400">Actions</div>
                  <div className="grid grid-cols-2 gap-2">
                    <button
                      className="rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-xs text-slate-200"
                      onClick={() => setDatasetViewerOpen(true)}
                      disabled={!dataset}
                    >
                      Preview
                    </button>
                    <button
                      className="rounded-xl border border-emerald-400/60 bg-emerald-500/20 px-3 py-2 text-xs font-semibold text-emerald-300"
                      onClick={() => void runPreprocess(true)}
                      disabled={!dataset || preprocessBusy}
                    >
                      {preprocessBusy ? 'Applying...' : 'Apply'}
                    </button>
                  </div>
                  <p className="mt-2 text-xs text-slate-400">Preview opens a floating dataset viewer for table/image inspection.</p>
                </div>

                {(datasetLoadProgress || preprocessProgress?.message) && (
                  <p className="mt-2 text-xs text-emerald-300">{datasetLoadProgress ?? preprocessProgress?.message}</p>
                )}
                {(datasetLoadError || kaggleError || preprocessError) && (
                  <p className="mt-2 text-xs text-rose-300">{datasetLoadError || kaggleError || preprocessError}</p>
                )}
              </motion.div>

              <motion.div
                animate={{ opacity: workflowColumnHover !== null && workflowColumnHover !== 2 ? 0.42 : 1 }}
                transition={{ duration: 0.35, ease: [0.4, 0, 0.2, 1] }}
                className={`wk-chain-col ${column2Valid ? 'wk-chain-col-active' : ''}`}
                onMouseEnter={() => setWorkflowColumnHover(2)}
                onMouseLeave={() => setWorkflowColumnHover(null)}
              >
                <div className="mb-4 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="flex h-8 w-8 items-center justify-center rounded-full border border-emerald-400/40 bg-emerald-500/15 text-sm font-bold text-emerald-300">
                      2
                    </span>
                    <h2 className="text-lg font-semibold tracking-tight">Model Architecture</h2>
                  </div>
                  <span className={`wk-number text-xs ${column2Valid ? 'text-emerald-300' : 'text-slate-400'}`}>
                    {column2Valid ? 'VALID' : 'PENDING'}
                  </span>
                </div>

                <div className="space-y-3">
                  <div>
                    <label className="wk-card-label mb-1 block">Method</label>
                    <select className="wk-input" value={modelMethod} onChange={(event) => setModelMethod(event.target.value as MethodMode)}>
                      <option value="supervised">Supervised</option>
                      <option value="unsupervised">Unsupervised</option>
                    </select>
                  </div>
                  <div>
                    <label className="wk-card-label mb-1 block">Task</label>
                    <select className="wk-input" value={methodTaskLabel} disabled={modelMethod === 'supervised'} onChange={() => undefined}>
                      <option>{methodTaskLabel}</option>
                    </select>
                  </div>
                  <div>
                    <label className="wk-card-label mb-1 block">Model</label>
                    <select
                      className="wk-input"
                      value={selectedStudioModel}
                      onChange={(event) => {
                        setModelChoice(event.target.value as ModelChoice);
                        setModelChoiceTouched(true);
                      }}
                    >
                      {studioModelOptions.map((algorithm, idx) => (
                        <option key={algorithm.id} value={algorithm.id}>
                          {idx === 0 && modelMethod === 'supervised' ? 'Recommended - ' : ''}
                          {displayAlgorithmLabel(
                            algorithm.id,
                            methodTaskLabel === 'Regression' ? 'regression' : methodTaskLabel === 'Clustering' ? 'unsupervised' : 'classification'
                          )}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div className="rounded-xl border border-white/10 bg-[#0E1420] px-3 py-2 text-sm text-slate-300">
                    {recommendedSuggestion
                      ? `${recommendedSuggestion.label} recommended for ${insights?.datasetSize ?? 'unknown'} ${insights?.modality ?? 'tabular'} data.`
                      : 'Waiting for recommendation engine.'}
                  </div>
                </div>
              </motion.div>

              <motion.div
                animate={{ opacity: workflowColumnHover !== null && workflowColumnHover !== 3 ? 0.42 : 1 }}
                transition={{ duration: 0.35, ease: [0.4, 0, 0.2, 1] }}
                className={`wk-chain-col ${column3Valid ? 'wk-chain-col-active' : ''}`}
                onMouseEnter={() => setWorkflowColumnHover(3)}
                onMouseLeave={() => setWorkflowColumnHover(null)}
              >
                <div className="mb-4 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="flex h-8 w-8 items-center justify-center rounded-full border border-emerald-400/40 bg-emerald-500/15 text-sm font-bold text-emerald-300">
                      3
                    </span>
                    <h2 className="text-lg font-semibold tracking-tight">Hyperparameters</h2>
                  </div>
                  <span className={`wk-number text-xs ${column3Valid ? 'text-emerald-300' : 'text-slate-400'}`}>
                    {column3Valid ? 'VALID' : 'PENDING'}
                  </span>
                </div>

                {!hyperAdvancedOpen && (
                  <div className="wk-panel-card mb-3 border border-emerald-500/15 bg-emerald-500/10">
                    <div className="wk-card-label mb-2">Training defaults</div>
                    <p className="mb-3 text-xs leading-relaxed text-slate-400">
                      Recommended starting values. Open Advanced when you need optimizers, schedulers, regularization, or architecture
                      controls.
                    </p>
                    <div className="grid gap-2 sm:grid-cols-3">
                      <div>
                        <label className="wk-card-label mb-1 block">Epochs</label>
                        <input
                          className="wk-input wk-number"
                          type="number"
                          min={1}
                          max={800}
                          value={preferences.epochs}
                          onChange={(event) => setPreferences((prev) => ({ ...prev, epochs: Math.max(1, Number(event.target.value) || 1) }))}
                        />
                      </div>
                      <div>
                        <label className="wk-card-label mb-1 block">Batch size</label>
                        <input
                          className="wk-input wk-number"
                          type="number"
                          min={1}
                          max={4096}
                          value={preferences.batchSize}
                          onChange={(event) => setPreferences((prev) => ({ ...prev, batchSize: Math.max(1, Number(event.target.value) || 1) }))}
                        />
                      </div>
                      <div>
                        <label className="wk-card-label mb-1 block">Learning rate</label>
                        <input
                          className="wk-input wk-number"
                          type="number"
                          step={0.00001}
                          min={0.000001}
                          max={1}
                          value={preferences.learningRate}
                          onChange={(event) =>
                            setPreferences((prev) => ({ ...prev, learningRate: Math.max(1e-6, Number(event.target.value) || 0.001) }))
                          }
                        />
                      </div>
                    </div>
                    <div className="mt-2 text-xs text-slate-500">
                      Optimizer: <span className="text-slate-300">{preferences.optimizer}</span> · Scheduler:{' '}
                      <span className="text-slate-300">{preferences.lrScheduler}</span>
                    </div>
                    <button
                      type="button"
                      className="mt-3 w-full rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-xs font-semibold text-emerald-300 transition hover:bg-white/10"
                      onClick={() => setHyperAdvancedOpen(true)}
                    >
                      Advanced tuning…
                    </button>
                  </div>
                )}
                {hyperAdvancedOpen && (
                <div className="space-y-3">
                  <div className="mb-1 flex items-center justify-between rounded-lg border border-white/10 bg-[#0E1420] px-3 py-2">
                    <span className="text-xs text-slate-500">Full hyperparameter control</span>
                    <button
                      type="button"
                      className="text-xs font-semibold text-emerald-400 transition hover:text-emerald-300"
                      onClick={() => setHyperAdvancedOpen(false)}
                    >
                      Hide advanced
                    </button>
                  </div>
                  <div className="wk-panel-card">
                    <div className="wk-card-label mb-2">1. Core Training Loop</div>
                    <div className="grid gap-2">
                      <div>
                        <label className="wk-card-label mb-1 block">Epochs / Max Steps</label>
                        <input
                          className="wk-input wk-number"
                          type="number"
                          min={1}
                          max={800}
                          value={preferences.epochs}
                          onChange={(event) => setPreferences((prev) => ({ ...prev, epochs: Math.max(1, Number(event.target.value) || 1) }))}
                        />
                      </div>
                      <div>
                        <label className="wk-card-label mb-1 block">Batch Size</label>
                        <input
                          className="wk-input wk-number"
                          type="number"
                          min={1}
                          max={4096}
                          value={preferences.batchSize}
                          onChange={(event) => setPreferences((prev) => ({ ...prev, batchSize: Math.max(1, Number(event.target.value) || 1) }))}
                        />
                      </div>
                      <label className="flex items-center gap-2 text-xs text-slate-300">
                        <input
                          type="checkbox"
                          checked={preferences.shuffleEachEpoch}
                          onChange={(event) => setPreferences((prev) => ({ ...prev, shuffleEachEpoch: event.target.checked }))}
                        />
                        Shuffle each epoch
                      </label>
                      <div>
                        <label className="wk-card-label mb-1 block">Early Stopping Patience</label>
                        <input
                          className="wk-input wk-number"
                          type="number"
                          min={0}
                          max={200}
                          value={preferences.earlyStoppingPatience}
                          onChange={(event) =>
                            setPreferences((prev) => ({
                              ...prev,
                              earlyStoppingPatience: Math.max(0, Number(event.target.value) || 0),
                            }))
                          }
                        />
                      </div>
                    </div>
                  </div>

                  <div className="wk-panel-card">
                    <div className="wk-card-label mb-2">2. Optimizer &amp; Convergence</div>
                    <div className="grid gap-2">
                      <div>
                        <label className="wk-card-label mb-1 block">Optimizer</label>
                        <select
                          className="wk-input"
                          value={preferences.optimizer}
                          onChange={(event) =>
                            setPreferences((prev) => ({
                              ...prev,
                              optimizer: event.target.value as TrainingPreferences['optimizer'],
                              neuralNetwork: {
                                ...prev.neuralNetwork,
                                optimizer: event.target.value as TrainingPreferences['optimizer'],
                              },
                            }))
                          }
                        >
                          <option value="adamw">AdamW</option>
                          <option value="sgd_momentum">SGD + Momentum</option>
                          <option value="adam">Adam</option>
                          <option value="adamax">Adamax</option>
                        </select>
                      </div>
                      <div>
                        <label className="wk-card-label mb-1 block">Learning Rate</label>
                        <input
                          className="wk-input wk-number"
                          type="number"
                          step={0.00001}
                          min={0.000001}
                          max={1}
                          value={preferences.learningRate}
                          onChange={(event) => setPreferences((prev) => ({ ...prev, learningRate: Math.max(1e-6, Number(event.target.value) || 0.001) }))}
                        />
                      </div>
                      <div>
                        <label className="wk-card-label mb-1 block">Weight Decay (L2)</label>
                        <input
                          className="wk-input wk-number"
                          type="number"
                          step={0.00001}
                          min={0}
                          max={1}
                          value={preferences.weightDecay}
                          onChange={(event) =>
                            setPreferences((prev) => ({
                              ...prev,
                              weightDecay: Math.max(0, Number(event.target.value) || 0),
                              neuralNetwork: { ...prev.neuralNetwork, weightDecay: Math.max(0, Number(event.target.value) || 0) },
                            }))
                          }
                        />
                      </div>
                      {preferences.optimizer === 'sgd_momentum' ? (
                        <div>
                          <label className="wk-card-label mb-1 block">Momentum</label>
                          <input
                            className="wk-input wk-number"
                            type="number"
                            step={0.01}
                            min={0}
                            max={0.999}
                            value={preferences.momentum}
                            onChange={(event) => setPreferences((prev) => ({ ...prev, momentum: Math.max(0, Math.min(0.999, Number(event.target.value) || 0)) }))}
                          />
                        </div>
                      ) : (
                        <div className="grid grid-cols-2 gap-2">
                          <div>
                            <label className="wk-card-label mb-1 block">Beta 1</label>
                            <input
                              className="wk-input wk-number"
                              type="number"
                              step={0.001}
                              min={0}
                              max={0.9999}
                              value={preferences.beta1}
                              onChange={(event) => setPreferences((prev) => ({ ...prev, beta1: Math.max(0, Math.min(0.9999, Number(event.target.value) || 0)) }))}
                            />
                          </div>
                          <div>
                            <label className="wk-card-label mb-1 block">Beta 2</label>
                            <input
                              className="wk-input wk-number"
                              type="number"
                              step={0.001}
                              min={0}
                              max={0.99999}
                              value={preferences.beta2}
                              onChange={(event) => setPreferences((prev) => ({ ...prev, beta2: Math.max(0, Math.min(0.99999, Number(event.target.value) || 0)) }))}
                            />
                          </div>
                        </div>
                      )}
                      <div>
                        <label className="wk-card-label mb-1 block">Learning Rate Scheduler</label>
                        <select
                          className="wk-input"
                          value={preferences.lrScheduler}
                          onChange={(event) => setPreferences((prev) => ({ ...prev, lrScheduler: event.target.value as TrainingPreferences['lrScheduler'] }))}
                        >
                          <option value="constant">Constant</option>
                          <option value="linear_decay">Linear Decay</option>
                          <option value="cosine_annealing">Cosine Annealing</option>
                          <option value="step_lr">StepLR</option>
                        </select>
                      </div>
                      <div>
                        <label className="wk-card-label mb-1 block">Warmup Steps</label>
                        <input
                          className="wk-input wk-number"
                          type="number"
                          min={0}
                          max={200}
                          value={preferences.warmupSteps}
                          onChange={(event) => setPreferences((prev) => ({ ...prev, warmupSteps: Math.max(0, Number(event.target.value) || 0) }))}
                        />
                      </div>
                      {preferences.lrScheduler === 'step_lr' && (
                        <div className="grid grid-cols-2 gap-2">
                          <div>
                            <label className="wk-card-label mb-1 block">Step Size</label>
                            <input
                              className="wk-input wk-number"
                              type="number"
                              min={1}
                              max={500}
                              value={preferences.schedulerStepSize}
                              onChange={(event) => setPreferences((prev) => ({ ...prev, schedulerStepSize: Math.max(1, Number(event.target.value) || 1) }))}
                            />
                          </div>
                          <div>
                            <label className="wk-card-label mb-1 block">Scheduler Gamma</label>
                            <input
                              className="wk-input wk-number"
                              type="number"
                              min={0.01}
                              max={0.99}
                              step={0.01}
                              value={preferences.schedulerGamma}
                              onChange={(event) => setPreferences((prev) => ({ ...prev, schedulerGamma: Math.max(0.01, Math.min(0.99, Number(event.target.value) || 0.5)) }))}
                            />
                          </div>
                        </div>
                      )}
                    </div>
                  </div>

                  <div className="wk-panel-card">
                    <div className="wk-card-label mb-2">3. Architecture &amp; Regularization</div>
                    <div className="grid gap-2">
                      <div>
                        <label className="wk-card-label mb-1 block">Activation Function</label>
                        <select
                          className="wk-input"
                          value={preferences.neuralNetwork.activation}
                          onChange={(event) => updateNeuralSettings({ activation: event.target.value as TrainingPreferences['neuralNetwork']['activation'] })}
                        >
                          <option value="relu">ReLU</option>
                          <option value="leaky_relu">LeakyReLU</option>
                          <option value="tanh">Tanh</option>
                          <option value="sigmoid">Sigmoid</option>
                          <option value="softmax">Softmax</option>
                        </select>
                      </div>
                      <div>
                        <label className="wk-card-label mb-1 block">Dropout Rate ({metricFixed4(preferences.neuralNetwork.dropoutRate)})</label>
                        <input
                          type="range"
                          min={0}
                          max={1}
                          step={0.01}
                          value={preferences.neuralNetwork.dropoutRate}
                          onChange={(event) => updateNeuralSettings({ dropoutRate: Math.max(0, Math.min(1, Number(event.target.value) || 0)) })}
                          className="w-full"
                        />
                      </div>
                      <label className="flex items-center gap-2 text-xs text-slate-300">
                        <input
                          type="checkbox"
                          checked={preferences.neuralNetwork.useBatchNorm}
                          onChange={(event) => updateNeuralSettings({ useBatchNorm: event.target.checked })}
                        />
                        Batch Normalization
                      </label>
                      <label className="flex items-center gap-2 text-xs text-slate-300">
                        <input
                          type="checkbox"
                          checked={preferences.neuralNetwork.useLayerNorm}
                          onChange={(event) => updateNeuralSettings({ useLayerNorm: event.target.checked })}
                        />
                        Layer Normalization
                      </label>
                      <div>
                        <label className="wk-card-label mb-1 block">Gradient Clipping</label>
                        <input
                          className="wk-input wk-number"
                          type="number"
                          min={0}
                          step={0.1}
                          max={100}
                          value={preferences.neuralNetwork.gradientClipping}
                          onChange={(event) => updateNeuralSettings({ gradientClipping: Math.max(0, Number(event.target.value) || 0) })}
                        />
                      </div>
                      {usesNeuralTuning(selectedStudioModel) && (
                        <div className="grid grid-cols-2 gap-2">
                          <div>
                            <label className="wk-card-label mb-1 block">Hidden Layers</label>
                            <input className="wk-input wk-number" type="number" min={1} max={8} value={preferences.neuralNetwork.hiddenLayers} onChange={(event) => updateNeuralSettings({ hiddenLayers: Number(event.target.value) || 1 })} />
                          </div>
                          <div>
                            <label className="wk-card-label mb-1 block">Neurons / Layer</label>
                            <input className="wk-input wk-number" type="number" min={8} max={2048} value={preferences.neuralNetwork.neuronsPerLayer} onChange={(event) => updateNeuralSettings({ neuronsPerLayer: Number(event.target.value) || 64 })} />
                          </div>
                        </div>
                      )}
                    </div>
                  </div>

                  <div key={selectedStudioModel} className="wk-panel-card animate-fade-slide">
                    <div className="wk-card-label mb-2">Model Specific</div>
                    {(selectedStudioModel === 'random_forest' || selectedStudioModel === 'decision_tree') && (
                      <>
                        <label className="wk-card-label mb-1 block">Tree Depth Hint</label>
                        <input type="range" min={2} max={40} value={treeDepthHint} onChange={(event) => setTreeDepthHint(Number(event.target.value))} className="w-full" />
                        <div className="wk-number text-xs text-slate-300">Depth {treeDepthHint}</div>
                      </>
                    )}
                    {selectedStudioModel === 'svm' && (
                      <>
                        <label className="wk-card-label mb-1 block">Kernel</label>
                        <select
                          className="wk-input"
                          value={preferences.algorithm.svmKernel}
                          onChange={(event) =>
                            setPreferences((prev) => ({
                              ...prev,
                              algorithm: { ...prev.algorithm, svmKernel: event.target.value as 'linear' | 'poly' | 'rbf' },
                            }))
                          }
                        >
                          <option value="linear">Linear</option>
                          <option value="poly">Polynomial</option>
                          <option value="rbf">RBF</option>
                        </select>
                      </>
                    )}
                    {selectedStudioModel === 'knn' && (
                      <>
                        <label className="wk-card-label mb-1 block">k Neighbors</label>
                        <input
                          className="wk-input wk-number"
                          type="number"
                          min={1}
                          max={99}
                          value={preferences.algorithm.knnNeighbors}
                          onChange={(event) =>
                            setPreferences((prev) => ({
                              ...prev,
                              algorithm: { ...prev.algorithm, knnNeighbors: Math.max(1, Number(event.target.value) || 1) },
                            }))
                          }
                        />
                      </>
                    )}
                    {selectedStudioModel === 'kmeans' && (
                      <>
                        <label className="wk-card-label mb-1 block">Clusters</label>
                        <input
                          className="wk-input wk-number"
                          type="number"
                          min={2}
                          max={128}
                          value={preferences.algorithm.kmeansClusters}
                          onChange={(event) =>
                            setPreferences((prev) => ({
                              ...prev,
                              algorithm: { ...prev.algorithm, kmeansClusters: Math.max(2, Number(event.target.value) || 2) },
                            }))
                          }
                        />
                      </>
                    )}
                    {selectedStudioModel === 'dbscan' && (
                      <div className="space-y-2">
                        <label className="wk-card-label mb-1 block">eps</label>
                        <input
                          className="wk-input wk-number"
                          type="number"
                          min={0.01}
                          max={20}
                          step={0.01}
                          value={preferences.algorithm.dbscanEpsilon}
                          onChange={(event) =>
                            setPreferences((prev) => ({
                              ...prev,
                              algorithm: { ...prev.algorithm, dbscanEpsilon: Math.max(0.01, Number(event.target.value) || 0.8) },
                            }))
                          }
                        />
                        <label className="wk-card-label mb-1 block">min_samples</label>
                        <input
                          className="wk-input wk-number"
                          type="number"
                          min={2}
                          max={50}
                          value={preferences.algorithm.dbscanMinSamples}
                          onChange={(event) =>
                            setPreferences((prev) => ({
                              ...prev,
                              algorithm: { ...prev.algorithm, dbscanMinSamples: Math.max(2, Number(event.target.value) || 2) },
                            }))
                          }
                        />
                      </div>
                    )}
                  </div>
                </div>
                )}
              </motion.div>

              <motion.div
                animate={{ opacity: workflowColumnHover !== null && workflowColumnHover !== 4 ? 0.42 : 1 }}
                transition={{ duration: 0.35, ease: [0.4, 0, 0.2, 1] }}
                className="wk-chain-col wk-chain-col-end"
                onMouseEnter={() => setWorkflowColumnHover(4)}
                onMouseLeave={() => setWorkflowColumnHover(null)}
              >
                <div className="mb-4 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="flex h-8 w-8 items-center justify-center rounded-full border border-emerald-400/40 bg-emerald-500/15 text-sm font-bold text-emerald-300">
                      4
                    </span>
                    <h2 className="text-lg font-semibold tracking-tight">Training &amp; Ignition</h2>
                  </div>
                  <span className={`wk-number text-xs ${trainingActive ? 'text-emerald-300' : 'text-slate-400'}`}>
                    {training.state.phase.toUpperCase()}
                  </span>
                </div>

                <div className="wk-panel-card mb-3">
                  <div className="wk-card-label mb-1">Ready Status</div>
                  <div className="text-sm text-slate-300">{training.state.statusMessage || 'Ready to Train'}</div>
                  <div className="mt-3 h-2 rounded-full bg-white/10">
                    <div className="h-2 rounded-full bg-emerald-400" style={{ width: `${training.state.progressPercent}%` }} />
                  </div>
                  <div className="wk-number mt-1 text-xs text-slate-400">{metricFixed4(training.state.progressPercent)}%</div>
                </div>

                <button
                  className="mb-2 w-full rounded-2xl border border-emerald-400/70 bg-emerald-500 px-4 py-4 text-lg font-bold text-[#062F25] shadow-[0_0_24px_rgba(16,185,129,0.45)] transition hover:brightness-105 disabled:opacity-60"
                  onClick={() => void startTraining()}
                  disabled={!processedDataset || preprocessBusy || datasetLoadBusy || trainingActive}
                >
                  {trainingActive ? 'Training Running...' : 'Start Training'}
                </button>
                <div className="grid grid-cols-2 gap-2">
                  <button className="rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-xs text-slate-200" onClick={training.stop}>
                    Stop
                  </button>
                  <button className="rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-xs text-slate-200" onClick={() => setStudioTab('training')}>
                    Monitor
                  </button>
                </div>

                <div className="mt-3 rounded-xl border border-white/10 bg-[#0E1420] p-3">
                  <div className="mb-2 text-xs uppercase tracking-[0.08em] text-slate-400">Live Logs</div>
                  <div className="max-h-40 overflow-auto font-mono text-[11px] leading-5 text-emerald-300">
                    {(training.state.logs.length ? training.state.logs.slice(-16) : ['[boot] waiting for training logs...']).map((entry, idx) => (
                      <div key={`${entry}_${idx}`}>{entry}</div>
                    ))}
                  </div>
                </div>
              </motion.div>
            </div>
          </section>
        )}

        {studioTab === 'training' && (
          <section className="animate-fade-slide space-y-4">
            <div className="wk-panel-card">
              <div className="mb-3 flex items-center justify-between">
                <h2 className="text-lg font-semibold">Training Monitor</h2>
                <span className="wk-number text-emerald-300">{training.state.resolvedModel || resolvedTrainingChoice}</span>
              </div>
              <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
                <div className="rounded-xl border border-white/10 bg-[#0E1420] p-3">
                  <div className="text-xs uppercase tracking-[0.08em] text-slate-400">Utilization</div>
                  <div className="wk-number mt-1 text-base text-slate-100">{metricFixed4(hardwareMetrics.utilizationPercent)} %</div>
                  <div className="mt-2 h-2 rounded-full bg-white/10">
                    <div className="h-2 rounded-full bg-emerald-400" style={{ width: `${Math.min(100, Math.max(0, hardwareMetrics.utilizationPercent))}%` }} />
                  </div>
                  <div className="mt-1 text-[11px] text-slate-500">Observed runtime activity</div>
                </div>
                <div className="rounded-xl border border-white/10 bg-[#0E1420] p-3">
                  <div className="text-xs uppercase tracking-[0.08em] text-slate-400">Thermal Load</div>
                  <div className="wk-number mt-1 text-base text-slate-100">{thermalStateLabel(hardwareMetrics.thermalState)}</div>
                  <div className="mt-2 h-2 rounded-full bg-white/10">
                    <div className="h-2 rounded-full bg-emerald-400" style={{ width: `${thermalStateToPercent(hardwareMetrics.thermalState)}%` }} />
                  </div>
                  <div className="mt-1 text-[11px] text-slate-500">Compute Pressure API</div>
                </div>
                <div className="rounded-xl border border-white/10 bg-[#0E1420] p-3">
                  <div className="text-xs uppercase tracking-[0.08em] text-slate-400">Memory In Use</div>
                  <div className="wk-number mt-1 text-base text-slate-100">{metricFixed4(hardwareMetrics.memoryMB)} MB</div>
                  <div className="mt-2 h-2 rounded-full bg-white/10">
                    <div
                      className="h-2 rounded-full bg-emerald-400"
                      style={{ width: `${memoryUsagePercent}%` }}
                    />
                  </div>
                  <div className="mt-1 text-[11px] text-slate-500">Heap + tensor estimate</div>
                </div>
                <div className="rounded-xl border border-white/10 bg-[#0E1420] p-3">
                  <div className="text-xs uppercase tracking-[0.08em] text-slate-400">Energy Impact</div>
                  <div className="wk-number mt-1 text-base text-slate-100">{energyImpactLabel(hardwareMetrics.energyImpact)}</div>
                  <div className="mt-2 h-2 rounded-full bg-white/10">
                    <div className="h-2 rounded-full bg-emerald-400" style={{ width: `${energyImpactToPercent(hardwareMetrics.energyImpact)}%` }} />
                  </div>
                  <div className="mt-1 text-[11px] text-slate-500">Estimated from load profile</div>
                </div>
              </div>
            </div>

            <div className="grid gap-4 lg:grid-cols-2">
              <LineChartCard title="Training Loss" values={trainingLossValues} color="#10B981" valueFormatter={(value) => metricFixed4(value)} />
              <LineChartCard title="Training Accuracy" values={trainingAccuracyValues} color="#34D399" valueFormatter={(value) => `${metricFixed4(value * 100)}%`} />
            </div>

            <div className="wk-panel-card">
              <div className="mb-2 flex items-center justify-between">
                <h3 className="text-base font-semibold">Terminal CLI</h3>
                <span className="wk-number text-xs text-slate-400">Runtime logs</span>
              </div>
              <div className="max-h-[340px] overflow-auto rounded-xl border border-white/10 bg-[#060A12] p-3 font-mono text-[12px] leading-6 text-emerald-300">
                {(training.state.logs.length ? training.state.logs : ['[boot] initializing local-first runtime...']).map((entry, idx) => (
                  <div key={`${entry}_${idx}`}>{entry}</div>
                ))}
              </div>
            </div>
          </section>
        )}

        {studioTab === 'export' && (
          <section className="animate-fade-slide space-y-4">
            <div className="wk-panel-card">
              <h2 className="text-lg font-semibold">Export Hub</h2>
              <p className="mt-1 text-sm text-slate-400">Model artifacts, metrics, and prediction subsets.</p>
            </div>

            {training.state.artifact ? (
              <>
                <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-5">
                  <button className="wk-panel-card text-left transition hover:border-emerald-400/40" onClick={() => training.exportModel('pth')}>
                    <div className="text-sm font-semibold">Export .pth</div>
                    <div className="mt-1 text-xs text-slate-400">PyTorch snapshot</div>
                  </button>
                  <button className="wk-panel-card text-left transition hover:border-emerald-400/40" onClick={() => training.exportModel('tensor')}>
                    <div className="text-sm font-semibold">Export .tensor</div>
                    <div className="mt-1 text-xs text-slate-400">Tensor payload</div>
                  </button>
                  <button className="wk-panel-card text-left transition hover:border-emerald-400/40" onClick={() => exportEvaluationMetrics('json')}>
                    <div className="text-sm font-semibold">Metrics JSON</div>
                    <div className="mt-1 text-xs text-slate-400">Structured report</div>
                  </button>
                  <button className="wk-panel-card text-left transition hover:border-emerald-400/40" onClick={() => exportEvaluationMetrics('csv')}>
                    <div className="text-sm font-semibold">Metrics CSV</div>
                    <div className="mt-1 text-xs text-slate-400">Flat sheet format</div>
                  </button>
                  <button className="wk-panel-card text-left transition hover:border-emerald-400/40" onClick={exportEvaluationPredictions}>
                    <div className="text-sm font-semibold">Predictions CSV</div>
                    <div className="mt-1 text-xs text-slate-400">Held-out subset</div>
                  </button>
                </div>
                {evaluationExportStatus && <p className="text-sm text-emerald-300">{evaluationExportStatus}</p>}
              </>
            ) : (
              <div className="wk-panel-card text-sm text-slate-400">Train a model first to unlock export options.</div>
            )}
          </section>
        )}

        {studioTab === 'inference' && (
          <section className="animate-fade-slide space-y-4">
            <div className="wk-panel-card">
              <h2 className="text-lg font-semibold">Inference Lab</h2>
              <p className="mt-1 text-sm text-slate-400">Held-out evaluation metrics, confusion analysis, and live inference for tabular/image data.</p>
            </div>
            {training.state.artifact && processedDataset ? (
              <>
                {evaluationSnapshot ? (
                  <>
                    <div className="wk-panel-card">
                      <div className="mb-2 flex items-center justify-between">
                        <h3 className="text-base font-semibold">Testing Evaluation Metrics</h3>
                        <div className="flex items-center gap-2">
                          <span className="wk-number text-xs text-slate-400">
                            Split {splitLabelForChoice(evaluationSnapshot.splitChoice)}
                          </span>
                          <span
                            className={`rounded-full border px-2 py-1 text-[10px] font-semibold tracking-[0.08em] ${
                              evaluationSnapshot.fitSignal === 'overfitting'
                                ? 'border-amber-400/50 bg-amber-500/10 text-amber-200'
                                : evaluationSnapshot.fitSignal === 'underfitting'
                                  ? 'border-sky-400/50 bg-sky-500/10 text-sky-200'
                                  : evaluationSnapshot.fitSignal === 'balanced'
                                    ? 'border-emerald-400/50 bg-emerald-500/10 text-emerald-200'
                                    : 'border-white/10 bg-white/5 text-slate-300'
                            }`}
                          >
                            Insight: {evaluationSnapshot.fitSignal.toUpperCase()}
                          </span>
                        </div>
                      </div>
                      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
                        {evaluationSnapshot.testMetrics?.kind === 'classification' ? (
                          <>
                            <div className="rounded-xl border border-white/10 bg-[#0E1420] p-3">
                              <div className="wk-card-label">Accuracy</div>
                              <div className="wk-number text-emerald-300">{metricFixed4((evaluationSnapshot.testMetrics?.accuracy ?? 0) * 100)}%</div>
                            </div>
                            <div className="rounded-xl border border-white/10 bg-[#0E1420] p-3">
                              <div className="wk-card-label">Precision</div>
                              <div className="wk-number text-emerald-300">{metricFixed4((evaluationSnapshot.testMetrics?.precision ?? 0) * 100)}%</div>
                            </div>
                            <div className="rounded-xl border border-white/10 bg-[#0E1420] p-3">
                              <div className="wk-card-label">Recall</div>
                              <div className="wk-number text-emerald-300">{metricFixed4((evaluationSnapshot.testMetrics?.recall ?? 0) * 100)}%</div>
                            </div>
                            <div className="rounded-xl border border-white/10 bg-[#0E1420] p-3">
                              <div className="wk-card-label">F1</div>
                              <div className="wk-number text-emerald-300">{metricFixed4((evaluationSnapshot.testMetrics?.f1 ?? 0) * 100)}%</div>
                            </div>
                          </>
                        ) : (
                          <>
                            <div className="rounded-xl border border-white/10 bg-[#0E1420] p-3">
                              <div className="wk-card-label">RMSE</div>
                              <div className="wk-number text-emerald-300">{metricFixed4(evaluationSnapshot.testMetrics?.rmse)}</div>
                            </div>
                            <div className="rounded-xl border border-white/10 bg-[#0E1420] p-3">
                              <div className="wk-card-label">MAE</div>
                              <div className="wk-number text-emerald-300">{metricFixed4(evaluationSnapshot.testMetrics?.mae)}</div>
                            </div>
                            <div className="rounded-xl border border-white/10 bg-[#0E1420] p-3">
                              <div className="wk-card-label">R²</div>
                              <div className="wk-number text-emerald-300">{metricFixed4(evaluationSnapshot.testMetrics?.r2)}</div>
                            </div>
                            <div className="rounded-xl border border-white/10 bg-[#0E1420] p-3">
                              <div className="wk-card-label">Fit Signal</div>
                              <div className="wk-number text-emerald-300">{evaluationSnapshot.fitSignal.toUpperCase()}</div>
                            </div>
                          </>
                        )}
                      </div>

                      {evaluationSnapshot.testMetrics?.kind === 'classification' && (
                        <div className="mt-4">
                          <div className="rounded-xl border border-white/10 bg-[#0E1420] p-3">
                            <div className="mb-2 text-sm font-semibold">Precision-Recall Graph</div>
                            {(() => {
                              const points = evaluationPrecisionRecallPoints;
                              if (!points.length) {
                                return <p className="text-xs text-slate-400">Precision-recall curve unavailable for this run.</p>;
                              }
                              const width = 320;
                              const height = 210;
                              const pad = 26;
                              const toX = (recall: number) => pad + recall * (width - pad * 2);
                              const toY = (precision: number) => height - pad - precision * (height - pad * 2);
                              const polyline = points
                                .map((point) => `${toX(point.recall).toFixed(2)},${toY(point.precision).toFixed(2)}`)
                                .join(' ');
                              return (
                                <svg viewBox={`0 0 ${width} ${height}`} className="h-[180px] w-full rounded-lg border border-white/10 bg-[#0A111A]">
                                  <line x1={pad} y1={height - pad} x2={width - pad} y2={height - pad} stroke="rgba(255,255,255,0.24)" />
                                  <line x1={pad} y1={pad} x2={pad} y2={height - pad} stroke="rgba(255,255,255,0.24)" />
                                  <polyline points={polyline} fill="none" stroke="#10B981" strokeWidth={2} />
                                  {points.slice(1, -1).map((point, idx) => (
                                    <circle
                                      key={`pr_point_${idx}_${point.recall.toFixed(3)}_${point.precision.toFixed(3)}`}
                                      cx={toX(point.recall)}
                                      cy={toY(point.precision)}
                                      r={2.8}
                                      fill="#10B981"
                                    />
                                  ))}
                                  <text x={pad} y={16} fill="#94A3B8" fontSize="10">Precision</text>
                                  <text x={width - 56} y={height - 8} fill="#94A3B8" fontSize="10">Recall</text>
                                </svg>
                              );
                            })()}
                          </div>
                          <p className="mt-2 text-xs text-slate-300">{evaluationSnapshot.fitMessage}</p>
                        </div>
                      )}
                    </div>

                    <div className="grid gap-4 lg:grid-cols-2">
                      {evaluationSnapshot.testMetrics?.kind === 'classification' ? (
                        <motion.div layoutId="evaluation-confusion-card" className="wk-panel-card relative z-10 overflow-hidden">
                          <div className="mb-2 flex items-center justify-between">
                            <h3 className="text-sm font-semibold">Confusion Matrix</h3>
                            <button
                              className="rounded-lg border border-emerald-400/40 bg-emerald-500/10 px-2 py-1 text-xs text-emerald-200"
                              onClick={() => setEvaluationMatrixExpanded(true)}
                            >
                              Expand
                            </button>
                          </div>
                          {evaluationMiniMatrix.length ? (
                            <button
                              type="button"
                              onClick={() => setEvaluationMatrixExpanded(true)}
                              className="relative block w-full overflow-hidden rounded-xl border border-white/10 bg-[#0A111A] p-3 text-left"
                            >
                              <div className="space-y-1">
                                {evaluationMiniMatrix.map((row, rowIdx) => (
                                  <div className="flex gap-1" key={`mini_matrix_row_${rowIdx}`}>
                                    {row.map((value, colIdx) => (
                                      <span
                                        key={`mini_matrix_cell_${rowIdx}_${colIdx}`}
                                        className="flex h-8 min-w-[32px] items-center justify-center rounded border border-white/10 text-[11px] font-semibold"
                                        style={matrixCellStyle(value, evaluationConfusionMax, rowIdx, colIdx)}
                                      >
                                        {value}
                                      </span>
                                    ))}
                                  </div>
                                ))}
                              </div>
                              <div className="absolute inset-0 flex items-center justify-center bg-black/35 text-xs font-semibold text-slate-100 backdrop-blur-[1px]">
                                Click to expand full matrix
                              </div>
                            </button>
                          ) : (
                            <p className="text-sm text-slate-400">Confusion matrix unavailable for current run.</p>
                          )}
                          <div className="mt-2 text-xs text-slate-400">
                            {evaluationMatrixHasOverflow
                              ? 'Mini-map shown. Open expanded view to inspect all classes.'
                              : 'All classes fit in the mini-map preview.'}
                          </div>
                          {selectedConfusionCell && (
                            <div className="mt-3 rounded-lg border border-white/10 bg-[#0E1420] p-2 text-xs text-slate-300">
                              <div>
                                Selected: actual <strong>{labelNameAt(heldOutDataset?.labelNames, selectedConfusionCell.actual)}</strong>,
                                predicted <strong>{labelNameAt(heldOutDataset?.labelNames, selectedConfusionCell.predicted)}</strong>
                              </div>
                              <div className="mt-1 text-slate-400">{confusionCellSamples.length} samples in this cell.</div>
                            </div>
                          )}
                        </motion.div>
                      ) : (
                        <div className="wk-panel-card">
                          <h3 className="mb-2 text-sm font-semibold">Predicted vs Actual</h3>
                          {regressionScatterPoints.length > 0 ? (() => {
                            const values = regressionScatterPoints.flatMap((sample) => [sample.actualValue, sample.predictedValue]);
                            const min = Math.min(...values);
                            const max = Math.max(...values);
                            const range = Math.max(1e-6, max - min);
                            const size = 260;
                            return (
                              <svg viewBox={`0 0 ${size} ${size}`} className="h-[240px] w-full rounded-lg border border-white/10 bg-[#0A111A]">
                                <line x1={16} y1={size - 16} x2={size - 16} y2={16} stroke="rgba(255,255,255,0.25)" strokeDasharray="4 3" />
                                {regressionScatterPoints.map((sample) => {
                                  const x = 16 + (((sample.actualValue - min) / range) * (size - 32));
                                  const y = size - 16 - (((sample.predictedValue - min) / range) * (size - 32));
                                  return <circle key={`reg_scatter_${sample.rowIndex}`} cx={x} cy={y} r={3} fill="#10B981" opacity={0.78} />;
                                })}
                              </svg>
                            );
                          })() : (
                            <p className="text-sm text-slate-400">No regression points available.</p>
                          )}
                        </div>
                      )}

                      <div className="wk-panel-card relative z-20 isolate border-white/15 bg-[#0F1726]/95 backdrop-blur-md">
                        <div className="mb-2 flex items-center justify-between">
                          <h3 className="text-sm font-semibold">Correct / Wrong Prediction Preview</h3>
                          <div className="flex gap-2 text-xs">
                            <button
                              className={`rounded-lg border px-2 py-1 ${predictionFilter === 'all' ? 'border-emerald-400/60 bg-emerald-500/20 text-emerald-300' : 'border-white/10 bg-white/5 text-slate-300'}`}
                              onClick={() => setPredictionFilter('all')}
                            >
                              All
                            </button>
                            <button
                              className={`rounded-lg border px-2 py-1 ${predictionFilter === 'correct' ? 'border-emerald-400/60 bg-emerald-500/20 text-emerald-300' : 'border-white/10 bg-white/5 text-slate-300'}`}
                              onClick={() => setPredictionFilter('correct')}
                            >
                              Correct
                            </button>
                            <button
                              className={`rounded-lg border px-2 py-1 ${predictionFilter === 'incorrect' ? 'border-emerald-400/60 bg-emerald-500/20 text-emerald-300' : 'border-white/10 bg-white/5 text-slate-300'}`}
                              onClick={() => setPredictionFilter('incorrect')}
                            >
                              Wrong
                            </button>
                          </div>
                        </div>

                        {isImageEvaluation ? (
                          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
                            {filteredEvaluationSamples.slice(0, 12).map((sample) => {
                              const preview = sample.rawRow ? renderPixelRowPreview(sample.rawRow) : null;
                              return (
                                <div
                                  key={`infer_sample_${sample.rowIndex}`}
                                  className={`rounded-xl border p-2 ${sample.correct ? 'border-emerald-400/40 bg-emerald-500/10' : 'border-rose-400/30 bg-rose-500/10'}`}
                                >
                                  <div className="mb-2 aspect-square overflow-hidden rounded-lg border border-white/10" style={imagePlaceholderStyle(sample.rawRow)}>
                                    {preview ? (
                                      <img src={preview} alt={`sample_${sample.rowIndex + 1}`} className="h-full w-full object-cover" />
                                    ) : (
                                      <div className="flex h-full items-center justify-center text-xs text-slate-500">No preview</div>
                                    )}
                                  </div>
                                  <div className="text-xs text-slate-300">Truth: {sample.actualLabel}</div>
                                  <div className="text-xs text-slate-300">Pred: {sample.predictedLabel}</div>
                                </div>
                              );
                            })}
                          </div>
                        ) : (
                          <div className="max-h-[280px] overflow-auto rounded-lg border border-white/10 bg-[#0D131E]">
                            <table className="min-w-full text-xs">
                              <thead className="bg-[#0E1420] text-slate-400">
                                <tr>
                                  <th className="px-2 py-2 text-left">#</th>
                                  <th className="px-2 py-2 text-left">Truth</th>
                                  <th className="px-2 py-2 text-left">Prediction</th>
                                  {evaluationSnapshot.testMetrics?.kind === 'regression' && <th className="px-2 py-2 text-left">Abs Error</th>}
                                  <th className="px-2 py-2 text-left">Status</th>
                                </tr>
                              </thead>
                              <tbody>
                                {filteredEvaluationSamples.slice(0, 24).map((sample) => (
                                  <tr key={`pred_row_${sample.rowIndex}`} className="border-t border-white/10">
                                    <td className="px-2 py-2">{sample.rowIndex + 1}</td>
                                    <td className="px-2 py-2">{sample.actualLabel}</td>
                                    <td className="px-2 py-2">{sample.predictedLabel}</td>
                                    {evaluationSnapshot.testMetrics?.kind === 'regression' && (
                                      <td className="px-2 py-2">{metricFixed4(sample.absoluteError)}</td>
                                    )}
                                    <td className="px-2 py-2">{sample.correct ? 'Correct' : 'Wrong'}</td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        )}
                      </div>
                    </div>
                  </>
                ) : (
                  <div className="wk-panel-card text-sm text-slate-400">Run training to populate held-out evaluation metrics and previews.</div>
                )}

                <div className="grid gap-4 lg:grid-cols-2">
                  <div className="wk-panel-card">
                    <div className="mb-2 text-sm font-semibold">Held-Out Sample Inference</div>
                    <select className="wk-input" value={liveInferenceTestIndex} onChange={(event) => setLiveInferenceTestIndex(Number(event.target.value) || 0)}>
                      {(heldOutDataset?.features ?? []).slice(0, 300).map((_, index) => (
                        <option key={`holdout_${index}`} value={index}>
                          Test sample #{index + 1}
                        </option>
                      ))}
                    </select>
                    <button className="mt-2 w-full rounded-xl border border-emerald-400/60 bg-emerald-500/20 px-3 py-2 text-sm font-semibold text-emerald-300" onClick={onRunLiveInferenceFromTestSample}>
                      Run Selected Test Sample
                    </button>
                    {inferenceResult && <p className="mt-2 text-sm text-emerald-300">{inferenceResult}</p>}
                    {inferenceError && <p className="mt-2 text-sm text-rose-300">{inferenceError}</p>}
                  </div>
                  <div className="wk-panel-card">
                    <div className="mb-2 text-sm font-semibold">Custom Sample Inference</div>
                    <textarea
                      className="wk-input min-h-[120px] font-mono text-xs"
                      value={inferenceInput}
                      onChange={(event) => setInferenceInput(event.target.value)}
                      placeholder={processedDataset.featureNames.length > 0 ? processedDataset.featureNames.map(() => '0').join(', ') : '{}'}
                    />
                    <div className="mt-2 grid grid-cols-2 gap-2">
                      <button className="rounded-xl border border-emerald-400/60 bg-emerald-500/20 px-3 py-2 text-sm font-semibold text-emerald-300" onClick={onRunInference}>
                        Run
                      </button>
                      <button className="rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-sm text-slate-200" onClick={() => inferenceFileInputRef.current?.click()}>
                        Upload JSON
                      </button>
                    </div>
                    <input
                      ref={inferenceFileInputRef}
                      type="file"
                      accept=".json,application/json"
                      className="hidden"
                      onChange={(event) => {
                        const file = event.target.files?.[0];
                        if (file) void onUploadInferenceSample(file);
                        event.currentTarget.value = '';
                      }}
                    />
                  </div>
                </div>
              </>
            ) : (
              <div className="wk-panel-card text-sm text-slate-400">Train a model first to unlock inference.</div>
            )}
          </section>
        )}

        <AnimatePresence>
          {evaluationMatrixExpanded && evaluationSnapshot?.testMetrics?.kind === 'classification' && (
            <motion.div
              className="fixed inset-0 z-[75] flex items-center justify-center bg-black/70 px-4 py-4 backdrop-blur-sm"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setEvaluationMatrixExpanded(false)}
            >
              <motion.div
                layoutId="evaluation-confusion-card"
                className="flex h-[86vh] w-full max-w-[1320px] flex-col rounded-2xl border border-white/10 bg-[#0B0E14] p-4 shadow-[0_20px_60px_rgba(0,0,0,0.45)]"
                onClick={(event) => event.stopPropagation()}
              >
                <div className="mb-3 flex items-center justify-between">
                  <div>
                    <h3 className="text-lg font-semibold">Confusion Matrix (Expanded)</h3>
                    <p className="text-xs text-slate-400">Scroll to inspect all classes. Click a cell to inspect matching samples.</p>
                  </div>
                  <button
                    className="rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-sm text-slate-200"
                    onClick={() => setEvaluationMatrixExpanded(false)}
                  >
                    Close
                  </button>
                </div>

                <div className="flex-1 overflow-auto rounded-xl border border-white/10 bg-[#0A111A] p-3">
                  <div className="min-w-max space-y-1">
                    {(evaluationSnapshot.testMetrics.confusionMatrix ?? []).map((row, rowIdx) => (
                      <div className="flex gap-1" key={`expanded_matrix_row_${rowIdx}`}>
                        {row.map((value, colIdx) => {
                          const activeCell =
                            selectedConfusionCell?.actual === rowIdx && selectedConfusionCell?.predicted === colIdx;
                          return (
                            <button
                              type="button"
                              key={`expanded_matrix_cell_${rowIdx}_${colIdx}`}
                              onClick={() => setSelectedConfusionCell({ actual: rowIdx, predicted: colIdx })}
                              className={`h-10 min-w-[42px] rounded border text-sm font-semibold ${
                                activeCell ? 'border-emerald-300 ring-1 ring-emerald-400/40' : 'border-white/10'
                              }`}
                              style={matrixCellStyle(value, evaluationConfusionMax, rowIdx, colIdx)}
                            >
                              {value}
                            </button>
                          );
                        })}
                      </div>
                    ))}
                  </div>
                </div>

                {selectedConfusionCell && (
                  <div className="mt-3 rounded-lg border border-white/10 bg-[#0E1420] p-3 text-xs text-slate-300">
                    <div>
                      Selected: actual <strong>{labelNameAt(heldOutDataset?.labelNames, selectedConfusionCell.actual)}</strong>,
                      predicted <strong>{labelNameAt(heldOutDataset?.labelNames, selectedConfusionCell.predicted)}</strong>
                    </div>
                    <div className="mt-1 text-slate-400">{confusionCellSamples.length} samples in this cell.</div>
                  </div>
                )}
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>

        {training.state.phase === 'completed' && studioTab !== 'export' && !datasetViewerOpen && !evaluationMatrixExpanded && (
          <button
            className="fixed bottom-6 right-6 z-40 rounded-full border border-emerald-400/70 bg-emerald-500 px-5 py-3 text-sm font-bold text-[#062F25] shadow-[0_0_22px_rgba(16,185,129,0.48)] transition hover:brightness-105"
            onClick={() => setStudioTab('export')}
            title="Open export tab"
          >
            Export
          </button>
        )}

        {datasetViewerOpen && dataset && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 px-4 py-4 backdrop-blur-sm">
            <div className="h-full w-full max-w-[1400px] rounded-2xl border border-white/10 bg-[#0B0E14] p-4">
              <div className="mb-3 flex items-center justify-between">
                <div>
                  <h3 className="text-lg font-semibold">Dataset Viewer</h3>
                  <p className="text-xs text-slate-400">{dataset.source.name} - {dataset.rows.length.toLocaleString()} rows</p>
                </div>
                <button className="rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-sm text-slate-200" onClick={() => setDatasetViewerOpen(false)}>
                  Close
                </button>
              </div>

              {datasetLooksImage && (
                <div className="mb-3 flex gap-2">
                  <button
                    className={`rounded-lg border px-3 py-1 text-xs ${
                      datasetViewerMode === 'table'
                        ? 'border-emerald-400/60 bg-emerald-500/20 text-emerald-300'
                        : 'border-white/10 bg-white/5 text-slate-300'
                    }`}
                    onClick={() => setDatasetViewerMode('table')}
                  >
                    Table
                  </button>
                  <button
                    className={`rounded-lg border px-3 py-1 text-xs ${
                      datasetViewerMode === 'images'
                        ? 'border-emerald-400/60 bg-emerald-500/20 text-emerald-300'
                        : 'border-white/10 bg-white/5 text-slate-300'
                    }`}
                    onClick={() => setDatasetViewerMode('images')}
                  >
                    Images
                  </button>
                </div>
              )}

              {datasetViewerMode === 'images' && datasetLooksImage ? (
                <div className="grid max-h-[80vh] grid-cols-2 gap-3 overflow-auto pr-1 sm:grid-cols-3 lg:grid-cols-5 xl:grid-cols-6">
                  {dataset.rows.slice(0, 240).map((row, index) => {
                    const preview = renderPixelRowPreview(row);
                    return (
                      <div key={`gallery_${index}`} className="rounded-xl border border-white/10 bg-[#111723] p-2">
                        <div className="mb-2 aspect-square overflow-hidden rounded-lg border border-white/10" style={imagePlaceholderStyle(row)}>
                          {preview ? (
                            <img src={preview} alt={`dataset_${index + 1}`} className="h-full w-full object-cover" />
                          ) : (
                            <div className="flex h-full items-center justify-center text-xs text-slate-500">No preview</div>
                          )}
                        </div>
                        <div className="truncate text-xs text-slate-400">{String(row?.image_name ?? row?.image_path ?? `sample_${index + 1}`)}</div>
                      </div>
                    );
                  })}
                </div>
              ) : (
                <>
                  <div className="mb-2 flex items-center justify-between text-xs text-slate-400">
                    <span>Page {datasetViewerPage + 1} / {viewerPageCount}</span>
                    <div className="flex gap-2">
                      <button className="rounded-lg border border-white/10 bg-white/5 px-2 py-1" disabled={datasetViewerPage === 0} onClick={() => setDatasetViewerPage((prev) => Math.max(0, prev - 1))}>
                        Prev
                      </button>
                      <button className="rounded-lg border border-white/10 bg-white/5 px-2 py-1" disabled={datasetViewerPage >= viewerPageCount - 1} onClick={() => setDatasetViewerPage((prev) => Math.min(viewerPageCount - 1, prev + 1))}>
                        Next
                      </button>
                    </div>
                  </div>
                  <DataTable rows={viewerRows} columns={columns} className="max-h-[75vh]" />
                </>
              )}
            </div>
          </div>
        )}

        {false && (
          <>
            {/*
        <header className="wk-header">
          <div className="wk-brand-block">
            <h1>Browser-First AI Platform</h1>
            <p>Local-first ML Studio</p>
          </div>
          <div className="wk-header-nav">
            <Stepper step={step} onSelectStep={setStep} />
          </div>
          <div className="wk-header-right">
            <span className="wk-capability">
              {caps?.tierName ?? 'Detecting hardware...'}
            </span>
            <span className="wk-capability">
              Hardware: {caps?.gpuName ?? 'Detecting...'}
              {caps?.maxMemoryMB ? ` (${Math.round(caps?.maxMemoryMB ?? 0)}MB)` : ''}
            </span>
            <span className="wk-capability wk-privacy-badge">Privacy: Data stays on device</span>
            <button className="btn btn-secondary btn-sm" onClick={handleStartNewRun}>
              New Training Run
            </button>
            {runId && (
              <button className="wk-run-id" onClick={copyRunIdToClipboard} title="Copy run id">
                <span>Run ID: {runId}</span>
                <span>Copy</span>
              </button>
            )}
          </div>
        </header>

        <section className="wk-panel">
          {step === 'dataset' && (
            <div className="wk-dataset-screen">
              <h2>Add Your Data</h2>
              <p className="wk-subtitle">Choose where your training data comes from.</p>

              <div className="wk-source-options">
                <button
                  className={`wk-option-card ${activeSource === 'upload' ? 'active' : ''}`}
                  onClick={() => setActiveSource('upload')}
                >
                  <strong>Upload File</strong>
                  <span>CSV, JSON, or Images ZIP</span>
                </button>
                <button
                  className={`wk-option-card ${activeSource === 'kaggle' ? 'active' : ''}`}
                  onClick={() => setActiveSource('kaggle')}
                >
                  <strong>Import from Kaggle</strong>
                  <span>Search datasets and preview before import</span>
                </button>
                <button
                  className={`wk-option-card ${activeSource === 'external' ? 'active' : ''}`}
                  onClick={() => setActiveSource('external')}
                >
                  <strong>Connect Data Source</strong>
                  <span>Future connector layer (warehouse, APIs)</span>
                </button>
              </div>

              {activeSource === 'upload' && (
                <div>
                  <div
                    className={`wk-dropzone ${dragActive ? 'dragging' : ''}`}
                    onDragOver={(event) => {
                      event.preventDefault();
                      if (datasetLoadBusy || kaggleLoading) return;
                      setDragActive(true);
                    }}
                    onDragLeave={() => setDragActive(false)}
                    onDrop={onDrop}
                    onClick={() => {
                      if (datasetLoadBusy || kaggleLoading) return;
                      fileInputRef.current?.click();
                    }}
                  >
                    <p>Drag and drop your dataset here</p>
                    <span>Supported: CSV, JSON, ZIP (image folders)</span>
                    <button className="btn btn-primary" type="button">
                      {datasetLoadBusy || kaggleLoading ? 'Importing...' : 'Choose File'}
                    </button>
                  </div>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".csv,.json,.zip"
                    style={{ display: 'none' }}
                    disabled={datasetLoadBusy || kaggleLoading}
                    onChange={(event) => {
                      if (datasetLoadBusy || kaggleLoading) return;
                      const file = event.target.files?.[0];
                      if (file) void handleUpload(file);
                    }}
                  />
                  <div className="wk-supported-types">
                    <span>CSV</span>
                    <span>JSON</span>
                    <span>Images ZIP</span>
                  </div>
                </div>
              )}

              {activeSource === 'kaggle' && (
                <div className="wk-kaggle-box">
                  <div className="wk-panel-block">
                    <div className="wk-block-header">
                      <h3>Kaggle Authentication</h3>
                      <small>OAuth token flow (recommended) or legacy key</small>
                    </div>
                    <div className="wk-model-cards">
                      <button
                        className={`wk-model-card ${kaggleAuthMode === 'oauth_token' ? 'active' : ''}`}
                        onClick={() => {
                          setKaggleAuthMode('oauth_token');
                          setKaggleConnected(false);
                          setKaggleConnectionInfo(null);
                          setKaggleError(null);
                        }}
                      >
                        <h3>OAuth Token</h3>
                        <small>Official Kaggle API token (KAGGLE_API_TOKEN / KGAT...).</small>
                      </button>
                      <button
                        className={`wk-model-card ${kaggleAuthMode === 'legacy_key' ? 'active' : ''}`}
                        onClick={() => {
                          setKaggleAuthMode('legacy_key');
                          setKaggleConnected(false);
                          setKaggleConnectionInfo(null);
                          setKaggleError(null);
                        }}
                      >
                        <h3>Legacy API Key</h3>
                        <small>Username + kaggle.json key pair compatibility mode.</small>
                      </button>
                    </div>

                    {kaggleAuthMode === 'oauth_token' ? (
                      <div style={{ marginTop: 10 }}>
                        <div className="form-group">
                          <label className="label">Kaggle OAuth API Token</label>
                          <input
                            className="input"
                            value={kaggleApiToken}
                            onChange={(event) => {
                              setKaggleApiToken(event.target.value);
                              setKaggleConnected(false);
                              setKaggleConnectionInfo(null);
                              setKaggleError(null);
                            }}
                            type="password"
                            placeholder="KGAT..."
                          />
                        </div>
                        <div className="wk-inline-actions">
                          <button className="btn btn-secondary" onClick={() => void openKaggleOAuthStart()}>
                            Open Kaggle Authorization
                          </button>
                          <button
                            className="btn btn-primary"
                            onClick={() => void handleKaggleOAuthConnect()}
                            disabled={kaggleLoading || !kaggleApiToken.trim()}
                          >
                            {kaggleLoading ? 'Connecting...' : 'Connect OAuth'}
                          </button>
                        </div>
                        {kaggleConnectionInfo && (
                          <p className={kaggleConnected ? 'wk-success' : 'wk-hint'}>{kaggleConnectionInfo}</p>
                        )}
                      </div>
                    ) : (
                      <div className="wk-row-grid" style={{ marginTop: 10 }}>
                        <div className="form-group">
                          <label className="label">Kaggle Username</label>
                          <input
                            className="input"
                            value={kaggleUsername}
                            onChange={(event) => setKaggleUsername(event.target.value)}
                            placeholder="your_kaggle_username"
                          />
                        </div>
                        <div className="form-group">
                          <label className="label">Kaggle API Key</label>
                          <input
                            className="input"
                            value={kaggleKey}
                            onChange={(event) => setKaggleKey(event.target.value)}
                            type="password"
                            placeholder="kaggle_api_key"
                          />
                        </div>
                      </div>
                    )}
                  </div>
                  <div className="wk-kaggle-search-row">
                    <input
                      className="input"
                      value={kaggleQuery}
                      onChange={(event) => setKaggleQuery(event.target.value)}
                      placeholder="Search Kaggle datasets or paste dataset URL"
                    />
                    <button className="btn btn-primary" onClick={handleKaggleSearch} disabled={kaggleLoading}>
                      {kaggleLoading ? 'Searching...' : 'Search'}
                    </button>
                  </div>

                  {kaggleResults.length > 0 && (
                    <div className="wk-kaggle-results">
                      {kaggleResults.map((result) => (
                        <button
                          key={result.ref}
                          className={`wk-kaggle-result ${selectedKaggleRef === result.ref ? 'active' : ''}`}
                          onClick={() => void handleKaggleRefSelect(result.ref)}
                        >
                          <strong>{result.title || result.ref}</strong>
                          <span>{result.subtitle || result.ref}</span>
                          <small>
                            {bytesToHuman(result.totalBytes)} · {result.downloadCount ?? 0} downloads
                          </small>
                        </button>
                      ))}
                    </div>
                  )}

                  {selectedKaggleRef && (
                    <div className="wk-kaggle-files">
                      <label className="label">Dataset files</label>
                      <select
                        className="select"
                        value={selectedKaggleFile}
                        onChange={(event) => setSelectedKaggleFile(event.target.value)}
                      >
                        <option value="">Use entire dataset bundle</option>
                        {kaggleFiles.map((file) => (
                          <option key={file.name} value={file.name}>
                            {file.name} ({bytesToHuman(file.totalBytes)})
                          </option>
                        ))}
                      </select>
                      <button className="btn btn-primary" type="button" onClick={() => void handleKaggleUseDataset()} disabled={kaggleLoading}>
                        {kaggleLoading ? 'Importing…' : 'Use dataset'}
                      </button>
                      {kaggleInlineStatus && <p className="wk-hint">{kaggleInlineStatus}</p>}
                      {kaggleUseCompleteMessage && !kaggleInlineStatus && (
                        <p className="wk-success">{kaggleUseCompleteMessage}</p>
                      )}
                    </div>
                  )}
                </div>
              )}

              {activeSource === 'external' && (
                <div className="wk-coming-soon">
                  <h4>Connect Data Source</h4>
                  <p>Use File System Access API to connect a local folder with CSV/JSON or image subfolders.</p>
                  <div className="wk-inline-actions">
                    <button
                      className="btn btn-primary"
                      onClick={() => void handleConnectDirectory()}
                      disabled={datasetLoadBusy || kaggleLoading}
                    >
                      {datasetLoadBusy || kaggleLoading ? 'Connecting...' : 'Select Directory'}
                    </button>
                  </div>
                </div>
              )}

              {datasetLoadProgress && (
                <p className={datasetLoadBusy || kaggleLoading ? 'wk-hint' : 'wk-success'}>{datasetLoadProgress}</p>
              )}
              {(datasetLoadError || kaggleError) && <p className="wk-error">{datasetLoadError || kaggleError}</p>}
            </div>
          )}

          {step === 'preprocess' && (
            dataset ? (
            <div className="wk-preprocess-screen">
              <h2>Understand Your Data</h2>
              <p className="wk-subtitle">Preview columns, inspect automatic insights, and apply one-click fixes.</p>

              <div className="wk-preprocess-grid">
                <div className="wk-panel-block">
                  <div className="wk-block-header">
                    <h3>Data Preview Table</h3>
                    <small>First {Math.min(50, previewRows.length)} rows</small>
                  </div>
                  <DataTable
                    rows={previewRows}
                    columns={columns}
                    targetColumn={preprocessConfig.targetColumn}
                    onHeaderClick={(column) => {
                      setPreprocessConfig((prev) => ({ ...prev, targetColumn: column }));
                    }}
                  />
                </div>

                <div className="wk-side-stack">
                  <div className="wk-panel-block">
                    <div className="wk-block-header">
                      <h3>Auto Insights</h3>
                    </div>
                    {insights ? (
                      <div className="wk-insight-list">
                        <div>Learning mode: {insights.learningMode}</div>
                        <div>Task: {insights.taskType}</div>
                        <div>Detected target: {insights.detectedTarget || 'not found'}</div>
                        <div>Data shape: {insights.modality}</div>
                        <div>
                          Dataset size: {insights.datasetSize} ({insights.rowCount.toLocaleString()} rows, {insights.featureCount} columns)
                        </div>
                        <div>Missing values in: {insights.missingColumns.length} columns</div>
                        <div>Categorical columns: {insights.categoricalColumns.length}</div>
                        {insights.recommendedAlgorithm && (
                          <div>
                            Recommended algorithm:{' '}
                            <strong>
                              {insights.suggestedAlgorithms.find((item) => item.id === insights.recommendedAlgorithm)?.label ??
                                insights.recommendedAlgorithm}
                            </strong>
                          </div>
                        )}
                        <div className="wk-hint">{insights.recommendationReason}</div>
                        {insights.warnings.map((warning) => (
                          <div key={warning} className="wk-warning">
                            {warning}
                          </div>
                        ))}
                      </div>
                    ) : (
                      <p className="wk-hint">Insights unavailable.</p>
                    )}
                  </div>

                  <div className="wk-panel-block">
                    <div className="wk-block-header">
                      <h3>Suggested Fixes</h3>
                    </div>
                    <div className="wk-fix-buttons">
                      <button
                        className={`wk-fix-btn ${preprocessConfig.fixMissingValues ? 'active' : ''}`}
                        onClick={() =>
                          setPreprocessConfig((prev) => ({ ...prev, fixMissingValues: !prev.fixMissingValues }))
                        }
                      >
                        Fix Missing Values
                      </button>
                      <button
                        className={`wk-fix-btn ${preprocessConfig.encodeCategories ? 'active' : ''}`}
                        onClick={() =>
                          setPreprocessConfig((prev) => ({ ...prev, encodeCategories: !prev.encodeCategories }))
                        }
                      >
                        Encode Categories
                      </button>
                      <button
                        className={`wk-fix-btn ${preprocessConfig.normalizeData ? 'active' : ''}`}
                        onClick={() =>
                          setPreprocessConfig((prev) => ({ ...prev, normalizeData: !prev.normalizeData }))
                        }
                      >
                        Normalize Data
                      </button>
                      {insights?.modality === 'image' && (
                        <button
                          className={`wk-fix-btn ${preprocessConfig.augmentImageData ? 'active' : ''}`}
                          onClick={() =>
                            setPreprocessConfig((prev) => ({
                              ...prev,
                              augmentImageData: !prev.augmentImageData,
                              imageAugmentationFactor: prev.augmentImageData
                                ? 1
                                : prev.imageAugmentationFactor === 3
                                  ? 3
                                  : 2,
                            }))
                          }
                        >
                          Augment Image Data
                        </button>
                      )}
                    </div>
                    {insights?.modality === 'image' && preprocessConfig.augmentImageData && (
                      <div className="wk-row-grid" style={{ marginTop: 10 }}>
                        <div className="form-group">
                          <label className="label">Augmentation Factor</label>
                          <select
                            className="select"
                            value={preprocessConfig.imageAugmentationFactor}
                            onChange={(event) =>
                              setPreprocessConfig((prev) => ({
                                ...prev,
                                imageAugmentationFactor: Number(event.target.value) as 1 | 2 | 3,
                              }))
                            }
                          >
                            <option value={1}>1x (disabled)</option>
                            <option value={2}>2x dataset size</option>
                            <option value={3}>3x dataset size</option>
                          </select>
                        </div>
                        <div className="form-group">
                          <label className="label">Augmentation Noise</label>
                          <input
                            className="input"
                            type="number"
                            min={0}
                            max={0.5}
                            step={0.01}
                            value={preprocessConfig.imageAugmentationNoise}
                            onChange={(event) =>
                              setPreprocessConfig((prev) => ({
                                ...prev,
                                imageAugmentationNoise: Math.max(0, Math.min(0.5, Number(event.target.value) || 0)),
                              }))
                            }
                          />
                        </div>
                      </div>
                    )}
                    {preprocessNotice && <p className="wk-hint">{preprocessNotice}</p>}

                    <div className="wk-split-panel">
                      <div className="wk-label-small">Training / Testing Split</div>
                      <div className="wk-split-options">
                        <button
                          className={`wk-chip ${splitChoice === '80_20' ? 'active' : ''}`}
                          onClick={() => setSplitChoice('80_20')}
                        >
                          80% / 20%
                        </button>
                        <button
                          className={`wk-chip ${splitChoice === '90_10' ? 'active' : ''}`}
                          onClick={() => setSplitChoice('90_10')}
                        >
                          90% / 10%
                        </button>
                      </div>
                      <p className="wk-hint" style={{ marginTop: 6 }}>
                        Held-out test data is evaluated in Step 7 to detect overfitting/underfitting.
                      </p>
                    </div>

                    <div className="wk-inline-actions">
                      <button className="btn btn-secondary" onClick={() => void runPreprocess(false)} disabled={preprocessBusy}>
                        {preprocessBusy
                          ? `Previewing${preprocessProgress?.percent ? ` (${preprocessProgress.percent}%)` : '...'}`
                          : 'Preview before/after'}
                      </button>
                    </div>
                    {preprocessBusy && preprocessProgress?.message && (
                      <p className="wk-hint">{preprocessProgress.message}</p>
                    )}

                    {processedDataset && (
                      <div className="wk-before-after">
                        <div>Missing values before: {processedDataset.stats.beforeMissing}</div>
                        <div>Missing values after: {processedDataset.stats.afterMissing}</div>
                        <div>Encoded columns: {processedDataset.stats.encodedColumns.length}</div>
                        <div>Normalized columns: {processedDataset.stats.normalizedColumns.length}</div>
                      </div>
                    )}

                    <button
                      className="wk-advanced-toggle"
                      onClick={() => setShowAdvancedPreprocess((prev) => !prev)}
                    >
                      {showAdvancedPreprocess ? 'Hide Advanced' : 'Advanced: manual column editing'}
                    </button>
                    {showAdvancedPreprocess && (
                      <div className="wk-advanced-block">
                        <div className="wk-label-small">Target column</div>
                        <select
                          className="select"
                          value={preprocessConfig.targetColumn ?? ''}
                          onChange={(event) =>
                            setPreprocessConfig((prev) => ({ ...prev, targetColumn: event.target.value || null }))
                          }
                        >
                          <option value="">Select target</option>
                          {columns.map((column) => (
                            <option key={column} value={column}>
                              {column}
                            </option>
                          ))}
                        </select>

                        <div className="wk-label-small">Drop columns</div>
                        <div className="wk-chip-wrap">
                          {profiles.map((profile) => {
                            const dropped = preprocessConfig.droppedColumns.includes(profile.name);
                            return (
                              <button
                                key={profile.name}
                                className={`wk-chip ${dropped ? 'dropped' : ''}`}
                                onClick={() =>
                                  setPreprocessConfig((prev) => ({
                                    ...prev,
                                    droppedColumns: dropped
                                      ? prev.droppedColumns.filter((column) => column !== profile.name)
                                      : [...prev.droppedColumns, profile.name],
                                  }))
                                }
                              >
                                {profile.name}
                              </button>
                            );
                          })}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {preprocessError && <p className="wk-error">{preprocessError}</p>}

              <div className="wk-screen-footer">
                <button className="btn btn-secondary" onClick={() => setStep('dataset')}>
                  Back
                </button>
                <button className="btn btn-primary" onClick={() => void runPreprocess(true)} disabled={preprocessBusy}>
                  {preprocessBusy ? 'Processing...' : 'Continue'}
                </button>
              </div>
            </div>
            ) : (
              <StepPlaceholder
                title="No dataset loaded yet"
                message="You can still navigate tabs freely. Load data in 'Add Your Data' whenever you are ready."
                actionLabel="Go to Dataset Selection"
                onAction={() => setStep('dataset')}
              />
            )
          )}

          {step === 'model' && (
            <div className="wk-model-screen">
              <h2>Choose Your Model</h2>
              <p className="wk-subtitle">Auto-suggested from target type, data modality, and dataset size. You can override.</p>

              {insights ? (
                <>
                  <div className="wk-panel-block">
                    <div className="wk-block-header">
                      <h3>Why this is recommended</h3>
                      <small>Dataset-aware recommendation engine</small>
                    </div>
                    <p className="wk-hint" style={{ marginTop: 0 }}>
                      {insights.recommendationReason}
                    </p>
                    <div className="wk-inline-actions">
                      <span className="wk-chip">Mode: {insights.learningMode}</span>
                      <span className="wk-chip">Task: {insights.taskType}</span>
                      <span className="wk-chip">Shape: {insights.modality}</span>
                      <span className="wk-chip">Size: {insights.datasetSize}</span>
                      {recommendedAlgorithm &&
                        recommendedSuggestion &&
                        isSelectableSuggestion(recommendedSuggestion) &&
                        effectiveModelChoice !== recommendedAlgorithm && (
                        <button
                          className="btn btn-secondary"
                          onClick={() => {
                            setModelChoice(recommendedAlgorithm);
                            setModelChoiceTouched(false);
                            setPreferences((prev) =>
                              insights ? applyRecommendedPreferencePreset(prev, recommendedAlgorithm, insights) : prev
                            );
                          }}
                        >
                          Use Recommended
                        </button>
                      )}
                    </div>
                  </div>

                  <div className="wk-model-section-grid">
                    <div className="wk-panel-block">
                      <div className="wk-block-header">
                        <h3>Compatible Algorithms</h3>
                        <small>{taskMode.toUpperCase()} mode</small>
                      </div>
                      <p className="wk-hint" style={{ marginTop: 0 }}>
                        {taskModeDescription(taskMode)}
                      </p>
                      <div className="wk-model-cards wk-model-cards-dense">
                        {compatibleAlgorithms.length ? (
                          compatibleAlgorithms.map((algorithm) => {
                            const active = effectiveModelChoice === algorithm.id;
                            const selectable = isSelectableSuggestion(algorithm);
                            const reason = suggestedReasonById.get(algorithm.id) ?? algorithm.reason;
                            const uiMeta = ALGORITHM_UI_META[algorithm.id];
                            return (
                              <button
                                key={algorithm.id}
                                className={`wk-model-card wk-model-card-dense ${active ? 'active' : ''} ${!selectable ? 'wk-model-card-disabled' : ''}`}
                                disabled={!selectable}
                                onClick={() => {
                                  setModelChoice(algorithm.id);
                                  setModelChoiceTouched(true);
                                  setPreferences((prev) =>
                                    insights ? applyRecommendedPreferencePreset(prev, algorithm.id, insights) : prev
                                  );
                                }}
                              >
                                <div className="wk-model-card-head">
                                  <h3>{displayAlgorithmLabel(algorithm.id, taskMode)}</h3>
                                  {insights.recommendedAlgorithm === algorithm.id && <span className="wk-pill">Recommended</span>}
                                </div>
                                <small>{reason}</small>
                                <div className="wk-chip-wrap">
                                  <span className="wk-chip">{uiMeta.group}</span>
                                  <span className="wk-chip">{algorithm.speed}</span>
                                  <span className="wk-chip">Interp: {uiMeta.interpretability}</span>
                                  <span className="wk-chip">Memory: {uiMeta.memory}</span>
                                  <span className={`wk-chip wk-support-${algorithm.runtimeSupport}`}>{algorithm.runtimeSupport}</span>
                                </div>
                              </button>
                            );
                          })
                        ) : (
                          <div className="wk-empty-mini">No compatible algorithms for the current task mode.</div>
                        )}
                      </div>
                    </div>

                    <div className="wk-panel-block">
                      <div className="wk-block-header">
                        <h3>Other Algorithms</h3>
                        <small>Visible but disabled</small>
                      </div>
                      <div className="wk-model-cards wk-model-cards-dense">
                        {incompatibleAlgorithms.length ? (
                          incompatibleAlgorithms.map((algorithm) => {
                            const compatibility = compatibilityById.get(algorithm.id);
                            const uiMeta = ALGORITHM_UI_META[algorithm.id];
                            return (
                              <button
                                key={algorithm.id}
                                className="wk-model-card wk-model-card-dense wk-model-card-disabled"
                                disabled
                                title={compatibility?.reason}
                              >
                                <div className="wk-model-card-head">
                                  <h3>{displayAlgorithmLabel(algorithm.id, taskMode)}</h3>
                                </div>
                                <small>{compatibility?.reason ?? 'Not compatible with current task mode.'}</small>
                                <div className="wk-chip-wrap">
                                  <span className="wk-chip">{uiMeta.group}</span>
                                  <span className="wk-chip">{algorithm.speed}</span>
                                  <span className="wk-chip">Interp: {uiMeta.interpretability}</span>
                                  <span className="wk-chip">Memory: {uiMeta.memory}</span>
                                  <span className={`wk-chip wk-support-${algorithm.runtimeSupport}`}>{algorithm.runtimeSupport}</span>
                                </div>
                              </button>
                            );
                          })
                        ) : (
                          <div className="wk-empty-mini">All algorithms are currently compatible.</div>
                        )}
                      </div>
                    </div>
                  </div>

                  <button className="wk-advanced-toggle" onClick={() => setShowAdvancedModelConfig((prev) => !prev)}>
                    {showAdvancedModelConfig ? 'Hide algorithm advanced config' : 'Optional advanced algorithm config'}
                  </button>
                  {showAdvancedModelConfig && (
                    <div className="wk-panel-block" style={{ marginTop: 12 }}>
                      <div className="wk-block-header">
                        <h3>Algorithm-Specific Controls</h3>
                        <small>Optional expert settings (hidden by default)</small>
                      </div>

                      {(
                        effectiveModelChoice === 'knn' ||
                        effectiveModelChoice === 'svm' ||
                        effectiveModelChoice === 'decision_tree' ||
                        effectiveModelChoice === 'random_forest' ||
                        effectiveModelChoice === 'kmeans' ||
                        effectiveModelChoice === 'dbscan'
                      ) && (
                        <div className="wk-row-grid">
                          {(effectiveModelChoice === 'knn') && (
                            <>
                              <div className="form-group">
                                <label className="label">KNN Neighbors</label>
                                <input
                                  className="input"
                                  type="number"
                                  min={1}
                                  max={100}
                                  value={preferences.algorithm.knnNeighbors}
                                  onChange={(event) =>
                                    setPreferences((prev) => ({
                                      ...prev,
                                      algorithm: {
                                        ...prev.algorithm,
                                        knnNeighbors: Math.max(1, Number(event.target.value) || 1),
                                      },
                                    }))
                                  }
                                />
                              </div>
                              <div className="form-group">
                                <label className="label">KNN Distance Metric</label>
                                <select
                                  className="select"
                                  value={preferences.algorithm.knnDistanceMetric}
                                  onChange={(event) =>
                                    setPreferences((prev) => ({
                                      ...prev,
                                      algorithm: {
                                        ...prev.algorithm,
                                        knnDistanceMetric: event.target.value as 'euclidean' | 'manhattan' | 'cosine',
                                      },
                                    }))
                                  }
                                >
                                  <option value="euclidean">Euclidean</option>
                                  <option value="manhattan">Manhattan</option>
                                  <option value="cosine">Cosine</option>
                                </select>
                              </div>
                            </>
                          )}

                          {(effectiveModelChoice === 'svm') && (
                            <div className="form-group">
                              <label className="label">SVM Kernel</label>
                              <select
                                className="select"
                                value={preferences.algorithm.svmKernel}
                                onChange={(event) =>
                                  setPreferences((prev) => ({
                                    ...prev,
                                    algorithm: {
                                      ...prev.algorithm,
                                      svmKernel: event.target.value as 'rbf' | 'linear' | 'poly',
                                    },
                                  }))
                                }
                              >
                                <option value="rbf">RBF</option>
                                <option value="linear">Linear</option>
                                <option value="poly">Polynomial</option>
                              </select>
                            </div>
                          )}

                          {(effectiveModelChoice === 'decision_tree' || effectiveModelChoice === 'random_forest') && (
                            <div className="form-group">
                              <label className="label">Tree Depth Hint</label>
                              <input
                                className="input"
                                type="number"
                                min={2}
                                max={64}
                                value={treeDepthHint}
                                onChange={(event) => setTreeDepthHint(Number(event.target.value) || 8)}
                              />
                            </div>
                          )}

                          {(effectiveModelChoice === 'kmeans') && (
                            <div className="form-group">
                              <label className="label">K-Means Clusters</label>
                              <input
                                className="input"
                                type="number"
                                min={2}
                                max={128}
                                value={preferences.algorithm.kmeansClusters}
                                onChange={(event) =>
                                  setPreferences((prev) => ({
                                    ...prev,
                                    algorithm: {
                                      ...prev.algorithm,
                                      kmeansClusters: Math.max(2, Number(event.target.value) || 2),
                                    },
                                  }))
                                }
                              />
                            </div>
                          )}

                          {(effectiveModelChoice === 'dbscan') && (
                            <>
                              <div className="form-group">
                                <label className="label">DBSCAN Epsilon</label>
                                <input
                                  className="input"
                                  type="number"
                                  min={0.01}
                                  max={10}
                                  step={0.01}
                                  value={preferences.algorithm.dbscanEpsilon}
                                  onChange={(event) =>
                                    setPreferences((prev) => ({
                                      ...prev,
                                      algorithm: {
                                        ...prev.algorithm,
                                        dbscanEpsilon: Math.max(0.01, Number(event.target.value) || 0.01),
                                      },
                                    }))
                                  }
                                />
                              </div>
                              <div className="form-group">
                                <label className="label">DBSCAN Min Samples</label>
                                <input
                                  className="input"
                                  type="number"
                                  min={2}
                                  max={128}
                                  value={preferences.algorithm.dbscanMinSamples}
                                  onChange={(event) =>
                                    setPreferences((prev) => ({
                                      ...prev,
                                      algorithm: {
                                        ...prev.algorithm,
                                        dbscanMinSamples: Math.max(2, Number(event.target.value) || 2),
                                      },
                                    }))
                                  }
                                />
                              </div>
                            </>
                          )}
                        </div>
                      )}

                      {selectedAlgorithmInfo && usesNeuralTuning(selectedAlgorithmInfo.id) && (
                        <p className="wk-hint">
                          Neural-compatible models share advanced tuning controls. Fine-tune epochs, learning-rate, and
                          batch-size in the next <strong>Training Preferences</strong> step.
                        </p>
                      )}
                      {selectedAlgorithmInfo &&
                        (selectedAlgorithmInfo.id === 'rnn' ||
                          selectedAlgorithmInfo.id === 'lstm' ||
                          selectedAlgorithmInfo.id === 'gru') && (
                          <p className="wk-hint">
                            Recurrent models run with native sequence logic in the worker. Hidden-size and optimization
                            preferences are shared with the next <strong>Training Preferences</strong> step.
                          </p>
                        )}
                    </div>
                  )}
                </>
              ) : (
                <StepPlaceholder
                  title="Model recommendations are waiting for data analysis"
                  message="Upload and preprocess data first so the engine can detect task type, modality, and suggest algorithms."
                  actionLabel="Go to Understand Your Data"
                  onAction={() => setStep('preprocess')}
                />
              )}

              <div className="wk-screen-footer">
                <button className="btn btn-secondary" onClick={() => setStep('preprocess')}>
                  Back
                </button>
                <button className="btn btn-primary" onClick={() => setStep('setup')}>
                  Train Model
                </button>
              </div>
            </div>
          )}

          {step === 'setup' && (
            <div className="wk-setup-screen">
              <h2>Training Preferences</h2>
              <p className="wk-subtitle">Adjust speed vs quality. Advanced settings are optional.</p>
              <p className="wk-hint">
                Training will run with <strong>{resolvedTrainingChoice}</strong>. If a prior checkpoint exists for this run,
                you will be prompted to resume or start fresh.
              </p>

              <div className="wk-panel-block">
                <label className="label">Speed &lt;-&gt; Accuracy</label>
                <input
                  type="range"
                  min={0}
                  max={100}
                  value={preferences.speedVsAccuracy}
                  onChange={(event) =>
                    setPreferences((prev) => ({ ...prev, speedVsAccuracy: Number(event.target.value) }))
                  }
                />
                <div className="wk-range-labels">
                  <span>Speed</span>
                  <span>{preferences.speedVsAccuracy}</span>
                  <span>Accuracy</span>
                </div>
              </div>

              <div className="wk-row-grid">
                <label className="wk-toggle">
                  <input
                    type="checkbox"
                    checked={preferences.useMoreCompute}
                    onChange={(event) =>
                      setPreferences((prev) => ({ ...prev, useMoreCompute: event.target.checked }))
                    }
                  />
                  Use more compute
                </label>
                <label className="wk-toggle">
                  <input
                    type="checkbox"
                    checked={preferences.optimizeForSmallerModel}
                    onChange={(event) =>
                      setPreferences((prev) => ({ ...prev, optimizeForSmallerModel: event.target.checked }))
                    }
                  />
                  Optimize for smaller model
                </label>
              </div>

              <div className="wk-panel-block" style={{ marginTop: 12 }}>
                <div className="wk-block-header">
                  <h3>Hybrid Execution Runtime</h3>
                  <small>Always-on orchestration path for best performance routing</small>
                </div>
                <div className="wk-runtime-stack">
                  <div className="wk-runtime-layer">
                    <strong>UI Thread</strong>
                    <span>Coordinates UX and model workflow</span>
                  </div>
                  <div className="wk-runtime-layer">
                    <strong>Web Worker</strong>
                    <span>Runs training orchestration off the main thread</span>
                  </div>
                  <div className="wk-runtime-layer">
                    <strong>WASM Runtime</strong>
                    <span>Executes compute functions with policy guardrails</span>
                  </div>
                  <div className="wk-runtime-layer wk-runtime-layer-base">
                    <strong>WebGPU</strong>
                    <span>
                      {caps?.webgpu ? 'GPU acceleration active' : 'GPU not available, fallback path active'}
                    </span>
                    <span className={`wk-runtime-dot ${caps?.webgpu ? 'active' : 'inactive'}`} />
                  </div>
                </div>
                <p className="wk-hint">
                  WebGPU is preferred when available. If not, runtime falls back through WebGL2/WASM SIMD/CPU while
                  preserving Worker + WASM orchestration.
                </p>
              </div>

              <button className="wk-advanced-toggle" onClick={() => setShowAdvancedTraining((prev) => !prev)}>
                {showAdvancedTraining ? 'Hide Advanced' : 'Advanced training controls'}
              </button>
              {showAdvancedTraining && (
                <div style={{ marginTop: 12 }}>
                  <div className="wk-row-grid">
                    <div className="form-group">
                      <label className="label">Epochs</label>
                      <input
                        className="input"
                        type="number"
                        min={1}
                        max={200}
                        value={preferences.epochs}
                        onChange={(event) =>
                          setPreferences((prev) => ({ ...prev, epochs: Number(event.target.value) || 1 }))
                        }
                      />
                    </div>
                    <div className="form-group">
                      <label className="label">Learning Rate</label>
                      <input
                        className="input"
                        type="number"
                        step={0.0001}
                        min={0.00001}
                        max={1}
                        value={preferences.learningRate}
                        onChange={(event) =>
                          setPreferences((prev) => ({ ...prev, learningRate: Number(event.target.value) || 0.001 }))
                        }
                      />
                    </div>
                    <div className="form-group">
                      <label className="label">Batch Size</label>
                      <input
                        className="input"
                        type="number"
                        min={8}
                        max={1024}
                        value={preferences.batchSize}
                        onChange={(event) =>
                          setPreferences((prev) => ({ ...prev, batchSize: Number(event.target.value) || 32 }))
                        }
                      />
                    </div>
                  </div>

                  {selectedAlgorithmInfo && usesNeuralTuning(selectedAlgorithmInfo.id) && (
                    <div className="wk-panel-block" style={{ marginTop: 10 }}>
                      <div className="wk-block-header">
                        <h3>Neural Network Advanced</h3>
                        <small>Edit architecture, activation, normalization, and regularization.</small>
                      </div>
                      <div className="wk-row-grid">
                        <div className="form-group">
                          <label className="label">Hidden Layers</label>
                          <input
                            className="input"
                            type="number"
                            min={1}
                            max={8}
                            value={preferences.neuralNetwork.hiddenLayers}
                            onChange={(event) => updateNeuralSettings({ hiddenLayers: Number(event.target.value) || 1 })}
                          />
                        </div>
                        <div className="form-group">
                          <label className="label">Neurons / Hidden Layer</label>
                          <input
                            className="input"
                            type="number"
                            min={8}
                            max={2048}
                            value={preferences.neuralNetwork.neuronsPerLayer}
                            onChange={(event) =>
                              updateNeuralSettings({ neuronsPerLayer: Number(event.target.value) || 64 })
                            }
                          />
                        </div>
                        <div className="form-group">
                          <label className="label">Activation</label>
                          <select
                            className="select"
                            value={preferences.neuralNetwork.activation}
                            onChange={(event) =>
                              updateNeuralSettings({
                                activation: event.target.value as TrainingPreferences['neuralNetwork']['activation'],
                              })
                            }
                          >
                            <option value="relu">ReLU</option>
                            <option value="leaky_relu">LeakyReLU</option>
                            <option value="tanh">Tanh</option>
                            <option value="sigmoid">Sigmoid</option>
                            <option value="softmax">Softmax</option>
                          </select>
                        </div>
                        <div className="form-group">
                          <label className="label">Optimizer</label>
                          <select
                            className="select"
                            value={preferences.optimizer}
                            onChange={(event) =>
                              setPreferences((prev) => ({
                                ...prev,
                                optimizer: event.target.value as TrainingPreferences['optimizer'],
                                neuralNetwork: { ...prev.neuralNetwork, optimizer: event.target.value as TrainingPreferences['optimizer'] },
                              }))
                            }
                          >
                            <option value="adamw">AdamW</option>
                            <option value="sgd_momentum">SGD + Momentum</option>
                            <option value="adam">Adam</option>
                            <option value="adamax">Adamax</option>
                          </select>
                        </div>
                        <div className="form-group">
                          <label className="label">Dropout Rate</label>
                          <input
                            className="input"
                            type="number"
                            step={0.01}
                            min={0}
                            max={0.8}
                            value={preferences.neuralNetwork.dropoutRate}
                            onChange={(event) =>
                              updateNeuralSettings({ dropoutRate: Math.max(0, Number(event.target.value) || 0) })
                            }
                          />
                        </div>
                        <div className="form-group">
                          <label className="label">Weight Decay (L2)</label>
                          <input
                            className="input"
                            type="number"
                            step={0.0001}
                            min={0}
                            max={1}
                            value={preferences.weightDecay}
                            onChange={(event) =>
                              setPreferences((prev) => ({
                                ...prev,
                                weightDecay: Math.max(0, Number(event.target.value) || 0),
                                neuralNetwork: { ...prev.neuralNetwork, weightDecay: Math.max(0, Number(event.target.value) || 0) },
                              }))
                            }
                          />
                        </div>
                      </div>
                      <div className="wk-row-grid">
                        <label className="wk-toggle">
                          <input
                            type="checkbox"
                            checked={preferences.neuralNetwork.useBatchNorm}
                            onChange={(event) => updateNeuralSettings({ useBatchNorm: event.target.checked })}
                          />
                          Enable BatchNorm between hidden layers
                        </label>
                      </div>
                    </div>
                  )}

                  <div className="wk-panel-block" style={{ marginTop: 10 }}>
                    <div className="wk-block-header">
                      <h3>WASM Function Advanced</h3>
                      <small>Template mode is safe; Eject mode enables executable policy code.</small>
                    </div>
                    <div className="wk-inline-actions">
                      <button
                        className={`btn ${wasmEditor.advancedMode === 'template' ? 'btn-primary' : 'btn-secondary'}`}
                        onClick={switchToTemplateMode}
                        disabled={wasmEditor.advancedMode === 'template'}
                      >
                        Template Mode
                      </button>
                      <button
                        className={`btn ${wasmEditor.advancedMode === 'eject' ? 'btn-primary' : 'btn-secondary'}`}
                        onClick={switchToEjectMode}
                        disabled={wasmEditor.advancedMode === 'eject'}
                      >
                        Eject to Executable Editor
                      </button>
                      {wasmEditor.advancedMode === 'eject' && (
                        <button className="btn btn-secondary" onClick={resetExecutableFromTemplate}>
                          Reset Code from Template
                        </button>
                      )}
                    </div>

                    {wasmEditor.advancedMode === 'template' ? (
                      <div className="wk-row-grid" style={{ marginTop: 10 }}>
                        <div className="form-group">
                          <label className="label">Function Name</label>
                          <input
                            className="input"
                            value={wasmTemplate.functionName}
                            onChange={(event) => updateWasmTemplate({ functionName: event.target.value || 'train_batch' })}
                          />
                        </div>
                        <div className="form-group">
                          <label className="label">Invocation Timeout (ms)</label>
                          <input
                            className="input"
                            type="number"
                            min={100}
                            max={120000}
                            value={wasmTemplate.invocationTimeoutMs}
                            onChange={(event) =>
                              updateWasmTemplate({
                                invocationTimeoutMs: Number(event.target.value) || 10000,
                              })
                            }
                          />
                        </div>
                        <div className="form-group">
                          <label className="label">Retry Count</label>
                          <input
                            className="input"
                            type="number"
                            min={0}
                            max={8}
                            value={wasmTemplate.retryCount}
                            onChange={(event) => updateWasmTemplate({ retryCount: Number(event.target.value) || 0 })}
                          />
                        </div>
                        <div className="form-group">
                          <label className="label">Memory Budget (MB)</label>
                          <input
                            className="input"
                            type="number"
                            min={64}
                            max={16384}
                            value={wasmTemplate.memoryBudgetMB}
                            onChange={(event) =>
                              updateWasmTemplate({ memoryBudgetMB: Number(event.target.value) || 512 })
                            }
                          />
                        </div>
                        <div className="form-group">
                          <label className="label">Shard Count</label>
                          <input
                            className="input"
                            type="number"
                            min={1}
                            max={32}
                            value={wasmTemplate.shardCount}
                            onChange={(event) => updateWasmTemplate({ shardCount: Number(event.target.value) || 1 })}
                          />
                        </div>
                        <div className="form-group">
                          <label className="label">Checkpoint Every N Epochs</label>
                          <input
                            className="input"
                            type="number"
                            min={1}
                            max={50}
                            value={wasmTemplate.checkpointEveryNEpochs}
                            onChange={(event) =>
                              updateWasmTemplate({
                                checkpointEveryNEpochs: Number(event.target.value) || 1,
                              })
                            }
                          />
                        </div>
                        <div className="form-group">
                          <label className="label">Gradient Clip Value (0=off)</label>
                          <input
                            className="input"
                            type="number"
                            step={0.1}
                            min={0}
                            max={1000}
                            value={wasmTemplate.gradientClipValue}
                            onChange={(event) =>
                              updateWasmTemplate({
                                gradientClipValue: Number(event.target.value) || 0,
                              })
                            }
                          />
                        </div>
                        <div className="form-group">
                          <label className="label">Cold Start Delay (ms)</label>
                          <input
                            className="input"
                            type="number"
                            min={0}
                            max={10000}
                            value={wasmTemplate.coldStartMs}
                            onChange={(event) => updateWasmTemplate({ coldStartMs: Number(event.target.value) || 0 })}
                          />
                        </div>
                      </div>
                    ) : (
                      <div className="wk-editor-wrap">
                        <p className="wk-warning">
                          Eject mode executes policy code inside the training worker. Guardrails are enforced, and
                          invalid code blocks training start.
                        </p>
                        <label className="label">Executable WASM Policy</label>
                        <textarea
                          className="input wk-textarea wk-code-editor"
                          value={wasmEditor.executableCode}
                          onChange={(event) =>
                            setPreferences((prev) => ({
                              ...prev,
                              runtime: {
                                ...prev.runtime,
                                wasmEditor: {
                                  ...prev.runtime.wasmEditor,
                                  executableCode: event.target.value,
                                },
                              },
                            }))
                          }
                        />
                        <p className="wk-hint">
                          Return a policy object (full or partial) from this code. Template defaults remain available via
                          <code> api.base</code>.
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {preprocessError && <p className="wk-error">{preprocessError}</p>}

              <div className="wk-screen-footer">
                <button className="btn btn-secondary" onClick={() => setStep('model')}>
                  Back
                </button>
                <button className="btn btn-primary" onClick={() => void startTraining()}>
                  Start Training
                </button>
              </div>
            </div>
          )}

          {step === 'training' && (
            <div className="wk-training-screen">
              <h2>Training in Progress</h2>
              <p className="wk-subtitle">{training.state.statusMessage || 'Training model...'}</p>

              <div className="wk-progress-wrap">
                <div className="wk-progress-head">
                  <span>{training.state.resolvedModel || 'model'}</span>
                  <span>{training.state.progressPercent}%</span>
                </div>
                <div className="progress-bar">
                  <div className="progress-fill" style={{ width: `${training.state.progressPercent}%` }} />
                </div>
              </div>

              <div className="wk-training-metrics">
                <div className="metric-card">
                  <div className="metric-value">{latestMetric ? `${(latestMetric.accuracy * 100).toFixed(1)}%` : '—'}</div>
                  <div className="metric-label">Accuracy</div>
                </div>
                <div className="metric-card">
                  <div className="metric-value">{latestMetric ? latestMetric.loss.toFixed(4) : '—'}</div>
                  <div className="metric-label">Loss</div>
                </div>
                <div className="metric-card">
                  <div className="metric-value">{training.state.backend || '—'}</div>
                  <div className="metric-label">Backend</div>
                </div>
              </div>

              <Curve points={training.state.curve} />

              <div className="wk-panel-block" style={{ marginTop: 12 }}>
                <div className="wk-block-header">
                  <h3>Training CLI Output</h3>
                  <small>Unified logs from all algorithm trainers</small>
                </div>
                <div className="wk-log-box">
                  {(training.state.logs.length ? training.state.logs : ['Initializing training...']).map((entry, idx) => {
                    const match = entry.match(/^\[(.+?)\]\s?(.*)$/);
                    return (
                      <div key={`${entry}_${idx}`}>
                        {match ? (
                          <>
                            <span className="wk-log-time">[{match[1]}]</span> {match[2]}
                          </>
                        ) : (
                          entry
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>

              {training.state.error && <p className="wk-error">{training.state.error}</p>}

              <div className="wk-screen-footer">
                <button className="btn btn-danger" onClick={training.stop}>
                  Stop Training
                </button>
                {training.state.phase === 'completed' && (
                  <button className="btn btn-primary" onClick={() => setStep('results')}>
                    View Results
                  </button>
                )}
              </div>
            </div>
          )}

          {step === 'results' && (
            training.state.metrics ? (
            <div className="wk-results-screen">
              <h2>Your Model is Ready</h2>
              <p className="wk-subtitle">
                {training.state.metrics.kind === 'regression'
                  ? 'Regression summary and feature insights.'
                  : 'Performance summary, confusion matrix, and feature insights.'}
              </p>

              <div className="wk-metric-grid">
                {metricCards.map((card) => (
                  <div key={card.label} className="metric-card">
                    <div className="metric-value">{card.value}</div>
                    <div className="metric-label">{card.label}</div>
                  </div>
                ))}
              </div>

              <div className="wk-results-grid">
                {training.state.metrics.kind === 'classification' && (
                  <div className="wk-panel-block">
                    <h3>Confusion Matrix</h3>
                    <div className="wk-matrix">
                      {(training.state.metrics.confusionMatrix ?? []).map((row, rowIdx) => (
                        <div className="wk-matrix-row" key={`row_${rowIdx}`}>
                          {row.map((value, colIdx) => (
                            <div
                              className={`wk-matrix-cell ${rowIdx === colIdx ? 'diag' : ''}`}
                              key={`cell_${rowIdx}_${colIdx}`}
                              style={matrixHeatStyle(value, trainingConfusionMax)}
                            >
                              {value}
                            </div>
                          ))}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {training.state.metrics.kind === 'regression' && (
                  <div className="wk-panel-block">
                    <h3>Regression Summary</h3>
                    <p className="wk-hint">
                      RMSE measures average prediction error magnitude, MAE shows absolute error, and R² indicates explained variance.
                    </p>
                    <div className="wk-before-after">
                      <div>RMSE: {(training.state.metrics.rmse ?? 0).toFixed(6)}</div>
                      <div>MAE: {(training.state.metrics.mae ?? 0).toFixed(6)}</div>
                      <div>R²: {(training.state.metrics.r2 ?? 0).toFixed(4)}</div>
                    </div>
                  </div>
                )}

                <div className="wk-panel-block">
                  <h3>Top Features</h3>
                  <ol className="wk-feature-list">
                    {training.state.metrics.featureImportance.slice(0, 10).map((feature) => (
                      <li key={feature.feature}>
                        <span>{feature.feature}</span>
                        <strong>{(feature.score * 100).toFixed(2)}%</strong>
                      </li>
                    ))}
                  </ol>
                </div>
              </div>

              <div className="wk-screen-footer">
                <button className="btn btn-secondary" onClick={() => setStep('training')}>
                  Back
                </button>
                <button className="btn btn-primary" onClick={() => setStep('export')}>
                  Open Export / Inference
                </button>
              </div>
            </div>
            ) : (
              <StepPlaceholder
                title="Results are not ready yet"
                message="Run training first, then this tab will show metrics, confusion matrix, and feature insights."
                actionLabel="Open Training"
                onAction={() => setStep('training')}
              />
            )
          )}

          {step === 'export' && (
            training.state.artifact && processedDataset ? (
            <div className="wk-export-screen">
              <h2>Export / Inference</h2>
              <p className="wk-subtitle">
                Evaluate on held-out test data, inspect prediction quality, detect fit behavior, and export model artifacts.
              </p>

              <div className="wk-panel-block">
                <div className="wk-block-header">
                  <h3>Split Selection Reminder</h3>
                  <small>Train/Test partition used for this run</small>
                </div>
                <div className="wk-inline-actions">
                  <span className="wk-chip">Training: {splitCounts?.train ?? 0} rows</span>
                  <span className="wk-chip">Testing: {splitCounts?.test ?? 0} rows</span>
                  <span className="wk-chip">Split: {splitLabelForChoice(splitUsed ?? splitChoice)}</span>
                </div>
                <div className="wk-split-options" style={{ marginTop: 10 }}>
                  <button className={`wk-chip ${splitChoice === '80_20' ? 'active' : ''}`} onClick={() => setSplitChoice('80_20')}>
                    80% / 20%
                  </button>
                  <button className={`wk-chip ${splitChoice === '90_10' ? 'active' : ''}`} onClick={() => setSplitChoice('90_10')}>
                    90% / 10%
                  </button>
                </div>
                {splitUsed && splitChoice !== splitUsed && (
                  <p className="wk-hint">Split updated. Start a new training run to apply this new partition.</p>
                )}
              </div>

              {evaluationSnapshot ? (
                <>
                  <div className="wk-panel-block" style={{ marginTop: 12 }}>
                    <div className="wk-block-header">
                      <h3>Evaluation Metrics Panel</h3>
                      <small>Held-out testing dataset metrics</small>
                    </div>
                    <div className="wk-metric-grid">
                      {evaluationSnapshot.testMetrics?.kind === 'classification' ? (
                        <>
                          <div className="metric-card">
                            <div className="metric-value">{((evaluationSnapshot.testMetrics?.accuracy ?? 0) * 100).toFixed(2)}%</div>
                            <div className="metric-label">Test Accuracy</div>
                          </div>
                          <div className="metric-card">
                            <div className="metric-value">{((evaluationSnapshot.testMetrics?.precision ?? 0) * 100).toFixed(2)}%</div>
                            <div className="metric-label">Test Precision</div>
                          </div>
                          <div className="metric-card">
                            <div className="metric-value">{((evaluationSnapshot.testMetrics?.recall ?? 0) * 100).toFixed(2)}%</div>
                            <div className="metric-label">Test Recall</div>
                          </div>
                          <div className="metric-card">
                            <div className="metric-value">{((evaluationSnapshot.testMetrics?.f1 ?? 0) * 100).toFixed(2)}%</div>
                            <div className="metric-label">Test F1</div>
                          </div>
                        </>
                      ) : (
                        <>
                          <div className="metric-card">
                            <div className="metric-value">{(evaluationSnapshot.testMetrics?.rmse ?? 0).toFixed(6)}</div>
                            <div className="metric-label">Test RMSE</div>
                          </div>
                          <div className="metric-card">
                            <div className="metric-value">{(evaluationSnapshot.testMetrics?.mae ?? 0).toFixed(6)}</div>
                            <div className="metric-label">Test MAE</div>
                          </div>
                          <div className="metric-card">
                            <div className="metric-value">{(evaluationSnapshot.testMetrics?.r2 ?? 0).toFixed(4)}</div>
                            <div className="metric-label">Test R²</div>
                          </div>
                          <div className="metric-card">
                            <div className="metric-value">
                              {(evaluationSnapshot.trainMetrics?.kind === 'regression'
                                ? evaluationSnapshot.trainMetrics.r2 ?? 0
                                : evaluationSnapshot.trainMetrics?.accuracy ?? 0
                              ).toFixed(4)}
                            </div>
                            <div className="metric-label">Train Reference</div>
                          </div>
                        </>
                      )}
                    </div>
                    <div className={`wk-fit-banner wk-fit-${evaluationSnapshot.fitSignal}`}>
                      <strong>{evaluationSnapshot.fitSignal.toUpperCase()}</strong>
                      <span>{evaluationSnapshot.fitMessage}</span>
                    </div>
                  </div>

                  <div className="wk-results-grid">
                    {evaluationSnapshot.testMetrics?.kind === 'classification' && (
                      <div className="wk-panel-block">
                        <h3>Confusion Matrix</h3>
                        <p className="wk-hint">Click a cell to inspect sample details.</p>
                        <div className="wk-matrix">
                          {(evaluationSnapshot.testMetrics?.confusionMatrix ?? []).map((row, rowIdx) => (
                            <div className="wk-matrix-row" key={`eval_row_${rowIdx}`}>
                              {row.map((value, colIdx) => {
                                const activeCell =
                                  selectedConfusionCell?.actual === rowIdx && selectedConfusionCell?.predicted === colIdx;
                                return (
                                  <button
                                    type="button"
                                    className={`wk-matrix-cell ${rowIdx === colIdx ? 'diag' : ''} ${activeCell ? 'active' : ''}`}
                                    key={`eval_cell_${rowIdx}_${colIdx}`}
                                    onClick={() => setSelectedConfusionCell({ actual: rowIdx, predicted: colIdx })}
                                    style={matrixHeatStyle(value, evaluationConfusionMax)}
                                  >
                                    {value}
                                  </button>
                                );
                              })}
                            </div>
                          ))}
                        </div>
                        {selectedConfusionCell && (
                          <>
                            <div className="wk-hint">
                              Selected cell: actual <strong>{heldOutDataset?.labelNames[selectedConfusionCell.actual] ?? selectedConfusionCell.actual}</strong>
                              , predicted{' '}
                              <strong>{heldOutDataset?.labelNames[selectedConfusionCell.predicted] ?? selectedConfusionCell.predicted}</strong>
                              {' '}({confusionCellSamples.length} samples)
                            </div>
                            <div className="wk-confusion-detail-list">
                              {confusionCellSamples.slice(0, 8).map((sample) => (
                                <div key={`cell_sample_${sample.rowIndex}`}>
                                  Row #{sample.rowIndex + 1} - {String(sample.rawRow?.image_name ?? sample.rawRow?.id ?? sample.rawRow?.name ?? 'sample')}
                                </div>
                              ))}
                              {confusionCellSamples.length > 8 && (
                                <div>...and {confusionCellSamples.length - 8} more rows</div>
                              )}
                            </div>
                          </>
                        )}
                      </div>
                    )}

                    {evaluationSnapshot.testMetrics?.kind === 'regression' && (
                      <div className="wk-panel-block">
                        <h3>Predicted vs Actual</h3>
                        {regressionScatterPoints.length > 0 ? (() => {
                          const values = regressionScatterPoints.flatMap((sample) => [sample.actualValue, sample.predictedValue]);
                          const min = Math.min(...values);
                          const max = Math.max(...values);
                          const range = Math.max(1e-6, max - min);
                          const size = 240;
                          return (
                            <svg viewBox={`0 0 ${size} ${size}`} className="wk-scatter-plot">
                              <line x1={16} y1={size - 16} x2={size - 16} y2={16} stroke="var(--border-bright)" strokeDasharray="4 3" />
                              {regressionScatterPoints.map((sample) => {
                                const x = 16 + (((sample.actualValue - min) / range) * (size - 32));
                                const y = size - 16 - (((sample.predictedValue - min) / range) * (size - 32));
                                return <circle key={`scatter_${sample.rowIndex}`} cx={x} cy={y} r={3} fill="var(--accent)" opacity={0.75} />;
                              })}
                            </svg>
                          );
                        })() : (
                          <p className="wk-hint">No regression points available for scatter visualization.</p>
                        )}
                      </div>
                    )}

                    <div className="wk-panel-block">
                      <h3>Sample Predictions</h3>
                      <div className="wk-inline-actions">
                        <button className={`wk-chip ${predictionFilter === 'all' ? 'active' : ''}`} onClick={() => setPredictionFilter('all')}>
                          All
                        </button>
                        <button className={`wk-chip ${predictionFilter === 'correct' ? 'active' : ''}`} onClick={() => setPredictionFilter('correct')}>
                          Correct
                        </button>
                        <button className={`wk-chip ${predictionFilter === 'incorrect' ? 'active' : ''}`} onClick={() => setPredictionFilter('incorrect')}>
                          Incorrect
                        </button>
                      </div>

                      {isImageEvaluation ? (
                        <div className="wk-image-pred-grid">
                          {filteredEvaluationSamples.slice(0, 12).map((sample) => {
                            const previewSource = sample.rawRow ? renderPixelRowPreview(sample.rawRow) : null;
                            return (
                              <div key={`img_eval_${sample.rowIndex}`} className={`wk-image-pred-card ${sample.correct ? 'correct' : 'incorrect'}`}>
                                <div className="wk-image-thumb" style={imagePlaceholderStyle(sample.rawRow)}>
                                  {previewSource ? (
                                    <img src={previewSource} alt={`sample_${sample.rowIndex + 1}`} className="wk-image-thumb-img" />
                                  ) : (
                                    <span className="wk-hint">Preview unavailable</span>
                                  )}
                                </div>
                                <div className="wk-image-meta">
                                  <strong>{String(sample.rawRow?.image_name ?? `sample_${sample.rowIndex + 1}`)}</strong>
                                  <small>Truth: {sample.actualLabel}</small>
                                  <small>Pred: {sample.predictedLabel}</small>
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      ) : (
                        <div className="wk-table-wrapper">
                          <table className="wk-table">
                            <thead>
                              <tr>
                                <th>#</th>
                                <th>Ground Truth</th>
                                <th>Prediction</th>
                                {evaluationSnapshot.testMetrics?.kind === 'regression' && <th>Absolute Error</th>}
                                <th>Status</th>
                              </tr>
                            </thead>
                            <tbody>
                              {filteredEvaluationSamples.slice(0, 20).map((sample) => (
                                <tr key={`tab_eval_${sample.rowIndex}`}>
                                  <td>{sample.rowIndex + 1}</td>
                                  <td>{sample.actualLabel}</td>
                                  <td>{sample.predictedLabel}</td>
                                  {evaluationSnapshot.testMetrics?.kind === 'regression' && (
                                    <td>{(sample.absoluteError ?? 0).toFixed(6)}</td>
                                  )}
                                  <td>{sample.correct ? '✅ Correct' : '❌ Miss'}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      )}
                    </div>
                  </div>
                </>
              ) : (
                <p className="wk-hint">Run training with a split to populate held-out evaluation metrics.</p>
              )}

              <div className="wk-panel-block" style={{ marginTop: 12 }}>
                <div className="wk-block-header">
                  <h3>Export Panel</h3>
                  <small>Model + evaluation exports</small>
                </div>
                <div className="wk-export-options">
                  <button className="wk-export-card" onClick={() => training.exportModel('pth')}>
                    <strong>Export .pth</strong>
                    <span>Weight snapshot with metadata</span>
                  </button>
                  <button className="wk-export-card" onClick={() => training.exportModel('tensor')}>
                    <strong>Export .tensor</strong>
                    <span>Raw tensor-friendly JSON payload</span>
                  </button>
                  <button className="wk-export-card" onClick={() => exportEvaluationMetrics('json')}>
                    <strong>Export metrics JSON</strong>
                    <span>Train/test metrics and fit diagnostics</span>
                  </button>
                  <button className="wk-export-card" onClick={() => exportEvaluationMetrics('csv')}>
                    <strong>Export metrics CSV</strong>
                    <span>Flat metrics for external analysis</span>
                  </button>
                  <button className="wk-export-card" onClick={exportEvaluationPredictions}>
                    <strong>Export predictions CSV</strong>
                    <span>Current filtered subset from evaluation table</span>
                  </button>
                </div>
                {evaluationExportStatus && <p className="wk-success">{evaluationExportStatus}</p>}
              </div>

              <div className="wk-results-grid">
                <div className="wk-panel-block">
                  <h3>Live Inference Demo</h3>
                  <p className="wk-hint">Pick a held-out sample for instant in-browser inference.</p>
                  <select
                    className="select"
                    value={liveInferenceTestIndex}
                    onChange={(event) => setLiveInferenceTestIndex(Number(event.target.value) || 0)}
                    disabled={!heldOutDataset || heldOutDataset.features.length === 0}
                  >
                    {(heldOutDataset?.features ?? []).slice(0, 200).map((_, index) => (
                      <option key={`live_sample_${index}`} value={index}>
                        Test sample #{index + 1}
                      </option>
                    ))}
                  </select>
                  <div className="wk-inline-actions">
                    <button className="btn btn-primary" onClick={onRunLiveInferenceFromTestSample}>
                      Run Selected Test Sample
                    </button>
                  </div>
                </div>

                <div className="wk-panel-block">
                  <h3>Custom Sample Inference</h3>
                  <p className="wk-hint">
                    Enter comma-separated values ({processedDataset.featureNames.length}) or a JSON object keyed by feature name.
                  </p>
                  <textarea
                    className="input wk-textarea"
                    value={inferenceInput}
                    onChange={(event) => setInferenceInput(event.target.value)}
                    placeholder={
                      processedDataset.featureNames.length > 0
                        ? processedDataset.featureNames.map(() => '0').join(', ')
                        : '{}'
                    }
                  />
                  <div className="wk-inline-actions">
                    <button className="btn btn-primary" onClick={onRunInference}>
                      Run Custom Sample
                    </button>
                    <button className="btn btn-secondary" onClick={() => inferenceFileInputRef.current?.click()}>
                      Upload JSON Sample
                    </button>
                    <input
                      ref={inferenceFileInputRef}
                      type="file"
                      accept=".json,application/json"
                      style={{ display: 'none' }}
                      onChange={(event) => {
                        const file = event.target.files?.[0];
                        if (file) void onUploadInferenceSample(file);
                        event.currentTarget.value = '';
                      }}
                    />
                  </div>
                  {inferenceResult && <p className="wk-success">{inferenceResult}</p>}
                  {inferenceError && <p className="wk-error">{inferenceError}</p>}
                </div>
              </div>

              <div className="wk-screen-footer">
                <button className="btn btn-secondary" onClick={() => setStep('results')}>
                  Back
                </button>
                <button
                  className="btn btn-danger"
                  onClick={() => {
                    if (runId) {
                      training.clearCheckpoint(runId);
                      clearStoredRunVersion(runId);
                    }
                    void clearPendingRunState().catch(() => {
                      // ignore IndexedDB write failures
                    });
                    setRunId('');
                    setStep('dataset');
                  }}
                >
                  New Training Run
                </button>
              </div>
            </div>
            ) : (
              <StepPlaceholder
                title="Nothing to export yet"
                message="Complete training to unlock held-out evaluation, exports, and live inference."
                actionLabel="Go to Training"
                onAction={() => setStep('training')}
              />
            )
          )}
        </section>

        <footer className="wk-footer">
          <span>
            Step {currentStepIndex + 1} of {STEPS.length}
          </span>
          {dataset && (
            <span>
              Dataset: {dataset.source.name} ({dataset.rows.length.toLocaleString()} rows)
            </span>
          )}
        </footer>
            */}
          </>
        )}
      </main>
    </>
  );
}
