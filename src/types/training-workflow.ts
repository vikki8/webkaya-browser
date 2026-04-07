import { Capabilities } from '../engine/capability-detect';
import { AlgorithmId, ProcessedDataset } from './data';

export type ModelChoice = AlgorithmId | 'recommended' | 'auto';
export type ResolvedModel =
  | 'random_forest'
  | 'neural_network'
  | 'decision_tree'
  | 'linear_regression'
  | 'logistic_regression'
  | 'svm'
  | 'knn'
  | 'kmeans'
  | 'dbscan'
  | 'rnn'
  | 'lstm'
  | 'gru';
export type ExportFormat = 'pth' | 'tensor' | 'onnx' | 'kaya';
export type RuntimePipeline = 'hybrid_worker_wasm_webgpu';
export type WasmEditorMode = 'template' | 'eject';
export type NeuralActivation = 'relu' | 'leaky_relu' | 'tanh' | 'sigmoid' | 'softmax';
export type NeuralOptimizer = 'adamw' | 'sgd_momentum' | 'adam' | 'adamax';
export type LearningRateScheduler = 'constant' | 'linear_decay' | 'cosine_annealing' | 'step_lr';
export type KnnDistanceMetric = 'euclidean' | 'manhattan' | 'cosine';
export type SvmKernel = 'linear' | 'rbf' | 'poly';

export interface WasmFunctionTemplateConfig {
  functionName: string;
  invocationTimeoutMs: number;
  retryCount: number;
  memoryBudgetMB: number;
  shardCount: number;
  checkpointEveryNEpochs: number;
  gradientClipValue: number;
  coldStartMs: number;
}

export interface WasmFunctionEditorState {
  advancedMode: WasmEditorMode;
  templateConfig: WasmFunctionTemplateConfig;
  executableCode: string;
}

export interface HybridRuntimeSettings {
  pipeline: RuntimePipeline;
  wasmEditor: WasmFunctionEditorState;
}

export interface WasmFunctionInvocationPolicy extends WasmFunctionTemplateConfig {}

export interface NeuralNetworkSettings {
  hiddenLayers: number;
  neuronsPerLayer: number;
  activation: NeuralActivation;
  useBatchNorm: boolean;
  useLayerNorm: boolean;
  dropoutRate: number;
  gradientClipping: number;
  optimizer?: NeuralOptimizer;
  weightDecay?: number;
}

export interface AlgorithmSettings {
  knnNeighbors: number;
  knnDistanceMetric: KnnDistanceMetric;
  svmKernel: SvmKernel;
  kmeansClusters: number;
  dbscanEpsilon: number;
  dbscanMinSamples: number;
}

export interface TrainingPreferences {
  speedVsAccuracy: number; // 0..100
  useMoreCompute: boolean;
  optimizeForSmallerModel: boolean;
  epochs: number;
  learningRate: number;
  batchSize: number;
  shuffleEachEpoch: boolean;
  earlyStoppingPatience: number;
  optimizer: NeuralOptimizer;
  weightDecay: number;
  momentum: number;
  beta1: number;
  beta2: number;
  lrScheduler: LearningRateScheduler;
  warmupSteps: number;
  schedulerStepSize: number;
  schedulerGamma: number;
  neuralNetwork: NeuralNetworkSettings;
  algorithm: AlgorithmSettings;
  runtime: HybridRuntimeSettings;
}

export interface TrainingCurvePoint {
  step: number;
  epoch: number;
  loss: number;
  accuracy: number;
}

export interface ModelMetrics {
  kind: 'classification' | 'regression';
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1?: number;
  confusionMatrix?: number[][];
  mae?: number;
  rmse?: number;
  r2?: number;
  featureImportance: Array<{ feature: string; score: number }>;
}

export interface DatasetTransferPayload {
  rowCount: number;
  featureCount: number;
  featuresBuffer: SharedArrayBuffer;
  labelsBuffer: SharedArrayBuffer;
  regressionTargetsBuffer?: SharedArrayBuffer;
}

export interface TrainedModelArtifact {
  modelType: ResolvedModel;
  backend: string;
  trainedAt: string;
  featureNames: string[];
  labelNames: string[];
  modelData: Record<string, unknown>;
}

export type TrainingPhase =
  | 'idle'
  | 'cleaning_data'
  | 'training_model'
  | 'optimizing_parameters'
  | 'completed'
  | 'error';

export type MainToTrainingWorkerMessage =
  | {
      type: 'init';
      payload: {
        runId: string;
        dataset: ProcessedDataset;
        datasetTransfer?: DatasetTransferPayload | null;
        modelChoice: ModelChoice;
        preferences: TrainingPreferences;
        capabilities: Capabilities | null;
      };
    }
  | { type: 'start' }
  | { type: 'stop' }
  | { type: 'export'; payload: { format: ExportFormat } }
  | { type: 'clear_checkpoint'; payload: { runId: string } };

export type WorkerToTrainingMainMessage =
  | { type: 'ready'; payload: { resumed: boolean; resolvedModel: ResolvedModel; backend: string } }
  | { type: 'status'; payload: { phase: TrainingPhase; message: string } }
  | { type: 'progress'; payload: TrainingCurvePoint & { percent: number } }
  | {
      type: 'training_complete';
      payload: {
        metrics: ModelMetrics;
        artifact: TrainedModelArtifact;
      };
    }
  | { type: 'export_ready'; payload: { format: ExportFormat; filename: string; blob: Blob } }
  | { type: 'error'; payload: { message: string } }
  | { type: 'log'; payload: { message: string } };
