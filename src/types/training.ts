export interface TrainingConfig {
  learningRate: number;
  batchSize: number;
  epochs: number;
  optimizer: 'sgd' | 'adam';
  lossFunction: 'cross_entropy' | 'mse';
  dataset: 'mnist' | 'cifar10' | 'synthetic' | 'custom';
}

export interface TrainingMetrics {
  epoch: number;
  step: number;
  loss: number;
  accuracy: number;
  learningRate: number;
  epochTime: number;
  totalTime: number;
}

export interface TrainingState {
  status: 'idle' | 'loading' | 'training' | 'paused' | 'completed' | 'error';
  currentEpoch: number;
  totalEpochs: number;
  currentStep: number;
  totalSteps: number;
  metrics: TrainingMetrics[];
  error?: string;
}

export const DEFAULT_TRAINING_CONFIG: TrainingConfig = {
  learningRate: 0.001,
  batchSize: 32,
  epochs: 10,
  optimizer: 'adam',
  lossFunction: 'cross_entropy',
  dataset: 'mnist',
};
