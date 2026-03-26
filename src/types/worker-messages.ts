import { ModelGraph } from './model';
import { TrainingConfig, TrainingMetrics } from './training';

export type MainToWorkerMessage =
  | { type: 'init'; payload: { graph: ModelGraph; config: TrainingConfig } }
  | { type: 'start' }
  | { type: 'pause' }
  | { type: 'resume' }
  | { type: 'stop' }
  | { type: 'export' };

export type WorkerToMainMessage =
  | { type: 'ready' }
  | { type: 'status'; payload: { status: string } }
  | { type: 'progress'; payload: TrainingMetrics }
  | { type: 'epoch_complete'; payload: TrainingMetrics }
  | { type: 'training_complete'; payload: { finalMetrics: TrainingMetrics } }
  | { type: 'error'; payload: { message: string } }
  | { type: 'export_ready'; payload: { blob: Blob; filename: string } }
  | { type: 'log'; payload: { message: string } };
