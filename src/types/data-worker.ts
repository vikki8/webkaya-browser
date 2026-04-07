import { AutoInsights, DataRow, PreprocessingConfig, ProcessedDataset } from './data';

export interface DataPreprocessProgress {
  stage: 'analyzing' | 'fix_missing' | 'encode' | 'normalize' | 'augment' | 'build_features' | 'complete';
  message: string;
  percent: number;
}

export type MainToDataWorkerMessage =
  | {
      type: 'analyze_and_preprocess';
      payload: {
        rows: DataRow[];
        columns: string[];
        config: PreprocessingConfig;
        previewOnly?: boolean;
      };
    }
  | { type: 'stop' };

export type WorkerToDataMainMessage =
  | {
      type: 'preprocess_progress';
      payload: DataPreprocessProgress;
    }
  | {
      type: 'preprocess_complete';
      payload: {
        insights: AutoInsights;
        previewRows: DataRow[];
        processed: ProcessedDataset;
      };
    }
  | { type: 'error'; payload: { message: string } }
  | { type: 'log'; payload: { message: string } };
