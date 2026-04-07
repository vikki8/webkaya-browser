import { analyzeDataset } from './insights';
import { preprocessDataset } from './preprocess';
import { MainToDataWorkerMessage, WorkerToDataMainMessage } from '../types/data-worker';

function post(message: WorkerToDataMainMessage) {
  (self as unknown as Worker).postMessage(message);
}

self.onmessage = (event: MessageEvent<MainToDataWorkerMessage>) => {
  const message = event.data;
  if (message.type === 'stop') {
    post({ type: 'log', payload: { message: 'Data worker stopped.' } });
    return;
  }

  if (message.type === 'analyze_and_preprocess') {
    try {
      const { rows, columns, config, previewOnly } = message.payload;
      const { insights } = analyzeDataset(rows, columns, {
        selectedTarget: config.targetColumn,
      });
      const resolvedConfig = {
        ...config,
        targetColumn: config.targetColumn ?? insights.detectedTarget,
      };
      const result = preprocessDataset(rows, columns, resolvedConfig, {
        previewOnly: Boolean(previewOnly),
        onProgress: (progress) => {
          post({
            type: 'preprocess_progress',
            payload: progress,
          });
        },
      });
      post({
        type: 'preprocess_complete',
        payload: {
          insights: result.insights,
          previewRows: result.previewRows,
          processed: result.processed,
        },
      });
    } catch (error: any) {
      post({ type: 'error', payload: { message: error?.message ?? 'Data preprocessing failed.' } });
    }
  }
};
