import { useCallback, useRef, useEffect } from 'react';
import { useStudioStore } from '../store';
import { WorkerToMainMessage } from '../../types/worker-messages';

export function useTrainingWorker() {
  const workerRef = useRef<Worker | null>(null);
  const {
    graph, config, setTrainingStatus, addMetrics, setError, addLog, resetTraining,
  } = useStudioStore();

  useEffect(() => {
    return () => {
      workerRef.current?.terminate();
    };
  }, []);

  const initAndStart = useCallback(() => {
    workerRef.current?.terminate();

    resetTraining();
    setTrainingStatus('loading');
    addLog('Initializing training worker...');

    const worker = new Worker(new URL('../../engine/trainer.ts', import.meta.url));
    workerRef.current = worker;

    worker.onmessage = (e: MessageEvent<WorkerToMainMessage>) => {
      const msg = e.data;
      switch (msg.type) {
        case 'ready':
          addLog('Model built. Starting training...');
          worker.postMessage({ type: 'start' });
          setTrainingStatus('training');
          break;
        case 'status':
          setTrainingStatus(msg.payload.status as any);
          break;
        case 'progress':
          addMetrics(msg.payload);
          break;
        case 'epoch_complete':
          addMetrics(msg.payload);
          addLog(`Epoch ${msg.payload.epoch + 1}: loss=${msg.payload.loss.toFixed(4)} acc=${(msg.payload.accuracy * 100).toFixed(1)}%`);
          break;
        case 'training_complete':
          addMetrics(msg.payload.finalMetrics);
          setTrainingStatus('completed');
          addLog(`Training complete! Final acc: ${(msg.payload.finalMetrics.accuracy * 100).toFixed(1)}%`);
          break;
        case 'error':
          setError(msg.payload.message);
          addLog(`ERROR: ${msg.payload.message}`);
          break;
        case 'log':
          addLog(msg.payload.message);
          break;
        case 'export_ready': {
          const url = URL.createObjectURL(msg.payload.blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = msg.payload.filename;
          a.click();
          URL.revokeObjectURL(url);
          addLog(`Exported: ${msg.payload.filename}`);
          break;
        }
      }
    };

    worker.onerror = (err) => {
      setError(err.message || 'Worker error');
      addLog(`Worker error: ${err.message}`);
    };

    worker.postMessage({ type: 'init', payload: { graph, config } });
  }, [graph, config, setTrainingStatus, addMetrics, setError, addLog, resetTraining]);

  const pause = useCallback(() => {
    workerRef.current?.postMessage({ type: 'pause' });
  }, []);

  const resume = useCallback(() => {
    workerRef.current?.postMessage({ type: 'resume' });
  }, []);

  const stop = useCallback(() => {
    workerRef.current?.postMessage({ type: 'stop' });
    setTrainingStatus('idle');
  }, [setTrainingStatus]);

  const exportModel = useCallback(() => {
    workerRef.current?.postMessage({ type: 'export' });
  }, []);

  return { initAndStart, pause, resume, stop, exportModel };
}
