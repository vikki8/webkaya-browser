import { useCallback, useEffect, useRef, useState } from 'react';
import { AutoInsights, DataRow, PreprocessingConfig, ProcessedDataset } from '../../types/data';
import { DataPreprocessProgress, MainToDataWorkerMessage, WorkerToDataMainMessage } from '../../types/data-worker';

interface DataWorkerResult {
  insights: AutoInsights;
  previewRows: DataRow[];
  processed: ProcessedDataset;
}

export function useDataWorker() {
  const workerRef = useRef<Worker | null>(null);
  const timeoutRef = useRef<number | null>(null);
  const resolverRef = useRef<{
    resolve: (value: DataWorkerResult) => void;
    reject: (reason?: unknown) => void;
  } | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<DataPreprocessProgress | null>(null);

  const clearPending = useCallback(() => {
    if (timeoutRef.current !== null) {
      window.clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
    resolverRef.current = null;
  }, []);

  useEffect(() => {
    const worker = new Worker(new URL('../../data/data-worker.ts', import.meta.url));
    workerRef.current = worker;

    worker.onmessage = (event: MessageEvent<WorkerToDataMainMessage>) => {
      const msg = event.data;
      if (msg.type === 'preprocess_progress') {
        setProgress(msg.payload);
      } else if (msg.type === 'preprocess_complete') {
        setBusy(false);
        setError(null);
        setProgress(null);
        resolverRef.current?.resolve(msg.payload);
        clearPending();
      } else if (msg.type === 'error') {
        setBusy(false);
        setError(msg.payload.message);
        setProgress(null);
        resolverRef.current?.reject(new Error(msg.payload.message));
        clearPending();
      }
    };

    worker.onerror = (event) => {
      setBusy(false);
      const message = event.message || 'Data worker failed.';
      setError(message);
       setProgress(null);
      resolverRef.current?.reject(new Error(message));
      clearPending();
    };

    return () => {
      workerRef.current?.postMessage({ type: 'stop' } satisfies MainToDataWorkerMessage);
      workerRef.current?.terminate();
      workerRef.current = null;
      clearPending();
    };
  }, [clearPending]);

  const preprocess = useCallback(
    (
      rows: DataRow[],
      columns: string[],
      config: PreprocessingConfig,
      options?: { previewOnly?: boolean; timeoutMs?: number }
    ) =>
      new Promise<DataWorkerResult>((resolve, reject) => {
        if (!workerRef.current) {
          reject(new Error('Data worker not initialized.'));
          return;
        }
        const timeoutMs = Math.max(10_000, options?.timeoutMs ?? 180_000);
        setBusy(true);
        setError(null);
        setProgress({
          stage: 'analyzing',
          message: options?.previewOnly ? 'Preparing preview...' : 'Preparing preprocessing...',
          percent: 1,
        });
        resolverRef.current = { resolve, reject };
        timeoutRef.current = window.setTimeout(() => {
          setBusy(false);
          setProgress(null);
          const timeoutError = new Error(
            'Preprocessing timed out. Try disabling normalization for very large datasets, then retry.'
          );
          setError(timeoutError.message);
          resolverRef.current?.reject(timeoutError);
          clearPending();
        }, timeoutMs);
        workerRef.current.postMessage({
          type: 'analyze_and_preprocess',
          payload: {
            rows,
            columns,
            config,
            previewOnly: Boolean(options?.previewOnly),
          },
        } satisfies MainToDataWorkerMessage);
      }),
    [clearPending]
  );

  return { preprocess, busy, error, progress };
}
