import { useCallback, useEffect, useRef, useState } from 'react';
import { Capabilities } from '../../engine/capability-detect';
import { ProcessedDataset } from '../../types/data';
import {
  ExportFormat,
  DatasetTransferPayload,
  ModelChoice,
  ModelMetrics,
  ResolvedModel,
  TrainedModelArtifact,
  TrainingCurvePoint,
  TrainingPhase,
  TrainingPreferences,
} from '../../types/training-workflow';
import { MainToTrainingWorkerMessage, WorkerToTrainingMainMessage } from '../../types/workerMessages';

interface WorkflowTrainingState {
  phase: TrainingPhase;
  statusMessage: string;
  curve: TrainingCurvePoint[];
  progressPercent: number;
  backend: string | null;
  resolvedModel: ResolvedModel | null;
  logs: string[];
  metrics: ModelMetrics | null;
  artifact: TrainedModelArtifact | null;
  error: string | null;
  ready: boolean;
}

const initialState: WorkflowTrainingState = {
  phase: 'idle',
  statusMessage: '',
  curve: [],
  progressPercent: 0,
  backend: null,
  resolvedModel: null,
  logs: [],
  metrics: null,
  artifact: null,
  error: null,
  ready: false,
};

export function useWorkflowTraining() {
  const workerRef = useRef<Worker | null>(null);
  const initTimeoutRef = useRef<number | null>(null);
  const initResolverRef = useRef<{
    resolve: () => void;
    reject: (reason?: unknown) => void;
  } | null>(null);
  const [state, setState] = useState<WorkflowTrainingState>(initialState);

  useEffect(() => {
    const worker = new Worker(new URL('../../engine/training-worker.ts', import.meta.url));
    workerRef.current = worker;

    worker.onmessage = (event: MessageEvent<WorkerToTrainingMainMessage>) => {
      const msg = event.data;
      if (msg.type === 'ready') {
        if (initTimeoutRef.current !== null) {
          window.clearTimeout(initTimeoutRef.current);
          initTimeoutRef.current = null;
        }
        setState((prev) => ({
          ...prev,
          ready: true,
          resolvedModel: msg.payload.resolvedModel,
          backend: msg.payload.backend,
          statusMessage: msg.payload.resumed
            ? 'Existing checkpoint detected for this run. Training can resume from saved progress.'
            : prev.statusMessage,
          error: null,
        }));
        initResolverRef.current?.resolve();
        initResolverRef.current = null;
        return;
      }

      if (msg.type === 'status') {
        setState((prev) => ({
          ...prev,
          phase: msg.payload.phase,
          statusMessage: msg.payload.message,
        }));
        return;
      }

      if (msg.type === 'progress') {
        setState((prev) => ({
          ...prev,
          curve: [...prev.curve.slice(-599), msg.payload],
          progressPercent: msg.payload.percent,
        }));
        return;
      }

      if (msg.type === 'training_complete') {
        setState((prev) => ({
          ...prev,
          phase: 'completed',
          metrics: msg.payload.metrics,
          artifact: msg.payload.artifact,
          progressPercent: 100,
        }));
        return;
      }

      if (msg.type === 'log') {
        setState((prev) => ({
          ...prev,
          logs: [...prev.logs.slice(-249), `[${new Date().toLocaleTimeString()}] ${msg.payload.message}`],
        }));
        return;
      }

      if (msg.type === 'error') {
        if (initTimeoutRef.current !== null) {
          window.clearTimeout(initTimeoutRef.current);
          initTimeoutRef.current = null;
        }
        setState((prev) => ({
          ...prev,
          phase: 'error',
          error: msg.payload.message,
        }));
        initResolverRef.current?.reject(new Error(msg.payload.message));
        initResolverRef.current = null;
        return;
      }

      if (msg.type === 'export_ready') {
        const url = URL.createObjectURL(msg.payload.blob);
        const anchor = document.createElement('a');
        anchor.href = url;
        anchor.download = msg.payload.filename;
        anchor.click();
        URL.revokeObjectURL(url);
      }
    };

    worker.onerror = (event) => {
      if (initTimeoutRef.current !== null) {
        window.clearTimeout(initTimeoutRef.current);
        initTimeoutRef.current = null;
      }
      const message = event.message || 'Training worker failed.';
      setState((prev) => ({ ...prev, phase: 'error', error: message }));
      initResolverRef.current?.reject(new Error(message));
      initResolverRef.current = null;
    };

    return () => {
      if (initTimeoutRef.current !== null) {
        window.clearTimeout(initTimeoutRef.current);
        initTimeoutRef.current = null;
      }
      workerRef.current?.terminate();
      workerRef.current = null;
    };
  }, []);

  const initialize = useCallback(
    (
      runId: string,
      dataset: ProcessedDataset,
      datasetTransfer: DatasetTransferPayload | null,
      modelChoice: ModelChoice,
      preferences: TrainingPreferences,
      capabilities: Capabilities | null
    ) =>
      new Promise<void>((resolve, reject) => {
        if (!workerRef.current) {
          reject(new Error('Training worker not initialized.'));
          return;
        }
        if (initResolverRef.current) {
          reject(new Error('Training worker initialization is already in progress.'));
          return;
        }
        initResolverRef.current = { resolve, reject };
        initTimeoutRef.current = window.setTimeout(() => {
          const timeoutError = new Error('Training worker initialization timed out.');
          initResolverRef.current?.reject(timeoutError);
          initResolverRef.current = null;
          initTimeoutRef.current = null;
          setState((prev) => ({ ...prev, phase: 'error', error: timeoutError.message }));
        }, 30_000);
        setState(initialState);
        workerRef.current.postMessage({
          type: 'init',
          payload: { runId, dataset, datasetTransfer, modelChoice, preferences, capabilities },
        } satisfies MainToTrainingWorkerMessage);
      }),
    []
  );

  const start = useCallback(() => {
    workerRef.current?.postMessage({ type: 'start' } satisfies MainToTrainingWorkerMessage);
  }, []);

  const stop = useCallback(() => {
    setState((prev) => ({
      ...prev,
      statusMessage: prev.phase === 'completed' ? prev.statusMessage : 'Stopping training...',
    }));
    workerRef.current?.postMessage({ type: 'stop' } satisfies MainToTrainingWorkerMessage);
  }, []);

  const exportModel = useCallback((format: ExportFormat) => {
    workerRef.current?.postMessage({ type: 'export', payload: { format } } satisfies MainToTrainingWorkerMessage);
  }, []);

  const clearCheckpoint = useCallback((runId: string) => {
    workerRef.current?.postMessage({ type: 'clear_checkpoint', payload: { runId } } satisfies MainToTrainingWorkerMessage);
  }, []);

  return {
    state,
    initialize,
    start,
    stop,
    exportModel,
    clearCheckpoint,
  };
}
