import JSZip from 'jszip';
import { ProcessedDataset } from '../types/data';
import {
  ExportFormat,
  ModelMetrics,
  TrainedModelArtifact,
  TrainingPreferences,
} from '../types/training-workflow';
import { serializeModelToOnnx } from './onnx-serializer';

export interface ExportContext {
  runId: string;
  artifact: TrainedModelArtifact;
  metrics: ModelMetrics;
  dataset: ProcessedDataset;
  preferences: TrainingPreferences;
}

function toArrayBuffer(bytes: Uint8Array): ArrayBuffer {
  const buffer = new ArrayBuffer(bytes.byteLength);
  new Uint8Array(buffer).set(bytes);
  return buffer;
}

function buildMetadataJson(ctx: ExportContext): string {
  return JSON.stringify(
    {
      runId: ctx.runId,
      modelType: ctx.artifact.modelType,
      backend: ctx.artifact.backend,
      trainedAt: ctx.artifact.trainedAt,
      metrics: ctx.metrics,
      labelNames: ctx.dataset.labelNames,
      featureNames: ctx.dataset.featureNames,
      targetColumn: ctx.dataset.targetColumn,
      datasetProblemType: ctx.dataset.problemType,
    },
    null,
    2
  );
}

function buildConfigJson(ctx: ExportContext): string {
  return JSON.stringify(
    {
      trainingPreferences: ctx.preferences,
      preprocessing: ctx.dataset.preprocessing,
      preprocessingStats: ctx.dataset.stats,
    },
    null,
    2
  );
}

function buildPthContent(ctx: ExportContext): string {
  return JSON.stringify(
    {
      format: 'browser-first-ai.pth.json',
      artifact: ctx.artifact,
      metadata: JSON.parse(buildMetadataJson(ctx)),
      config: JSON.parse(buildConfigJson(ctx)),
    },
    null,
    2
  );
}

function buildTensorContent(ctx: ExportContext): string {
  return JSON.stringify(
    {
      format: 'browser-first-ai.tensor.v1',
      modelType: ctx.artifact.modelType,
      backend: ctx.artifact.backend,
      trainedAt: ctx.artifact.trainedAt,
      featureNames: ctx.dataset.featureNames,
      labelNames: ctx.dataset.labelNames,
      modelData: ctx.artifact.modelData,
    },
    null,
    2
  );
}

export async function exportModelArtifact(
  format: ExportFormat,
  context: ExportContext
): Promise<{ filename: string; blob: Blob }> {
  const safeName = context.dataset.targetColumn.replace(/[^a-z0-9_-]/gi, '_').toLowerCase() || 'model';

  if (format === 'pth') {
    return {
      filename: `${safeName}.pth`,
      blob: new Blob([buildPthContent(context)], { type: 'application/json' }),
    };
  }

  if (format === 'tensor') {
    return {
      filename: `${safeName}.tensor`,
      blob: new Blob([buildTensorContent(context)], { type: 'application/json' }),
    };
  }

  if (format === 'onnx') {
    const onnxBinary = serializeModelToOnnx(context);
    return {
      filename: `${safeName}.onnx`,
      blob: new Blob([toArrayBuffer(onnxBinary)], { type: 'application/octet-stream' }),
    };
  }

  const onnxBinary = serializeModelToOnnx(context);
  const zip = new JSZip();
  zip.file('model.onnx', onnxBinary);
  zip.file('metadata.json', buildMetadataJson(context));
  zip.file('config.json', buildConfigJson(context));
  zip.file('model.pth.json', buildPthContent(context));
  const blob = await zip.generateAsync({ type: 'blob' });
  return {
    filename: `${safeName}.kaya`,
    blob,
  };
}
