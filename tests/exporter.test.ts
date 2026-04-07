import { describe, expect, test } from 'vitest';
import { unzipSync } from 'fflate';
import { exportModelArtifact } from '../src/engine/exporter';
import { onnx } from 'onnx-proto';

describe('model exporter', () => {
  const context = {
    runId: 'wk_test_run',
    artifact: {
      modelType: 'random_forest' as const,
      backend: 'CPU',
      trainedAt: new Date('2026-03-26T00:00:00.000Z').toISOString(),
      featureNames: ['f1', 'f2'],
      labelNames: ['a', 'b'],
      modelData: {
        trees: [
          {
            prediction: 0,
            featureIndex: 0,
            threshold: 0.5,
            left: {
              prediction: 0,
              featureIndex: null,
              threshold: 0,
              left: null,
              right: null,
            },
            right: {
              prediction: 1,
              featureIndex: null,
              threshold: 0,
              left: null,
              right: null,
            },
          },
        ],
        classes: [0, 1],
        featureImportance: [0.4, 0.6],
      },
    },
    metrics: {
      kind: 'classification' as const,
      accuracy: 0.9,
      precision: 0.89,
      recall: 0.88,
      f1: 0.885,
      confusionMatrix: [
        [9, 1],
        [1, 9],
      ],
      featureImportance: [
        { feature: 'f2', score: 0.6 },
        { feature: 'f1', score: 0.4 },
      ],
    },
    dataset: {
      featureNames: ['f1', 'f2'],
      features: [[0, 1]],
      labels: [1],
      labelNames: ['a', 'b'],
      targetColumn: 'label',
      problemType: 'classification' as const,
      sampleRows: [{ f1: 0, f2: 1, label: 'b' }],
      preprocessing: {
        fixMissingValues: true,
        encodeCategories: true,
        normalizeData: true,
        augmentImageData: false,
        imageAugmentationFactor: 1 as const,
        imageAugmentationNoise: 0.06,
        droppedColumns: [],
        targetColumn: 'label',
      },
      stats: {
        beforeMissing: 0,
        afterMissing: 0,
        encodedColumns: [],
        normalizedColumns: [],
        droppedColumns: [],
      },
    },
    preferences: {
      speedVsAccuracy: 50,
      useMoreCompute: false,
      optimizeForSmallerModel: false,
      epochs: 8,
      learningRate: 0.001,
      batchSize: 32,
      shuffleEachEpoch: true,
      earlyStoppingPatience: 4,
      optimizer: 'adamw' as const,
      weightDecay: 0.0001,
      momentum: 0.9,
      beta1: 0.9,
      beta2: 0.999,
      lrScheduler: 'constant' as const,
      warmupSteps: 0,
      schedulerStepSize: 10,
      schedulerGamma: 0.5,
      algorithm: {
        knnNeighbors: 7,
        knnDistanceMetric: 'euclidean' as const,
        svmKernel: 'rbf' as const,
        kmeansClusters: 8,
        dbscanEpsilon: 0.8,
        dbscanMinSamples: 6,
      },
      neuralNetwork: {
        hiddenLayers: 2,
        neuronsPerLayer: 128,
        activation: 'relu' as const,
        useBatchNorm: false,
        useLayerNorm: false,
        dropoutRate: 0.1,
        gradientClipping: 1,
        optimizer: 'adamw' as const,
        weightDecay: 0.0001,
      },
      runtime: {
        pipeline: 'hybrid_worker_wasm_webgpu' as const,
        wasmEditor: {
          advancedMode: 'template' as const,
          executableCode: '',
          templateConfig: {
            functionName: 'train_batch',
            invocationTimeoutMs: 10000,
            retryCount: 1,
            memoryBudgetMB: 512,
            shardCount: 1,
            checkpointEveryNEpochs: 1,
            gradientClipValue: 0,
            coldStartMs: 50,
          },
        },
      },
    },
  };

  test('creates kaya bundle with required files', async () => {
    const { filename, blob } = await exportModelArtifact('kaya', context);
    expect(filename.endsWith('.kaya')).toBe(true);

    const files = unzipSync(new Uint8Array(await blob.arrayBuffer()));
    expect(files['model.onnx']).toBeTruthy();
    expect(files['metadata.json']).toBeTruthy();
    expect(files['config.json']).toBeTruthy();

    const onnxBytes = files['model.onnx']!;
    const model = onnx.ModelProto.decode(onnxBytes);
    expect(model.graph?.node?.length).toBeGreaterThan(0);
    expect(model.opsetImport?.length).toBeGreaterThan(0);
  });
});
