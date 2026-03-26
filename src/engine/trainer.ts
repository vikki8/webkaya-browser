import { Tensor, crossEntropyLoss, mseLoss } from './tensor';
import { Sequential } from './nn';
import { SGD, Adam, Optimizer } from './optimizer';
import { Dataset, SyntheticMNIST, SyntheticCIFAR, SyntheticTabular } from './datasets';
import { buildModel, estimateModelSize } from './model-builder';
import { serializeCheckpoint, restoreCheckpoint, exportWeights, CheckpointData } from './checkpoint';
import { ModelGraph } from '../types/model';
import { TrainingConfig, TrainingMetrics } from '../types/training';
import { WorkerToMainMessage, MainToWorkerMessage } from '../types/worker-messages';

let model: Sequential | null = null;
let optimizer: Optimizer | null = null;
let dataset: Dataset | null = null;
let config: TrainingConfig | null = null;
let graph: ModelGraph | null = null;
let shouldStop = false;
let shouldPause = false;
let lastCheckpoint: CheckpointData | null = null;

function post(msg: WorkerToMainMessage) {
  (self as any).postMessage(msg);
}

function createDataset(name: string): Dataset {
  switch (name) {
    case 'mnist': return new SyntheticMNIST(2000);
    case 'cifar10': return new SyntheticCIFAR(1000);
    case 'synthetic': return new SyntheticTabular(1000, 16);
    default: return new SyntheticMNIST(2000);
  }
}

function createOptimizer(params: Tensor[], cfg: TrainingConfig): Optimizer {
  switch (cfg.optimizer) {
    case 'adam': return new Adam(params, cfg.learningRate);
    case 'sgd': return new SGD(params, cfg.learningRate);
    default: return new Adam(params, cfg.learningRate);
  }
}

function computeAccuracy(logits: Tensor, targets: Int32Array): number {
  const [batch, classes] = logits.shape;
  let correct = 0;
  for (let b = 0; b < batch; b++) {
    let maxVal = -Infinity, maxIdx = 0;
    for (let c = 0; c < classes; c++) {
      const val = logits.data[b * classes + c];
      if (val > maxVal) { maxVal = val; maxIdx = c; }
    }
    if (maxIdx === targets[b]) correct++;
  }
  return correct / batch;
}

async function train() {
  if (!model || !optimizer || !dataset || !config) {
    post({ type: 'error', payload: { message: 'Not initialized' } });
    return;
  }

  shouldStop = false;
  shouldPause = false;

  const totalBatches = Math.ceil(dataset.numSamples / config.batchSize);
  const startTime = performance.now();
  let startEpoch = 0;

  if (lastCheckpoint) {
    try {
      restoreCheckpoint(model, lastCheckpoint);
      startEpoch = lastCheckpoint.epoch;
      post({ type: 'log', payload: { message: `Resumed from epoch ${startEpoch}` } });
    } catch (e) {
      post({ type: 'log', payload: { message: 'Could not restore checkpoint, starting fresh' } });
      startEpoch = 0;
    }
  }

  for (let epoch = startEpoch; epoch < config.epochs; epoch++) {
    if (shouldStop) break;
    const epochStart = performance.now();
    let epochLoss = 0;
    let epochAcc = 0;

    for (let batch = 0; batch < totalBatches; batch++) {
      if (shouldStop) break;
      while (shouldPause) {
        await new Promise(r => setTimeout(r, 100));
        if (shouldStop) break;
      }

      const { inputs, targets } = dataset.getBatch(batch, config.batchSize);

      optimizer.zeroGrad();
      const logits = model.forward(inputs, true);
      const loss = config.lossFunction === 'cross_entropy'
        ? crossEntropyLoss(logits, targets)
        : crossEntropyLoss(logits, targets);

      if (isNaN(loss.data[0]) || !isFinite(loss.data[0])) {
        post({ type: 'error', payload: { message: `NaN/Inf detected at epoch ${epoch}, step ${batch}. Try reducing learning rate.` } });
        return;
      }

      loss.backward();
      optimizer.step();

      const acc = computeAccuracy(logits, targets);
      epochLoss += loss.data[0];
      epochAcc += acc;

      const step = epoch * totalBatches + batch;
      const metrics: TrainingMetrics = {
        epoch,
        step,
        loss: loss.data[0],
        accuracy: acc,
        learningRate: config.learningRate,
        epochTime: performance.now() - epochStart,
        totalTime: performance.now() - startTime,
      };
      post({ type: 'progress', payload: metrics });

      // Yield to keep the worker responsive
      if (batch % 5 === 0) await new Promise(r => setTimeout(r, 0));
    }

    const avgLoss = epochLoss / totalBatches;
    const avgAcc = epochAcc / totalBatches;

    lastCheckpoint = serializeCheckpoint(model, epoch + 1, 0, avgLoss, avgAcc);

    const epochMetrics: TrainingMetrics = {
      epoch,
      step: (epoch + 1) * totalBatches,
      loss: avgLoss,
      accuracy: avgAcc,
      learningRate: config.learningRate,
      epochTime: performance.now() - epochStart,
      totalTime: performance.now() - startTime,
    };
    post({ type: 'epoch_complete', payload: epochMetrics });
  }

  if (!shouldStop) {
    const finalMetrics: TrainingMetrics = {
      epoch: config.epochs - 1,
      step: config.epochs * totalBatches,
      loss: lastCheckpoint?.loss ?? 0,
      accuracy: lastCheckpoint?.accuracy ?? 0,
      learningRate: config.learningRate,
      epochTime: 0,
      totalTime: performance.now() - startTime,
    };
    post({ type: 'training_complete', payload: { finalMetrics } });
  }
}

function handleExport() {
  if (!model || !graph) {
    post({ type: 'error', payload: { message: 'No model to export' } });
    return;
  }

  const { shapes, weights } = exportWeights(model);
  const metadata = {
    name: graph.name,
    layers: graph.layers,
    paramShapes: shapes,
    exportedAt: new Date().toISOString(),
    framework: 'webkaya',
    version: '0.1.0',
  };

  const metadataJson = JSON.stringify(metadata, null, 2);
  const metadataBlob = new Blob([metadataJson], { type: 'application/json' });

  const weightBuffers: ArrayBuffer[] = weights.map(w => w.buffer.slice(0));
  const totalSize = weightBuffers.reduce((a, b) => a + b.byteLength, 0);
  const combinedWeights = new Uint8Array(totalSize);
  let offset = 0;
  for (const buf of weightBuffers) {
    combinedWeights.set(new Uint8Array(buf), offset);
    offset += buf.byteLength;
  }

  const blob = new Blob([
    metadataJson, '\n---WEIGHTS---\n', combinedWeights
  ], { type: 'application/octet-stream' });

  post({ type: 'export_ready', payload: { blob, filename: `${graph.name || 'model'}.kaya` } });
}

self.onmessage = async (e: MessageEvent<MainToWorkerMessage>) => {
  const msg = e.data;

  switch (msg.type) {
    case 'init': {
      graph = msg.payload.graph;
      config = msg.payload.config;
      lastCheckpoint = null;

      try {
        const estimate = estimateModelSize(graph);
        post({ type: 'log', payload: { message: `Model: ${estimate.params.toLocaleString()} params, ~${estimate.memoryMB.toFixed(1)}MB memory` } });

        model = buildModel(graph);
        dataset = createDataset(config.dataset);
        optimizer = createOptimizer(model.parameters(), config);

        post({ type: 'log', payload: { message: `Dataset: ${dataset.numSamples} samples, ${dataset.numClasses} classes` } });
        post({ type: 'ready' });
      } catch (err: any) {
        post({ type: 'error', payload: { message: err.message || 'Init failed' } });
      }
      break;
    }
    case 'start':
      post({ type: 'status', payload: { status: 'training' } });
      await train();
      break;
    case 'pause':
      shouldPause = true;
      post({ type: 'status', payload: { status: 'paused' } });
      break;
    case 'resume':
      shouldPause = false;
      post({ type: 'status', payload: { status: 'training' } });
      break;
    case 'stop':
      shouldStop = true;
      shouldPause = false;
      post({ type: 'status', payload: { status: 'idle' } });
      break;
    case 'export':
      handleExport();
      break;
  }
};
