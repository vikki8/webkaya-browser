import { Tensor } from './tensor';
import { Sequential } from './nn';

export interface CheckpointData {
  epoch: number;
  step: number;
  loss: number;
  accuracy: number;
  paramShapes: number[][];
  paramData: ArrayBuffer[];
  timestamp: number;
}

export function serializeCheckpoint(
  model: Sequential, epoch: number, step: number, loss: number, accuracy: number
): CheckpointData {
  const params = model.parameters();
  return {
    epoch,
    step,
    loss,
    accuracy,
    paramShapes: params.map(p => [...p.shape]),
    paramData: params.map(p => p.data.buffer.slice(0)),
    timestamp: Date.now(),
  };
}

export function restoreCheckpoint(model: Sequential, checkpoint: CheckpointData) {
  const params = model.parameters();
  if (params.length !== checkpoint.paramData.length) {
    throw new Error(`Checkpoint param count mismatch: model has ${params.length}, checkpoint has ${checkpoint.paramData.length}`);
  }
  for (let i = 0; i < params.length; i++) {
    const src = new Float32Array(checkpoint.paramData[i]);
    if (src.length !== params[i].data.length) {
      throw new Error(`Param ${i} size mismatch: ${params[i].data.length} vs ${src.length}`);
    }
    params[i].data.set(src);
  }
}

export function exportWeights(model: Sequential): { shapes: number[][]; weights: Float32Array[] } {
  const params = model.parameters();
  return {
    shapes: params.map(p => [...p.shape]),
    weights: params.map(p => new Float32Array(p.data)),
  };
}
