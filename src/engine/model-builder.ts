import { ModelGraph, LayerConfig } from '../types/model';
import {
  Sequential, Layer, Conv2d, Linear, ReLU, Sigmoid, Tanh,
  MaxPool2d, AvgPool2d, BatchNorm2d, Dropout, Flatten, Softmax
} from './nn';

function buildLayer(config: LayerConfig): Layer {
  const p = config.params;
  switch (config.type) {
    case 'conv2d':
      return new Conv2d(
        p.inChannels as number, p.outChannels as number,
        p.kernelSize as number, p.stride as number, p.padding as number,
        config.name
      );
    case 'linear':
      return new Linear(p.inFeatures as number, p.outFeatures as number, config.name);
    case 'relu':
      return new ReLU();
    case 'sigmoid':
      return new Sigmoid();
    case 'tanh':
      return new Tanh();
    case 'maxpool2d':
      return new MaxPool2d(p.kernelSize as number, p.stride as number, config.name);
    case 'avgpool2d':
      return new AvgPool2d(p.kernelSize as number, p.stride as number, config.name);
    case 'batchnorm2d':
      return new BatchNorm2d(p.numFeatures as number, config.name);
    case 'dropout':
      return new Dropout(p.p as number, config.name);
    case 'flatten':
      return new Flatten();
    case 'softmax':
      return new Softmax();
    default:
      throw new Error(`Unknown layer type: ${config.type}`);
  }
}

export function buildModel(graph: ModelGraph): Sequential {
  const layers = graph.layers.map(buildLayer);
  return new Sequential(layers);
}

export function estimateModelSize(graph: ModelGraph): { params: number; memoryMB: number } {
  let totalParams = 0;

  for (const layer of graph.layers) {
    const p = layer.params;
    switch (layer.type) {
      case 'conv2d': {
        const kernelParams = (p.outChannels as number) * (p.inChannels as number) *
          (p.kernelSize as number) * (p.kernelSize as number);
        totalParams += kernelParams + (p.outChannels as number);
        break;
      }
      case 'linear':
        totalParams += (p.inFeatures as number) * (p.outFeatures as number) + (p.outFeatures as number);
        break;
      case 'batchnorm2d':
        totalParams += (p.numFeatures as number) * 2;
        break;
    }
  }

  // Adam stores 2 extra copies of parameters (m and v)
  const memoryMB = (totalParams * 4 * 3) / (1024 * 1024);

  return { params: totalParams, memoryMB };
}
