export type LayerType =
  | 'conv2d'
  | 'linear'
  | 'relu'
  | 'sigmoid'
  | 'tanh'
  | 'maxpool2d'
  | 'avgpool2d'
  | 'batchnorm2d'
  | 'dropout'
  | 'flatten'
  | 'softmax';

export interface LayerConfig {
  id: string;
  type: LayerType;
  params: Record<string, number | boolean>;
  name: string;
}

export interface Conv2dParams {
  inChannels: number;
  outChannels: number;
  kernelSize: number;
  stride: number;
  padding: number;
}

export interface LinearParams {
  inFeatures: number;
  outFeatures: number;
}

export interface MaxPool2dParams {
  kernelSize: number;
  stride: number;
}

export interface BatchNorm2dParams {
  numFeatures: number;
}

export interface DropoutParams {
  p: number;
}

export interface ModelGraph {
  layers: LayerConfig[];
  name: string;
}

export const LAYER_DEFAULTS: Record<LayerType, Record<string, number | boolean>> = {
  conv2d: { inChannels: 1, outChannels: 32, kernelSize: 3, stride: 1, padding: 1 },
  linear: { inFeatures: 128, outFeatures: 10 },
  relu: {},
  sigmoid: {},
  tanh: {},
  maxpool2d: { kernelSize: 2, stride: 2 },
  avgpool2d: { kernelSize: 2, stride: 2 },
  batchnorm2d: { numFeatures: 32 },
  dropout: { p: 0.5 },
  flatten: {},
  softmax: {},
};
