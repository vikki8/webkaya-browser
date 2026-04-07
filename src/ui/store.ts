import { create } from 'zustand';
import { LayerConfig, ModelGraph, LayerType, LAYER_DEFAULTS } from '../types/model';
import { TrainingConfig, TrainingState, TrainingMetrics, DEFAULT_TRAINING_CONFIG } from '../types/training';
import { v4 as uuid } from 'uuid';

interface StudioStore {
  // Model graph
  graph: ModelGraph;
  addLayer: (type: LayerType) => void;
  removeLayer: (id: string) => void;
  moveLayer: (fromIdx: number, toIdx: number) => void;
  updateLayerParam: (id: string, key: string, value: number | boolean) => void;
  setModelName: (name: string) => void;
  loadPreset: (preset: 'simple_cnn' | 'deep_cnn' | 'linear') => void;

  // Training config
  config: TrainingConfig;
  updateConfig: <K extends keyof TrainingConfig>(key: K, value: TrainingConfig[K]) => void;

  // Training state
  trainingState: TrainingState;
  setTrainingStatus: (status: TrainingState['status']) => void;
  addMetrics: (m: TrainingMetrics) => void;
  setError: (error: string) => void;
  resetTraining: () => void;

  // Logs
  logs: string[];
  addLog: (msg: string) => void;
  clearLogs: () => void;
}

function layerName(type: LayerType, layers: LayerConfig[]): string {
  const count = layers.filter(l => l.type === type).length;
  return `${type}_${count + 1}`;
}

const PRESET_SIMPLE_CNN: LayerConfig[] = [
  { id: uuid(), type: 'conv2d', name: 'conv1', params: { inChannels: 1, outChannels: 16, kernelSize: 3, stride: 1, padding: 1 } },
  { id: uuid(), type: 'relu', name: 'relu1', params: {} },
  { id: uuid(), type: 'maxpool2d', name: 'pool1', params: { kernelSize: 2, stride: 2 } },
  { id: uuid(), type: 'conv2d', name: 'conv2', params: { inChannels: 16, outChannels: 32, kernelSize: 3, stride: 1, padding: 1 } },
  { id: uuid(), type: 'relu', name: 'relu2', params: {} },
  { id: uuid(), type: 'maxpool2d', name: 'pool2', params: { kernelSize: 2, stride: 2 } },
  { id: uuid(), type: 'flatten', name: 'flatten', params: {} },
  { id: uuid(), type: 'linear', name: 'fc1', params: { inFeatures: 32 * 7 * 7, outFeatures: 128 } },
  { id: uuid(), type: 'relu', name: 'relu3', params: {} },
  { id: uuid(), type: 'linear', name: 'fc2', params: { inFeatures: 128, outFeatures: 10 } },
];

const PRESET_DEEP_CNN: LayerConfig[] = [
  { id: uuid(), type: 'conv2d', name: 'conv1', params: { inChannels: 1, outChannels: 32, kernelSize: 3, stride: 1, padding: 1 } },
  { id: uuid(), type: 'batchnorm2d', name: 'bn1', params: { numFeatures: 32 } },
  { id: uuid(), type: 'relu', name: 'relu1', params: {} },
  { id: uuid(), type: 'conv2d', name: 'conv2', params: { inChannels: 32, outChannels: 64, kernelSize: 3, stride: 1, padding: 1 } },
  { id: uuid(), type: 'batchnorm2d', name: 'bn2', params: { numFeatures: 64 } },
  { id: uuid(), type: 'relu', name: 'relu2', params: {} },
  { id: uuid(), type: 'maxpool2d', name: 'pool1', params: { kernelSize: 2, stride: 2 } },
  { id: uuid(), type: 'dropout', name: 'drop1', params: { p: 0.25 } },
  { id: uuid(), type: 'conv2d', name: 'conv3', params: { inChannels: 64, outChannels: 128, kernelSize: 3, stride: 1, padding: 1 } },
  { id: uuid(), type: 'batchnorm2d', name: 'bn3', params: { numFeatures: 128 } },
  { id: uuid(), type: 'relu', name: 'relu3', params: {} },
  { id: uuid(), type: 'maxpool2d', name: 'pool2', params: { kernelSize: 2, stride: 2 } },
  { id: uuid(), type: 'flatten', name: 'flatten', params: {} },
  { id: uuid(), type: 'linear', name: 'fc1', params: { inFeatures: 128 * 7 * 7, outFeatures: 256 } },
  { id: uuid(), type: 'relu', name: 'relu4', params: {} },
  { id: uuid(), type: 'dropout', name: 'drop2', params: { p: 0.5 } },
  { id: uuid(), type: 'linear', name: 'fc2', params: { inFeatures: 256, outFeatures: 10 } },
];

const PRESET_LINEAR: LayerConfig[] = [
  { id: uuid(), type: 'flatten', name: 'flatten', params: {} },
  { id: uuid(), type: 'linear', name: 'fc1', params: { inFeatures: 784, outFeatures: 128 } },
  { id: uuid(), type: 'relu', name: 'relu1', params: {} },
  { id: uuid(), type: 'linear', name: 'fc2', params: { inFeatures: 128, outFeatures: 10 } },
];

export const useStudioStore = create<StudioStore>((set, get) => ({
  graph: { name: 'MyModel', layers: [...PRESET_SIMPLE_CNN] },

  addLayer: (type) => set(s => ({
    graph: {
      ...s.graph,
      layers: [...s.graph.layers, {
        id: uuid(),
        type,
        name: layerName(type, s.graph.layers),
        params: { ...LAYER_DEFAULTS[type] },
      }],
    },
  })),

  removeLayer: (id) => set(s => ({
    graph: { ...s.graph, layers: s.graph.layers.filter(l => l.id !== id) },
  })),

  moveLayer: (from, to) => set(s => {
    const layers = [...s.graph.layers];
    const [moved] = layers.splice(from, 1);
    layers.splice(to, 0, moved);
    return { graph: { ...s.graph, layers } };
  }),

  updateLayerParam: (id, key, value) => set(s => ({
    graph: {
      ...s.graph,
      layers: s.graph.layers.map(l =>
        l.id === id ? { ...l, params: { ...l.params, [key]: value } } : l
      ),
    },
  })),

  setModelName: (name) => set(s => ({ graph: { ...s.graph, name } })),

  loadPreset: (preset) => set(() => {
    const presetMap = {
      simple_cnn: PRESET_SIMPLE_CNN,
      deep_cnn: PRESET_DEEP_CNN,
      linear: PRESET_LINEAR,
    };
    return {
      graph: { name: preset === 'simple_cnn' ? 'SimpleCNN' : preset === 'deep_cnn' ? 'DeepCNN' : 'LinearNet', layers: presetMap[preset].map(l => ({ ...l, id: uuid() })) },
    };
  }),

  config: { ...DEFAULT_TRAINING_CONFIG },
  updateConfig: (key, value) => set(s => ({ config: { ...s.config, [key]: value } })),

  trainingState: {
    status: 'idle',
    currentEpoch: 0,
    totalEpochs: 0,
    currentStep: 0,
    totalSteps: 0,
    metrics: [],
  },
  setTrainingStatus: (status) => set(s => ({ trainingState: { ...s.trainingState, status } })),
  addMetrics: (m) => set(s => ({
    trainingState: {
      ...s.trainingState,
      currentEpoch: m.epoch,
      currentStep: m.step,
      metrics: [...s.trainingState.metrics, m],
    },
  })),
  setError: (error) => set(s => ({
    trainingState: { ...s.trainingState, status: 'error', error },
  })),
  resetTraining: () => set(() => ({
    trainingState: {
      status: 'idle', currentEpoch: 0, totalEpochs: 0,
      currentStep: 0, totalSteps: 0, metrics: [],
    },
  })),

  logs: [],
  addLog: (msg) => set(s => ({ logs: [...s.logs.slice(-200), `[${new Date().toLocaleTimeString()}] ${msg}`] })),
  clearLogs: () => set({ logs: [] }),
}));
