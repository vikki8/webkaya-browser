import React from 'react';
import { useStudioStore } from '../store';
import { estimateModelSize } from '../../engine/model-builder';

export function ConfigPanel() {
  const { config, updateConfig, graph } = useStudioStore();
  const estimate = estimateModelSize(graph);

  return (
    <div>
      <div className="panel-header">Configuration</div>
      <div className="panel-content">
        <div className="section-title">Model Info</div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, marginBottom: 16 }}>
          <div className="metric-card" style={{ padding: 8 }}>
            <div className="metric-value" style={{ fontSize: 16 }}>{estimate.params.toLocaleString()}</div>
            <div className="metric-label">Parameters</div>
          </div>
          <div className="metric-card" style={{ padding: 8 }}>
            <div className="metric-value" style={{ fontSize: 16 }}>{estimate.memoryMB.toFixed(1)} MB</div>
            <div className="metric-label">Memory Est.</div>
          </div>
        </div>

        <div className="divider" />
        <div className="section-title">Dataset</div>
        <div className="form-group">
          <label className="label">Dataset</label>
          <select
            className="select"
            value={config.dataset}
            onChange={e => updateConfig('dataset', e.target.value as any)}
          >
            <option value="mnist">Synthetic MNIST (28x28, 10 classes)</option>
            <option value="cifar10">Synthetic CIFAR-10 (32x32 RGB, 10 classes)</option>
            <option value="synthetic">Synthetic Tabular (16 features, 4 classes)</option>
          </select>
        </div>

        <div className="divider" />
        <div className="section-title">Hyperparameters</div>

        <div className="form-group">
          <label className="label">Learning Rate</label>
          <input
            className="input"
            type="number"
            step={0.0001}
            min={0.00001}
            max={1}
            value={config.learningRate}
            onChange={e => updateConfig('learningRate', parseFloat(e.target.value) || 0.001)}
          />
        </div>

        <div className="form-group">
          <label className="label">Batch Size</label>
          <select
            className="select"
            value={config.batchSize}
            onChange={e => updateConfig('batchSize', parseInt(e.target.value))}
          >
            {[4, 8, 16, 32, 64, 128].map(v => (
              <option key={v} value={v}>{v}</option>
            ))}
          </select>
        </div>

        <div className="form-group">
          <label className="label">Epochs</label>
          <input
            className="input"
            type="number"
            min={1}
            max={100}
            value={config.epochs}
            onChange={e => updateConfig('epochs', parseInt(e.target.value) || 1)}
          />
        </div>

        <div className="form-group">
          <label className="label">Optimizer</label>
          <select
            className="select"
            value={config.optimizer}
            onChange={e => updateConfig('optimizer', e.target.value as any)}
          >
            <option value="adam">Adam</option>
            <option value="sgd">SGD</option>
          </select>
        </div>

        <div className="form-group">
          <label className="label">Loss Function</label>
          <select
            className="select"
            value={config.lossFunction}
            onChange={e => updateConfig('lossFunction', e.target.value as any)}
          >
            <option value="cross_entropy">Cross Entropy</option>
            <option value="mse">MSE</option>
          </select>
        </div>
      </div>
    </div>
  );
}
