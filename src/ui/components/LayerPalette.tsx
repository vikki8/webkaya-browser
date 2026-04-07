import React from 'react';
import { LayerType } from '../../types/model';
import { useStudioStore } from '../store';

const LAYER_CATALOG: { type: LayerType; label: string; color: string; abbr: string; category: string }[] = [
  { type: 'conv2d', label: 'Conv2d', color: '#6366f1', abbr: 'CV', category: 'Convolution' },
  { type: 'linear', label: 'Linear', color: '#06b6d4', abbr: 'FC', category: 'Dense' },
  { type: 'relu', label: 'ReLU', color: '#22c55e', abbr: 'Re', category: 'Activation' },
  { type: 'sigmoid', label: 'Sigmoid', color: '#22c55e', abbr: 'Sg', category: 'Activation' },
  { type: 'tanh', label: 'Tanh', color: '#22c55e', abbr: 'Th', category: 'Activation' },
  { type: 'softmax', label: 'Softmax', color: '#22c55e', abbr: 'SM', category: 'Activation' },
  { type: 'maxpool2d', label: 'MaxPool2d', color: '#a855f7', abbr: 'MP', category: 'Pooling' },
  { type: 'avgpool2d', label: 'AvgPool2d', color: '#a855f7', abbr: 'AP', category: 'Pooling' },
  { type: 'batchnorm2d', label: 'BatchNorm2d', color: '#f59e0b', abbr: 'BN', category: 'Normalization' },
  { type: 'dropout', label: 'Dropout', color: '#ef4444', abbr: 'DO', category: 'Regularization' },
  { type: 'flatten', label: 'Flatten', color: '#9898b0', abbr: 'FL', category: 'Reshape' },
];

export function LayerPalette() {
  const addLayer = useStudioStore(s => s.addLayer);
  const loadPreset = useStudioStore(s => s.loadPreset);

  const categories = LAYER_CATALOG.reduce<Record<string, typeof LAYER_CATALOG>>((acc, item) => {
    (acc[item.category] = acc[item.category] || []).push(item);
    return acc;
  }, {});

  return (
    <div>
      <div className="panel-header">Layers</div>
      <div className="panel-content">
        <div className="section-title">Presets</div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4, marginBottom: 16 }}>
          <button className="btn btn-secondary btn-sm" onClick={() => loadPreset('simple_cnn')}>
            Simple CNN
          </button>
          <button className="btn btn-secondary btn-sm" onClick={() => loadPreset('deep_cnn')}>
            Deep CNN
          </button>
          <button className="btn btn-secondary btn-sm" onClick={() => loadPreset('linear')}>
            Linear Net
          </button>
        </div>

        <div className="divider" />

        {Object.entries(categories).map(([cat, items]) => (
          <div key={cat}>
            <div className="label" style={{ marginTop: 8, marginBottom: 6 }}>{cat}</div>
            {items.map(item => (
              <div
                key={item.type}
                className="palette-item"
                onClick={() => addLayer(item.type)}
                title={`Add ${item.label} layer`}
              >
                <div
                  className="palette-icon"
                  style={{ background: `${item.color}22`, color: item.color }}
                >
                  {item.abbr}
                </div>
                {item.label}
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}
