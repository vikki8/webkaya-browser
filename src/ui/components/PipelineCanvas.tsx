import React, { useState, useCallback } from 'react';
import { useStudioStore } from '../store';
import { LayerConfig } from '../../types/model';

const LAYER_COLORS: Record<string, string> = {
  conv2d: '#6366f1',
  linear: '#06b6d4',
  relu: '#22c55e',
  sigmoid: '#22c55e',
  tanh: '#22c55e',
  softmax: '#22c55e',
  maxpool2d: '#a855f7',
  avgpool2d: '#a855f7',
  batchnorm2d: '#f59e0b',
  dropout: '#ef4444',
  flatten: '#9898b0',
};

function LayerParamEditor({ layer }: { layer: LayerConfig }) {
  const updateLayerParam = useStudioStore(s => s.updateLayerParam);
  const entries = Object.entries(layer.params);
  if (entries.length === 0) return null;

  return (
    <div className="layer-params">
      {entries.map(([key, val]) => (
        <div key={key} className="layer-param-group">
          <span className="label">{key}</span>
          <input
            className="input"
            type="number"
            value={val as number}
            step={key === 'p' ? 0.05 : 1}
            min={key === 'p' ? 0 : 1}
            max={key === 'p' ? 1 : undefined}
            onChange={e => updateLayerParam(layer.id, key, parseFloat(e.target.value) || 0)}
            style={{ padding: '3px 6px', fontSize: 11 }}
          />
        </div>
      ))}
    </div>
  );
}

export function PipelineCanvas() {
  const { graph, removeLayer, moveLayer, setModelName } = useStudioStore();
  const [dragIdx, setDragIdx] = useState<number | null>(null);
  const [dragOverIdx, setDragOverIdx] = useState<number | null>(null);

  const handleDragStart = useCallback((idx: number) => {
    setDragIdx(idx);
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent, idx: number) => {
    e.preventDefault();
    setDragOverIdx(idx);
  }, []);

  const handleDrop = useCallback((idx: number) => {
    if (dragIdx !== null && dragIdx !== idx) {
      moveLayer(dragIdx, idx);
    }
    setDragIdx(null);
    setDragOverIdx(null);
  }, [dragIdx, moveLayer]);

  const handleDragEnd = useCallback(() => {
    setDragIdx(null);
    setDragOverIdx(null);
  }, []);

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
        <input
          className="input"
          value={graph.name}
          onChange={e => setModelName(e.target.value)}
          style={{ maxWidth: 200, fontWeight: 600 }}
          placeholder="Model Name"
        />
        <span className="badge badge-info">{graph.layers.length} layers</span>
      </div>

      {graph.layers.length === 0 ? (
        <div className="empty-state">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M12 5v14M5 12h14" />
          </svg>
          <p style={{ marginTop: 12 }}>Add layers from the palette</p>
          <p style={{ fontSize: 11, marginTop: 4 }}>or choose a preset to get started</p>
        </div>
      ) : (
        <div>
          {graph.layers.map((layer, idx) => {
            const color = LAYER_COLORS[layer.type] || '#6366f1';
            return (
              <React.Fragment key={layer.id}>
                <div
                  className={`layer-card${dragIdx === idx ? ' dragging' : ''}`}
                  draggable
                  onDragStart={() => handleDragStart(idx)}
                  onDragOver={(e) => handleDragOver(e, idx)}
                  onDrop={() => handleDrop(idx)}
                  onDragEnd={handleDragEnd}
                  style={{
                    borderLeftWidth: 3,
                    borderLeftColor: color,
                    ...(dragOverIdx === idx && dragIdx !== idx ? { borderTopColor: 'var(--accent)', borderTopWidth: 2 } : {}),
                  }}
                >
                  <div className="layer-card-header">
                    <div>
                      <span className="layer-type" style={{ color }}>{layer.type}</span>
                      <span className="layer-name" style={{ marginLeft: 8 }}>{layer.name}</span>
                    </div>
                    <div style={{ display: 'flex', gap: 4 }}>
                      <button
                        className="btn-icon"
                        style={{ width: 24, height: 24, fontSize: 14 }}
                        onClick={() => removeLayer(layer.id)}
                        title="Remove layer"
                      >
                        &times;
                      </button>
                    </div>
                  </div>
                  <LayerParamEditor layer={layer} />
                </div>
                {idx < graph.layers.length - 1 && (
                  <div style={{
                    display: 'flex', justifyContent: 'center', padding: '2px 0',
                    color: 'var(--text-muted)', fontSize: 16
                  }}>
                    &#8595;
                  </div>
                )}
              </React.Fragment>
            );
          })}
        </div>
      )}
    </div>
  );
}
