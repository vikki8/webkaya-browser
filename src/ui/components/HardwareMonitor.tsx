import React from 'react';
import { useCapabilities } from '../hooks/useCapabilities';

export function HardwareMonitor() {
  const { caps, loading } = useCapabilities();

  if (loading) {
    return (
      <div className="panel-header">
        <span className="animate-pulse">Detecting hardware...</span>
      </div>
    );
  }

  if (!caps) return null;

  return (
    <div>
      <div className="panel-header">Hardware</div>
      <div className="panel-content">
        <div className={`hardware-badge tier-${caps.tier}`} style={{ marginBottom: 12 }}>
          <span style={{ width: 8, height: 8, borderRadius: '50%', background: 'currentColor' }} />
          Tier {caps.tier}: {caps.tierName}
        </div>

        <div style={{ display: 'grid', gap: 8 }}>
          <FeatureRow label="WebGPU" available={caps.webgpu} />
          <FeatureRow label="WebGL2" available={caps.webgl2} />
          <FeatureRow label="WASM SIMD" available={caps.wasmSimd} />
          <FeatureRow label="SharedArrayBuffer" available={caps.sharedArrayBuffer} />
          <FeatureRow label="OffscreenCanvas" available={caps.offscreenCanvas} />
          <FeatureRow label="OPFS" available={caps.opfs} />
        </div>

        {caps.gpuName && (
          <div style={{ marginTop: 12, fontSize: 11, color: 'var(--text-muted)' }}>
            GPU: {caps.gpuName}
          </div>
        )}

        <div style={{ marginTop: 8, fontSize: 11, color: 'var(--text-muted)' }}>
          Est. Memory: {caps.maxMemoryMB} MB
        </div>
      </div>
    </div>
  );
}

function FeatureRow({ label, available }: { label: string; available: boolean }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: 12 }}>
      <span style={{ color: 'var(--text-secondary)' }}>{label}</span>
      <span style={{
        color: available ? 'var(--success)' : 'var(--text-muted)',
        fontSize: 11,
        fontWeight: 500,
      }}>
        {available ? '● Available' : '○ N/A'}
      </span>
    </div>
  );
}
