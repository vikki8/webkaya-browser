import React, { useRef, useEffect } from 'react';
import { useStudioStore } from '../store';

function MetricCard({ value, label, color }: { value: string; label: string; color?: string }) {
  return (
    <div className="metric-card">
      <div className="metric-value" style={{ color: color || 'var(--text-primary)' }}>{value}</div>
      <div className="metric-label">{label}</div>
    </div>
  );
}

function LossChart() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const metrics = useStudioStore(s => s.trainingState.metrics);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || metrics.length < 2) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const h = rect.height;
    const pad = { top: 10, right: 10, bottom: 20, left: 40 };
    const plotW = w - pad.left - pad.right;
    const plotH = h - pad.top - pad.bottom;

    ctx.clearRect(0, 0, w, h);

    // Only plot epoch-level metrics (every N steps)
    const step = Math.max(1, Math.floor(metrics.length / 200));
    const sampled = metrics.filter((_, i) => i % step === 0 || i === metrics.length - 1);

    const losses = sampled.map(m => m.loss);
    const accs = sampled.map(m => m.accuracy);

    const minLoss = Math.min(...losses);
    const maxLoss = Math.max(...losses) || 1;
    const lossRange = maxLoss - minLoss || 1;

    // Grid
    ctx.strokeStyle = '#2a2a45';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + (plotH / 4) * i;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(w - pad.right, y);
      ctx.stroke();
    }

    // Loss line
    ctx.strokeStyle = '#6366f1';
    ctx.lineWidth = 2;
    ctx.beginPath();
    sampled.forEach((_, i) => {
      const x = pad.left + (i / (sampled.length - 1)) * plotW;
      const y = pad.top + (1 - (losses[i] - minLoss) / lossRange) * plotH;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Accuracy line
    ctx.strokeStyle = '#22c55e';
    ctx.lineWidth = 2;
    ctx.beginPath();
    sampled.forEach((_, i) => {
      const x = pad.left + (i / (sampled.length - 1)) * plotW;
      const y = pad.top + (1 - accs[i]) * plotH;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Labels
    ctx.fillStyle = '#9898b0';
    ctx.font = '10px Inter, sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(maxLoss.toFixed(2), pad.left - 4, pad.top + 10);
    ctx.fillText(minLoss.toFixed(2), pad.left - 4, pad.top + plotH);

    // Legend
    ctx.textAlign = 'left';
    ctx.fillStyle = '#6366f1';
    ctx.fillText('Loss', w - 70, pad.top + 12);
    ctx.fillStyle = '#22c55e';
    ctx.fillText('Accuracy', w - 70, pad.top + 24);

  }, [metrics]);

  return (
    <div className="chart-container">
      <canvas
        ref={canvasRef}
        style={{ width: '100%', height: '100%' }}
      />
    </div>
  );
}

export function TrainingDashboard() {
  const { trainingState } = useStudioStore();
  const { metrics, status, currentEpoch } = trainingState;
  const latest = metrics[metrics.length - 1];
  const totalEpochs = useStudioStore(s => s.config.epochs);

  const progress = totalEpochs > 0 ? ((currentEpoch + 1) / totalEpochs) * 100 : 0;

  return (
    <div>
      <div className="panel-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span>Training</span>
        <span className={`badge ${
          status === 'training' ? 'badge-success' :
          status === 'paused' ? 'badge-warning' :
          status === 'error' ? 'badge-error' :
          status === 'completed' ? 'badge-info' :
          'badge-accent'
        }`}>
          {status === 'training' && '● '}{status}
        </span>
      </div>
      <div className="panel-content">
        {status === 'training' && (
          <div style={{ marginBottom: 12 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
              <span className="label">Epoch {currentEpoch + 1} / {totalEpochs}</span>
              <span className="label">{progress.toFixed(0)}%</span>
            </div>
            <div className="progress-bar">
              <div className="progress-fill" style={{ width: `${Math.min(progress, 100)}%` }} />
            </div>
          </div>
        )}

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginBottom: 12 }}>
          <MetricCard
            value={latest ? latest.loss.toFixed(4) : '—'}
            label="Loss"
            color="var(--accent)"
          />
          <MetricCard
            value={latest ? `${(latest.accuracy * 100).toFixed(1)}%` : '—'}
            label="Accuracy"
            color="var(--success)"
          />
          <MetricCard
            value={latest ? `${(latest.epochTime / 1000).toFixed(1)}s` : '—'}
            label="Epoch Time"
          />
          <MetricCard
            value={latest ? `${(latest.totalTime / 1000).toFixed(0)}s` : '—'}
            label="Total Time"
          />
        </div>

        {metrics.length > 1 && <LossChart />}

        {status === 'idle' && metrics.length === 0 && (
          <div className="empty-state" style={{ height: 100 }}>
            <p>Configure your model and click Train</p>
          </div>
        )}
      </div>
    </div>
  );
}
