import React from 'react';
import Head from 'next/head';
import { LayerPalette } from '../ui/components/LayerPalette';
import { PipelineCanvas } from '../ui/components/PipelineCanvas';
import { TrainingDashboard } from '../ui/components/TrainingDashboard';
import { ConfigPanel } from '../ui/components/ConfigPanel';
import { HardwareMonitor } from '../ui/components/HardwareMonitor';
import { LogConsole } from '../ui/components/LogConsole';
import { useTrainingWorker } from '../ui/hooks/useTrainingWorker';
import { useStudioStore } from '../ui/store';

export default function Studio() {
  const { initAndStart, pause, resume, stop, exportModel } = useTrainingWorker();
  const status = useStudioStore(s => s.trainingState.status);
  const layerCount = useStudioStore(s => s.graph.layers.length);

  return (
    <>
      <Head>
        <title>WebKaya — Local-First ML Studio</title>
        <meta name="description" content="Train neural networks directly in your browser" />
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet" />
      </Head>

      <div className="studio-layout">
        {/* Header */}
        <header className="header">
          <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
            <h1>WebKaya</h1>
            <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>Local-First ML Studio</span>
          </div>
          <div className="header-controls">
            {status === 'idle' || status === 'completed' || status === 'error' ? (
              <button
                className="btn btn-primary"
                onClick={initAndStart}
                disabled={layerCount === 0}
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M8 5v14l11-7z"/>
                </svg>
                Train
              </button>
            ) : status === 'training' ? (
              <>
                <button className="btn btn-secondary" onClick={pause}>
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z"/>
                  </svg>
                  Pause
                </button>
                <button className="btn btn-danger" onClick={stop}>Stop</button>
              </>
            ) : status === 'paused' ? (
              <>
                <button className="btn btn-primary" onClick={resume}>
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M8 5v14l11-7z"/>
                  </svg>
                  Resume
                </button>
                <button className="btn btn-danger" onClick={stop}>Stop</button>
              </>
            ) : status === 'loading' ? (
              <button className="btn btn-secondary" disabled>
                <span className="animate-spin" style={{ display: 'inline-block', width: 14, height: 14, border: '2px solid var(--text-muted)', borderTopColor: 'var(--accent)', borderRadius: '50%' }} />
                Loading...
              </button>
            ) : null}

            {(status === 'completed' || status === 'paused') && (
              <button className="btn btn-secondary" onClick={exportModel}>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                  <polyline points="7 10 12 15 17 10"/>
                  <line x1="12" y1="15" x2="12" y2="3"/>
                </svg>
                Export .kaya
              </button>
            )}
          </div>
        </header>

        {/* Left Sidebar - Layer Palette */}
        <div className="sidebar-left">
          <LayerPalette />
        </div>

        {/* Main Area - Pipeline Canvas */}
        <div className="main-area">
          <PipelineCanvas />
        </div>

        {/* Right Sidebar - Config + Hardware */}
        <div className="sidebar-right">
          <ConfigPanel />
          <div style={{ marginTop: 12 }}>
            <HardwareMonitor />
          </div>
        </div>

        {/* Bottom Panel - Charts + Logs */}
        <div className="bottom-panel">
          <TrainingDashboard />
          <LogConsole />
        </div>
      </div>
    </>
  );
}
