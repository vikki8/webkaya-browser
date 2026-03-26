import React, { useRef, useEffect } from 'react';
import { useStudioStore } from '../store';

export function LogConsole() {
  const logs = useStudioStore(s => s.logs);
  const clearLogs = useStudioStore(s => s.clearLogs);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <div className="panel-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span>Console</span>
        <button className="btn btn-sm btn-secondary" onClick={clearLogs} style={{ padding: '2px 8px' }}>
          Clear
        </button>
      </div>
      <div className="log-container" style={{ flex: 1 }}>
        {logs.length === 0 ? (
          <span style={{ color: 'var(--text-muted)' }}>Waiting for events...</span>
        ) : (
          logs.map((log, i) => (
            <div key={i} className="log-entry" style={{
              color: log.includes('ERROR') ? 'var(--error)' :
                     log.includes('complete') ? 'var(--success)' :
                     'var(--text-secondary)'
            }}>
              {log}
            </div>
          ))
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
