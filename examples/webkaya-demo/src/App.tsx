import { useEffect, useRef, useState } from 'react';
import { DataAgent, PythonRunner, loadPyodideRuntime, planQuestion } from '@webkaya/sandbox/python';
import { ClaudeProvider, CodeAnalyst } from '@webkaya/sandbox/llm';
import { subscribe, watchDataset, type Ledger } from './privacy';

const SAMPLE = `employee,department,base_salary,bonus
Alice Chen,Engineering,185000,42000
Bob Martins,Engineering,162000,28000
Carla Diaz,Sales,141000,55000
Dan O'Neil,Sales,138000,49000
Erin Walsh,Marketing,128000,21000
Grace Lee,Engineering,204000,61000
Hassan Ali,Sales,133000,47000`;

const EXAMPLES = ['average base salary by department', 'highest total compensation', 'total bonus pool'];

interface RunView {
  code: string;
  value?: unknown;
  error?: string;
  ms: number;
  note?: string;
}

export function App() {
  const runnerRef = useRef<PythonRunner | null>(null);
  const loadingRef = useRef<Promise<PythonRunner> | null>(null);
  const analystRef = useRef<CodeAnalyst | null>(null);
  const agentRef = useRef<DataAgent | null>(null);

  const [status, setStatus] = useState('loading runtime…');
  const [columns, setColumns] = useState<string[]>([]);
  const [fileLabel, setFileLabel] = useState<string | null>(null);
  const [question, setQuestion] = useState('');
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<RunView | null>(null);
  const [ledger, setLedger] = useState<Ledger | null>(null);
  const [keySet, setKeySet] = useState(false);

  useEffect(() => subscribe(setLedger), []);
  useEffect(() => { void ensureRuntime(); }, []);

  function ensureRuntime(): Promise<PythonRunner> {
    if (!loadingRef.current) {
      loadingRef.current = (async () => {
        setStatus('loading Python runtime…');
        const runner = new PythonRunner(await loadPyodideRuntime());
        runnerRef.current = runner;
        refreshAgent();
        setStatus('ready');
        return runner;
      })();
    }
    return loadingRef.current;
  }

  function refreshAgent() {
    agentRef.current =
      analystRef.current && runnerRef.current
        ? new DataAgent(analystRef.current, runnerRef.current, { maxAttempts: 3 })
        : null;
  }

  async function loadCsv(text: string, label: string) {
    watchDataset(text);
    const runner = await ensureRuntime();
    await runner.loadDataframe('df', text);
    const cols = (text.split(/\r?\n/)[0] || '').split(',').map((c) => c.trim()).filter(Boolean);
    const rows = Math.max(0, text.trim().split(/\r?\n/).length - 1);
    setColumns(cols);
    setFileLabel(`${label} · ${rows} rows`);
  }

  function setApiKey(key: string) {
    const trimmed = key.trim();
    if (trimmed) {
      const provider = new ClaudeProvider({ apiKey: trimmed, dangerouslyAllowBrowser: true });
      analystRef.current = new CodeAnalyst({ provider, language: 'python' });
      setKeySet(true);
    } else {
      analystRef.current = null;
      setKeySet(false);
    }
    refreshAgent();
  }

  async function ask(q: string) {
    const query = q.trim();
    const runner = runnerRef.current;
    if (!query || busy || !runner) return;
    setBusy(true);
    try {
      if (agentRef.current) {
        setStatus('writing code…');
        const o = await agentRef.current.ask(query, columns);
        const ms = o.attempts.reduce((s, a) => s + (a.result.durationMs || 0), 0);
        setResult({
          code: o.code,
          value: o.ok ? o.value : undefined,
          error: o.ok ? undefined : o.attempts.at(-1)?.result.error,
          ms,
          note: o.attempts.length > 1 ? `${o.attempts.length} attempts` : undefined,
        });
      } else {
        const plan = planQuestion(query, columns);
        setStatus('running…');
        const r = await runner.run(plan.code);
        setResult({ code: plan.code, value: r.ok ? r.value : undefined, error: r.ok ? undefined : r.error, ms: r.durationMs });
      }
    } catch (e) {
      setResult({ code: '', error: (e as Error).message, ms: 0 });
    } finally {
      setBusy(false);
      setStatus('ready');
    }
  }

  const loaded = columns.length > 0;

  return (
    <div className="app">
      <header>
        <div className="wordmark">WebKaya</div>
        <div className="tagline">Ask questions about a CSV. The code runs in your browser.</div>
      </header>

      <section className="panel">
        <h2>Data</h2>
        <div className="row">
          <button onClick={() => loadCsv(SAMPLE, 'sample.csv')}>Use sample</button>
          <button className="ghost" onClick={pickFile}>Choose file…</button>
          {fileLabel && <span className="meta">{fileLabel}</span>}
        </div>
        {loaded && (
          <div className="cols">
            {columns.map((c) => <span key={c} className="col">{c}</span>)}
          </div>
        )}
      </section>

      <section className="panel">
        <h2>Ask</h2>
        <div className="row">
          <input
            type="text"
            placeholder="e.g. average base salary by department"
            value={question}
            disabled={!loaded || busy}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter') ask(question); }}
          />
          <button disabled={!loaded || busy} onClick={() => ask(question)}>{busy ? '…' : 'Ask'}</button>
        </div>
        <div className="cols" style={{ marginTop: '.7rem' }}>
          {EXAMPLES.map((q) => (
            <button key={q} className="chip" disabled={!loaded || busy} onClick={() => { setQuestion(q); ask(q); }}>{q}</button>
          ))}
        </div>
      </section>

      {result && (
        <section className="panel">
          <h2>Result{result.note ? ` · ${result.note}` : ''}</h2>
          <div className="answer">{result.error ? <span className="err">{result.error}</span> : renderValue(result.value)}</div>
          {result.code && (
            <details>
              <summary>Generated code · ran in {Math.round(result.ms)} ms</summary>
              <pre>{result.code}</pre>
            </details>
          )}
        </section>
      )}

      <section className="panel">
        <h2>Network</h2>
        {ledger && ledger.records.length > 0 ? (
          ledger.records.map((r, i) => (
            <div className="net-row" key={i}>
              <div>
                <div className="host">{r.method} {r.host}</div>
                <div className="purpose">{r.purpose}</div>
              </div>
              <div className="bytes">{r.bytes ? `${r.bytes.toLocaleString()} B` : '—'}</div>
            </div>
          ))
        ) : (
          <div className="meta">No outbound requests yet.</div>
        )}
        <div className="net-foot">
          Your data sent:{' '}
          {ledger && ledger.dataBytesSent > 0
            ? <span className="warn">{ledger.dataBytesSent} B</span>
            : <span className="ok">0 B</span>}
          {ledger && ledger.questionBytes > 0 && (
            <span className="meta"> · question to model: {ledger.questionBytes.toLocaleString()} B</span>
          )}
        </div>
      </section>

      <details className="panel" style={{ paddingTop: '.9rem' }}>
        <summary style={{ margin: 0 }}>Use Claude (optional)</summary>
        <div className="row" style={{ marginTop: '.8rem' }}>
          <input type="password" placeholder="Anthropic API key — sk-ant-…" autoComplete="off"
                 onChange={(e) => setApiKey(e.target.value)} />
          <span className="meta">{keySet ? 'on' : 'off'}</span>
        </div>
        <p className="meta" style={{ margin: '.5rem 0 0' }}>
          Claude writes the analysis code; only the question is sent. Used directly from this tab — use a revocable key.
        </p>
      </details>

      <div className="status">{status}</div>
    </div>
  );

  function pickFile() {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.csv,text/csv';
    input.onchange = async () => {
      const file = input.files?.[0];
      if (file) await loadCsv(await file.text(), file.name);
    };
    input.click();
  }
}

function renderValue(value: unknown) {
  if (Array.isArray(value) && value.length && typeof value[0] === 'object' && value[0] !== null) {
    const cols = Object.keys(value[0] as Record<string, unknown>);
    return (
      <table>
        <thead><tr>{cols.map((c) => <th key={c}>{c}</th>)}</tr></thead>
        <tbody>
          {(value as Record<string, unknown>[]).map((row, i) => (
            <tr key={i}>{cols.map((c) => <td key={c}>{String(row[c])}</td>)}</tr>
          ))}
        </tbody>
      </table>
    );
  }
  if (value && typeof value === 'object') return <pre>{JSON.stringify(value, null, 2)}</pre>;
  return <pre>{String(value)}</pre>;
}
