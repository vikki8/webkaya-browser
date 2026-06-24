// Private Data Analyst — the killer demo for WebKaya's client-side thesis.
//
// An AI writes Python; it runs in THIS browser tab over the user's file; the
// data never leaves the device — and the live egress ledger (privacy.js) proves
// it by monitoring every network request the page makes.
//
// Run `npm run build` at the repo root first (this imports from ../../dist),
// then serve the repo over http and open this folder.
import './privacy.js';
import { onLedger, watchDataset } from './privacy.js';
import { planQuestion } from '../../dist/python/planner.js';
import { PythonRunner, loadPyodideRuntime } from '../../dist/python/pyodide-runner.js';
import { DataAgent } from '../../dist/python/data-agent.js';
import { ClaudeProvider, CodeAnalyst } from '../../dist/llm/index.js';

const $ = (id) => document.getElementById(id);

// A deliberately sensitive sample, so the privacy stakes are obvious at a glance.
const SAMPLE_CSV = `employee,department,base_salary,bonus,ssn_last4
Alice Chen,Engineering,185000,42000,4821
Bob Martins,Engineering,162000,28000,7193
Carla Diaz,Sales,141000,55000,3360
Dan O'Neil,Sales,138000,49000,9047
Erin Walsh,Marketing,128000,21000,5582
Frank Yu,Marketing,121000,18000,6614
Grace Lee,Engineering,204000,61000,2208
Hassan Ali,Sales,133000,47000,8875`;

let analyst = null;   // CodeAnalyst when an API key is set
let dataAgent = null; // DataAgent (Claude + Pyodide loop) when both ready
let runner = null;
let columns = [];
let runs = 0;

function setStatus(text) { $('status').textContent = text; }

function parseColumns(csvText) {
  const header = (csvText.split(/\r?\n/)[0] || '').trim();
  return header ? header.split(',').map((c) => c.trim()) : [];
}

async function ensureRuntime() {
  if (runner) return runner;
  setStatus('loading Python runtime…');
  const py = await loadPyodideRuntime();
  runner = new PythonRunner(py);
  refreshAgent();
  setStatus('ready');
  return runner;
}

function refreshAgent() {
  dataAgent = analyst && runner ? new DataAgent(analyst, runner, { maxAttempts: 3 }) : null;
}

async function loadCsv(csvText, label) {
  watchDataset(csvText); // tell the egress monitor what to watch for
  await ensureRuntime();
  await runner.loadDataframe('df', csvText);
  columns = parseColumns(csvText);
  const rowCount = Math.max(0, csvText.trim().split(/\r?\n/).length - 1);
  $('file-status').textContent = `${label} — ${rowCount} rows · ${columns.join(', ')}`;
  $('question').disabled = false;
  $('ask').disabled = false;
}

function renderResult(value) {
  const el = $('result');
  if (Array.isArray(value) && value.length && typeof value[0] === 'object') {
    const cols = Object.keys(value[0]);
    el.innerHTML =
      '<table><thead><tr>' + cols.map((c) => `<th>${c}</th>`).join('') + '</tr></thead><tbody>' +
      value.map((row) => '<tr>' + cols.map((c) => `<td>${row[c]}</td>`).join('') + '</tr>').join('') +
      '</tbody></table>';
  } else if (value && typeof value === 'object') {
    el.innerHTML = '<pre>' + JSON.stringify(value, null, 2) + '</pre>';
  } else {
    el.innerHTML = '<pre>' + String(value) + '</pre>';
  }
}

function logRun(question, explanation, result) {
  runs += 1;
  const entry = document.createElement('div');
  entry.className = 'log-entry';
  entry.innerHTML =
    `<strong>#${runs}</strong> ${question} ` +
    `<span class="${result.ok ? 'ok' : 'err'}">(${result.ok ? 'ok' : 'error'}, ${Math.round(result.durationMs || 0)}ms)</span>` +
    `<br><span class="muted">${explanation || ''}</span>`;
  $('log').prepend(entry);
}

async function askWithAgent(question) {
  const outcome = await dataAgent.ask(question, columns);
  const durationMs = outcome.attempts.reduce((sum, a) => sum + (a.result.durationMs || 0), 0);
  const note = outcome.attempts.length > 1 ? ` (${outcome.attempts.length} attempts)` : '';
  return {
    code: outcome.code,
    explanation: outcome.explanation + note,
    result: { ok: outcome.ok, value: outcome.value, error: outcome.attempts.at(-1)?.result.error, durationMs },
  };
}

async function ask(question) {
  if (!runner || !question.trim()) return;
  $('result-card').hidden = false;

  let attempt;
  try {
    if (dataAgent) {
      setStatus('Claude is writing the analysis…');
      attempt = await askWithAgent(question);
    } else {
      const plan = planQuestion(question, columns);
      setStatus('running locally…');
      const result = await runner.run(plan.code);
      attempt = { code: plan.code, explanation: plan.explanation, result };
    }
  } catch (error) {
    $('code').textContent = '';
    $('result').innerHTML = `<span class="err">Code generation failed: ${error.message}</span>`;
    setStatus('ready');
    return;
  }

  $('code').textContent = attempt.code;
  $('timing').textContent = `Ran locally in ${Math.round(attempt.result.durationMs || 0)} ms.`;
  if (attempt.result.ok) renderResult(attempt.result.value);
  else $('result').innerHTML = `<span class="err">${attempt.result.error}</span>`;
  logRun(question, attempt.explanation, attempt.result);
  setStatus('ready');
}

// --- The hero: render the live privacy ledger on every network event ---
onLedger((s) => {
  const ledger = $('ledger');
  const breach = s.dataLeakCount > 0;
  ledger.classList.toggle('breach', breach);
  $('data-bytes').textContent = breach ? `${s.dataBytesSent} bytes` : '0 bytes';

  $('verify').innerHTML = breach
    ? `<span class="err">⚠ ${s.dataLeakCount} request(s) contained your data.</span>`
    : `<span class="ok">✓ verified against ${s.requestCount} outbound request${s.requestCount === 1 ? '' : 's'}</span> — none contained your data.` +
      (s.questionBytes ? ` Claude received ${s.questionBytes.toLocaleString()} bytes (your question + instructions, no data).` : '');

  $('egress').innerHTML = s.records.length
    ? s.records.map((r) =>
        `<tr><td>${r.method}</td><td>${r.host}</td><td>${r.purpose}</td>` +
        `<td>${r.bytes ? r.bytes.toLocaleString() + ' B' : '—'}</td></tr>`).join('')
    : '<tr><td class="muted" colspan="4">No outbound requests yet.</td></tr>';

  $('contrast').innerHTML = s.datasetBytes
    ? `A typical hosted code interpreter uploads your whole file (<b>${(s.datasetBytes / 1024).toFixed(1)} KB</b>) ` +
      `to its servers on every question. Here, that number is <b style="color:var(--accent)">0 KB</b>.`
    : 'Load a dataset and the contrast appears here.';
});

$('apikey').addEventListener('change', () => {
  const key = $('apikey').value.trim();
  if (key) {
    analyst = new CodeAnalyst({ provider: new ClaudeProvider({ apiKey: key, dangerouslyAllowBrowser: true }), language: 'python' });
    $('planner-mode').textContent = 'Engine: Claude (opus-4-8) + repair loop';
    $('planner-mode').className = 'ok';
  } else {
    analyst = null;
    $('planner-mode').textContent = 'Engine: built-in (no key)';
    $('planner-mode').className = 'muted';
  }
  refreshAgent();
});

$('pick').addEventListener('click', async () => {
  try {
    if (window.showOpenFilePicker) {
      const [handle] = await window.showOpenFilePicker({ types: [{ description: 'CSV', accept: { 'text/csv': ['.csv'] } }] });
      const file = await handle.getFile();
      await loadCsv(await file.text(), file.name);
    } else {
      const input = document.createElement('input');
      input.type = 'file';
      input.accept = '.csv,text/csv';
      input.onchange = async () => {
        const file = input.files?.[0];
        if (file) await loadCsv(await file.text(), file.name);
      };
      input.click();
    }
  } catch (error) {
    if (error?.name !== 'AbortError') setStatus('file error: ' + error);
  }
});

$('sample').addEventListener('click', () => loadCsv(SAMPLE_CSV, 'employee_comp.csv'));
$('ask').addEventListener('click', () => ask($('question').value));
$('question').addEventListener('keydown', (e) => { if (e.key === 'Enter') ask($('question').value); });
document.querySelectorAll('.examples button').forEach((btn) =>
  btn.addEventListener('click', () => { $('question').value = btn.dataset.q; ask(btn.dataset.q); })
);

// Warm up the runtime so the first question is fast.
ensureRuntime().catch((e) => setStatus('runtime load failed: ' + e));
