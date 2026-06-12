// Local Data Analyst demo — agent-generated Python over a local CSV, in-browser.
//
// Run `npm run build` at the repo root first (this imports from ../../dist),
// then serve the repo over http and open examples/local-data-analyst/index.html.
import { planQuestion } from '../../dist/python/planner.js';
import { PythonRunner, loadPyodideRuntime } from '../../dist/python/pyodide-runner.js';
import { ClaudeProvider, CodeAnalyst } from '../../dist/llm/index.js';

const $ = (id) => document.getElementById(id);
let analyst = null; // CodeAnalyst when an API key is provided, else null (built-in planner)
const SAMPLE_CSV = `country,region,revenue,units
USA,Americas,120,40
Canada,Americas,80,30
Germany,EMEA,95,25
France,EMEA,70,20
Japan,APAC,110,35
Australia,APAC,60,15`;

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
  setStatus('ready');
  return runner;
}

async function loadCsv(csvText, label) {
  await ensureRuntime();
  await runner.loadDataframe('df', csvText);
  columns = parseColumns(csvText);
  const rowCount = Math.max(0, csvText.trim().split(/\r?\n/).length - 1);
  $('rows').textContent = rowCount.toLocaleString();
  $('file-status').textContent = `${label} — ${rowCount} rows, columns: ${columns.join(', ')}`;
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

function logRun(question, plan, result) {
  runs += 1;
  const entry = document.createElement('div');
  entry.className = 'log-entry';
  entry.innerHTML =
    `<strong>#${runs}</strong> ${question} ` +
    `<span class="${result.ok ? 'ok' : 'err'}">(${result.ok ? 'ok' : 'error'}, ${Math.round(result.durationMs)}ms)</span>` +
    `<br><span class="muted">${plan.explanation}</span>`;
  $('log').prepend(entry);
}

async function planFor(question) {
  // With a key: Claude writes the Python (the question text is sent to the
  // Anthropic API — never the data). Otherwise the built-in planner runs fully
  // offline.
  if (analyst) {
    setStatus('asking Claude to write the Python…');
    return analyst.plan(question, columns);
  }
  return planQuestion(question, columns);
}

async function ask(question) {
  if (!runner || !question.trim()) return;
  let plan;
  try {
    plan = await planFor(question);
  } catch (error) {
    $('result-card').hidden = false;
    $('code').textContent = '';
    $('result').innerHTML = `<span class="err">Code generation failed: ${error.message}</span>`;
    setStatus('ready');
    return;
  }

  $('code').textContent = plan.code;
  $('result-card').hidden = false;
  setStatus('running Python locally…');
  let result = await runner.run(plan.code);

  // In Claude mode, give the model one shot at repairing a failing run.
  if (!result.ok && analyst) {
    setStatus('Python failed — asking Claude to fix it…');
    try {
      plan = await analyst.repair(question, columns, plan.code, result.error || 'unknown error');
      $('code').textContent = plan.code;
      result = await runner.run(plan.code);
    } catch (error) {
      result = { ...result, error: `repair failed: ${error.message}` };
    }
  }

  $('last-ms').textContent = Math.round(result.durationMs || 0);
  if (result.ok) {
    renderResult(result.value);
  } else {
    $('result').innerHTML = `<span class="err">${result.error}</span>`;
  }
  logRun(question, plan, result);
  // The privacy invariant: the data never left the device (only the question
  // text reaches Claude in key mode).
  $('bytes-sent').textContent = '0';
  setStatus('ready');
}

$('apikey').addEventListener('change', () => {
  const key = $('apikey').value.trim();
  if (key) {
    const provider = new ClaudeProvider({ apiKey: key, dangerouslyAllowBrowser: true });
    analyst = new CodeAnalyst({ provider, language: 'python' });
    $('planner-mode').textContent = 'Planner: Claude (opus-4-8)';
    $('planner-mode').className = 'ok';
  } else {
    analyst = null;
    $('planner-mode').textContent = 'Planner: built-in (no key)';
    $('planner-mode').className = 'muted';
  }
});

$('pick').addEventListener('click', async () => {
  try {
    if (window.showOpenFilePicker) {
      const [handle] = await window.showOpenFilePicker({
        types: [{ description: 'CSV', accept: { 'text/csv': ['.csv'] } }],
      });
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

$('sample').addEventListener('click', () => loadCsv(SAMPLE_CSV, 'sample-sales.csv'));
$('ask').addEventListener('click', () => ask($('question').value));
$('question').addEventListener('keydown', (e) => { if (e.key === 'Enter') ask($('question').value); });
document.querySelectorAll('.examples button').forEach((btn) =>
  btn.addEventListener('click', () => {
    $('question').value = btn.dataset.q;
    ask(btn.dataset.q);
  })
);

// Warm up the runtime in the background so the first question is fast.
ensureRuntime().catch((e) => setStatus('runtime load failed: ' + e));
