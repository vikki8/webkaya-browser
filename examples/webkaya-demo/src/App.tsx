import { useState } from 'react';
import { IsolatedOrchestrator, type AgentRun } from '@webkaya/sandbox';
import { ClaudeProvider } from '@webkaya/sandbox/llm';

const workerFactory = () =>
  new Worker(new URL('../../../src/runtime/worker/worker-entry.ts', import.meta.url), { type: 'module' });

// The task: total these sales by region. Split into 3 shards of 3 rows.
const DATA = [
  { region: 'EMEA', amount: 95 }, { region: 'APAC', amount: 110 }, { region: 'AMER', amount: 120 },
  { region: 'EMEA', amount: 70 }, { region: 'APAC', amount: 60 }, { region: 'AMER', amount: 80 },
  { region: 'EMEA', amount: 40 }, { region: 'APAC', amount: 35 }, { region: 'AMER', amount: 55 },
];

type Role = 'mapper' | 'reducer' | 'reporter';

const REFERENCE: Record<Role, string> = {
  mapper: `
var rows = (ctx.args.input && ctx.args.input.rows) || [];
var shard = ctx.args.input.shard, sums = {};
for (var i=0;i<rows.length;i++){ var r=rows[i]; sums[r.region]=(sums[r.region]||0)+r.amount; }
var writes = {};
for (var k in sums) writes['partial:'+shard+':'+k] = sums[k];
return { writes: writes, output: sums };`,
  reducer: `
var read = ctx.args.read || {}, totals = {};
for (var key in read){ var region=key.split(':')[2]; totals[region]=(totals[region]||0)+Number(read[key]); }
var writes = {};
for (var r in totals) writes['total:'+r] = totals[r];
return { writes: writes, output: totals };`,
  reporter: `
var read = ctx.args.read || {}, grand=0, top=null, topV=-1, parts=[];
for (var key in read){ var region=key.split(':')[1]; var v=Number(read[key]); grand+=v; parts.push(region+' '+v); if(v>topV){topV=v;top=region;} }
return { writes: { 'report:summary': 'Total sales '+grand+' ('+parts.join(', ')+'). Top region: '+top+'.' }, output: { grand: grand } };`,
};

const ROLE_SYSTEM: Record<Role, string> = {
  mapper: `You are one of several isolated agents. Write the BODY of a function taking \`ctx\`.
\`ctx.args.input\` = { shard: string, rows: [{region, amount}] }. Sum amount by region.
You can ONLY share results by returning writes. Return EXACTLY: { writes: { "partial:<shard>:<region>": <sum> , ... }, output: <anything> }.
Example: return { writes: { "partial:A:EMEA": 95, "partial:A:APAC": 110 }, output: {} };
No imports — builtins and ctx only.`,
  reducer: `You are an isolated agent. Write the BODY of a function taking \`ctx\`.
\`ctx.args.read\` = an object whose keys look like "partial:<shard>:<region>" mapping to numeric strings (written by upstream agents). Sum them per region.
Return EXACTLY: { writes: { "total:<region>": <sum>, ... }, output: <anything> }.
Example: return { writes: { "total:EMEA": 205 }, output: {} };
No imports — builtins and ctx only.`,
  reporter: `You are an isolated agent. Write the BODY of a function taking \`ctx\`.
\`ctx.args.read\` = an object of keys "total:<region>" to numeric strings. Compute the grand total and the top region.
Return EXACTLY: { writes: { "report:summary": "<one readable sentence>" }, output: <anything> }.
No imports — builtins and ctx only.`,
};

interface Step { name: string; role: Role; phase: string; reads?: string[]; input?: unknown; }
const SHARDS = [DATA.slice(0, 3), DATA.slice(3, 6), DATA.slice(6, 9)];
const STEPS: Step[] = [
  ...SHARDS.map((rows, i) => ({ name: `agent ${i + 1}`, role: 'mapper' as Role, phase: 'map', input: { shard: 'ABC'[i], rows } })),
  { name: 'combiner', role: 'reducer', phase: 'reduce', reads: ['partial:*'] },
  { name: 'reporter', role: 'reporter', phase: 'report', reads: ['total:*'] },
];
const PHASES = [
  { id: 'map', title: '1 · Split the work', sub: 'Three agents each get a third of the rows and post their subtotals.' },
  { id: 'reduce', title: '2 · Combine', sub: 'One agent reads every subtotal off the board and adds them up by region.' },
  { id: 'report', title: '3 · Report', sub: 'One agent reads the totals and writes the final summary.' },
];

type Status = 'idle' | 'running' | 'done' | 'error';
interface AgentView {
  name: string; role: Role; phase: string; status: Status; note?: string; error?: string;
  summary?: string; reads?: Record<string, string | null>; writes?: Record<string, string>; code?: string; ai?: boolean;
}
interface BoardEntry { key: string; value: string; by: string; }

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const region = (key: string) => (key.startsWith('partial:') ? key.split(':')[2] : key.split(':')[1]);
const prefix = (k: string) => k.split(':')[0];

function describe(role: Role, run: AgentRun): string {
  const w = run.writes;
  if (role === 'mapper') {
    const parts = Object.entries(w).map(([k, v]) => `${region(k)} ${v}`);
    return `Summed its 3 rows → ${parts.join(', ')}`;
  }
  if (role === 'reducer') {
    const parts = Object.entries(w).map(([k, v]) => `${region(k)} ${v}`);
    return `Read ${Object.keys(run.reads).length} subtotals → totals ${parts.join(', ')}`;
  }
  return `Read the totals and wrote the summary`;
}

export function App() {
  const init = (): AgentView[] => STEPS.map((s) => ({ name: s.name, role: s.role, phase: s.phase, status: 'idle' }));
  const [agents, setAgents] = useState<AgentView[]>(init);
  const [board, setBoard] = useState<BoardEntry[]>([]);
  const [running, setRunning] = useState(false);
  const [report, setReport] = useState<string | null>(null);
  const [active, setActive] = useState<string | null>(null);
  const [apiKey, setApiKey] = useState('');

  const patch = (name: string, p: Partial<AgentView>) =>
    setAgents((prev) => prev.map((a) => (a.name === name ? { ...a, ...p } : a)));

  async function run() {
    if (running) return;
    setRunning(true); setReport(null); setBoard([]); setAgents(init());

    const provider = apiKey.trim().length > 12 ? new ClaudeProvider({ apiKey: apiKey.trim(), dangerouslyAllowBrowser: true }) : null;
    const orch = new IsolatedOrchestrator({ runtime: 'worker', workerFactory });
    const boardSoFar: BoardEntry[] = [];

    for (const phase of PHASES) {
      setActive(phase.id);
      for (const step of STEPS.filter((s) => s.phase === phase.id)) {
        patch(step.name, { status: 'running', note: provider ? 'an agent is writing its code…' : undefined });

        let handler = REFERENCE[step.role]; let ai = false;
        if (provider) {
          try {
            const gen = await provider.generateCode({ system: ROLE_SYSTEM[step.role], prompt: 'Write the function body.' });
            handler = gen.code; ai = true;
          } catch { patch(step.name, { note: 'Claude unavailable — using built-in code' }); }
        }
        await sleep(350);

        let r: AgentRun | null = null;
        try { r = await orch.runAgent({ name: step.name, handler, reads: step.reads, input: step.input }, phase.id); }
        catch (e) { patch(step.name, { status: 'error', error: (e as Error).message }); continue; }

        // If Claude produced code that wrote nothing, fall back to the built-in so the demo always works.
        if (ai && r.ok && Object.keys(r.writes).length === 0) {
          patch(step.name, { note: 'Claude’s code produced no result — using built-in' });
          r = await orch.runAgent({ name: step.name, handler: REFERENCE[step.role], reads: step.reads, input: step.input }, phase.id);
          ai = false; handler = REFERENCE[step.role];
        }

        if (!r.ok) { patch(step.name, { status: 'error', error: r.error, code: handler }); continue; }

        for (const [key, value] of Object.entries(r.writes)) boardSoFar.push({ key, value, by: step.name });
        setBoard(boardSoFar.slice().sort((a, b) => a.key.localeCompare(b.key)));
        patch(step.name, { status: 'done', summary: describe(step.role, r), reads: r.reads, writes: r.writes, code: handler, ai, note: undefined });
        await sleep(250);
      }
    }
    setActive(null);
    setReport(orch.board.get('report:summary'));
    setRunning(false);
  }

  return (
    <div className="app">
      <div className="accent" />
      <header>
        <div className="wordmark">WebKaya</div>
        <h1>A team of agents that never talk to each other</h1>
        <p className="lede">
          Five agents add up some sales figures together. Each one runs <strong>fully isolated</strong> —
          it can’t see this page and can’t call the other agents. The only thing they share is a
          <strong> blackboard</strong>: each agent reads what the others left there, does its bit, and
          writes its result back. Watch the board fill in.
        </p>
      </header>

      <section className="panel data">
        <h2>The data · 9 sales records</h2>
        <div className="rows">
          {DATA.map((d, i) => (
            <span className={`pill shard-${Math.floor(i / 3)}`} key={i}>{d.region} {d.amount}</span>
          ))}
        </div>
        <div className="muted">Colour = which agent gets that row. Goal: total by region.</div>
      </section>

      <section className="controls panel">
        <div className="row">
          <button className="cta" onClick={run} disabled={running}>{running ? 'Running…' : '▶  Run'}</button>
          <input className="key" type="password" value={apiKey} onChange={(e) => setApiKey(e.target.value)}
                 placeholder="Optional: Anthropic key — let each agent write its own code" autoComplete="off" />
        </div>
      </section>

      <div className="grid">
        <section className="flow">
          {PHASES.map((phase, pi) => (
            <div key={phase.id}>
              <div className={`phase ${active === phase.id ? 'active' : ''}`}>
                <div className="phase-head"><span className="phase-title">{phase.title}</span><span className="phase-sub">{phase.sub}</span></div>
                <div className="cards">
                  {agents.filter((a) => a.phase === phase.id).map((a) => (
                    <div className={`card ${a.status}`} key={a.name}>
                      <div className="card-top">
                        <span className="dot" />
                        <span className="name">{a.name}</span>
                        {a.ai && <span className="badge ai">Claude wrote this</span>}
                        <span className="tag">isolated</span>
                      </div>
                      <div className="card-body">
                        {a.status === 'idle' && <span className="muted">waiting</span>}
                        {a.status === 'running' && <span className="muted">{a.note ?? 'working…'}</span>}
                        {a.status === 'error' && <span className="err">✗ {a.error ?? 'failed'}</span>}
                        {a.status === 'done' && (
                          <>
                            <div className="summary">{a.summary}</div>
                            {a.note && <div className="muted small">{a.note}</div>}
                            <details>
                              <summary>details</summary>
                              {a.reads && Object.keys(a.reads).length > 0 && (
                                <div className="io"><b>read</b> {Object.entries(a.reads).map(([k, v]) => <code key={k}>{k}={v}</code>)}</div>
                              )}
                              <div className="io"><b>wrote</b> {Object.entries(a.writes ?? {}).map(([k, v]) => <code key={k}>{k}={v}</code>)}</div>
                              {a.code && <pre>{a.code.trim()}</pre>}
                            </details>
                          </>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
              {pi < PHASES.length - 1 && <div className="connector" aria-hidden />}
            </div>
          ))}
        </section>

        <section className="board panel">
          <h2>Shared blackboard <span className="count">{board.length}</span></h2>
          {board.length === 0 ? (
            <div className="muted">empty — press Run</div>
          ) : (
            board.map((e) => (
              <div className={`kv ${prefix(e.key)}`} key={e.key}>
                <span className="k">{e.key}</span>
                <span className="kv-right"><span className="v">{e.value}</span><span className="by">{e.by}</span></span>
              </div>
            ))
          )}
        </section>
      </div>

      {report && <section className="panel result"><h2>Result</h2><div>{report}</div></section>}
    </div>
  );
}
