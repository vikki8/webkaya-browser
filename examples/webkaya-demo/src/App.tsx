import { useState } from 'react';
import { IsolatedOrchestrator, type AgentRun } from '@webkaya/sandbox';
import { ClaudeProvider } from '@webkaya/sandbox/llm';

// Real Web Worker per agent — reliable browser isolation. Each agent runs on
// its own thread with no access to this page (no window/DOM), and agents never
// reach each other.
const workerFactory = () =>
  new Worker(new URL('../../../src/runtime/worker/worker-entry.ts', import.meta.url), { type: 'module' });

const DATA = [
  { region: 'EMEA', amount: 95 }, { region: 'APAC', amount: 110 }, { region: 'AMER', amount: 120 },
  { region: 'EMEA', amount: 70 }, { region: 'APAC', amount: 60 }, { region: 'AMER', amount: 80 },
  { region: 'EMEA', amount: 40 }, { region: 'APAC', amount: 35 }, { region: 'AMER', amount: 55 },
];

type Role = 'mapper' | 'reducer' | 'reporter';

// Fallback handlers, used when no API key is set. With a key, Claude writes these.
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
var read = ctx.args.read || {}, grand=0, top=null, topV=-1, n=0;
for (var key in read){ var region=key.split(':')[1]; var v=Number(read[key]); grand+=v; n++; if(v>topV){topV=v;top=region;} }
return { writes: { 'report:summary': 'Total '+grand+' across '+n+' regions; top '+top+' ('+topV+')' }, output: { grand: grand, top: top } };`,
};

const ROLE_SYSTEM: Record<Role, string> = {
  mapper: `Write a WebKaya agent (function body, arg \`ctx\`). \`ctx.args.input\` is { shard: string, rows: [{region, amount}] }. Sum amounts by region. Return { writes, output } where writes maps 'partial:<shard>:<region>' to the sum (writes is the only way to share results). No imports; builtins and ctx only. Keep it short.`,
  reducer: `Write a WebKaya agent (function body, arg \`ctx\`). \`ctx.args.read\` is an object of blackboard keys named 'partial:<shard>:<region>' to numeric string values written by upstream agents. Sum them per region. Return { writes, output } where writes maps 'total:<region>' to the regional total. No imports; builtins and ctx only. Keep it short.`,
  reporter: `Write a WebKaya agent (function body, arg \`ctx\`). \`ctx.args.read\` is an object of keys 'total:<region>' to numeric string values. Compute the grand total and the top region. Return { writes: { 'report:summary': <a one-line string> }, output: {...} }. No imports; builtins and ctx only. Keep it short.`,
};

interface Step { name: string; role: Role; phase: string; reads?: string[]; input?: unknown; }

function buildSteps(): { phases: { id: string; label: string }[]; steps: Step[] } {
  const shards = [DATA.slice(0, 3), DATA.slice(3, 6), DATA.slice(6, 9)];
  const steps: Step[] = [
    ...shards.map((rows, i) => ({ name: `mapper-${'ABC'[i]}`, role: 'mapper' as Role, phase: 'map', input: { shard: 'ABC'[i], rows } })),
    { name: 'reducer', role: 'reducer', phase: 'reduce', reads: ['partial:*'] },
    { name: 'reporter', role: 'reporter', phase: 'report', reads: ['total:*'] },
  ];
  return {
    phases: [
      { id: 'map', label: 'Map · 3 agents, one per shard' },
      { id: 'reduce', label: 'Reduce · reads the mappers’ writes' },
      { id: 'report', label: 'Report · reads the reducer’s writes' },
    ],
    steps,
  };
}

type Status = 'idle' | 'running' | 'done' | 'error';
interface AgentView { name: string; role: Role; phase: string; status: Status; note?: string; error?: string; reads?: number; writes?: string[]; ai?: boolean; }
interface KV { key: string; value: string; }

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const prefix = (k: string) => k.split(':')[0];

export function App() {
  const { phases, steps } = buildSteps();
  const initial = (): AgentView[] => steps.map((s) => ({ name: s.name, role: s.role, phase: s.phase, status: 'idle' }));
  const [agents, setAgents] = useState<AgentView[]>(initial);
  const [board, setBoard] = useState<KV[]>([]);
  const [running, setRunning] = useState(false);
  const [report, setReport] = useState<string | null>(null);
  const [activePhase, setActivePhase] = useState<string | null>(null);
  const [apiKey, setApiKey] = useState('');
  const [tokens, setTokens] = useState<{ in: number; out: number } | null>(null);

  const patch = (name: string, p: Partial<AgentView>) =>
    setAgents((prev) => prev.map((a) => (a.name === name ? { ...a, ...p } : a)));

  async function run() {
    if (running) return;
    setRunning(true);
    setReport(null);
    setBoard([]);
    setTokens(null);
    setAgents(initial());

    const useClaude = apiKey.trim().length > 12;
    const provider = useClaude ? new ClaudeProvider({ apiKey: apiKey.trim(), dangerouslyAllowBrowser: true }) : null;
    let tin = 0, tout = 0;

    const orch = new IsolatedOrchestrator({ runtime: 'worker', workerFactory });
    const refresh = () => setBoard(orch.board.keys().sort().map((key) => ({ key, value: orch.board.get(key) ?? '' })));

    for (const phase of phases) {
      setActivePhase(phase.id);
      for (const step of steps.filter((s) => s.phase === phase.id)) {
        patch(step.name, { status: 'running' });

        let handler = REFERENCE[step.role];
        let ai = false;
        if (provider) {
          patch(step.name, { note: 'writing its own code…' });
          try {
            const gen = await provider.generateCode({ system: ROLE_SYSTEM[step.role], prompt: 'Write the agent.' });
            handler = gen.code;
            tin += gen.usage?.inputTokens ?? 0;
            tout += gen.usage?.outputTokens ?? 0;
            ai = true;
          } catch (e) {
            patch(step.name, { note: 'Claude unavailable — using built-in' });
          }
        }

        await sleep(250);
        let r: AgentRun;
        try {
          r = await orch.runAgent({ name: step.name, handler, reads: step.reads, input: step.input }, phase.id);
        } catch (e) {
          patch(step.name, { status: 'error', error: (e as Error).message, ai });
          continue;
        }
        patch(step.name, {
          status: r.ok ? 'done' : 'error',
          error: r.ok ? undefined : r.error,
          reads: Object.keys(r.reads).length,
          writes: Object.keys(r.writes),
          ai,
          note: undefined,
        });
        refresh();
        await sleep(180);
      }
    }

    setActivePhase(null);
    setReport(orch.board.get('report:summary'));
    if (useClaude) setTokens({ in: tin, out: tout });
    setRunning(false);
  }

  return (
    <div className="app">
      <div className="accent" />
      <header>
        <div className="wordmark">WebKaya</div>
        <h1>Isolated agents, one shared blackboard</h1>
        <p className="lede">
          A map → reduce → report pipeline. Each agent runs in its own isolated worker — no access
          to this page, no way to call another agent. They coordinate only by reading and writing a
          shared blackboard.
        </p>
      </header>

      <section className="controls panel">
        <div className="row">
          <button className="cta" onClick={run} disabled={running}>{running ? 'Running…' : '▶  Run pipeline'}</button>
          <input
            className="key"
            type="password"
            placeholder="Anthropic API key — let the agents write their own code (optional)"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            autoComplete="off"
          />
        </div>
        <div className="stat-row">
          <span className="stat">engine · Web Worker (isolated)</span>
          <span className="stat">{apiKey.trim().length > 12 ? 'agents · written by Claude' : 'agents · built-in'}</span>
          {tokens && <span className="stat">tokens · {tokens.in.toLocaleString()} in / {tokens.out.toLocaleString()} out</span>}
        </div>
      </section>

      <div className="grid">
        <section className="flow">
          {phases.map((phase, pi) => (
            <div key={phase.id}>
              <div className={`phase ${activePhase === phase.id ? 'active' : ''}`}>
                <div className="phase-head">{phase.label}</div>
                <div className="cards">
                  {agents.filter((a) => a.phase === phase.id).map((a) => (
                    <div className={`card ${a.status}`} key={a.name}>
                      <div className="card-top">
                        <span className="dot" />
                        <span className="name">{a.name}</span>
                        {a.ai && <span className="badge ai">Claude</span>}
                        <span className="badge">isolated</span>
                      </div>
                      <div className="card-meta">
                        {a.status === 'idle' && 'waiting'}
                        {a.status === 'running' && (a.note ?? 'running…')}
                        {a.status === 'done' && `read ${a.reads} · wrote ${a.writes?.length ?? 0}`}
                        {a.status === 'error' && <span className="err">✗ {a.error ?? 'failed'}</span>}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
              {pi < phases.length - 1 && <div className="connector" aria-hidden />}
            </div>
          ))}
        </section>

        <section className="board panel">
          <h2>Shared blackboard <span className="count">{board.length}</span></h2>
          {board.length === 0 ? (
            <div className="empty">empty — run the pipeline</div>
          ) : (
            board.map((e) => (
              <div className={`kv ${prefix(e.key)}`} key={e.key}>
                <span className="k">{e.key}</span>
                <span className="v">{e.value}</span>
              </div>
            ))
          )}
          {report && <div className="result">{report}</div>}
        </section>
      </div>
    </div>
  );
}
