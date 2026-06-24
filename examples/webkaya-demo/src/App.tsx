import { useState } from 'react';
import { IsolatedOrchestrator, type AgentRun, type AgentSpec } from '@webkaya/sandbox';

// --- The scenario: a map → reduce → report pipeline run by isolated agents ---
// that coordinate only through a shared blackboard. Each agent's logic is plain
// guest JS; it could equally be written by an LLM.

const DATA = [
  { region: 'EMEA', amount: 95 }, { region: 'APAC', amount: 110 }, { region: 'AMER', amount: 120 },
  { region: 'EMEA', amount: 70 }, { region: 'APAC', amount: 60 }, { region: 'AMER', amount: 80 },
  { region: 'EMEA', amount: 40 }, { region: 'APAC', amount: 35 }, { region: 'AMER', amount: 55 },
];

const MAPPER = `
var rows = (ctx.args.input && ctx.args.input.rows) || [];
var shard = ctx.args.input.shard;
var sums = {};
for (var i = 0; i < rows.length; i++) { var r = rows[i]; sums[r.region] = (sums[r.region] || 0) + r.amount; }
var writes = {};
for (var k in sums) writes['partial:' + shard + ':' + k] = sums[k];
return { writes: writes, output: sums };
`;

const REDUCER = `
var read = ctx.args.read || {};
var totals = {};
for (var key in read) { var region = key.split(':')[2]; totals[region] = (totals[region] || 0) + Number(read[key]); }
var writes = {};
for (var r in totals) writes['total:' + r] = totals[r];
return { writes: writes, output: totals };
`;

const REPORTER = `
var read = ctx.args.read || {};
var grand = 0, top = null, topV = -1, n = 0;
for (var key in read) { var region = key.split(':')[1]; var v = Number(read[key]); grand += v; n++; if (v > topV) { topV = v; top = region; } }
return { writes: { 'report:summary': 'Total ' + grand + ' across ' + n + ' regions; top ' + top + ' (' + topV + ')' }, output: { grand: grand, top: top } };
`;

interface Phase { id: string; label: string; specs: AgentSpec[]; }

function buildPhases(): Phase[] {
  const shards = [DATA.slice(0, 3), DATA.slice(3, 6), DATA.slice(6, 9)];
  return [
    {
      id: 'map', label: 'Map · 3 isolated agents',
      specs: shards.map((rows, i) => ({
        name: `mapper-${'ABC'[i]}`, handler: MAPPER, input: { shard: 'ABC'[i], rows },
      })),
    },
    { id: 'reduce', label: 'Reduce · reads the mappers’ writes', specs: [{ name: 'reducer', handler: REDUCER, reads: ['partial:*'] }] },
    { id: 'report', label: 'Report · reads the reducer’s writes', specs: [{ name: 'reporter', handler: REPORTER, reads: ['total:*'] }] },
  ];
}

type Status = 'idle' | 'running' | 'done' | 'error';
interface AgentView { name: string; phase: string; status: Status; readCount?: number; writes?: string[]; }
interface BoardEntry { key: string; value: string; }

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const prefixOf = (k: string) => k.split(':')[0];

export function App() {
  const phases = buildPhases();
  const [agents, setAgents] = useState<AgentView[]>(
    phases.flatMap((p) => p.specs.map((s) => ({ name: s.name, phase: p.id, status: 'idle' as Status }))),
  );
  const [board, setBoard] = useState<BoardEntry[]>([]);
  const [running, setRunning] = useState(false);
  const [report, setReport] = useState<string | null>(null);
  const [activePhase, setActivePhase] = useState<string | null>(null);

  function patch(name: string, p: Partial<AgentView>) {
    setAgents((prev) => prev.map((a) => (a.name === name ? { ...a, ...p } : a)));
  }

  async function run() {
    if (running) return;
    setRunning(true);
    setReport(null);
    setBoard([]);
    setAgents((prev) => prev.map((a) => ({ ...a, status: 'idle', readCount: undefined, writes: undefined })));

    const orch = new IsolatedOrchestrator();
    const refreshBoard = () => setBoard(orch.board.keys().sort().map((key) => ({ key, value: orch.board.get(key) ?? '' })));

    for (const phase of phases) {
      setActivePhase(phase.id);
      for (const spec of phase.specs) {
        patch(spec.name, { status: 'running' });
        await sleep(380);
        let run: AgentRun;
        try {
          run = await orch.runAgent(spec, phase.id);
        } catch (e) {
          patch(spec.name, { status: 'error' });
          continue;
        }
        patch(spec.name, {
          status: run.ok ? 'done' : 'error',
          readCount: Object.keys(run.reads).length,
          writes: Object.keys(run.writes),
        });
        refreshBoard();
        await sleep(220);
      }
    }

    setActivePhase(null);
    setReport(orch.board.get('report:summary'));
    setRunning(false);
  }

  return (
    <div className="app">
      <header>
        <div className="wordmark">WebKaya</div>
        <div className="tagline">Isolated agents, coordinating through a shared blackboard.</div>
      </header>

      <p className="note">
        Each agent runs in its own WebAssembly realm — no host access, no agent-to-agent calls. The
        only thing they share is the blackboard: an agent reads what earlier agents wrote, and the
        orchestrator applies the writes it returns. Coordination through memory, not direct I/O.
      </p>

      <div className="row" style={{ marginBottom: '1.25rem' }}>
        <button onClick={run} disabled={running}>{running ? 'Running…' : 'Run pipeline'}</button>
        {activePhase && <span className="meta">phase: {activePhase}</span>}
      </div>

      <div className="grid">
        <section className="lanes">
          {phases.map((phase) => (
            <div className={`lane ${activePhase === phase.id ? 'active' : ''}`} key={phase.id}>
              <div className="lane-head">{phase.label}</div>
              <div className="cards">
                {agents.filter((a) => a.phase === phase.id).map((a) => (
                  <div className={`card ${a.status}`} key={a.name}>
                    <div className="card-top">
                      <span className="dot" />
                      <span className="name">{a.name}</span>
                      <span className="badge">WASM-isolated</span>
                    </div>
                    <div className="card-meta">
                      {a.status === 'idle' && 'waiting'}
                      {a.status === 'running' && 'running…'}
                      {a.status === 'done' && `read ${a.readCount} · wrote ${a.writes?.length ?? 0}`}
                      {a.status === 'error' && 'error'}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </section>

        <section className="board panel">
          <h2>Shared blackboard</h2>
          {board.length === 0 ? (
            <div className="meta">empty — run the pipeline</div>
          ) : (
            board.map((e) => (
              <div className={`kv ${prefixOf(e.key)}`} key={e.key}>
                <span className="k">{e.key}</span>
                <span className="v">{e.value}</span>
              </div>
            ))
          )}
        </section>
      </div>

      {report && (
        <section className="panel report">
          <h2>Result</h2>
          <div>{report}</div>
        </section>
      )}

      <div className="status">
        {running ? 'orchestrating…' : 'ready'} · agents coordinate via the blackboard only
      </div>
    </div>
  );
}
