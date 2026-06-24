import { describe, expect, it } from 'vitest';
import { IsolatedOrchestrator, OrchestratorEvent } from '../src/orchestration/blackboard';

// A mapper: sums its shard's rows by region, writes partials to the board.
const MAPPER = `
var rows = (ctx.args.input && ctx.args.input.rows) || [];
var shard = ctx.args.input.shard;
var sums = {};
for (var i = 0; i < rows.length; i++) { var r = rows[i]; sums[r.region] = (sums[r.region] || 0) + r.amount; }
var writes = {};
for (var k in sums) writes['partial:' + shard + ':' + k] = sums[k];
return { writes: writes, output: { shard: shard, sums: sums } };
`;

// A reducer: reads ALL partials the mappers wrote and totals them by region.
const REDUCER = `
var read = ctx.args.read || {};
var totals = {};
for (var key in read) { var region = key.split(':')[2]; totals[region] = (totals[region] || 0) + Number(read[key]); }
var writes = {};
for (var r in totals) writes['total:' + r] = totals[r];
return { writes: writes, output: { regions: Object.keys(totals).length } };
`;

describe('IsolatedOrchestrator: isolated multi-agent coordination', () => {
  it('coordinates a map-reduce across phases through the blackboard', async () => {
    const orch = new IsolatedOrchestrator();

    // Phase 1: two isolated mappers, each its own shard.
    await orch.runPhase('map', [
      { name: 'mapper-A', handler: MAPPER, input: { shard: 'A', rows: [{ region: 'EMEA', amount: 95 }, { region: 'APAC', amount: 110 }] } },
      { name: 'mapper-B', handler: MAPPER, input: { shard: 'B', rows: [{ region: 'EMEA', amount: 70 }, { region: 'APAC', amount: 60 }] } },
    ]);

    // Phase 2: a reducer that builds on the mappers' outputs via the board.
    const [reducer] = await orch.runPhase('reduce', [
      { name: 'reducer', handler: REDUCER, reads: ['partial:*'] },
    ]);

    expect(reducer.ok).toBe(true);
    // The reducer read exactly the four partials the mappers wrote.
    expect(Object.keys(reducer.reads).sort()).toEqual([
      'partial:A:APAC', 'partial:A:EMEA', 'partial:B:APAC', 'partial:B:EMEA',
    ]);
    // And produced correct cross-shard totals on the board.
    expect(orch.board.get('total:EMEA')).toBe('165'); // 95 + 70
    expect(orch.board.get('total:APAC')).toBe('170'); // 110 + 60
  });

  it('isolates each agent from the host realm (WASM)', async () => {
    const orch = new IsolatedOrchestrator();
    const [run] = await orch.runPhase('probe', [
      { name: 'probe', handler: 'return { output: [typeof fetch, typeof process, typeof window].join(",") };' },
    ]);
    expect(run.ok).toBe(true);
    expect(run.output).toBe('undefined,undefined,undefined');
  });

  it('brokers memory: an agent sees only the keys it requested, not the whole board', async () => {
    const orch = new IsolatedOrchestrator();
    orch.board.set('secret', 'do-not-leak');
    orch.board.set('visible', 'ok');
    const [run] = await orch.runPhase('read', [
      // Asks for 'visible' only; tries to read 'secret' too.
      { name: 'reader', handler: 'return { output: { visible: ctx.args.read.visible, secret: ctx.args.read.secret || "not-given" } };', reads: ['visible'] },
    ]);
    expect(run.output).toEqual({ visible: 'ok', secret: 'not-given' });
  });

  it('applies agent writes to the board only on success; a thrown agent writes nothing', async () => {
    const orch = new IsolatedOrchestrator();
    const [bad] = await orch.runPhase('fail', [
      { name: 'bad', handler: 'var x = { writes: { k: "1" } }; throw new Error("boom");' },
    ]);
    expect(bad.ok).toBe(false);
    expect(bad.error).toMatch(/boom/);
    expect(orch.board.get('k')).toBeNull();
  });

  it('emits lifecycle events for live visualization', async () => {
    const events: OrchestratorEvent[] = [];
    const orch = new IsolatedOrchestrator({ onEvent: (e) => events.push(e) });
    await orch.runPhase('p', [{ name: 'a', handler: 'return { writes: { x: "1" } };' }]);
    expect(events.map((e) => e.type)).toEqual(['phase:start', 'agent:start', 'agent:done', 'phase:done']);
  });
});
