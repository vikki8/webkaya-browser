import { describe, expect, it } from 'vitest';
import { Sandbox } from '../src/sandbox/sandbox';
import { MemorySnapshotStore } from '../src/sandbox/snapshot-store';
import { EbpfMap } from '../src/ebpf/maps';
import { assemble, op } from '../src/ebpf/asm';
import { HELPERS } from '../src/ebpf/vm';

const fastPolicy = { coldStartMs: 0, retryCount: 0, timeoutMs: 2_000 };

function createSandbox(options: { onLog?: (m: string) => void } = {}) {
  return Sandbox.create({
    policy: fastPolicy,
    initialState: {},
    store: new MemorySnapshotStore(),
    onLog: options.onLog,
  });
}

// map_add(fd 0, key 0, 1); return 0 (allow)
const countingProgram = assemble([
  op.movImm(1, 0),
  op.movImm(2, 0),
  op.movImm(3, 1),
  op.call(HELPERS.MAP_ADD),
  op.movImm(0, 0),
  op.exit(),
]);

describe('sandbox probes: observation', () => {
  it('counts runs at run:start', async () => {
    const counters = new EbpfMap();
    const box = await createSandbox();
    box.attachProbe('run:start', { name: 'run-counter', program: countingProgram, maps: [counters] });

    await box.run('return 1;');
    await box.run('return 2;');
    await box.run('return 3;');
    expect(counters.get(0n)).toBe(3n);
  });

  it('buckets run outcomes by the ok field at run:end', async () => {
    // key = ctx.ok (offset 8), delta 1
    const program = assemble([
      op.ldxdw(2, 1, 8),
      op.movImm(1, 0),
      op.movImm(3, 1),
      op.call(HELPERS.MAP_ADD),
      op.movImm(0, 0),
      op.exit(),
    ]);
    const outcomes = new EbpfMap();
    const box = await createSandbox();
    box.attachProbe('run:end', { name: 'outcome-counter', program, maps: [outcomes] });

    await box.run('return 1;');
    await box.run('throw new Error("boom");');
    expect(outcomes.get(1n)).toBe(1n);
    expect(outcomes.get(0n)).toBe(1n);
  });

  it('observes snapshot and log tracepoints', async () => {
    const snapshots = new EbpfMap();
    const logLines = new EbpfMap();
    const box = await createSandbox();
    box.attachProbe('snapshot', { program: countingProgram, maps: [snapshots] });
    box.attachProbe('log', { program: countingProgram, maps: [logLines] });

    await box.run('ctx.log("a"); ctx.log("b"); return 1;');
    await box.snapshot('checkpoint');
    expect(logLines.get(0n)).toBe(2n);
    expect(snapshots.get(0n)).toBe(1n);
  });

  it('routes the trace helper to the sandbox log', async () => {
    const logs: string[] = [];
    const box = await createSandbox({ onLog: (m) => logs.push(m) });
    const program = assemble([op.movImm(1, 123), op.call(HELPERS.TRACE), op.movImm(0, 0), op.exit()]);
    box.attachProbe('run:start', { name: 'tracer', program });

    await box.run('return 1;');
    expect(logs.some((m) => m.includes('[probe tracer] trace: 123'))).toBe(true);
  });
});

describe('sandbox probes: admission control', () => {
  // Deny when codeLength (offset 8) > 64.
  const maxCodeLength = assemble([
    op.ldxdw(2, 1, 8),
    op.movImm(0, 0),
    op.jleImm(2, 64, 1),
    op.movImm(0, 1),
    op.exit(),
  ]);

  it('vetoes runs that exceed the probe limit', async () => {
    const box = await createSandbox();
    box.attachProbe('run:start', { name: 'max-code-length', program: maxCodeLength });

    const allowed = await box.run('return 1;');
    expect(allowed.ok).toBe(true);

    const denied = await box.run(`ctx.state.x = 1; ${' '.repeat(80)} return 2;`);
    expect(denied.ok).toBe(false);
    expect(denied.error).toMatch(/denied by probe "max-code-length"/i);
    expect(box.getState()).toEqual({});
  });

  it('records denied runs in the run log', async () => {
    const box = await createSandbox();
    box.attachProbe('run:start', { name: 'max-code-length', program: maxCodeLength });
    await box.run(`return 1; ${' '.repeat(80)}`);
    const log = box.getRunLog();
    expect(log).toHaveLength(1);
    expect(log[0].ok).toBe(false);
  });

  it('fails open by default and closed when requested', async () => {
    // Out-of-bounds read: always errors at runtime.
    const crashing = assemble([op.ldxdw(0, 1, 4096), op.exit()]);

    const lenient = await createSandbox();
    lenient.attachProbe('run:start', { name: 'crashy', program: crashing });
    expect((await lenient.run('return 1;')).ok).toBe(true);

    const strict = await createSandbox();
    strict.attachProbe('run:start', { name: 'crashy', program: crashing, failClosed: true });
    const denied = await strict.run('return 1;');
    expect(denied.ok).toBe(false);
    expect(denied.error).toMatch(/denied by probe "crashy"/i);
  });

  it('stops firing after detach', async () => {
    const counters = new EbpfMap();
    const box = await createSandbox();
    const id = box.attachProbe('run:start', { program: countingProgram, maps: [counters] });

    await box.run('return 1;');
    expect(box.detachProbe(id)).toBe(true);
    await box.run('return 2;');
    expect(counters.get(0n)).toBe(1n);
  });

  it('rejects invalid programs at attach time', async () => {
    const box = await createSandbox();
    expect(() => box.attachProbe('run:start', { program: assemble([op.movImm(0, 0)]) })).toThrow(/end with exit/);
  });
});
