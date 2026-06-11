import { describe, expect, it } from 'vitest';
import { Sandbox } from '../src/sandbox/sandbox';
import { MemorySnapshotStore } from '../src/sandbox/snapshot-store';

const fastPolicy = { coldStartMs: 0, retryCount: 0, timeoutMs: 2_000 };

function createSandbox(initialState: Record<string, unknown> = {}) {
  return Sandbox.create({
    policy: fastPolicy,
    initialState,
    store: new MemorySnapshotStore(),
  });
}

describe('Sandbox.run', () => {
  it('executes guest code against sandbox state', async () => {
    const box = await createSandbox({ counter: 1 });
    const result = await box.run('ctx.state.counter = ctx.state.counter + 41; return ctx.state.counter;');
    expect(result.ok).toBe(true);
    expect(result.value).toBe(42);
    expect(box.getState()).toEqual({ counter: 42 });
  });

  it('passes args and captures logs', async () => {
    const box = await createSandbox();
    const result = await box.run('ctx.log("hello " + ctx.args.who); return ctx.args.who;', {
      args: { who: 'agent' },
    });
    expect(result.value).toBe('agent');
    expect(result.logs).toEqual(['hello agent']);
  });

  it('returns ok:false for disallowed code without mutating state', async () => {
    const box = await createSandbox({ safe: true });
    const result = await box.run('fetch("https://example.com"); return 1;');
    expect(result.ok).toBe(false);
    expect(result.error).toMatch(/disallowed token/);
    expect(box.getState()).toEqual({ safe: true });
  });

  it('does not commit state from a failed run', async () => {
    const box = await createSandbox({ value: 'original' });
    const result = await box.run('ctx.state.value = "corrupted"; throw new Error("boom");');
    expect(result.ok).toBe(false);
    expect(result.error).toMatch(/boom/);
    expect(box.getState()).toEqual({ value: 'original' });
  });

  it('enforces the memory budget', async () => {
    const box = await createSandbox();
    const result = await box.run('return 1;', { estimatedMemoryMB: 100_000 });
    expect(result.ok).toBe(false);
    expect(result.error).toMatch(/above sandbox budget/);
  });
});

describe('Sandbox snapshots and forks', () => {
  it('snapshots and restores state through the store', async () => {
    const store = new MemorySnapshotStore();
    const box = await Sandbox.create({ policy: fastPolicy, initialState: { n: 0 }, store });
    await box.run('ctx.state.n = 7;');
    const snap = await box.snapshot('after-seven');

    const restored = await Sandbox.restore(snap.id, { policy: fastPolicy, store });
    expect(restored.getState()).toEqual({ n: 7 });
    expect(restored.parentSnapshotId).toBe(snap.id);
  });

  it('forks diverge without affecting the parent', async () => {
    const box = await createSandbox({ branch: 'parent' });
    const fork = await box.fork();
    await fork.run('ctx.state.branch = "fork";');
    expect(fork.getState()).toEqual({ branch: 'fork' });
    expect(box.getState()).toEqual({ branch: 'parent' });
  });

  it('auto-snapshots on the configured cadence', async () => {
    const store = new MemorySnapshotStore();
    const box = await Sandbox.create({
      policy: { ...fastPolicy, snapshotEveryNRuns: 2 },
      initialState: {},
      store,
    });
    await box.run('ctx.state.a = 1;');
    expect(await store.list()).toHaveLength(0);
    await box.run('ctx.state.b = 2;');
    expect(await store.list()).toHaveLength(1);
  });
});

describe('Sandbox.replay', () => {
  it('reproduces the run sequence from initial state', async () => {
    const box = await createSandbox({ total: 0 });
    await box.run('ctx.state.total = ctx.state.total + ctx.args;', { args: 10 });
    await box.run('ctx.state.total = ctx.state.total * 2;');

    const { results, finalState } = await box.replay();
    expect(results).toHaveLength(2);
    expect(results.every((r) => r.ok)).toBe(true);
    expect(finalState).toEqual({ total: 20 });
    expect(finalState).toEqual(box.getState());
  });
});

describe('Sandbox.dispose', () => {
  it('rejects use after dispose', async () => {
    const box = await createSandbox();
    box.dispose();
    await expect(box.run('return 1;')).rejects.toThrow(/disposed/);
  });
});
