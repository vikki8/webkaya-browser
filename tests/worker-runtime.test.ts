import { describe, expect, it } from 'vitest';
import { Sandbox } from '../src/sandbox/sandbox';
import { MemorySnapshotStore } from '../src/sandbox/snapshot-store';
import { runGuestRequest } from '../src/runtime/worker/worker-core';
import { LoopbackTransport } from '../src/runtime/worker/transport';
import { GuestRunPolicy } from '../src/types/protocol';

const policy: GuestRunPolicy = {
  entrypoint: 'main',
  timeoutMs: 2_000,
  retryCount: 0,
  memoryBudgetMB: 512,
  coldStartMs: 0,
  maxGuestCodeLength: 20_000,
};

describe('worker core (runGuestRequest)', () => {
  it('executes guest code against serialized state', async () => {
    const outcome = await runGuestRequest({
      code: "ctx.state.n = (ctx.state.n || 0) + 41; return ctx.state.n;",
      name: 'r',
      args: undefined,
      estimatedMemoryMB: 0,
      state: { n: 1 },
      policy,
    });
    expect(outcome.ok).toBe(true);
    expect(outcome.value).toBe(42);
    expect(outcome.state).toEqual({ n: 42 });
  });

  it('captures logs and passes args', async () => {
    const outcome = await runGuestRequest({
      code: "ctx.log('hi ' + ctx.args.who); return ctx.args.who;",
      name: 'r',
      args: { who: 'agent' },
      estimatedMemoryMB: 0,
      state: {},
      policy,
    });
    expect(outcome.value).toBe('agent');
    expect(outcome.logs).toEqual(['hi agent']);
  });

  it('returns the original state untouched on failure', async () => {
    const outcome = await runGuestRequest({
      code: "ctx.state.v = 'corrupted'; throw new Error('boom');",
      name: 'r',
      args: undefined,
      estimatedMemoryMB: 0,
      state: { v: 'original' },
      policy,
    });
    expect(outcome.ok).toBe(false);
    expect(outcome.error).toMatch(/boom/);
    expect(outcome.state).toEqual({ v: 'original' });
  });

  it('enforces the memory budget and code-safety scan', async () => {
    const overBudget = await runGuestRequest({
      code: 'return 1;', name: 'r', args: undefined, estimatedMemoryMB: 99_999, state: {}, policy,
    });
    expect(overBudget.ok).toBe(false);
    expect(overBudget.error).toMatch(/above sandbox budget/);

    const unsafe = await runGuestRequest({
      code: 'fetch("https://x"); return 1;', name: 'r', args: undefined, estimatedMemoryMB: 0, state: {}, policy,
    });
    expect(unsafe.ok).toBe(false);
    expect(unsafe.error).toMatch(/disallowed token/);
  });
});

describe('LoopbackTransport serialization', () => {
  it('rejects non-serializable state across the boundary', async () => {
    const transport = new LoopbackTransport();
    // structuredClone of a function-bearing message throws, surfacing the same
    // failure a real Worker postMessage would produce.
    expect(() =>
      transport.post({
        type: 'run',
        id: 'x',
        code: 'return 1;',
        name: 'r',
        estimatedMemoryMB: 0,
        state: { fn: () => 1 } as unknown as Record<string, unknown>,
        policy,
      })
    ).toThrow();
  });
});

describe('Sandbox in worker mode (loopback)', () => {
  function workerBox(initialState: Record<string, unknown> = {}) {
    return Sandbox.create({
      runtime: 'worker',
      policy: { coldStartMs: 0, retryCount: 0, timeoutMs: 2_000 },
      initialState,
      store: new MemorySnapshotStore(),
    });
  }

  it('reports worker runtime mode', async () => {
    const box = await workerBox();
    expect(box.runtime).toBe('worker');
  });

  it('runs guest code off the inline path and commits state on success', async () => {
    const box = await workerBox({ counter: 0 });
    const result = await box.run('ctx.state.counter += 10; return ctx.state.counter;');
    expect(result.ok).toBe(true);
    expect(result.value).toBe(10);
    expect(box.getState()).toEqual({ counter: 10 });
  });

  it('does not commit state from a failed run', async () => {
    const box = await workerBox({ v: 'original' });
    const result = await box.run("ctx.state.v = 'corrupted'; throw new Error('boom');");
    expect(result.ok).toBe(false);
    expect(box.getState()).toEqual({ v: 'original' });
  });

  it('still fires admission-control probes before executing', async () => {
    const box = await workerBox();
    // Reuse the eBPF deny-by-code-length probe shape via a simple veto: attach a
    // probe that denies any run (program returns 1).
    const { assemble, op } = await import('../src/ebpf/asm');
    box.attachProbe('run:start', {
      name: 'deny-all',
      program: assemble([op.movImm(0, 1), op.exit()]),
    });
    const result = await box.run('return 1;');
    expect(result.ok).toBe(false);
    expect(result.error).toMatch(/denied by probe "deny-all"/i);
  });

  it('produces identical results to inline mode (parity)', async () => {
    const code = "ctx.state.items = ctx.state.items || []; ctx.state.items.push(ctx.args); return ctx.state.items.length;";
    const inline = await Sandbox.create({ runtime: 'inline', policy: { coldStartMs: 0 }, store: new MemorySnapshotStore(), initialState: {} });
    const worker = await workerBox({});
    const a = await inline.run(code, { args: 'x' });
    const b = await worker.run(code, { args: 'x' });
    expect(b.ok).toBe(a.ok);
    expect(b.value).toBe(a.value);
    expect(worker.getState()).toEqual(inline.getState());
  });
});
