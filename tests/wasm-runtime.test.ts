import { describe, expect, it } from 'vitest';
import { Sandbox } from '../src/sandbox/sandbox';
import { MemorySnapshotStore } from '../src/sandbox/snapshot-store';

function wasmBox(initialState: Record<string, unknown> = {}) {
  return Sandbox.create({
    runtime: 'wasm',
    policy: { coldStartMs: 0, retryCount: 0, timeoutMs: 1_000, memoryBudgetMB: 128 },
    initialState,
    store: new MemorySnapshotStore(),
  });
}

describe('Sandbox in WASM mode (QuickJS)', () => {
  it('reports wasm runtime mode', async () => {
    expect((await wasmBox()).runtime).toBe('wasm');
  });

  it('runs guest code and commits state on success', async () => {
    const box = await wasmBox({ counter: 1 });
    const result = await box.run('ctx.state.counter += 41; ctx.log("done"); return ctx.state.counter;');
    expect(result.ok).toBe(true);
    expect(result.value).toBe(42);
    expect(result.logs).toEqual(['done']);
    expect(box.getState()).toEqual({ counter: 42 });
  });

  it('passes args and returns structured values', async () => {
    const box = await wasmBox();
    const result = await box.run('return { sum: ctx.args.a + ctx.args.b, who: ctx.args.who };', {
      args: { a: 40, b: 2, who: 'guest' },
    });
    expect(result.value).toEqual({ sum: 42, who: 'guest' });
  });

  it('isolates the guest from the host realm (no globals reachable)', async () => {
    const box = await wasmBox();
    // These bare identifiers dodge the host token-scanner (which is a separate,
    // belt-and-suspenders defense) so this asserts the realm boundary itself:
    // inside QuickJS none of the host capabilities exist.
    const result = await box.run(
      'return [typeof fetch, typeof process, typeof window, typeof document].join(",");'
    );
    expect(result.ok).toBe(true);
    expect(result.value).toBe('undefined,undefined,undefined,undefined');
  });

  it('does not commit state from a failed run', async () => {
    const box = await wasmBox({ v: 'original' });
    const result = await box.run('ctx.state.v = "corrupted"; throw new Error("boom");');
    expect(result.ok).toBe(false);
    expect(result.error).toMatch(/boom/);
    expect(box.getState()).toEqual({ v: 'original' });
  });

  it('interrupts a guest that exceeds its time budget (infinite loop)', async () => {
    const box = await Sandbox.create({
      runtime: 'wasm',
      policy: { coldStartMs: 0, retryCount: 0, timeoutMs: 150, memoryBudgetMB: 64 },
      store: new MemorySnapshotStore(),
    });
    const result = await box.run('while (true) {}');
    expect(result.ok).toBe(false);
    expect(result.error).toMatch(/interrupted|time budget/i);
  });

  it('still fires admission-control probes before executing', async () => {
    const box = await wasmBox();
    const { assemble, op } = await import('../src/ebpf/asm');
    box.attachProbe('run:start', { name: 'deny-all', program: assemble([op.movImm(0, 1), op.exit()]) });
    const result = await box.run('return 1;');
    expect(result.ok).toBe(false);
    expect(result.error).toMatch(/denied by probe "deny-all"/i);
  });

  it('produces the same result as inline mode for ordinary code (parity)', async () => {
    const code = 'ctx.state.items = ctx.state.items || []; ctx.state.items.push(ctx.args); return ctx.state.items.length;';
    const inline = await Sandbox.create({ runtime: 'inline', policy: { coldStartMs: 0 }, store: new MemorySnapshotStore(), initialState: {} });
    const wasm = await wasmBox({});
    const a = await inline.run(code, { args: 'x' });
    const b = await wasm.run(code, { args: 'x' });
    expect(b.ok).toBe(a.ok);
    expect(b.value).toBe(a.value);
    expect(wasm.getState()).toEqual(inline.getState());
  });
});
