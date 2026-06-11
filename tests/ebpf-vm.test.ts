import { describe, expect, it } from 'vitest';
import { EbpfMap } from '../src/ebpf/maps';
import { assemble, insn, op } from '../src/ebpf/asm';
import { EbpfVm, HELPERS, verifyProgram } from '../src/ebpf/vm';

function run(chunks: Uint8Array[], ctx = new ArrayBuffer(0), env = {}): bigint {
  return new EbpfVm(assemble(chunks)).run(ctx, env);
}

describe('EbpfVm arithmetic and control flow', () => {
  it('computes with immediates and registers', () => {
    expect(run([op.movImm(0, 40), op.addImm(0, 2), op.exit()])).toBe(42n);
    expect(run([op.movImm(0, 6), op.mulImm(0, 7), op.exit()])).toBe(42n);
    expect(run([op.movImm(1, 30), op.movImm(2, 12), op.movReg(0, 1), op.addReg(0, 2), op.exit()])).toBe(42n);
  });

  it('loads 64-bit immediates via lddw', () => {
    expect(run([op.lddw(0, 0xdeadbeefcafeban), op.exit()])).toBe(0xdeadbeefcafeban);
  });

  it('takes and skips conditional branches', () => {
    const clampAt10 = (input: number) =>
      run([op.movImm(0, input), op.jgtImm(0, 10, 1), op.ja(1), op.movImm(0, 10), op.exit()]);
    expect(clampAt10(3)).toBe(3n);
    expect(clampAt10(30)).toBe(10n);
  });

  it('treats division and modulo by zero per the eBPF spec', () => {
    expect(run([op.movImm(0, 5), op.movImm(2, 0), op.divReg(0, 2), op.exit()])).toBe(0n);
    expect(run([op.movImm(0, 5), op.movImm(2, 0), op.modReg(0, 2), op.exit()])).toBe(5n);
  });

  it('zero-extends 32-bit ALU results', () => {
    expect(run([op.mov32Imm(0, -1), op.exit()])).toBe(0xffffffffn);
  });

  it('aborts runaway programs at the instruction limit', () => {
    const vm = new EbpfVm(assemble([op.ja(-1), op.exit()]), { maxInstructions: 1_000 });
    expect(() => vm.run(new ArrayBuffer(0))).toThrow(/instruction limit/);
  });
});

describe('EbpfVm memory', () => {
  it('reads the context struct through r1', () => {
    const ctx = new ArrayBuffer(16);
    const view = new DataView(ctx);
    view.setBigUint64(0, 7n, true);
    view.setBigUint64(8, 5n, true);
    const result = run([op.ldxdw(0, 1, 0), op.ldxdw(2, 1, 8), op.addReg(0, 2), op.exit()], ctx);
    expect(result).toBe(12n);
  });

  it('supports stack stores and loads through r10', () => {
    expect(run([op.movImm(1, 42), op.stxdw(10, -8, 1), op.ldxdw(0, 10, -8), op.exit()])).toBe(42n);
  });

  it('rejects writes to the read-only context', () => {
    const ctx = new ArrayBuffer(8);
    expect(() => run([op.movImm(2, 1), op.stxdw(1, 0, 2), op.exit()], ctx)).toThrow(/read-only/);
  });

  it('rejects out-of-bounds access', () => {
    const ctx = new ArrayBuffer(16);
    expect(() => run([op.ldxdw(0, 1, 64), op.exit()], ctx)).toThrow(/out-of-bounds/);
  });
});

describe('EbpfVm helpers', () => {
  it('emits values through the trace helper', () => {
    const traced: bigint[] = [];
    run([op.movImm(1, 123), op.call(HELPERS.TRACE), op.movImm(0, 0), op.exit()], new ArrayBuffer(0), {
      trace: (value: bigint) => traced.push(value),
    });
    expect(traced).toEqual([123n]);
  });

  it('round-trips values through maps', () => {
    const map = new EbpfMap();
    const result = run(
      [
        op.movImm(1, 0),
        op.movImm(2, 5),
        op.movImm(3, 77),
        op.call(HELPERS.MAP_SET),
        op.movImm(1, 0),
        op.movImm(2, 5),
        op.call(HELPERS.MAP_GET),
        op.exit(),
      ],
      new ArrayBuffer(0),
      { maps: [map] }
    );
    expect(result).toBe(77n);
    expect(map.get(5n)).toBe(77n);
  });

  it('accumulates counters across invocations via map_add', () => {
    const map = new EbpfMap();
    const vm = new EbpfVm(
      assemble([op.movImm(1, 0), op.movImm(2, 0), op.movImm(3, 1), op.call(HELPERS.MAP_ADD), op.movImm(0, 0), op.exit()])
    );
    vm.run(new ArrayBuffer(0), { maps: [map] });
    vm.run(new ArrayBuffer(0), { maps: [map] });
    expect(map.get(0n)).toBe(2n);
  });

  it('returns a positive timestamp from ktime_get_ns', () => {
    expect(run([op.call(HELPERS.KTIME_GET_NS), op.exit()])).toBeGreaterThan(0n);
  });
});

describe('verifyProgram', () => {
  it('rejects empty and misaligned programs', () => {
    expect(() => verifyProgram(new Uint8Array(0))).toThrow(/non-empty/);
    expect(() => verifyProgram(new Uint8Array(7))).toThrow(/multiple of 8/);
  });

  it('requires the program to end with exit', () => {
    expect(() => verifyProgram(assemble([op.movImm(0, 0)]))).toThrow(/end with exit/);
  });

  it('rejects jumps outside the program', () => {
    expect(() => verifyProgram(assemble([op.ja(5), op.exit()]))).toThrow(/invalid target/);
  });

  it('rejects writes to r10', () => {
    expect(() => verifyProgram(assemble([op.movImm(10, 1), op.exit()]))).toThrow(/read-only r10/);
  });

  it('rejects unknown helpers', () => {
    expect(() => verifyProgram(assemble([op.call(99), op.exit()]))).toThrow(/unknown helper/);
  });

  it('rejects a truncated lddw pair', () => {
    expect(() => verifyProgram(assemble([insn(0x18, 0, 0, 0, 1), op.exit()]))).toThrow(/second slot/);
  });
});
