import { EbpfMap } from './maps';

/**
 * Userspace eBPF virtual machine.
 *
 * Executes standard eBPF bytecode (64-bit registers r0-r10, 512-byte stack,
 * little-endian instruction encoding) with a static verifier pass and dynamic
 * bounds/instruction-count enforcement. This is the browser-tier execution
 * engine for sandbox probes; the same bytecode can attach to kernel
 * tracepoints (or ubpf) on a native tier.
 *
 * Supported ISA subset: ALU/ALU64 (except byte-swap), JMP/JMP32 conditionals,
 * MEM-mode loads/stores, lddw, CALL to the helpers below, EXIT.
 *
 * Helper ABI (diverges from kernel ABI: maps are addressed by fd index in r1,
 * not by pointer). Arguments r1-r5, result r0, r1-r5 clobbered by calls:
 * - MAP_GET(fd, key) -> value (0 if missing)
 * - MAP_SET(fd, key, value) -> 0 ok, max-u64 on capacity failure
 * - MAP_ADD(fd, key, delta) -> 0 ok, max-u64 on capacity failure
 * - TRACE(value) -> 0, emits value to the host trace callback
 * - KTIME_GET_NS() -> monotonic-ish timestamp in nanoseconds
 */
export const HELPERS = {
  MAP_GET: 1,
  MAP_SET: 2,
  MAP_ADD: 3,
  TRACE: 4,
  KTIME_GET_NS: 5,
} as const;

const KNOWN_HELPERS = new Set<number>(Object.values(HELPERS));
const MAX_PROGRAM_INSNS = 4096;
const DEFAULT_MAX_INSTRUCTIONS = 100_000;
const HARD_MAX_INSTRUCTIONS = 1_000_000;

const CTX_BASE = 0x10000000n;
const STACK_BASE = 0x20000000n;
const STACK_SIZE = 512;

export interface EbpfEnv {
  maps?: EbpfMap[];
  trace?: (value: bigint) => void;
}

export interface EbpfVmOptions {
  /** Runtime step budget per invocation (default 100k, capped at 1M). */
  maxInstructions?: number;
}

interface Insn {
  opcode: number;
  dst: number;
  src: number;
  off: number;
  imm: number;
}

function decode(program: Uint8Array): Insn[] {
  if (program.length === 0 || program.length % 8 !== 0) {
    throw new Error('eBPF program must be a non-empty multiple of 8 bytes.');
  }
  const view = new DataView(program.buffer, program.byteOffset, program.byteLength);
  const insns: Insn[] = [];
  for (let i = 0; i < program.length; i += 8) {
    insns.push({
      opcode: view.getUint8(i),
      dst: view.getUint8(i + 1) & 0x0f,
      src: (view.getUint8(i + 1) >> 4) & 0x0f,
      off: view.getInt16(i + 2, true),
      imm: view.getInt32(i + 4, true),
    });
  }
  return insns;
}

function checkInsn(insn: Insn, index: number, count: number, isLddwSecond: boolean[]): void {
  const { opcode, dst, src, off, imm } = insn;
  const cls = opcode & 0x07;
  if (src > 10) throw new Error(`eBPF verify: invalid source register r${src} at insn ${index}.`);

  switch (cls) {
    case 0x07: // ALU64
    case 0x04: { // ALU32
      const aluOp = opcode >> 4;
      if (aluOp > 0x0c) throw new Error(`eBPF verify: unsupported ALU op 0x${opcode.toString(16)} at insn ${index}.`);
      if (dst > 9) throw new Error(`eBPF verify: write to read-only r${dst} at insn ${index}.`);
      break;
    }
    case 0x05: { // JMP
      const jop = opcode >> 4;
      if (jop > 0x0d) throw new Error(`eBPF verify: unsupported JMP op 0x${opcode.toString(16)} at insn ${index}.`);
      if (jop === 0x08) {
        if (!KNOWN_HELPERS.has(imm)) throw new Error(`eBPF verify: unknown helper ${imm} at insn ${index}.`);
      } else if (jop !== 0x09) {
        const target = index + 1 + off;
        if (target < 0 || target >= count || isLddwSecond[target]) {
          throw new Error(`eBPF verify: jump to invalid target ${target} at insn ${index}.`);
        }
      }
      break;
    }
    case 0x06: { // JMP32 — conditionals only
      const jop = opcode >> 4;
      if (jop === 0x00 || jop === 0x08 || jop === 0x09 || jop > 0x0d) {
        throw new Error(`eBPF verify: unsupported JMP32 op 0x${opcode.toString(16)} at insn ${index}.`);
      }
      const target = index + 1 + off;
      if (target < 0 || target >= count || isLddwSecond[target]) {
        throw new Error(`eBPF verify: jump to invalid target ${target} at insn ${index}.`);
      }
      break;
    }
    case 0x01: // LDX
      if ((opcode & 0xe0) !== 0x60) throw new Error(`eBPF verify: unsupported load mode at insn ${index}.`);
      if (dst > 9) throw new Error(`eBPF verify: write to read-only r${dst} at insn ${index}.`);
      break;
    case 0x02: // ST
    case 0x03: // STX — dst is the address base register, r10 allowed
      if ((opcode & 0xe0) !== 0x60) throw new Error(`eBPF verify: unsupported store mode at insn ${index}.`);
      break;
    default: // 0x00 LD — only lddw
      if (opcode !== 0x18) throw new Error(`eBPF verify: unsupported LD op 0x${opcode.toString(16)} at insn ${index}.`);
      if (dst > 9) throw new Error(`eBPF verify: write to read-only r${dst} at insn ${index}.`);
      break;
  }
}

/**
 * Static checks: known opcodes, register bounds, in-range jump targets,
 * known helpers, well-formed lddw pairs, program ends with exit.
 * Memory bounds and termination are enforced dynamically by the VM.
 */
export function verifyProgram(program: Uint8Array): Insn[] {
  const insns = decode(program);
  if (insns.length > MAX_PROGRAM_INSNS) {
    throw new Error(`eBPF verify: program too large (${insns.length} insns, max ${MAX_PROGRAM_INSNS}).`);
  }

  const isLddwSecond = new Array<boolean>(insns.length).fill(false);
  for (let i = 0; i < insns.length; i++) {
    if (isLddwSecond[i]) continue;
    if (insns[i].opcode === 0x18) {
      if (i + 1 >= insns.length || insns[i + 1].opcode !== 0) {
        throw new Error(`eBPF verify: lddw at insn ${i} is missing its second slot.`);
      }
      isLddwSecond[i + 1] = true;
    }
  }

  for (let i = 0; i < insns.length; i++) {
    if (isLddwSecond[i]) continue;
    checkInsn(insns[i], i, insns.length, isLddwSecond);
  }

  const last = insns.length - 1;
  if (isLddwSecond[last] || insns[last].opcode !== 0x95) {
    throw new Error('eBPF verify: program must end with exit.');
  }
  return insns;
}

function sizeOf(opcode: number): number {
  switch (opcode & 0x18) {
    case 0x00: return 4; // W
    case 0x08: return 2; // H
    case 0x10: return 1; // B
    default: return 8;   // DW
  }
}

export class EbpfVm {
  private readonly insns: Insn[];
  private readonly maxInstructions: number;

  constructor(program: Uint8Array, options: EbpfVmOptions = {}) {
    this.insns = verifyProgram(program);
    this.maxInstructions = Math.min(
      Math.max(options.maxInstructions ?? DEFAULT_MAX_INSTRUCTIONS, 1),
      HARD_MAX_INSTRUCTIONS
    );
  }

  /** Execute the program with r1 pointing at `ctx`. Returns r0. */
  run(ctx: ArrayBuffer, env: EbpfEnv = {}): bigint {
    const ctxView = new DataView(ctx);
    const stack = new DataView(new ArrayBuffer(STACK_SIZE));
    const regs = new Array<bigint>(11).fill(0n);
    regs[1] = CTX_BASE;
    regs[10] = STACK_BASE + BigInt(STACK_SIZE);
    const maps = env.maps ?? [];

    const mem = (addr: bigint, size: number, write: boolean): { view: DataView; offset: number } => {
      const end = addr + BigInt(size);
      if (addr >= CTX_BASE && end <= CTX_BASE + BigInt(ctxView.byteLength)) {
        if (write) throw new Error('eBPF: context memory is read-only.');
        return { view: ctxView, offset: Number(addr - CTX_BASE) };
      }
      if (addr >= STACK_BASE && end <= STACK_BASE + BigInt(STACK_SIZE)) {
        return { view: stack, offset: Number(addr - STACK_BASE) };
      }
      throw new Error(`eBPF: out-of-bounds memory access at 0x${addr.toString(16)} (${size} bytes).`);
    };

    const load = (addr: bigint, size: number): bigint => {
      const { view, offset } = mem(addr, size, false);
      switch (size) {
        case 1: return BigInt(view.getUint8(offset));
        case 2: return BigInt(view.getUint16(offset, true));
        case 4: return BigInt(view.getUint32(offset, true));
        default: return view.getBigUint64(offset, true);
      }
    };

    const store = (addr: bigint, size: number, value: bigint): void => {
      const { view, offset } = mem(addr, size, true);
      const v = BigInt.asUintN(size * 8, value);
      switch (size) {
        case 1: view.setUint8(offset, Number(v)); break;
        case 2: view.setUint16(offset, Number(v), true); break;
        case 4: view.setUint32(offset, Number(v), true); break;
        default: view.setBigUint64(offset, v, true);
      }
    };

    const mapFor = (fd: bigint): EbpfMap => {
      const map = maps[Number(fd)];
      if (!map) throw new Error(`eBPF: no map at fd ${fd}.`);
      return map;
    };

    const callHelper = (id: number): bigint => {
      switch (id) {
        case HELPERS.MAP_GET:
          return mapFor(regs[1]).get(regs[2]) ?? 0n;
        case HELPERS.MAP_SET:
          return mapFor(regs[1]).set(regs[2], regs[3]) ? 0n : BigInt.asUintN(64, -1n);
        case HELPERS.MAP_ADD:
          return mapFor(regs[1]).add(regs[2], regs[3]) ? 0n : BigInt.asUintN(64, -1n);
        case HELPERS.TRACE:
          env.trace?.(regs[1]);
          return 0n;
        case HELPERS.KTIME_GET_NS: {
          const nowMs = typeof performance !== 'undefined' ? performance.now() : Date.now();
          return BigInt(Math.round(nowMs * 1e6));
        }
        default:
          throw new Error(`eBPF: unknown helper ${id}.`);
      }
    };

    let pc = 0;
    let steps = 0;
    for (;;) {
      if (++steps > this.maxInstructions) {
        throw new Error(`eBPF: instruction limit exceeded (${this.maxInstructions}).`);
      }
      const { opcode, dst, src, off, imm } = this.insns[pc];
      const cls = opcode & 0x07;

      switch (cls) {
        case 0x07: // ALU64
        case 0x04: { // ALU32
          const width = cls === 0x07 ? 64 : 32;
          const useReg = (opcode & 0x08) !== 0;
          const aluOp = opcode >> 4;
          const a = BigInt.asUintN(width, regs[dst]);
          const b = BigInt.asUintN(width, useReg ? regs[src] : BigInt(imm));
          const shift = b & BigInt(width - 1);
          let out: bigint;
          switch (aluOp) {
            case 0x0: out = a + b; break;
            case 0x1: out = a - b; break;
            case 0x2: out = a * b; break;
            case 0x3: out = b === 0n ? 0n : a / b; break;
            case 0x4: out = a | b; break;
            case 0x5: out = a & b; break;
            case 0x6: out = a << shift; break;
            case 0x7: out = a >> shift; break;
            case 0x8: out = -a; break;
            case 0x9: out = b === 0n ? a : a % b; break;
            case 0xa: out = a ^ b; break;
            case 0xb: out = b; break;
            default: out = BigInt.asIntN(width, a) >> shift; break; // 0xc ARSH
          }
          regs[dst] = BigInt.asUintN(width, out);
          pc++;
          break;
        }
        case 0x05: // JMP
        case 0x06: { // JMP32
          const jop = opcode >> 4;
          if (cls === 0x05 && jop === 0x08) { // CALL
            const result = callHelper(imm);
            regs[0] = BigInt.asUintN(64, result);
            regs[1] = regs[2] = regs[3] = regs[4] = regs[5] = 0n;
            pc++;
            break;
          }
          if (cls === 0x05 && jop === 0x09) return regs[0]; // EXIT
          if (cls === 0x05 && jop === 0x00) { // JA
            pc += 1 + off;
            break;
          }
          const width = cls === 0x05 ? 64 : 32;
          const useReg = (opcode & 0x08) !== 0;
          const ua = BigInt.asUintN(width, regs[dst]);
          const ub = BigInt.asUintN(width, useReg ? regs[src] : BigInt(imm));
          const sa = BigInt.asIntN(width, ua);
          const sb = BigInt.asIntN(width, ub);
          let take: boolean;
          switch (jop) {
            case 0x1: take = ua === ub; break;
            case 0x2: take = ua > ub; break;
            case 0x3: take = ua >= ub; break;
            case 0x4: take = (ua & ub) !== 0n; break;
            case 0x5: take = ua !== ub; break;
            case 0x6: take = sa > sb; break;
            case 0x7: take = sa >= sb; break;
            case 0xa: take = ua < ub; break;
            case 0xb: take = ua <= ub; break;
            case 0xc: take = sa < sb; break;
            default: take = sa <= sb; break; // 0xd JSLE
          }
          pc += take ? 1 + off : 1;
          break;
        }
        case 0x00: { // lddw
          const lo = BigInt(imm >>> 0);
          const hi = BigInt(this.insns[pc + 1].imm >>> 0);
          regs[dst] = lo | (hi << 32n);
          pc += 2;
          break;
        }
        case 0x01: // LDX
          regs[dst] = load(regs[src] + BigInt(off), sizeOf(opcode));
          pc++;
          break;
        case 0x02: // ST imm
          store(regs[dst] + BigInt(off), sizeOf(opcode), BigInt(imm));
          pc++;
          break;
        default: // 0x03 STX
          store(regs[dst] + BigInt(off), sizeOf(opcode), regs[src]);
          pc++;
          break;
      }
    }
  }
}
