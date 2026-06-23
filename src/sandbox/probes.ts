import { EbpfMap } from '../ebpf/maps.js';
import { EbpfVm } from '../ebpf/vm.js';

export type SandboxTracepoint = 'run:start' | 'run:end' | 'snapshot' | 'log';

/**
 * Field order of the context struct passed to probes at each tracepoint.
 * Each field is a little-endian u64; field N lives at byte offset N * 8.
 * A probe reads them via `ldxdw rX, [r1 + N*8]`.
 */
export const TRACEPOINT_LAYOUTS = {
  'run:start': ['runIndex', 'codeLength', 'estimatedMemoryMB', 'hasArgs', 'timestampMs'],
  'run:end': ['runIndex', 'ok', 'durationUs', 'logCount', 'codeLength'],
  snapshot: ['runCount', 'stateBytes'],
  log: ['runIndex', 'messageLength'],
} as const satisfies Record<SandboxTracepoint, readonly string[]>;

export interface ProbeOptions {
  /** Verified eBPF bytecode; `attachProbe` throws if verification fails. */
  program: Uint8Array;
  name?: string;
  /** Maps visible to the program, addressed by index (fd 0 = maps[0]). */
  maps?: EbpfMap[];
  /**
   * If the probe itself errors at a gating tracepoint, veto the operation
   * instead of the default log-and-allow.
   */
  failClosed?: boolean;
  maxInstructions?: number;
}

interface AttachedProbe {
  id: string;
  tracepoint: SandboxTracepoint;
  name: string;
  vm: EbpfVm;
  maps: EbpfMap[];
  failClosed: boolean;
}

let probeCounter = 0;

export class ProbeRegistry {
  private readonly probes: AttachedProbe[] = [];

  constructor(private readonly onLog: (message: string) => void) {}

  attach(tracepoint: SandboxTracepoint, options: ProbeOptions): string {
    const id = `probe-${++probeCounter}`;
    this.probes.push({
      id,
      tracepoint,
      name: options.name ?? id,
      vm: new EbpfVm(options.program, { maxInstructions: options.maxInstructions }),
      maps: options.maps ?? [],
      failClosed: options.failClosed ?? false,
    });
    return id;
  }

  detach(id: string): boolean {
    const index = this.probes.findIndex((probe) => probe.id === id);
    if (index === -1) return false;
    this.probes.splice(index, 1);
    return true;
  }

  /**
   * Run every probe attached to `tracepoint` against the encoded context.
   * Returns the name of the first probe that returned nonzero (or that
   * errored with failClosed set); the caller decides whether that vetoes
   * the operation. Probe errors never propagate to the sandbox.
   */
  fire(tracepoint: SandboxTracepoint, fields: bigint[]): string | null {
    const matching = this.probes.filter((probe) => probe.tracepoint === tracepoint);
    if (matching.length === 0) return null;

    const ctx = new ArrayBuffer(fields.length * 8);
    const view = new DataView(ctx);
    fields.forEach((value, i) => view.setBigUint64(i * 8, BigInt.asUintN(64, value), true));

    for (const probe of matching) {
      try {
        const ret = probe.vm.run(ctx, {
          maps: probe.maps,
          trace: (value) => this.onLog(`[probe ${probe.name}] trace: ${value}`),
        });
        if (ret !== 0n) return probe.name;
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        this.onLog(`[probe ${probe.name}] error: ${message}`);
        if (probe.failClosed) return probe.name;
      }
    }
    return null;
  }
}
