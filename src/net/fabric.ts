import { Sandbox } from '../sandbox/sandbox.js';
import { EbpfMap } from '../ebpf/maps.js';
import { EbpfVm } from '../ebpf/vm.js';
import { INGRESS_ADDR } from './hooks.js';

export type NetAddress = number;

export interface NetRequest {
  payload: unknown;
  port?: number;
  protocol?: number;
}

export interface NetResponse {
  ok: boolean;
  status: number;
  body?: unknown;
  error?: string;
  /** True when the request was dropped by network policy before delivery. */
  denied?: boolean;
  from: NetAddress;
  to: NetAddress;
}

interface Endpoint {
  address: NetAddress;
  name: string;
  sandbox: Sandbox;
  handler: string;
}

export interface JoinOptions {
  /**
   * Guest code that handles incoming requests. Receives the request as
   * `ctx.args` (`{ from, payload, port }`) and returns the response body.
   * Defaults to an echo handler.
   */
  handler?: string;
  name?: string;
}

const DEFAULT_HANDLER = 'return { echo: ctx.args.payload, from: ctx.args.from };';

/**
 * An in-process virtual network for sandboxes. Sandboxes `join` to receive an
 * address; `request` delivers traffic between them, enforcing a single eBPF
 * network-policy verdict on every hop. Counters live in eBPF maps so the same
 * observability model used for sandbox tracepoints applies to packets.
 *
 * The fabric models — it does not implement — TCP. On the server tier the same
 * policy bytecode attaches to kernel eBPF (tc/XDP) governing real pod traffic.
 */
export class SandboxFabric {
  private readonly endpoints = new Map<NetAddress, Endpoint>();
  private nextAddress: NetAddress = INGRESS_ADDR + 1;
  private policy: EbpfVm | null = null;
  private readonly onLog: (message: string) => void;

  /** fd 0: per-destination delivered count. fd 1: per-source dropped count. */
  readonly deliveredByDst = new EbpfMap();
  readonly droppedBySrc = new EbpfMap();

  constructor(options: { onLog?: (message: string) => void; policyProgram?: Uint8Array } = {}) {
    this.onLog = options.onLog ?? (() => {});
    if (options.policyProgram) this.setPolicy(options.policyProgram);
  }

  /** Install or clear the network-policy program (verified at install time). */
  setPolicy(program: Uint8Array | null): void {
    this.policy = program ? new EbpfVm(program) : null;
  }

  join(sandbox: Sandbox, options: JoinOptions = {}): NetAddress {
    const address = this.nextAddress++;
    this.endpoints.set(address, {
      address,
      name: options.name ?? `sandbox-${address}`,
      sandbox,
      handler: options.handler ?? DEFAULT_HANDLER,
    });
    return address;
  }

  leave(address: NetAddress): boolean {
    return this.endpoints.delete(address);
  }

  addresses(): NetAddress[] {
    return [...this.endpoints.keys()];
  }

  endpointName(address: NetAddress): string {
    return address === INGRESS_ADDR ? 'ingress' : this.endpoints.get(address)?.name ?? `addr-${address}`;
  }

  /**
   * Deliver a request from `from` to `to`. The policy hook runs first; a
   * nonzero verdict drops the request. Otherwise the destination sandbox runs
   * its handler (a governed run, so its own probes/timeout/memory budget all
   * apply) and the handler's return value becomes the response body.
   */
  async request(from: NetAddress, to: NetAddress, request: NetRequest): Promise<NetResponse> {
    const target = this.endpoints.get(to);
    const port = request.port ?? 0;
    const protocol = request.protocol ?? 0;
    const length = this.estimateLength(request.payload);

    if (!target) {
      this.droppedBySrc.add(BigInt(from), 1n);
      return { ok: false, status: 404, error: `No endpoint at address ${to}.`, from, to };
    }

    if (this.policy) {
      const verdict = this.runPolicy(from, to, port, length, protocol);
      if (verdict !== 0n) {
        this.droppedBySrc.add(BigInt(from), 1n);
        this.onLog(
          `[fabric] policy dropped ${this.endpointName(from)} -> ${this.endpointName(to)} (verdict ${verdict}).`
        );
        return { ok: false, status: 403, error: 'Denied by network policy.', denied: true, from, to };
      }
    }

    const result = await target.sandbox.run(target.handler, {
      name: `request:${this.endpointName(from)}->${target.name}`,
      args: { from, payload: request.payload, port },
    });
    this.deliveredByDst.add(BigInt(to), 1n);

    return {
      ok: result.ok,
      status: result.ok ? 200 : 500,
      body: result.ok ? result.value : undefined,
      error: result.ok ? undefined : result.error,
      from,
      to,
    };
  }

  private runPolicy(from: NetAddress, to: NetAddress, port: number, length: number, protocol: number): bigint {
    const ctx = new ArrayBuffer(40);
    const view = new DataView(ctx);
    view.setBigUint64(0, BigInt(from), true);
    view.setBigUint64(8, BigInt(to), true);
    view.setBigUint64(16, BigInt(port), true);
    view.setBigUint64(24, BigInt(length), true);
    view.setBigUint64(32, BigInt(protocol), true);
    try {
      return this.policy!.run(ctx, { trace: (v) => this.onLog(`[policy] trace: ${v}`) });
    } catch (error) {
      // A crashing policy fails closed: deny the request.
      this.onLog(`[fabric] policy error (failing closed): ${error instanceof Error ? error.message : error}`);
      return 1n;
    }
  }

  private estimateLength(payload: unknown): number {
    try {
      return JSON.stringify(payload)?.length ?? 0;
    } catch {
      return 0;
    }
  }
}
