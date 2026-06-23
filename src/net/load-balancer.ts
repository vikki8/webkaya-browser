import { EbpfMap } from '../ebpf/maps.js';
import { EbpfVm } from '../ebpf/vm.js';
import { NetAddress, NetResponse, SandboxFabric } from './fabric.js';
import { INGRESS_ADDR, roundRobinBalancer } from './hooks.js';

export interface IngressRequest {
  path?: string;
  payload?: unknown;
  srcPort?: number;
  /** Stable key for sticky balancing; derived from path+payload when omitted. */
  hash?: number;
}

export interface LoadBalancerOptions {
  /** Backend-selection program. Defaults to round-robin. */
  program?: Uint8Array;
  /** Maps for the program (fd 0...). Defaults to a single round-robin counter. */
  maps?: EbpfMap[];
  onLog?: (message: string) => void;
}

const MAX_U64 = (1n << 64n) - 1n;

function fnv1a(input: string): number {
  let hash = 0x811c9dc5;
  for (let i = 0; i < input.length; i++) {
    hash ^= input.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
  }
  return hash >>> 0;
}

/**
 * An eBPF load balancer that also serves as a minimal web server / ingress.
 * It selects a backend by running an eBPF program over the request tuple
 * (the browser-tier analogue of an XDP/Katran load balancer), and can serve
 * static routes itself before ever touching a backend — terminating "HTTP"
 * at the edge like NGINX serving static pages.
 */
export class LoadBalancer {
  readonly address: NetAddress = INGRESS_ADDR;
  private readonly fabric: SandboxFabric;
  private readonly backends: NetAddress[] = [];
  private vm: EbpfVm;
  private readonly maps: EbpfMap[];
  private readonly staticRoutes = new Map<string, unknown>();
  private readonly onLog: (message: string) => void;

  constructor(fabric: SandboxFabric, options: LoadBalancerOptions = {}) {
    this.fabric = fabric;
    this.maps = options.maps ?? [new EbpfMap()];
    this.vm = new EbpfVm(options.program ?? roundRobinBalancer());
    this.onLog = options.onLog ?? (() => {});
  }

  addBackend(address: NetAddress): void {
    if (!this.backends.includes(address)) this.backends.push(address);
  }

  removeBackend(address: NetAddress): void {
    const index = this.backends.indexOf(address);
    if (index !== -1) this.backends.splice(index, 1);
  }

  backendPool(): NetAddress[] {
    return [...this.backends];
  }

  /** Install a different balancing program (verified at install time). */
  setProgram(program: Uint8Array): void {
    this.vm = new EbpfVm(program);
  }

  /** Register a static route served directly by the ingress (web-server mode). */
  serveStatic(path: string, body: unknown): void {
    this.staticRoutes.set(path, body);
  }

  /**
   * Handle an incoming request. Static routes are served at the edge;
   * otherwise the eBPF program picks a backend and the fabric delivers the
   * request to it (subject to network policy).
   */
  async handle(request: IngressRequest): Promise<NetResponse> {
    const path = request.path ?? '/';

    if (this.staticRoutes.has(path)) {
      return {
        ok: true,
        status: 200,
        body: this.staticRoutes.get(path),
        from: this.address,
        to: this.address,
      };
    }

    if (this.backends.length === 0) {
      return { ok: false, status: 503, error: 'No backends available.', from: this.address, to: this.address };
    }

    const hash = request.hash ?? fnv1a(`${path}:${this.stringify(request.payload)}`);
    const index = this.selectBackend(request.srcPort ?? 0, fnv1a(path), hash);
    if (index === null) {
      return { ok: false, status: 503, error: 'Load balancer dropped the request.', from: this.address, to: this.address };
    }

    const backend = this.backends[index];
    this.onLog(`[lb] ${path} -> ${this.fabric.endpointName(backend)} (backend ${index}).`);
    return this.fabric.request(this.address, backend, { payload: request.payload, port: request.srcPort });
  }

  private selectBackend(srcPort: number, dstPortHash: number, requestHash: number): number | null {
    const ctx = new ArrayBuffer(40);
    const view = new DataView(ctx);
    view.setBigUint64(0, BigInt(this.address), true);
    view.setBigUint64(8, BigInt(srcPort), true);
    view.setBigUint64(16, BigInt(dstPortHash), true);
    view.setBigUint64(24, BigInt(requestHash), true);
    view.setBigUint64(32, BigInt(this.backends.length), true);

    let ret: bigint;
    try {
      ret = this.vm.run(ctx, { maps: this.maps, trace: (v) => this.onLog(`[lb] trace: ${v}`) });
    } catch (error) {
      this.onLog(`[lb] program error: ${error instanceof Error ? error.message : error}`);
      return null;
    }
    if (ret === MAX_U64) return null; // sentinel: drop
    return Number(ret % BigInt(this.backends.length));
  }

  private stringify(payload: unknown): string {
    try {
      return JSON.stringify(payload) ?? '';
    } catch {
      return '';
    }
  }
}
