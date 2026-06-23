import { GuestRunPolicy, HostToSandboxMessage, SandboxToHostMessage } from '../types/protocol.js';
import { BaseGuestContext, executeGuestCode } from '../runtime/guest-exec.js';
import { GuestInvoker } from '../runtime/guest-invoker.js';
import { SandboxPolicy } from '../types/policy.js';
import { LoopbackTransport, Transport, WorkerTransport } from '../runtime/worker/transport.js';

export interface ExecRequest {
  code: string;
  name: string;
  args: unknown;
  estimatedMemoryMB: number;
  /** Working state the guest mutates; the host has already cloned it. */
  state: Record<string, unknown>;
}

export interface ExecOutcome {
  ok: boolean;
  value?: unknown;
  error?: string;
  /** Resulting state — mutated copy on success, the input on failure. */
  state: Record<string, unknown>;
  logs: string[];
}

/** Extra `ctx` members exposed to guests (memory tiers); inline mode only. */
export interface InlineContextExtras {
  [key: string]: unknown;
}

export interface SandboxExecutor {
  execute(request: ExecRequest): Promise<ExecOutcome>;
  dispose(): void;
}

function toRunPolicy(policy: SandboxPolicy): GuestRunPolicy {
  return {
    entrypoint: policy.entrypoint,
    timeoutMs: policy.timeoutMs,
    retryCount: policy.retryCount,
    memoryBudgetMB: policy.memoryBudgetMB,
    coldStartMs: policy.coldStartMs,
    maxGuestCodeLength: policy.maxGuestCodeLength,
  };
}

/**
 * In-process execution. Supports live `ctx` extras (memory tiers) because the
 * guest runs in the host's realm. Cannot terminate a hung guest — its timeout
 * is cooperative (the guest must yield).
 */
export class InlineExecutor implements SandboxExecutor {
  private readonly invoker: GuestInvoker;

  constructor(
    private readonly policy: SandboxPolicy,
    private readonly extras: () => InlineContextExtras,
    onLog: (message: string) => void = () => {}
  ) {
    this.invoker = new GuestInvoker(policy, onLog, onLog);
  }

  async execute(request: ExecRequest): Promise<ExecOutcome> {
    const logs: string[] = [];
    const ctx: BaseGuestContext = {
      state: request.state,
      args: request.args,
      log: (message: string) => logs.push(String(message)),
      ...this.extras(),
    };
    try {
      const value = await this.invoker.invoke(request.name, request.estimatedMemoryMB, () =>
        executeGuestCode(request.code, ctx)
      );
      return { ok: true, value, state: request.state, logs };
    } catch (error) {
      return { ok: false, error: error instanceof Error ? error.message : String(error), state: request.state, logs };
    }
  }

  dispose(): void {
    /* nothing to release */
  }
}

/**
 * Off-main-thread (or loopback) execution over the message protocol. Gains
 * real isolation and, with a real Worker, hard timeout enforcement: a guest
 * that blows its budget is terminated and the worker respawned. Does not expose
 * live `ctx` extras — only serializable state crosses the boundary.
 */
export class WorkerExecutor implements SandboxExecutor {
  private transport: Transport | null = null;
  private readonly pending = new Map<string, (message: SandboxToHostMessage) => void>();
  private seq = 0;

  constructor(
    private readonly policy: SandboxPolicy,
    private readonly transportFactory: () => Transport
  ) {}

  private ensureTransport(): Transport {
    if (!this.transport) {
      this.transport = this.transportFactory();
      this.transport.onMessage((message) => {
        if (message.type === 'result') {
          const resolve = this.pending.get(message.id);
          if (resolve) {
            this.pending.delete(message.id);
            resolve(message);
          }
        }
      });
    }
    return this.transport;
  }

  async execute(request: ExecRequest): Promise<ExecOutcome> {
    const id = `run-${++this.seq}`;
    const message: HostToSandboxMessage = {
      type: 'run',
      id,
      code: request.code,
      name: request.name,
      args: request.args,
      estimatedMemoryMB: request.estimatedMemoryMB,
      state: request.state,
      policy: toRunPolicy(this.policy),
    };

    // Real Workers can be killed; give a grace window beyond the cooperative
    // timeout, then terminate-and-respawn so a wedged thread can't hang forever.
    const hardLimit = Math.max(100, this.policy.timeoutMs) + 1_000;

    return new Promise<ExecOutcome>((resolve) => {
      const transport = this.ensureTransport();
      let settled = false;
      const timer = setTimeout(() => {
        if (settled) return;
        settled = true;
        this.pending.delete(id);
        this.hardReset();
        resolve({
          ok: false,
          error: `Invocation "${request.name}" exceeded hard limit (${hardLimit}ms); worker terminated.`,
          state: request.state,
          logs: [],
        });
      }, hardLimit);

      this.pending.set(id, (response) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        if (response.type !== 'result') return;
        resolve({
          ok: response.ok,
          value: response.value,
          error: response.error,
          state: response.state,
          logs: response.logs,
        });
      });

      try {
        transport.post(message);
      } catch (error) {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        this.pending.delete(id);
        resolve({
          ok: false,
          error: error instanceof Error ? error.message : String(error),
          state: request.state,
          logs: [],
        });
      }
    });
  }

  private hardReset(): void {
    this.transport?.terminate();
    this.transport = null;
    for (const resolve of this.pending.values()) {
      resolve({ type: 'result', id: '', ok: false, error: 'Worker terminated.', logs: [], state: {}, durationMs: 0 });
    }
    this.pending.clear();
  }

  dispose(): void {
    this.transport?.terminate();
    this.transport = null;
    this.pending.clear();
  }
}

export type WorkerRuntimeMode = 'inline' | 'worker';

/**
 * Default transport selection: a real Worker when the caller supplies a factory
 * (apps pass `() => new Worker(new URL('@webkaya/sandbox/worker', import.meta.url),
 * { type: 'module' })`), otherwise an in-process loopback that runs the same
 * worker core without thread isolation.
 */
export function defaultTransportFactory(workerFactory?: () => Worker): () => Transport {
  if (workerFactory) {
    return () => new WorkerTransport(workerFactory());
  }
  return () => new LoopbackTransport();
}
