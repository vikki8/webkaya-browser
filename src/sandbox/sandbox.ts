import { SandboxPolicy } from '../types/policy.js';
import { Capabilities, detectCapabilities } from '../runtime/capability-detect.js';
import { ComputeBackendSelection, selectComputeBackend } from '../runtime/backends.js';
import { assertGuestCodeSafety, normalizePolicy } from '../runtime/policy.js';
import { ProbeOptions, ProbeRegistry, SandboxTracepoint } from './probes.js';
import { KVStore, MemoryBinding } from '../memory/tiered-memory.js';
import {
  InlineExecutor,
  SandboxExecutor,
  WorkerExecutor,
  SandboxRuntimeMode,
  defaultTransportFactory,
} from './executor.js';
import { QuickJsExecutor } from '../runtime/wasm/quickjs-executor.js';
import {
  createDefaultSnapshotStore,
  SandboxSnapshot,
  SnapshotStore,
} from './snapshot-store.js';

export interface SandboxOptions {
  policy?: Partial<SandboxPolicy>;
  initialState?: Record<string, unknown>;
  store?: SnapshotStore;
  onLog?: (message: string) => void;
  /**
   * Tiered key/value memory exposed to guest code as `ctx.local` (private to
   * this sandbox) and `ctx.global` (shared across the fabric). Reading global
   * memory is I/O, so it makes replay non-deterministic — see README.
   */
  memory?: MemoryBinding;
  /**
   * Execution mode:
   * - `inline` (default): guests run in the host realm behind a token-scanned
   *   `Function` boundary; supports the live `ctx.local` / `ctx.global` tiers.
   * - `worker`: guests run over the message protocol, off the main thread when
   *   `workerFactory` is supplied; thread isolation + terminate-on-timeout.
   * - `wasm`: guests run inside QuickJS compiled to WebAssembly — a real realm
   *   boundary (no host globals reachable) with enforced memory and time
   *   limits. Requires the optional `quickjs-emscripten` package.
   *
   * `worker` and `wasm` exchange only serializable state, so they do not expose
   * the live memory tiers.
   */
  runtime?: SandboxRuntimeMode;
  /** Factory for the backing Web Worker when `runtime: 'worker'`. */
  workerFactory?: () => Worker;
}

export interface RunOptions {
  name?: string;
  args?: unknown;
  estimatedMemoryMB?: number;
}

export interface RunResult {
  id: string;
  name: string;
  ok: boolean;
  value?: unknown;
  error?: string;
  logs: string[];
  startedAt: number;
  durationMs: number;
}

export interface RunRecord {
  id: string;
  name: string;
  code: string;
  args?: unknown;
  estimatedMemoryMB: number;
  startedAt: number;
  durationMs: number;
  ok: boolean;
}

/** The only surface guest code can touch. */
export interface GuestContext {
  state: Record<string, unknown>;
  args: unknown;
  log: (message: string) => void;
  /** Sandbox-private key/value tier (present only when memory is bound). */
  local?: KVStore;
  /** Fabric-shared key/value tier (present only when memory is bound). */
  global?: KVStore;
}

function newId(): string {
  try {
    return crypto.randomUUID();
  } catch {
    return `id-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
  }
}

/**
 * A client-side sandbox: runs guest code against an isolated state object
 * under a governance policy, records every run, and supports snapshot,
 * fork, restore, and replay.
 *
 * Isolation scales with `runtime`: `inline` (token-scanned `Function`),
 * `worker` (off-thread, terminable), or `wasm` (a true QuickJS-on-WebAssembly
 * realm with no host access and enforced memory/time limits).
 */
export class Sandbox {
  readonly id: string;
  readonly policy: SandboxPolicy;
  readonly capabilities: Capabilities | null;
  readonly backend: ComputeBackendSelection;
  readonly parentSnapshotId?: string;
  readonly runtime: SandboxRuntimeMode;

  private state: Record<string, unknown>;
  private readonly initialState: Record<string, unknown>;
  private readonly store: SnapshotStore;
  private readonly executor: SandboxExecutor;
  private readonly onLog: (message: string) => void;
  private readonly runLog: RunRecord[] = [];
  private readonly probes: ProbeRegistry;
  private readonly memory?: MemoryBinding;
  private readonly options: SandboxOptions;
  private disposed = false;

  private constructor(
    options: SandboxOptions,
    capabilities: Capabilities | null,
    parentSnapshotId?: string
  ) {
    this.id = newId();
    this.options = options;
    this.policy = normalizePolicy(options.policy);
    this.capabilities = capabilities;
    this.backend = selectComputeBackend(capabilities);
    this.parentSnapshotId = parentSnapshotId;
    this.runtime = options.runtime ?? 'inline';
    this.initialState = structuredClone(options.initialState ?? {});
    this.state = structuredClone(this.initialState);
    this.store = options.store ?? createDefaultSnapshotStore();
    this.onLog = options.onLog ?? (() => {});
    this.probes = new ProbeRegistry(this.onLog);
    this.memory = options.memory;

    if (this.runtime !== 'inline' && this.memory) {
      this.onLog(`${this.runtime} runtime does not expose ctx.local/ctx.global; memory tiers are inline-only.`);
    }
    if (this.runtime === 'worker') {
      this.executor = new WorkerExecutor(this.policy, defaultTransportFactory(options.workerFactory));
    } else if (this.runtime === 'wasm') {
      this.executor = new QuickJsExecutor(this.policy);
    } else {
      this.executor = new InlineExecutor(
        this.policy,
        () => (this.memory ? { local: this.memory.local, global: this.memory.global } : {}),
        this.onLog
      );
    }
  }

  static async create(options: SandboxOptions = {}): Promise<Sandbox> {
    let capabilities: Capabilities | null = null;
    try {
      capabilities = await detectCapabilities();
    } catch {
      capabilities = null;
    }
    return new Sandbox(options, capabilities);
  }

  /** Restore a sandbox from a persisted snapshot. */
  static async restore(snapshotId: string, options: SandboxOptions = {}): Promise<Sandbox> {
    const store = options.store ?? createDefaultSnapshotStore();
    const snapshot = await store.load(snapshotId);
    if (!snapshot) {
      throw new Error(`Snapshot "${snapshotId}" not found.`);
    }
    let capabilities: Capabilities | null = null;
    try {
      capabilities = await detectCapabilities();
    } catch {
      capabilities = null;
    }
    return new Sandbox(
      { ...options, store, initialState: snapshot.state },
      capabilities,
      snapshot.id
    );
  }

  getState(): Record<string, unknown> {
    return structuredClone(this.state);
  }

  getRunLog(): RunRecord[] {
    return this.runLog.map((record) => ({ ...record }));
  }

  /**
   * Attach a verified eBPF probe program to a sandbox tracepoint. Probes at
   * `run:start` act as admission control: a nonzero return value denies the
   * run. Probes at other tracepoints observe and update their maps; return
   * values are ignored. Throws if the program fails verification.
   * See TRACEPOINT_LAYOUTS for the context struct each tracepoint exposes.
   */
  attachProbe(tracepoint: SandboxTracepoint, options: ProbeOptions): string {
    this.assertUsable();
    return this.probes.attach(tracepoint, options);
  }

  detachProbe(id: string): boolean {
    return this.probes.detach(id);
  }

  /**
   * Execute guest code. The guest sees only `ctx` (state, args, log) and a
   * frozen standard library — no network, no DOM, no ambient globals. Failures
   * are returned as `ok: false` results, never thrown.
   */
  async run(code: string, options: RunOptions = {}): Promise<RunResult> {
    this.assertUsable();
    const runId = newId();
    const runIndex = this.runLog.length;
    const name = options.name ?? `${this.policy.entrypoint}#${runIndex + 1}`;
    const estimatedMemoryMB = options.estimatedMemoryMB ?? 0;
    const logs: string[] = [];
    const startedAt = Date.now();
    const startedClock = typeof performance !== 'undefined' ? performance.now() : startedAt;

    const finish = (ok: boolean, value?: unknown, error?: string): RunResult => {
      const durationMs =
        (typeof performance !== 'undefined' ? performance.now() : Date.now()) - startedClock;
      this.runLog.push({
        id: runId,
        name,
        code,
        args: options.args,
        estimatedMemoryMB,
        startedAt,
        durationMs,
        ok,
      });
      this.probes.fire('run:end', [
        BigInt(runIndex),
        ok ? 1n : 0n,
        BigInt(Math.round(durationMs * 1000)),
        BigInt(logs.length),
        BigInt(code.length),
      ]);
      return { id: runId, name, ok, value, error, logs, startedAt, durationMs };
    };

    let workingState: Record<string, unknown>;
    try {
      assertGuestCodeSafety(code, this.policy.maxGuestCodeLength);
      workingState = structuredClone(this.state);
    } catch (error) {
      return finish(false, undefined, error instanceof Error ? error.message : String(error));
    }

    const veto = this.probes.fire('run:start', [
      BigInt(runIndex),
      BigInt(code.length),
      BigInt(Math.round(estimatedMemoryMB)),
      options.args === undefined ? 0n : 1n,
      BigInt(startedAt),
    ]);
    if (veto) {
      return finish(false, undefined, `Run denied by probe "${veto}".`);
    }

    const outcome = await this.executor.execute({
      code,
      name,
      args: options.args,
      estimatedMemoryMB,
      state: workingState,
    });

    for (const message of outcome.logs) {
      logs.push(message);
      this.onLog(message);
      this.probes.fire('log', [BigInt(runIndex), BigInt(message.length)]);
    }

    if (outcome.ok) {
      // Commit state only on success so failed runs cannot corrupt the sandbox.
      this.state = outcome.state;
      const result = finish(true, outcome.value);
      await this.maybeAutoSnapshot();
      return result;
    }
    return finish(false, undefined, outcome.error);
  }

  /** Persist current state to the snapshot store and return the snapshot. */
  async snapshot(label?: string): Promise<SandboxSnapshot> {
    this.assertUsable();
    const snapshot: SandboxSnapshot = {
      id: newId(),
      label,
      createdAt: Date.now(),
      sandboxId: this.id,
      parentSnapshotId: this.parentSnapshotId,
      runCount: this.runLog.length,
      state: structuredClone(this.state),
    };
    await this.store.save(snapshot);
    let stateBytes = 0;
    try {
      stateBytes = JSON.stringify(snapshot.state)?.length ?? 0;
    } catch {
      stateBytes = 0;
    }
    this.probes.fire('snapshot', [BigInt(this.runLog.length), BigInt(stateBytes)]);
    return snapshot;
  }

  /**
   * Branch a new sandbox from the current state. The fork gets its own copy
   * of state and an empty run log; the parent is untouched. Probes are not
   * copied — attach them to the fork explicitly if needed.
   */
  async fork(options: Omit<SandboxOptions, 'initialState'> = {}): Promise<Sandbox> {
    this.assertUsable();
    const snapshot = await this.snapshot('fork-point');
    return new Sandbox(
      {
        policy: options.policy ?? this.policy,
        store: options.store ?? this.store,
        onLog: options.onLog ?? this.onLog,
        memory: options.memory ?? this.memory,
        runtime: options.runtime ?? this.runtime,
        workerFactory: options.workerFactory ?? this.options.workerFactory,
        initialState: structuredClone(this.state),
      },
      this.capabilities,
      snapshot.id
    );
  }

  /**
   * Re-execute this sandbox's recorded runs from its initial state in a fresh
   * sandbox, returning the replayed results. Useful for debugging a run
   * sequence or verifying that behavior is reproducible.
   */
  async replay(): Promise<{ results: RunResult[]; finalState: Record<string, unknown> }> {
    this.assertUsable();
    const replayBox = new Sandbox(
      {
        policy: { ...this.policy, coldStartMs: 0, snapshotEveryNRuns: 0 },
        store: this.store,
        onLog: () => {},
        memory: this.memory,
        initialState: structuredClone(this.initialState),
      },
      this.capabilities
    );
    const results: RunResult[] = [];
    for (const record of this.runLog) {
      results.push(
        await replayBox.run(record.code, {
          name: record.name,
          args: record.args,
          estimatedMemoryMB: record.estimatedMemoryMB,
        })
      );
    }
    return { results, finalState: replayBox.getState() };
  }

  dispose(): void {
    this.disposed = true;
    this.state = {};
    this.executor.dispose();
  }

  private assertUsable(): void {
    if (this.disposed) {
      throw new Error('Sandbox has been disposed.');
    }
  }

  private async maybeAutoSnapshot(): Promise<void> {
    const cadence = this.policy.snapshotEveryNRuns;
    if (cadence > 0 && this.runLog.length % cadence === 0) {
      await this.snapshot('auto');
    }
  }
}
