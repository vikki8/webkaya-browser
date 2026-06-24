import { Sandbox } from '../sandbox/sandbox.js';
import { KVStore, MemoryTier } from '../memory/tiered-memory.js';

/**
 * Multi-agent orchestration over a shared blackboard, with isolation.
 *
 * Each agent runs in its OWN sandbox (`runtime: 'wasm'` — a real WebAssembly
 * realm, no host access, no reference to any other agent). Agents never call
 * each other. The only coordination channel is a shared key/value blackboard,
 * and even that is brokered by the orchestrator: an agent receives a read-only
 * snapshot of the keys it asked for in `ctx.args.read`, and returns the writes
 * it wants to make (`{ writes, output }`); the orchestrator applies them. So an
 * agent can build on another agent's output — but only the slice it was handed,
 * and only by proposing writes, never by touching shared state directly.
 *
 * Phases run in order, so later phases see earlier phases' writes; agents
 * within a phase run concurrently and isolated.
 */

export interface AgentSpec {
  name: string;
  /**
   * Guest JS run in isolation. Receives `ctx.args = { read, input }` where
   * `read` is the requested blackboard snapshot and `input` is `spec.input`.
   * Returns `{ writes?: object, output?: any }`.
   */
  handler: string;
  /** Blackboard keys or `prefix:*` globs to snapshot into `ctx.args.read`. */
  reads?: string[];
  input?: unknown;
}

export interface AgentRun {
  name: string;
  ok: boolean;
  output?: unknown;
  reads: Record<string, string | null>;
  writes: Record<string, string>;
  error?: string;
  durationMs: number;
}

export type OrchestratorEvent =
  | { type: 'phase:start'; phase: string; agents: string[] }
  | { type: 'agent:start'; phase: string; name: string; reads: string[] }
  | { type: 'agent:done'; phase: string; run: AgentRun }
  | { type: 'phase:done'; phase: string };

export interface IsolatedOrchestratorOptions {
  /** Backing store for the blackboard (in-memory by default; pass a Redis-backed tier to distribute). */
  store?: KVStore;
  timeoutMs?: number;
  memoryBudgetMB?: number;
  onEvent?: (event: OrchestratorEvent) => void;
}

export class IsolatedOrchestrator {
  /** The shared blackboard — the only thing agents coordinate through. */
  readonly board: KVStore;
  private readonly timeoutMs: number;
  private readonly memoryBudgetMB: number;
  private readonly onEvent: (event: OrchestratorEvent) => void;

  constructor(options: IsolatedOrchestratorOptions = {}) {
    this.board = options.store ?? new MemoryTier();
    this.timeoutMs = options.timeoutMs ?? 2_000;
    this.memoryBudgetMB = options.memoryBudgetMB ?? 128;
    this.onEvent = options.onEvent ?? (() => {});
  }

  private snapshot(reads?: string[]): Record<string, string | null> {
    const out: Record<string, string | null> = {};
    for (const pattern of reads ?? []) {
      if (pattern.includes('*')) {
        for (const key of this.board.keys(pattern)) out[key] = this.board.get(key);
      } else {
        out[pattern] = this.board.get(pattern);
      }
    }
    return out;
  }

  /** Run a single agent in its own isolated sandbox, brokering its blackboard access. */
  async runAgent(spec: AgentSpec, phase = 'phase'): Promise<AgentRun> {
    const reads = this.snapshot(spec.reads);
    this.onEvent({ type: 'agent:start', phase, name: spec.name, reads: Object.keys(reads) });

    const box = await Sandbox.create({
      runtime: 'wasm',
      policy: { coldStartMs: 0, retryCount: 0, timeoutMs: this.timeoutMs, memoryBudgetMB: this.memoryBudgetMB },
    });

    let run: AgentRun;
    try {
      const result = await box.run(spec.handler, { name: spec.name, args: { read: reads, input: spec.input } });
      const writes: Record<string, string> = {};
      let output: unknown;
      if (result.ok) {
        const value = (result.value ?? {}) as { writes?: Record<string, unknown>; output?: unknown };
        output = value.output;
        for (const [key, raw] of Object.entries(value.writes ?? {})) {
          const stored = typeof raw === 'string' ? raw : JSON.stringify(raw);
          this.board.set(key, stored); // host-brokered write
          writes[key] = stored;
        }
      }
      run = {
        name: spec.name,
        ok: result.ok,
        output,
        reads,
        writes,
        error: result.ok ? undefined : result.error,
        durationMs: result.durationMs,
      };
    } finally {
      box.dispose();
    }

    this.onEvent({ type: 'agent:done', phase, run });
    return run;
  }

  /** Run a set of isolated agents concurrently as one phase. */
  async runPhase(phase: string, specs: AgentSpec[]): Promise<AgentRun[]> {
    this.onEvent({ type: 'phase:start', phase, agents: specs.map((s) => s.name) });
    const runs = await Promise.all(specs.map((spec) => this.runAgent(spec, phase)));
    this.onEvent({ type: 'phase:done', phase });
    return runs;
  }
}
