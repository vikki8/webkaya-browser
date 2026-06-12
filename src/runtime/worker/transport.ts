import { HostToSandboxMessage, SandboxToHostMessage } from '../../types/protocol';
import { runGuestRequest } from './worker-core';

/**
 * Carries protocol messages between the host and a sandbox runtime. The two
 * implementations differ only in where the guest runs: a real Web Worker
 * thread, or in-process (loopback). The host code above is identical for both.
 */
export interface Transport {
  post(message: HostToSandboxMessage): void;
  onMessage(handler: (message: SandboxToHostMessage) => void): void;
  /** Forcibly stop the runtime (terminates a real Worker; no-op for loopback). */
  terminate(): void;
}

function serialize<T>(value: T): T {
  return typeof structuredClone === 'function'
    ? structuredClone(value)
    : JSON.parse(JSON.stringify(value));
}

/**
 * Runs the worker core in the host's own thread. It faithfully exercises the
 * serializable protocol path (messages are structured-cloned in and out, so a
 * non-serializable state will fail here exactly as it would across a real
 * Worker boundary) but provides no thread isolation and cannot enforce a
 * hard timeout — it is the test and no-Worker fallback path.
 */
export class LoopbackTransport implements Transport {
  private handler: ((message: SandboxToHostMessage) => void) | null = null;

  post(message: HostToSandboxMessage): void {
    const cloned = serialize(message);
    if (cloned.type !== 'run') return;
    void runGuestRequest({
      code: cloned.code,
      name: cloned.name,
      args: cloned.args,
      estimatedMemoryMB: cloned.estimatedMemoryMB,
      state: cloned.state,
      policy: cloned.policy,
    }).then((outcome) => {
      const response: SandboxToHostMessage = {
        type: 'result',
        id: cloned.id,
        ok: outcome.ok,
        value: outcome.value,
        error: outcome.error,
        logs: outcome.logs,
        state: outcome.state,
        durationMs: 0,
      };
      this.handler?.(serialize(response));
    });
  }

  onMessage(handler: (message: SandboxToHostMessage) => void): void {
    this.handler = handler;
  }

  terminate(): void {
    this.handler = null;
  }
}

/**
 * Wraps a real Web Worker. The worker must load the module from
 * `runtime/worker/worker-entry`, which wires `self.onmessage` to the same
 * `runGuestRequest` core used by loopback. This path provides true off-main-
 * thread isolation; hard timeout enforcement (terminate-and-respawn) lives in
 * the executor above it.
 */
export class WorkerTransport implements Transport {
  constructor(private readonly worker: Worker) {}

  post(message: HostToSandboxMessage): void {
    this.worker.postMessage(message);
  }

  onMessage(handler: (message: SandboxToHostMessage) => void): void {
    this.worker.onmessage = (event: MessageEvent) => handler(event.data as SandboxToHostMessage);
  }

  terminate(): void {
    this.worker.terminate();
  }
}
