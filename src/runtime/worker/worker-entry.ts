/**
 * Web Worker entry point. In a real app this module is what you load into the
 * Worker:
 *
 *   const worker = new Worker(new URL('@webkaya/sandbox/worker', import.meta.url),
 *                             { type: 'module' });
 *
 * It wires the Worker's message port to the same `runGuestRequest` core that
 * the loopback transport runs in-process, so guest semantics are identical
 * whether or not execution is isolated on its own thread.
 */
import { HostToSandboxMessage, SandboxToHostMessage } from '../../types/protocol';
import { runGuestRequest } from './worker-core';

declare const self: {
  onmessage: ((event: MessageEvent) => void) | null;
  postMessage: (message: SandboxToHostMessage) => void;
};

export function installWorkerHandler(scope: typeof self): void {
  scope.onmessage = (event: MessageEvent) => {
    const message = event.data as HostToSandboxMessage;
    if (message.type !== 'run') return;
    const startedAt = typeof performance !== 'undefined' ? performance.now() : Date.now();
    void runGuestRequest({
      code: message.code,
      name: message.name,
      args: message.args,
      estimatedMemoryMB: message.estimatedMemoryMB,
      state: message.state,
      policy: message.policy,
    }).then((outcome) => {
      const now = typeof performance !== 'undefined' ? performance.now() : Date.now();
      scope.postMessage({
        type: 'result',
        id: message.id,
        ok: outcome.ok,
        value: outcome.value,
        error: outcome.error,
        logs: outcome.logs,
        state: outcome.state,
        durationMs: now - startedAt,
      });
    });
  };
}

if (typeof self !== 'undefined' && typeof (self as { postMessage?: unknown }).postMessage === 'function') {
  installWorkerHandler(self);
}
