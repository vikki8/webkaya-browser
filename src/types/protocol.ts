/**
 * Host <-> sandbox message protocol.
 *
 * The in-process executor and the (planned) Web Worker transport both speak
 * this protocol, so hosts can move a sandbox off the main thread without
 * changing application code.
 */

export type HostToSandboxMessage =
  | { type: 'run'; id: string; code: string; args?: unknown; name?: string; estimatedMemoryMB?: number }
  | { type: 'snapshot'; id: string; label?: string }
  | { type: 'dispose' };

export type SandboxToHostMessage =
  | {
      type: 'result';
      id: string;
      ok: boolean;
      value?: unknown;
      error?: string;
      logs: string[];
      durationMs: number;
    }
  | { type: 'log'; id: string; message: string }
  | { type: 'snapshot-created'; id: string; snapshotId: string };
