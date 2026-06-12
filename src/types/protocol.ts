/**
 * Host <-> sandbox message protocol.
 *
 * The in-process executor and the Web Worker transport both speak this
 * protocol, so a host can move a sandbox off the main thread without changing
 * application code. State crosses the boundary by structured clone, so it must
 * be serializable when worker mode is used.
 */

export interface GuestRunPolicy {
  entrypoint: string;
  timeoutMs: number;
  retryCount: number;
  memoryBudgetMB: number;
  coldStartMs: number;
  maxGuestCodeLength: number;
}

export type HostToSandboxMessage =
  | {
      type: 'run';
      id: string;
      code: string;
      name: string;
      args?: unknown;
      estimatedMemoryMB: number;
      state: Record<string, unknown>;
      policy: GuestRunPolicy;
    }
  | { type: 'dispose' };

export type SandboxToHostMessage =
  | {
      type: 'result';
      id: string;
      ok: boolean;
      value?: unknown;
      error?: string;
      logs: string[];
      state: Record<string, unknown>;
      durationMs: number;
    }
  | { type: 'log'; id: string; message: string };

