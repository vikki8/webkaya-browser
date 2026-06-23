import { GuestRunPolicy } from '../../types/protocol.js';
import { assertGuestCodeSafety } from '../policy.js';
import { executeGuestCode } from '../guest-exec.js';

/**
 * Environment-agnostic guest execution. This runs identically in a Web Worker
 * (the product path) and in-process (the loopback path used by tests and by
 * environments without Workers). It touches no DOM, no live host objects, and
 * only serializable data — everything it needs arrives in the request.
 */

export interface GuestRunRequest {
  code: string;
  name: string;
  args: unknown;
  estimatedMemoryMB: number;
  state: Record<string, unknown>;
  policy: GuestRunPolicy;
}

export interface GuestRunOutcome {
  ok: boolean;
  value?: unknown;
  error?: string;
  /** Resulting state — the mutated working copy on success, the input on failure. */
  state: Record<string, unknown>;
  logs: string[];
}

function clone<T>(value: T): T {
  return typeof structuredClone === 'function'
    ? structuredClone(value)
    : JSON.parse(JSON.stringify(value));
}

function runWithTimeout(timeoutMs: number, handler: () => unknown): Promise<unknown> {
  const limit = Math.max(100, timeoutMs);
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`Invocation timeout (${limit}ms)`)), limit);
    Promise.resolve()
      .then(handler)
      .then((value) => {
        clearTimeout(timer);
        resolve(value);
      })
      .catch((error) => {
        clearTimeout(timer);
        reject(error);
      });
  });
}

/**
 * Execute one guest run. Re-validates guest code safety and the memory budget
 * defensively (the host checks too), enforces the cooperative timeout and
 * retry count, and commits state only on success. Never throws — failures are
 * returned as `ok: false` with the original state preserved.
 */
export async function runGuestRequest(request: GuestRunRequest): Promise<GuestRunOutcome> {
  const { code, args, policy, estimatedMemoryMB } = request;
  const logs: string[] = [];
  const originalState = request.state;

  try {
    assertGuestCodeSafety(code, policy.maxGuestCodeLength);
    if (estimatedMemoryMB > policy.memoryBudgetMB) {
      throw new Error(
        `Invocation "${request.name}" requires ~${estimatedMemoryMB.toFixed(2)}MB, above sandbox budget ${policy.memoryBudgetMB}MB.`
      );
    }
  } catch (error) {
    return { ok: false, error: errorMessage(error), state: originalState, logs };
  }

  if (policy.coldStartMs > 0) {
    await new Promise((resolve) => setTimeout(resolve, Math.min(policy.coldStartMs, 10_000)));
  }

  const retries = Math.max(0, policy.retryCount);
  let lastError: unknown = null;
  for (let attempt = 0; attempt <= retries; attempt++) {
    const workingState = clone(originalState);
    const ctx = {
      state: workingState,
      args,
      log: (message: unknown) => logs.push(String(message)),
    };
    try {
      const value = await runWithTimeout(policy.timeoutMs, () => executeGuestCode(code, ctx));
      return { ok: true, value, state: workingState, logs };
    } catch (error) {
      lastError = error;
    }
  }
  return { ok: false, error: errorMessage(lastError), state: originalState, logs };
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
