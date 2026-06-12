/**
 * Guest code compilation, shared by the inline and worker execution paths so
 * both compile and invoke guest code with identical semantics. The guest sees
 * only the `ctx` object passed in — no ambient globals beyond the JS standard
 * library available in its realm.
 */

export interface BaseGuestContext {
  state: Record<string, unknown>;
  args: unknown;
  log: (message: string) => void;
  [key: string]: unknown;
}

/**
 * Compile and invoke guest code against `ctx`. Returns the guest's return
 * value (or a promise of it). Throws if the code throws or fails to compile —
 * callers are responsible for timeout, retry, and error capture.
 */
export function executeGuestCode(code: string, ctx: BaseGuestContext): unknown {
  // eslint-disable-next-line no-new-func
  const guest = new Function('ctx', `"use strict";\n${code}`);
  return guest(ctx);
}
