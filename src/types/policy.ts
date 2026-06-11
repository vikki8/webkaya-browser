export type PolicyMode = 'template' | 'eject';

/**
 * Governs how guest code is allowed to execute inside a sandbox.
 * All numeric fields are clamped by `normalizePolicy` — a policy obtained
 * from user input is always safe to hand to the runtime.
 */
export interface SandboxPolicy {
  /** Logical name reported in logs and errors for this sandbox's workload. */
  entrypoint: string;
  /** Wall-clock budget for a single guest invocation. */
  timeoutMs: number;
  /** Automatic retries after a failed or timed-out invocation. */
  retryCount: number;
  /** Upper bound a single invocation may declare as its estimated memory need. */
  memoryBudgetMB: number;
  /** Simulated cold-start delay before the first invocation (0 to disable). */
  coldStartMs: number;
  /** Maximum accepted guest source length, in characters. */
  maxGuestCodeLength: number;
  /** Auto-snapshot cadence: persist sandbox state every N successful runs (0 to disable). */
  snapshotEveryNRuns: number;
}

/**
 * Policy-as-code editor state. `template` mode uses the structured config
 * directly; `eject` mode runs `executableCode` against a constrained API to
 * produce policy overrides.
 */
export interface PolicyEditorState {
  mode: PolicyMode;
  template: SandboxPolicy;
  executableCode: string;
}
