import { PolicyEditorState, SandboxPolicy } from '../types/policy.js';

export const MAX_POLICY_CODE_LENGTH = 20_000;

/**
 * Tokens disallowed in any guest-supplied source (policy code and run code).
 * The sandbox provides no ambient I/O: guests interact only through `ctx`.
 * This keeps runs local, replayable, and free of hidden network dependencies.
 */
export const DISALLOWED_GUEST_TOKENS = [
  'import ',
  'require(',
  'fetch(',
  'XMLHttpRequest',
  'WebSocket',
  'Worker(',
  'new Function',
  'eval(',
  'globalThis',
  'window.',
  'document.',
  'navigator.',
  'self.',
  'postMessage(',
];

export const DEFAULT_SANDBOX_POLICY: SandboxPolicy = {
  entrypoint: 'main',
  timeoutMs: 10_000,
  retryCount: 1,
  memoryBudgetMB: 512,
  coldStartMs: 0,
  maxGuestCodeLength: 20_000,
  snapshotEveryNRuns: 0,
};

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function toNumber(value: unknown, fallback: number): number {
  const num = typeof value === 'number' ? value : Number(value);
  return Number.isFinite(num) ? num : fallback;
}

function sanitizeEntrypoint(raw: unknown): string {
  if (typeof raw !== 'string') return DEFAULT_SANDBOX_POLICY.entrypoint;
  const cleaned = raw
    .trim()
    .replace(/\s+/g, '_')
    .replace(/[^a-zA-Z0-9_]/g, '_');
  return cleaned || DEFAULT_SANDBOX_POLICY.entrypoint;
}

export function normalizePolicy(policy?: Partial<SandboxPolicy> | null): SandboxPolicy {
  const merged = { ...DEFAULT_SANDBOX_POLICY, ...(policy ?? {}) };
  return {
    entrypoint: sanitizeEntrypoint(merged.entrypoint),
    timeoutMs: clamp(Math.round(toNumber(merged.timeoutMs, 10_000)), 100, 120_000),
    retryCount: clamp(Math.round(toNumber(merged.retryCount, 1)), 0, 8),
    memoryBudgetMB: clamp(Math.round(toNumber(merged.memoryBudgetMB, 512)), 64, 16_384),
    coldStartMs: clamp(Math.round(toNumber(merged.coldStartMs, 0)), 0, 10_000),
    maxGuestCodeLength: clamp(Math.round(toNumber(merged.maxGuestCodeLength, 20_000)), 256, 200_000),
    snapshotEveryNRuns: clamp(Math.round(toNumber(merged.snapshotEveryNRuns, 0)), 0, 1_000),
  };
}

export function validatePolicy(policy: SandboxPolicy): string[] {
  const errors: string[] = [];
  if (!policy.entrypoint.trim()) errors.push('Entrypoint name is required.');
  if (policy.timeoutMs < 100 || policy.timeoutMs > 120_000) {
    errors.push('Timeout must be between 100 and 120000 ms.');
  }
  if (policy.retryCount < 0 || policy.retryCount > 8) errors.push('Retry count must be between 0 and 8.');
  if (policy.memoryBudgetMB < 64 || policy.memoryBudgetMB > 16_384) {
    errors.push('Memory budget must be between 64MB and 16384MB.');
  }
  if (policy.coldStartMs < 0 || policy.coldStartMs > 10_000) {
    errors.push('Cold start delay must be between 0 and 10000 ms.');
  }
  if (policy.maxGuestCodeLength < 256 || policy.maxGuestCodeLength > 200_000) {
    errors.push('Max guest code length must be between 256 and 200000 characters.');
  }
  if (policy.snapshotEveryNRuns < 0 || policy.snapshotEveryNRuns > 1_000) {
    errors.push('Snapshot cadence must be between 0 and 1000 runs.');
  }
  return errors;
}

export function assertGuestCodeSafety(code: string, maxLength = MAX_POLICY_CODE_LENGTH): void {
  if (code.length > maxLength) {
    throw new Error(`Guest code too large (max ${maxLength} characters).`);
  }
  for (const token of DISALLOWED_GUEST_TOKENS) {
    if (code.includes(token)) {
      throw new Error(`Guest code uses disallowed token: "${token.trim()}".`);
    }
  }
}

export function generatePolicyCodeFromTemplate(template: Partial<SandboxPolicy>): string {
  const cfg = normalizePolicy(template);
  return `// WebKaya Sandbox — ejected policy
// "api.base" is the safe baseline from template mode.
// Return a partial policy object to override fields.

const policy = {
  entrypoint: ${JSON.stringify(cfg.entrypoint)},
  timeoutMs: ${cfg.timeoutMs},
  retryCount: ${cfg.retryCount},
  memoryBudgetMB: ${cfg.memoryBudgetMB},
  coldStartMs: ${cfg.coldStartMs},
  maxGuestCodeLength: ${cfg.maxGuestCodeLength},
  snapshotEveryNRuns: ${cfg.snapshotEveryNRuns},
};

// Example tweak:
// policy.timeoutMs = api.int(policy.timeoutMs + 500, 100, 120000);

return policy;
`;
}

export function normalizePolicyEditorState(editor?: Partial<PolicyEditorState> | null): PolicyEditorState {
  const mode = editor?.mode === 'eject' ? 'eject' : 'template';
  const template = normalizePolicy(editor?.template);
  const code =
    typeof editor?.executableCode === 'string' && editor.executableCode.trim()
      ? editor.executableCode
      : generatePolicyCodeFromTemplate(template);
  return { mode, template, executableCode: code };
}

export function compilePolicyCode(code: string, basePolicy: Partial<SandboxPolicy>): SandboxPolicy {
  const trimmed = code.trim();
  if (!trimmed) {
    throw new Error('Ejected policy code is empty. Provide code or switch back to template mode.');
  }
  assertGuestCodeSafety(trimmed);
  const base = normalizePolicy(basePolicy);
  const api = {
    base: { ...base },
    clamp: (value: number, min: number, max: number) => clamp(value, min, max),
    int: (value: number, min: number, max: number) => clamp(Math.round(value), min, max),
    num: (value: number, min: number, max: number) => clamp(value, min, max),
  };

  let result: unknown;
  try {
    // eslint-disable-next-line no-new-func
    const executable = new Function('api', `"use strict";\n${trimmed}`);
    result = executable(api);
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown compile error.';
    throw new Error(`Policy compile error: ${message}`);
  }

  if (!result || typeof result !== 'object' || Array.isArray(result)) {
    throw new Error('Ejected policy code must return an object with policy overrides.');
  }

  const normalized = normalizePolicy({ ...base, ...(result as Partial<SandboxPolicy>) });
  const validationErrors = validatePolicy(normalized);
  if (validationErrors.length) {
    throw new Error(`Ejected policy invalid: ${validationErrors.join(' ')}`);
  }
  return normalized;
}

export function resolvePolicy(editorState: PolicyEditorState): SandboxPolicy {
  const normalized = normalizePolicyEditorState(editorState);
  if (normalized.mode === 'eject') {
    return compilePolicyCode(normalized.executableCode, normalized.template);
  }
  const validationErrors = validatePolicy(normalized.template);
  if (validationErrors.length) {
    throw new Error(`Template policy invalid: ${validationErrors.join(' ')}`);
  }
  return normalizePolicy(normalized.template);
}
