import { describe, expect, it } from 'vitest';
import {
  compilePolicyCode,
  DEFAULT_SANDBOX_POLICY,
  assertGuestCodeSafety,
  generatePolicyCodeFromTemplate,
  normalizePolicy,
  resolvePolicy,
  validatePolicy,
} from '../src/runtime/policy';

describe('normalizePolicy', () => {
  it('returns defaults for empty input', () => {
    expect(normalizePolicy()).toEqual(DEFAULT_SANDBOX_POLICY);
    expect(normalizePolicy(null)).toEqual(DEFAULT_SANDBOX_POLICY);
  });

  it('clamps out-of-range values', () => {
    const policy = normalizePolicy({
      timeoutMs: 999_999,
      retryCount: -5,
      memoryBudgetMB: 1,
      coldStartMs: 99_999,
      maxGuestCodeLength: 10,
      snapshotEveryNRuns: 5_000,
    });
    expect(policy.timeoutMs).toBe(120_000);
    expect(policy.retryCount).toBe(0);
    expect(policy.memoryBudgetMB).toBe(64);
    expect(policy.coldStartMs).toBe(10_000);
    expect(policy.maxGuestCodeLength).toBe(256);
    expect(policy.snapshotEveryNRuns).toBe(1_000);
  });

  it('sanitizes entrypoint names', () => {
    expect(normalizePolicy({ entrypoint: 'my agent step!' }).entrypoint).toBe('my_agent_step_');
    expect(normalizePolicy({ entrypoint: '   ' }).entrypoint).toBe('main');
  });
});

describe('validatePolicy', () => {
  it('accepts the defaults', () => {
    expect(validatePolicy(DEFAULT_SANDBOX_POLICY)).toEqual([]);
  });

  it('reports out-of-range fields', () => {
    const errors = validatePolicy({ ...DEFAULT_SANDBOX_POLICY, timeoutMs: 5, retryCount: 99 });
    expect(errors.some((e) => e.includes('Timeout'))).toBe(true);
    expect(errors.some((e) => e.includes('Retry count'))).toBe(true);
  });
});

describe('assertGuestCodeSafety', () => {
  it('allows plain computation', () => {
    expect(() => assertGuestCodeSafety('const x = 1 + 1; return x;')).not.toThrow();
  });

  it('rejects disallowed tokens', () => {
    expect(() => assertGuestCodeSafety('fetch("https://example.com")')).toThrow(/disallowed token/);
    expect(() => assertGuestCodeSafety('eval("1")')).toThrow(/disallowed token/);
    expect(() => assertGuestCodeSafety('window.location.href')).toThrow(/disallowed token/);
  });

  it('rejects oversized code', () => {
    expect(() => assertGuestCodeSafety('x'.repeat(500), 256)).toThrow(/too large/);
  });
});

describe('compilePolicyCode', () => {
  it('applies overrides from ejected code', () => {
    const policy = compilePolicyCode('return { timeoutMs: api.int(api.base.timeoutMs + 500, 100, 120000) };', {});
    expect(policy.timeoutMs).toBe(10_500);
  });

  it('rejects non-object returns', () => {
    expect(() => compilePolicyCode('return 42;', {})).toThrow(/must return an object/);
  });

  it('rejects unsafe tokens in policy code', () => {
    expect(() => compilePolicyCode('return fetch("/x");', {})).toThrow(/disallowed token/);
  });
});

describe('resolvePolicy', () => {
  it('resolves template mode directly', () => {
    const policy = resolvePolicy({
      mode: 'template',
      template: { ...DEFAULT_SANDBOX_POLICY, retryCount: 3 },
      executableCode: '',
    });
    expect(policy.retryCount).toBe(3);
  });

  it('round-trips generated eject code', () => {
    const template = { ...DEFAULT_SANDBOX_POLICY, memoryBudgetMB: 1024 };
    const code = generatePolicyCodeFromTemplate(template);
    const policy = resolvePolicy({ mode: 'eject', template, executableCode: code });
    expect(policy.memoryBudgetMB).toBe(1024);
  });
});
