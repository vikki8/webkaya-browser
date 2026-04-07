import { describe, expect, test } from 'vitest';
import {
  compileExecutablePolicy,
  DEFAULT_WASM_TEMPLATE_CONFIG,
  normalizeWasmEditorState,
  resolveWasmInvocationPolicy,
  validateTemplateConfig,
} from '../src/engine/wasm-function-editor';

describe('wasm function editor', () => {
  test('normalizes editor state and generates starter executable code', () => {
    const state = normalizeWasmEditorState();

    expect(state.advancedMode).toBe('template');
    expect(state.templateConfig.functionName).toBe('train_batch');
    expect(state.executableCode).toContain('return policy;');
  });

  test('compiles executable policy overrides from eject code', () => {
    const policy = compileExecutablePolicy(
      `
const policy = { ...api.base };
policy.functionName = 'train_batch_fast';
policy.retryCount = api.int(api.base.retryCount + 2, 0, 8);
policy.shardCount = api.int(6, 1, 32);
return policy;
`,
      DEFAULT_WASM_TEMPLATE_CONFIG
    );

    expect(policy.functionName).toBe('train_batch_fast');
    expect(policy.retryCount).toBe(3);
    expect(policy.shardCount).toBe(6);
  });

  test('rejects unsafe executable code tokens', () => {
    expect(() =>
      compileExecutablePolicy(
        `
return fetch('https://example.com');
`,
        DEFAULT_WASM_TEMPLATE_CONFIG
      )
    ).toThrow(/disallowed token/i);
  });

  test('resolves template and eject policies with guardrails', () => {
    const templatePolicy = resolveWasmInvocationPolicy({
      advancedMode: 'template',
      executableCode: '',
      templateConfig: {
        ...DEFAULT_WASM_TEMPLATE_CONFIG,
        memoryBudgetMB: 1024,
      },
    });
    expect(templatePolicy.memoryBudgetMB).toBe(1024);
    expect(validateTemplateConfig(templatePolicy)).toHaveLength(0);

    const ejectPolicy = resolveWasmInvocationPolicy({
      advancedMode: 'eject',
      templateConfig: DEFAULT_WASM_TEMPLATE_CONFIG,
      executableCode: `
const policy = { ...api.base };
policy.invocationTimeoutMs = api.int(api.base.invocationTimeoutMs + 2500, 100, 120000);
return policy;
`,
    });
    expect(ejectPolicy.invocationTimeoutMs).toBe(12_500);
  });
});

