import {
  WasmFunctionEditorState,
  WasmFunctionInvocationPolicy,
  WasmFunctionTemplateConfig,
} from '../types/training-workflow';

const MAX_EXECUTABLE_CODE_LENGTH = 20_000;
const UNSAFE_EXECUTABLE_TOKENS = [
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

export const DEFAULT_WASM_TEMPLATE_CONFIG: WasmFunctionTemplateConfig = {
  functionName: 'train_batch',
  invocationTimeoutMs: 10_000,
  retryCount: 1,
  memoryBudgetMB: 512,
  shardCount: 1,
  checkpointEveryNEpochs: 1,
  gradientClipValue: 0,
  coldStartMs: 50,
};

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function toNumber(value: unknown, fallback: number): number {
  const num = typeof value === 'number' ? value : Number(value);
  return Number.isFinite(num) ? num : fallback;
}

function sanitizeFunctionName(raw: unknown): string {
  if (typeof raw !== 'string') return DEFAULT_WASM_TEMPLATE_CONFIG.functionName;
  const cleaned = raw
    .trim()
    .replace(/\s+/g, '_')
    .replace(/[^a-zA-Z0-9_]/g, '_');
  return cleaned || DEFAULT_WASM_TEMPLATE_CONFIG.functionName;
}

export function normalizeTemplateConfig(
  config?: Partial<WasmFunctionTemplateConfig> | null
): WasmFunctionTemplateConfig {
  const merged = { ...DEFAULT_WASM_TEMPLATE_CONFIG, ...(config ?? {}) };
  return {
    functionName: sanitizeFunctionName(merged.functionName),
    invocationTimeoutMs: clamp(Math.round(toNumber(merged.invocationTimeoutMs, 10_000)), 100, 120_000),
    retryCount: clamp(Math.round(toNumber(merged.retryCount, 1)), 0, 8),
    memoryBudgetMB: clamp(Math.round(toNumber(merged.memoryBudgetMB, 512)), 64, 16_384),
    shardCount: clamp(Math.round(toNumber(merged.shardCount, 1)), 1, 32),
    checkpointEveryNEpochs: clamp(Math.round(toNumber(merged.checkpointEveryNEpochs, 1)), 1, 50),
    gradientClipValue: clamp(toNumber(merged.gradientClipValue, 0), 0, 1_000),
    coldStartMs: clamp(Math.round(toNumber(merged.coldStartMs, 50)), 0, 10_000),
  };
}

export function validateTemplateConfig(config: WasmFunctionTemplateConfig): string[] {
  const errors: string[] = [];
  if (!config.functionName.trim()) errors.push('Function name is required.');
  if (config.invocationTimeoutMs < 100 || config.invocationTimeoutMs > 120_000) {
    errors.push('Invocation timeout must be between 100 and 120000 ms.');
  }
  if (config.retryCount < 0 || config.retryCount > 8) errors.push('Retry count must be between 0 and 8.');
  if (config.memoryBudgetMB < 64 || config.memoryBudgetMB > 16_384) {
    errors.push('Memory budget must be between 64MB and 16384MB.');
  }
  if (config.shardCount < 1 || config.shardCount > 32) errors.push('Shard count must be between 1 and 32.');
  if (config.checkpointEveryNEpochs < 1 || config.checkpointEveryNEpochs > 50) {
    errors.push('Checkpoint cadence must be between 1 and 50 epochs.');
  }
  if (config.gradientClipValue < 0 || config.gradientClipValue > 1_000) {
    errors.push('Gradient clip value must be between 0 and 1000.');
  }
  if (config.coldStartMs < 0 || config.coldStartMs > 10_000) {
    errors.push('Cold start delay must be between 0 and 10000 ms.');
  }
  return errors;
}

export function generateExecutableFromTemplate(templateConfig: WasmFunctionTemplateConfig): string {
  const cfg = normalizeTemplateConfig(templateConfig);
  return `// WebKaya Eject Mode - Executable WASM policy
// Use "api.base" as the safe baseline from Template Mode.
// Return a partial policy object to override fields.

const policy = {
  functionName: ${JSON.stringify(cfg.functionName)},
  invocationTimeoutMs: ${cfg.invocationTimeoutMs},
  retryCount: ${cfg.retryCount},
  memoryBudgetMB: ${cfg.memoryBudgetMB},
  shardCount: ${cfg.shardCount},
  checkpointEveryNEpochs: ${cfg.checkpointEveryNEpochs},
  gradientClipValue: ${cfg.gradientClipValue},
  coldStartMs: ${cfg.coldStartMs},
};

// Example tweak:
// policy.invocationTimeoutMs = api.int(policy.invocationTimeoutMs + 500, 100, 120000);

return policy;
`;
}

export function normalizeWasmEditorState(editor?: Partial<WasmFunctionEditorState> | null): WasmFunctionEditorState {
  const advancedMode = editor?.advancedMode === 'eject' ? 'eject' : 'template';
  const templateConfig = normalizeTemplateConfig(editor?.templateConfig);
  const code =
    typeof editor?.executableCode === 'string' && editor.executableCode.trim()
      ? editor.executableCode
      : generateExecutableFromTemplate(templateConfig);
  return {
    advancedMode,
    templateConfig,
    executableCode: code,
  };
}

function assertExecutableSafety(code: string): void {
  if (code.length > MAX_EXECUTABLE_CODE_LENGTH) {
    throw new Error(`WASM editor code too large (max ${MAX_EXECUTABLE_CODE_LENGTH} characters).`);
  }
  for (const token of UNSAFE_EXECUTABLE_TOKENS) {
    if (code.includes(token)) {
      throw new Error(`WASM editor code uses disallowed token: "${token.trim()}".`);
    }
  }
}

export function compileExecutablePolicy(
  code: string,
  templateConfig: WasmFunctionTemplateConfig
): WasmFunctionInvocationPolicy {
  const trimmed = code.trim();
  if (!trimmed) {
    throw new Error('Executable WASM policy is empty. Provide code or switch back to Template mode.');
  }
  assertExecutableSafety(trimmed);
  const base = normalizeTemplateConfig(templateConfig);
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
    throw new Error(`WASM editor compile error: ${message}`);
  }

  if (!result || typeof result !== 'object' || Array.isArray(result)) {
    throw new Error('Executable WASM policy must return an object with policy overrides.');
  }

  const normalized = normalizeTemplateConfig({
    ...base,
    ...(result as Partial<WasmFunctionInvocationPolicy>),
  });
  const validationErrors = validateTemplateConfig(normalized);
  if (validationErrors.length) {
    throw new Error(`WASM editor policy invalid: ${validationErrors.join(' ')}`);
  }
  return normalized;
}

export function resolveWasmInvocationPolicy(editorState: WasmFunctionEditorState): WasmFunctionInvocationPolicy {
  const normalizedEditor = normalizeWasmEditorState(editorState);
  if (normalizedEditor.advancedMode === 'eject') {
    return compileExecutablePolicy(normalizedEditor.executableCode, normalizedEditor.templateConfig);
  }
  const validationErrors = validateTemplateConfig(normalizedEditor.templateConfig);
  if (validationErrors.length) {
    throw new Error(`Template configuration invalid: ${validationErrors.join(' ')}`);
  }
  return normalizeTemplateConfig(normalizedEditor.templateConfig);
}

