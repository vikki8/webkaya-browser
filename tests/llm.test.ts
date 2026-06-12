import { describe, expect, it, vi } from 'vitest';
import { CodeAnalyst } from '../src/llm/code-analyst';
import { CodeAgent } from '../src/llm/code-agent';
import { ClaudeProvider } from '../src/llm/claude';
import { LlmProvider } from '../src/llm/provider';
import { Sandbox } from '../src/sandbox/sandbox';
import { MemorySnapshotStore } from '../src/sandbox/snapshot-store';

class FakeProvider implements LlmProvider {
  readonly name = 'fake';
  lastSystem = '';
  lastPrompt = '';
  constructor(private readonly reply: { code: string; explanation: string }) {}
  async generateCode(request: { system: string; prompt: string }) {
    this.lastSystem = request.system;
    this.lastPrompt = request.prompt;
    return { ...this.reply };
  }
}

describe('CodeAnalyst', () => {
  it('builds a Python plan and passes columns into the prompt', async () => {
    const provider = new FakeProvider({ code: 'df["revenue"].mean()', explanation: 'mean revenue' });
    const analyst = new CodeAnalyst({ provider, language: 'python' });
    const plan = await analyst.plan('average revenue', ['country', 'revenue']);

    expect(plan.language).toBe('python');
    expect(plan.code).toBe('df["revenue"].mean()');
    expect(provider.lastSystem).toContain('pandas DataFrame');
    expect(provider.lastPrompt).toContain('country, revenue');
    expect(provider.lastPrompt).toContain('average revenue');
  });

  it('uses the JavaScript system prompt and ctx.state.rows for js language', async () => {
    const provider = new FakeProvider({ code: 'return ctx.state.rows.length;', explanation: 'count' });
    const analyst = new CodeAnalyst({ provider, language: 'javascript' });
    const plan = await analyst.plan('how many rows', ['a', 'b']);

    expect(plan.code).toBe('return ctx.state.rows.length;');
    expect(provider.lastSystem).toContain('ctx.state.rows');
  });

  it('includes the failing code and error in a repair prompt', async () => {
    const provider = new FakeProvider({ code: 'fixed', explanation: 'fixed it' });
    const analyst = new CodeAnalyst({ provider });
    await analyst.repair('avg revenue', ['revenue'], 'df["rev"].mean()', 'KeyError: rev');

    expect(provider.lastPrompt).toContain('The previous attempt failed');
    expect(provider.lastPrompt).toContain('df["rev"].mean()');
    expect(provider.lastPrompt).toContain('KeyError: rev');
  });

  it('honors a custom dataframe name', async () => {
    const provider = new FakeProvider({ code: 'sales.head()', explanation: 'head' });
    const analyst = new CodeAnalyst({ provider, dataframeName: 'sales' });
    await analyst.plan('preview', ['x']);
    expect(provider.lastSystem).toContain('`sales`');
  });
});

/**
 * Stub the Anthropic client so the provider's parsing and error mapping are
 * tested without network calls or API credits. We monkeypatch the private
 * getClient so loadAnthropic() (the dynamic import) is never reached.
 */
function stubProvider(create: (...args: unknown[]) => unknown): ClaudeProvider {
  const provider = new ClaudeProvider({ apiKey: 'test' });
  (provider as unknown as { getClient: () => Promise<unknown> }).getClient = async () => ({
    messages: { create },
  });
  return provider;
}

describe('ClaudeProvider response handling', () => {
  it('parses a structured JSON code response', async () => {
    const provider = stubProvider(async () => ({
      stop_reason: 'end_turn',
      content: [{ type: 'text', text: JSON.stringify({ code: 'len(df)', explanation: 'rows' }) }],
      usage: { input_tokens: 12, output_tokens: 8 },
    }));

    const result = await provider.generateCode({ system: 's', prompt: 'p' });
    expect(result.code).toBe('len(df)');
    expect(result.explanation).toBe('rows');
    expect(result.usage).toEqual({ inputTokens: 12, outputTokens: 8 });
  });

  it('throws a clear error on a refusal', async () => {
    const provider = stubProvider(async () => ({ stop_reason: 'refusal', content: [], usage: {} }));
    await expect(provider.generateCode({ system: 's', prompt: 'p' })).rejects.toThrow(/declined/);
  });

  it('throws when the model hits the token limit', async () => {
    const provider = stubProvider(async () => ({
      stop_reason: 'max_tokens',
      content: [{ type: 'text', text: '{' }],
      usage: { input_tokens: 1, output_tokens: 1 },
    }));
    await expect(provider.generateCode({ system: 's', prompt: 'p' })).rejects.toThrow(/token limit/);
  });

  it('throws when the response is not valid JSON', async () => {
    const provider = stubProvider(async () => ({
      stop_reason: 'end_turn',
      content: [{ type: 'text', text: 'not json' }],
      usage: { input_tokens: 1, output_tokens: 1 },
    }));
    await expect(provider.generateCode({ system: 's', prompt: 'p' })).rejects.toThrow(/not valid JSON/);
  });

  it('defaults the model to claude-opus-4-8', () => {
    expect(new ClaudeProvider().model).toBe('claude-opus-4-8');
    expect(new ClaudeProvider({ model: 'claude-sonnet-4-6' }).model).toBe('claude-sonnet-4-6');
  });
});

/** Provider that returns a scripted sequence of code snippets, one per call. */
class ScriptedProvider implements LlmProvider {
  readonly name = 'scripted';
  calls = 0;
  lastPrompt = '';
  constructor(private readonly snippets: string[]) {}
  async generateCode(request: { system: string; prompt: string }) {
    this.lastPrompt = request.prompt;
    const code = this.snippets[Math.min(this.calls, this.snippets.length - 1)];
    this.calls += 1;
    return { code, explanation: `attempt ${this.calls}`, usage: { inputTokens: 5, outputTokens: 3 } };
  }
}

async function agentSandbox() {
  return Sandbox.create({
    policy: { coldStartMs: 0, retryCount: 0, timeoutMs: 2_000 },
    initialState: {},
    store: new MemorySnapshotStore(),
  });
}

describe('CodeAgent generate -> run -> repair loop', () => {
  it('runs generated code in the sandbox and returns the result', async () => {
    const provider = new ScriptedProvider(['ctx.state.n = 41 + 1; return ctx.state.n;']);
    const agent = new CodeAgent(provider, await agentSandbox());
    const outcome = await agent.run('compute the answer');

    expect(outcome.ok).toBe(true);
    expect(outcome.result.value).toBe(42);
    expect(outcome.attempts).toHaveLength(1);
    expect(outcome.usage).toEqual({ inputTokens: 5, outputTokens: 3 });
  });

  it('feeds the error back and repairs on a second attempt', async () => {
    const provider = new ScriptedProvider([
      "throw new Error('boom');",
      'return 7;',
    ]);
    const agent = new CodeAgent(provider, await agentSandbox(), { maxAttempts: 3 });
    const outcome = await agent.run('do the thing');

    expect(outcome.ok).toBe(true);
    expect(outcome.result.value).toBe(7);
    expect(outcome.attempts).toHaveLength(2);
    // Usage accumulates across both model calls.
    expect(outcome.usage).toEqual({ inputTokens: 10, outputTokens: 6 });
    // The repair prompt carried the previous error.
    expect(provider.lastPrompt).toContain('boom');
  });

  it('gives up after maxAttempts and reports the last failure', async () => {
    const provider = new ScriptedProvider(["throw new Error('always fails');"]);
    const agent = new CodeAgent(provider, await agentSandbox(), { maxAttempts: 2 });
    const outcome = await agent.run('impossible');

    expect(outcome.ok).toBe(false);
    expect(outcome.attempts).toHaveLength(2);
    expect(outcome.result.error).toMatch(/always fails/);
  });

  it('surfaces a probe veto as a failure the model can repair', async () => {
    const { assemble, op } = await import('../src/ebpf/asm');
    const sandbox = await agentSandbox();
    // Deny any run whose code is longer than 16 chars, forcing a shorter retry.
    sandbox.attachProbe('run:start', {
      name: 'max-len',
      program: assemble([op.ldxdw(2, 1, 8), op.movImm(0, 0), op.jleImm(2, 16, 1), op.movImm(0, 1), op.exit()]),
    });
    const provider = new ScriptedProvider([
      'return 123456789012345678;', // > 16 chars -> vetoed
      'return 1;', // short -> allowed
    ]);
    const agent = new CodeAgent(provider, sandbox, { maxAttempts: 2 });
    const outcome = await agent.run('return a number');

    expect(outcome.ok).toBe(true);
    expect(outcome.attempts[0].result.error).toMatch(/denied by probe/i);
    expect(provider.lastPrompt).toMatch(/denied by probe/i);
  });
});

// Keep vitest from complaining about an unused import in some configs.
void vi;
