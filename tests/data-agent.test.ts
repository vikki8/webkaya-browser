import { describe, expect, it } from 'vitest';
import { DataAgent } from '../src/python/data-agent';
import { PythonRunner, PyodideLike } from '../src/python/pyodide-runner';
import { CodeAnalyst } from '../src/llm/code-analyst';
import { LlmProvider } from '../src/llm/provider';

/** Returns a scripted code snippet per call; records prompts for assertions. */
class ScriptedProvider implements LlmProvider {
  readonly name = 'scripted';
  calls = 0;
  prompts: string[] = [];
  constructor(private readonly snippets: string[]) {}
  async generateCode(request: { system: string; prompt: string }) {
    this.prompts.push(request.prompt);
    const code = this.snippets[Math.min(this.calls, this.snippets.length - 1)];
    this.calls += 1;
    return { code, explanation: `attempt ${this.calls}`, usage: { inputTokens: 4, outputTokens: 2 } };
  }
}

/**
 * Fake Pyodide whose `runPythonAsync` throws for any code containing "BROKEN"
 * and otherwise returns a canned value — enough to drive the agent's
 * run/repair branches without a real CPython runtime.
 */
class FakePyodide implements PyodideLike {
  globals = { _m: new Map<string, unknown>(), set(k: string, v: unknown) { this._m.set(k, v); }, get(k: string) { return this._m.get(k); } };
  async runPythonAsync(code: string): Promise<unknown> {
    if (code.includes('BROKEN')) throw new Error('NameError: name BROKEN is not defined');
    return 42;
  }
}

function makeAgent(snippets: string[], maxAttempts = 3) {
  const provider = new ScriptedProvider(snippets);
  const analyst = new CodeAnalyst({ provider, language: 'python' });
  const runner = new PythonRunner(new FakePyodide());
  return { agent: new DataAgent(analyst, runner, { maxAttempts }), provider };
}

describe('DataAgent local-data loop', () => {
  it('runs generated pandas and returns the value on first success', async () => {
    const { agent } = makeAgent(['df["x"].mean()']);
    const outcome = await agent.ask('average x', ['x']);

    expect(outcome.ok).toBe(true);
    expect(outcome.value).toBe(42);
    expect(outcome.attempts).toHaveLength(1);
    expect(outcome.usage).toEqual({ inputTokens: 4, outputTokens: 2 });
  });

  it('repairs after a failing run, feeding the error back to the model', async () => {
    const { agent, provider } = makeAgent(['BROKEN code', 'df["x"].mean()']);
    const outcome = await agent.ask('average x', ['x']);

    expect(outcome.ok).toBe(true);
    expect(outcome.value).toBe(42);
    expect(outcome.attempts).toHaveLength(2);
    expect(outcome.usage).toEqual({ inputTokens: 8, outputTokens: 4 });
    // The repair prompt carried the previous code and Python error.
    expect(provider.prompts[1]).toContain('previous attempt failed');
    expect(provider.prompts[1]).toContain('NameError');
  });

  it('gives up after maxAttempts and reports the last failure', async () => {
    const { agent } = makeAgent(['BROKEN forever'], 2);
    const outcome = await agent.ask('do it', ['x']);

    expect(outcome.ok).toBe(false);
    expect(outcome.attempts).toHaveLength(2);
    expect(outcome.attempts[1].result.error).toMatch(/NameError/);
  });
});
