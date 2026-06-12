import { RunResult, Sandbox } from '../sandbox/sandbox';
import { DISALLOWED_GUEST_TOKENS } from '../runtime/policy';
import { LlmProvider, LlmUsage } from './provider';

/**
 * The agent loop the product exists for: an LLM writes guest code, the
 * sandbox runs it under policy (probes, timeout, memory budget), and
 * failures — including probe vetoes and token-scan rejections — are fed
 * back to the model for another attempt. Every attempt is recorded in the
 * sandbox's run log, so the whole loop is replayable.
 */

export interface CodeAgentOptions {
  /** Maximum generate -> run -> retry cycles (default 3). */
  maxAttempts?: number;
  onLog?: (message: string) => void;
}

export interface AgentAttempt {
  code: string;
  explanation: string;
  result: RunResult;
}

export interface AgentOutcome {
  ok: boolean;
  /** The final (successful or last-failed) run. */
  result: RunResult;
  code: string;
  explanation: string;
  attempts: AgentAttempt[];
  usage: LlmUsage;
}

const GUEST_SYSTEM_PROMPT = `You write JavaScript that runs inside the WebKaya browser sandbox.

Environment contract:
- Your code is the body of a function receiving a single argument \`ctx\`.
- \`ctx.state\` is a plain mutable object holding the sandbox's persistent state. Mutations are committed only if your code completes without throwing.
- \`ctx.args\` carries the input for this run (may be undefined).
- \`ctx.log(message)\` records a line of output.
- End with \`return <value>;\` to produce the run's result.

Hard constraints (the sandbox rejects code containing these tokens, so never use them): ${DISALLOWED_GUEST_TOKENS.map((t) => `\`${t.trim()}\``).join(', ')}.
There is no network, DOM, module system, or filesystem — only \`ctx\` and the JavaScript standard library. Keep the code short and self-contained.`;

function addUsage(into: LlmUsage, add?: LlmUsage): void {
  if (!add) return;
  into.inputTokens += add.inputTokens;
  into.outputTokens += add.outputTokens;
}

/**
 * Drives an LLM to write guest code, runs it in a sandbox, and repairs on
 * failure. The sandbox is supplied by the caller, so all of its governance
 * (eBPF probes, snapshot cadence, worker isolation, memory tiers) applies to
 * the agent's code automatically.
 */
export class CodeAgent {
  private readonly maxAttempts: number;
  private readonly onLog: (message: string) => void;

  constructor(
    private readonly provider: LlmProvider,
    private readonly sandbox: Sandbox,
    options: CodeAgentOptions = {}
  ) {
    this.maxAttempts = Math.max(1, options.maxAttempts ?? 3);
    this.onLog = options.onLog ?? (() => {});
  }

  /**
   * Generate code for `task`, run it, and retry with the error fed back until
   * it succeeds or `maxAttempts` is reached.
   */
  async run(task: string, options: { args?: unknown } = {}): Promise<AgentOutcome> {
    const usage: LlmUsage = { inputTokens: 0, outputTokens: 0 };
    const attempts: AgentAttempt[] = [];
    let prompt = `Task: ${task}\n\nWrite the sandbox function body that accomplishes it.`;

    for (let attempt = 1; attempt <= this.maxAttempts; attempt++) {
      this.onLog(`[agent] generating code (attempt ${attempt}/${this.maxAttempts})`);
      const generated = await this.provider.generateCode({ system: GUEST_SYSTEM_PROMPT, prompt });
      addUsage(usage, generated.usage);

      const result = await this.sandbox.run(generated.code, {
        name: `agent-attempt-${attempt}`,
        args: options.args,
      });
      attempts.push({ code: generated.code, explanation: generated.explanation, result });

      if (result.ok) {
        this.onLog(`[agent] succeeded on attempt ${attempt}`);
        return {
          ok: true,
          result,
          code: generated.code,
          explanation: generated.explanation,
          attempts,
          usage,
        };
      }

      this.onLog(`[agent] attempt ${attempt} failed: ${result.error}`);
      prompt = `Task: ${task}\n\nThe previous attempt failed. Fix it.\n\nPrevious code:\n${generated.code}\n\nError:\n${result.error}`;
    }

    const last = attempts[attempts.length - 1];
    return {
      ok: false,
      result: last.result,
      code: last.code,
      explanation: last.explanation,
      attempts,
      usage,
    };
  }
}
