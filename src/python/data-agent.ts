import { CodeAnalyst } from '../llm/code-analyst';
import { LlmUsage } from '../llm/provider';
import { PythonRunner, PythonRunResult } from './pyodide-runner';

/**
 * The local-data analyst loop, end to end: an LLM writes pandas for a
 * question, the code runs in Pyodide over the user's in-browser DataFrame,
 * and a failing run is fed back to the model for repair — the Python
 * counterpart to `CodeAgent` (which drives JS guests through a `Sandbox`).
 *
 * Only the question and any error text reach the model; the data never leaves
 * the device.
 */

export interface DataAgentOptions {
  /** Maximum generate -> run -> repair cycles (default 3). */
  maxAttempts?: number;
  onLog?: (message: string) => void;
}

export interface DataAgentAttempt {
  code: string;
  explanation: string;
  result: PythonRunResult;
}

export interface DataAgentOutcome {
  ok: boolean;
  value?: unknown;
  stdout: string;
  code: string;
  explanation: string;
  attempts: DataAgentAttempt[];
  usage: LlmUsage;
}

function addUsage(into: LlmUsage, add?: LlmUsage): void {
  if (!add) return;
  into.inputTokens += add.inputTokens;
  into.outputTokens += add.outputTokens;
}

export class DataAgent {
  private readonly maxAttempts: number;
  private readonly onLog: (message: string) => void;

  constructor(
    private readonly analyst: CodeAnalyst,
    private readonly runner: PythonRunner,
    options: DataAgentOptions = {}
  ) {
    this.maxAttempts = Math.max(1, options.maxAttempts ?? 3);
    this.onLog = options.onLog ?? (() => {});
  }

  /**
   * Answer `question` against a DataFrame with the given `columns`: generate
   * pandas, run it locally, and repair with the error fed back until it
   * succeeds or `maxAttempts` is reached.
   */
  async ask(question: string, columns: string[]): Promise<DataAgentOutcome> {
    const usage: LlmUsage = { inputTokens: 0, outputTokens: 0 };
    const attempts: DataAgentAttempt[] = [];

    this.onLog('[data-agent] writing code (attempt 1)');
    let plan = await this.analyst.plan(question, columns);
    addUsage(usage, plan.usage);

    for (let attempt = 1; attempt <= this.maxAttempts; attempt++) {
      const result = await this.runner.run(plan.code);
      attempts.push({ code: plan.code, explanation: plan.explanation, result });

      if (result.ok) {
        this.onLog(`[data-agent] succeeded on attempt ${attempt}`);
        return {
          ok: true,
          value: result.value,
          stdout: result.stdout,
          code: plan.code,
          explanation: plan.explanation,
          attempts,
          usage,
        };
      }

      this.onLog(`[data-agent] attempt ${attempt} failed: ${result.error}`);
      if (attempt < this.maxAttempts) {
        this.onLog(`[data-agent] asking the model to repair (attempt ${attempt + 1})`);
        plan = await this.analyst.repair(question, columns, plan.code, result.error ?? 'unknown error');
        addUsage(usage, plan.usage);
      }
    }

    const last = attempts[attempts.length - 1];
    return {
      ok: false,
      value: undefined,
      stdout: last.result.stdout,
      code: last.code,
      explanation: last.explanation,
      attempts,
      usage,
    };
  }
}
