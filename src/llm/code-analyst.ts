import { CodeGenResult, LlmProvider } from './provider';

/**
 * Turns a natural-language question about a dataset into executable code, using
 * any `LlmProvider`. This is the production replacement for the deterministic
 * `planQuestion` planner: same job (question + columns -> code), real model.
 *
 * It is execution-agnostic — it produces code, it does not run it. Feed the
 * result to `PythonRunner.run` (Python) or `Sandbox.run` (JS), so the sandbox's
 * policy, timeout, and metering still govern execution.
 */

export type GuestLanguage = 'python' | 'javascript';

export interface AnalystOptions {
  provider: LlmProvider;
  language?: GuestLanguage;
  /** DataFrame variable name (python) — defaults to "df". */
  dataframeName?: string;
}

export interface AnalysisPlan extends CodeGenResult {
  language: GuestLanguage;
}

const PYTHON_SYSTEM = (df: string) => `You generate Python code that answers questions about a pandas DataFrame.

Environment:
- A DataFrame named \`${df}\` is already loaded. Do not read files or import data.
- pandas is available as \`pd\`. You may use numpy as \`np\`.
- The code runs in a sandboxed in-browser Python runtime over the user's local data; it has no network or filesystem access.

Rules:
- The final expression must evaluate to the answer (a scalar, list, or dict). Prefer \`.to_dict()\` / \`.to_dict(orient="records")\` for tabular results so the value serializes cleanly.
- Do not call print(); return the value as the last expression.
- Do not define functions or classes unless strictly necessary. Keep it to a few lines.
- Never fabricate columns — use only the columns provided.`;

const JS_SYSTEM = `You generate JavaScript that answers questions about an in-memory dataset.

Environment:
- The code runs inside a sandboxed guest with a single \`ctx\` object: \`ctx.state\` (the data), \`ctx.args\`, and \`ctx.log(msg)\`.
- The dataset is an array of row objects at \`ctx.state.rows\`. Column names are provided below.
- There is no network, DOM, or filesystem access; no \`fetch\`, \`require\`, or \`import\`.

Rules:
- End with \`return <answer>;\` — a number, array, or object.
- Keep it to a few lines of plain ES2022. No external libraries.
- Never fabricate columns — use only the columns provided.`;

function buildPrompt(question: string, columns: string[], language: GuestLanguage, df: string): string {
  const cols = columns.length ? columns.join(', ') : '(unknown — inspect the data)';
  const ref = language === 'python' ? `the DataFrame \`${df}\`` : '`ctx.state.rows`';
  return `Columns in ${ref}: ${cols}\n\nQuestion: ${question}\n\nGenerate code that answers it.`;
}

export class CodeAnalyst {
  private readonly provider: LlmProvider;
  private readonly language: GuestLanguage;
  private readonly dataframeName: string;

  constructor(options: AnalystOptions) {
    this.provider = options.provider;
    this.language = options.language ?? 'python';
    this.dataframeName = options.dataframeName ?? 'df';
  }

  /** Generate code answering `question` against a dataset with `columns`. */
  async plan(question: string, columns: string[]): Promise<AnalysisPlan> {
    const system = this.language === 'python' ? PYTHON_SYSTEM(this.dataframeName) : JS_SYSTEM;
    const prompt = buildPrompt(question, columns, this.language, this.dataframeName);
    const result = await this.provider.generateCode({ system, prompt });
    return { ...result, language: this.language };
  }

  /**
   * Ask the model to repair code that failed, given the error. Returns a fresh
   * plan; pair with the sandbox's run-record/replay trail for an agentic
   * generate → run → repair loop.
   */
  async repair(
    question: string,
    columns: string[],
    failedCode: string,
    error: string
  ): Promise<AnalysisPlan> {
    const system = this.language === 'python' ? PYTHON_SYSTEM(this.dataframeName) : JS_SYSTEM;
    const prompt = `${buildPrompt(question, columns, this.language, this.dataframeName)}

The previous attempt failed. Fix it.

Previous code:
${failedCode}

Error:
${error}`;
    const result = await this.provider.generateCode({ system, prompt });
    return { ...result, language: this.language };
  }
}
