/**
 * Provider-agnostic LLM interface for code generation.
 *
 * The sandbox doesn't care which model writes the guest code — it governs
 * whatever runs. `ClaudeProvider` is the first implementation; other model
 * APIs implement the same two-method surface to plug in.
 */

export interface CodeGenRequest {
  /** Stable instructions describing the target environment and constraints. */
  system: string;
  /** The task (and, on retries, the failing code + error) to generate code for. */
  prompt: string;
  maxTokens?: number;
}

export interface LlmUsage {
  inputTokens: number;
  outputTokens: number;
}

export interface CodeGenResult {
  code: string;
  explanation: string;
  usage?: LlmUsage;
}

export interface LlmProvider {
  readonly name: string;
  generateCode(request: CodeGenRequest): Promise<CodeGenResult>;
}
