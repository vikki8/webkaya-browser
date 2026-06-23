import type Anthropic from '@anthropic-ai/sdk';
import { CodeGenRequest, CodeGenResult, LlmProvider } from './provider.js';

/**
 * Claude-backed code generation via the official Anthropic SDK.
 *
 * The SDK is an optional peer dependency loaded on first use, so the core
 * package stays dependency-free for users who don't enable LLM features.
 * Defaults: `claude-opus-4-8`, adaptive thinking, structured JSON output
 * ({code, explanation}) enforced by the API.
 */

export interface ClaudeProviderOptions {
  /** Defaults to the ANTHROPIC_API_KEY environment variable (Node only). */
  apiKey?: string;
  /** Model ID; defaults to claude-opus-4-8. */
  model?: string;
  maxTokens?: number;
  /**
   * Required to run in a browser tab (the demo's mode). Exposes the API key
   * to the page — use a key you can revoke, never a production key.
   */
  dangerouslyAllowBrowser?: boolean;
}

const CODE_SCHEMA = {
  type: 'object',
  properties: {
    code: { type: 'string', description: 'The generated source code, ready to execute.' },
    explanation: { type: 'string', description: 'One or two sentences describing what the code does.' },
  },
  required: ['code', 'explanation'],
  additionalProperties: false,
} as const;

type AnthropicModule = typeof import('@anthropic-ai/sdk');
let anthropicModule: AnthropicModule | null = null;

async function loadAnthropic(): Promise<AnthropicModule> {
  if (!anthropicModule) {
    try {
      anthropicModule = await import('@anthropic-ai/sdk');
    } catch {
      throw new Error(
        'ClaudeProvider requires the optional "@anthropic-ai/sdk" package. Install it with: npm install @anthropic-ai/sdk'
      );
    }
  }
  return anthropicModule;
}

export class ClaudeProvider implements LlmProvider {
  readonly name = 'claude';
  readonly model: string;
  private readonly options: ClaudeProviderOptions;
  // Typed as `unknown` so the Anthropic SDK type doesn't leak into the public
  // .d.ts — consumers who don't use LLM features needn't install the SDK.
  private client: unknown = null;

  constructor(options: ClaudeProviderOptions = {}) {
    this.options = options;
    this.model = options.model ?? 'claude-opus-4-8';
  }

  private async getClient(): Promise<Anthropic> {
    if (!this.client) {
      const mod = await loadAnthropic();
      this.client = new mod.default({
        apiKey: this.options.apiKey,
        dangerouslyAllowBrowser:
          this.options.dangerouslyAllowBrowser ?? (typeof window !== 'undefined' ? true : undefined),
      });
    }
    return this.client as Anthropic;
  }

  async generateCode(request: CodeGenRequest): Promise<CodeGenResult> {
    const mod = await loadAnthropic();
    const client = await this.getClient();

    let response: Anthropic.Message;
    try {
      response = await client.messages.create({
        model: this.model,
        max_tokens: request.maxTokens ?? this.options.maxTokens ?? 4096,
        thinking: { type: 'adaptive' },
        system: request.system,
        output_config: { format: { type: 'json_schema', schema: CODE_SCHEMA } },
        messages: [{ role: 'user', content: request.prompt }],
      } as Anthropic.MessageCreateParamsNonStreaming);
    } catch (error) {
      if (error instanceof mod.default.AuthenticationError) {
        throw new Error('Claude API: invalid or missing API key.');
      }
      if (error instanceof mod.default.RateLimitError) {
        throw new Error('Claude API: rate limited — wait a moment and retry.');
      }
      if (error instanceof mod.default.APIError) {
        throw new Error(`Claude API error ${error.status}: ${error.message}`);
      }
      throw error;
    }

    if (response.stop_reason === 'refusal') {
      throw new Error('Claude declined to generate code for this request.');
    }
    if (response.stop_reason === 'max_tokens') {
      throw new Error('Claude hit the output token limit — raise maxTokens and retry.');
    }

    const text = response.content.find(
      (block): block is Anthropic.TextBlock => block.type === 'text'
    )?.text;
    if (!text) {
      throw new Error('Claude returned no text content.');
    }

    let parsed: { code?: unknown; explanation?: unknown };
    try {
      parsed = JSON.parse(text) as { code?: unknown; explanation?: unknown };
    } catch {
      throw new Error('Claude returned output that was not valid JSON.');
    }
    if (typeof parsed.code !== 'string' || !parsed.code.trim()) {
      throw new Error('Claude returned no code.');
    }

    return {
      code: parsed.code,
      explanation: typeof parsed.explanation === 'string' ? parsed.explanation : '',
      usage: {
        inputTokens: response.usage.input_tokens,
        outputTokens: response.usage.output_tokens,
      },
    };
  }
}
