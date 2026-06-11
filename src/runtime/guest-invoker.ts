import { SandboxPolicy } from '../types/policy';

const WASM_SIMD_TEST = new Uint8Array([
  0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11,
]);

/**
 * Executes guest invocations under a sandbox policy: warm-up, per-invocation
 * timeout, declared-memory budgeting, and bounded retries.
 */
export class GuestInvoker {
  private warmed = false;
  private readonly policy: SandboxPolicy;
  private readonly log: (message: string) => void;
  private readonly warn: (message: string) => void;

  constructor(
    policy: SandboxPolicy,
    log: (message: string) => void = () => {},
    warn: (message: string) => void = () => {}
  ) {
    this.policy = policy;
    this.log = log;
    this.warn = warn;
  }

  isWasmSupported(): boolean {
    try {
      return typeof WebAssembly !== 'undefined' && WebAssembly.validate(WASM_SIMD_TEST);
    } catch {
      return false;
    }
  }

  async warmupIfNeeded(): Promise<void> {
    if (this.warmed) return;
    if (this.policy.coldStartMs > 0) {
      await new Promise((resolve) => setTimeout(resolve, this.policy.coldStartMs));
    }
    this.warmed = true;
    this.log(`Sandbox runtime warm-up complete for "${this.policy.entrypoint}".`);
  }

  private runWithTimeout<T>(handler: () => Promise<T> | T): Promise<T> {
    const timeoutMs = Math.max(100, this.policy.timeoutMs);
    return new Promise<T>((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error(`Invocation timeout (${timeoutMs}ms)`)), timeoutMs);
      Promise.resolve()
        .then(() => handler())
        .then((value) => {
          clearTimeout(timer);
          resolve(value);
        })
        .catch((error) => {
          clearTimeout(timer);
          reject(error);
        });
    });
  }

  async invoke<T>(
    invocationName: string,
    estimatedMemoryMB: number,
    handler: () => Promise<T> | T
  ): Promise<T> {
    await this.warmupIfNeeded();

    if (estimatedMemoryMB > this.policy.memoryBudgetMB) {
      throw new Error(
        `Invocation "${invocationName}" requires ~${estimatedMemoryMB.toFixed(2)}MB, above sandbox budget ${this.policy.memoryBudgetMB}MB.`
      );
    }

    const retries = Math.max(0, this.policy.retryCount);
    let lastError: unknown = null;
    for (let attempt = 0; attempt <= retries; attempt++) {
      try {
        return await this.runWithTimeout(handler);
      } catch (error) {
        lastError = error;
        const attemptNumber = attempt + 1;
        if (attemptNumber <= retries) {
          this.warn(`Invocation "${invocationName}" failed on attempt ${attemptNumber}. Retrying...`);
        }
      }
    }
    throw lastError instanceof Error ? lastError : new Error('Sandbox invocation failed.');
  }
}
