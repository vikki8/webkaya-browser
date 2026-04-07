import { WasmFunctionInvocationPolicy } from '../types/training-workflow';

const DEFAULT_SIMD_TEST = new Uint8Array([
  0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11,
]);

export class WasmFunctionRuntime {
  private warmed = false;
  private readonly config: WasmFunctionInvocationPolicy;
  private readonly log: (message: string) => void;
  private readonly warn: (message: string) => void;

  constructor(
    config: WasmFunctionInvocationPolicy,
    log: (message: string) => void,
    warn: (message: string) => void
  ) {
    this.config = config;
    this.log = log;
    this.warn = warn;
  }

  isWasmSupported(): boolean {
    try {
      return typeof WebAssembly !== 'undefined' && WebAssembly.validate(DEFAULT_SIMD_TEST);
    } catch {
      return false;
    }
  }

  async warmupIfNeeded(): Promise<void> {
    if (this.warmed) return;
    if (this.config.coldStartMs > 0) {
      await new Promise((resolve) => setTimeout(resolve, this.config.coldStartMs));
    }
    this.warmed = true;
    this.log(`WASM function runtime warm-up complete for "${this.config.functionName}".`);
  }

  private runWithTimeout<T>(handler: () => Promise<T> | T): Promise<T> {
    const timeoutMs = Math.max(100, this.config.invocationTimeoutMs);
    return new Promise<T>((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error(`Function timeout (${timeoutMs}ms)`)), timeoutMs);
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

    if (estimatedMemoryMB > this.config.memoryBudgetMB) {
      throw new Error(
        `Invocation "${invocationName}" requires ~${estimatedMemoryMB.toFixed(2)}MB, above function budget ${this.config.memoryBudgetMB}MB.`
      );
    }

    const retries = Math.max(0, this.config.retryCount);
    let lastError: unknown = null;
    for (let attempt = 0; attempt <= retries; attempt++) {
      try {
        return await this.runWithTimeout(handler);
      } catch (error) {
        lastError = error;
        const attemptNumber = attempt + 1;
        if (attemptNumber <= retries) {
          this.warn(`Function "${invocationName}" failed on attempt ${attemptNumber}. Retrying...`);
        }
      }
    }
    throw lastError instanceof Error ? lastError : new Error('WASM function invocation failed.');
  }
}
