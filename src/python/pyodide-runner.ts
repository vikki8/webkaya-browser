/**
 * Python guest runtime via Pyodide (CPython compiled to WASM).
 *
 * This is the runtime behind the "analyze my local data" wedge: a user's CSV
 * becomes a pandas DataFrame inside the browser, agent-generated Python runs
 * over it, and the data never leaves the device. Pyodide is loaded at runtime
 * from a CDN (or a self-hosted indexURL), so the core SDK keeps zero install
 * dependencies.
 *
 * The runtime is abstracted behind `PyodideLike` so the orchestration here is
 * testable with a fake; real Python execution is browser-only.
 */

export interface PyodideLike {
  runPythonAsync(code: string): Promise<unknown>;
  globals: {
    set(name: string, value: unknown): void;
    get(name: string): unknown;
  };
  loadPackage?(names: string | string[]): Promise<void>;
  setStdout?(options: { batched: (text: string) => void }): void;
  setStderr?(options: { batched: (text: string) => void }): void;
}

export interface PythonRunResult {
  ok: boolean;
  value?: unknown;
  stdout: string;
  error?: string;
  durationMs: number;
}

function now(): number {
  return typeof performance !== 'undefined' ? performance.now() : Date.now();
}

function toPlainJs(value: unknown): unknown {
  if (value && typeof (value as { toJs?: unknown }).toJs === 'function') {
    try {
      return (value as { toJs: (opts?: unknown) => unknown }).toJs({ dict_converter: Object.fromEntries });
    } catch {
      return value;
    }
  }
  return value;
}

/**
 * Wraps a loaded Pyodide instance with stdout capture, a one-call helper to
 * load a CSV into a named DataFrame, and a result-returning `run`.
 */
export class PythonRunner {
  private stdoutBuffer = '';

  constructor(private readonly py: PyodideLike) {
    this.py.setStdout?.({ batched: (text) => { this.stdoutBuffer += text; } });
    this.py.setStderr?.({ batched: (text) => { this.stdoutBuffer += text; } });
  }

  /** Read CSV text into a pandas DataFrame bound to `name` in the Python globals. */
  async loadDataframe(name: string, csvText: string): Promise<void> {
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(name)) {
      throw new Error(`Invalid DataFrame name: "${name}".`);
    }
    await this.py.loadPackage?.('pandas');
    this.py.globals.set('__webkaya_csv', csvText);
    await this.py.runPythonAsync(
      `import pandas as pd, io\n${name} = pd.read_csv(io.StringIO(__webkaya_csv))\n`
    );
  }

  /** Run Python; the value of the final expression is returned (converted to JS). */
  async run(code: string): Promise<PythonRunResult> {
    this.stdoutBuffer = '';
    const start = now();
    try {
      const value = await this.py.runPythonAsync(code);
      return { ok: true, value: toPlainJs(value), stdout: this.stdoutBuffer, durationMs: now() - start };
    } catch (error) {
      return {
        ok: false,
        error: error instanceof Error ? error.message : String(error),
        stdout: this.stdoutBuffer,
        durationMs: now() - start,
      };
    }
  }
}

export interface LoadPyodideOptions {
  /** Pyodide distribution URL; defaults to the jsDelivr CDN. */
  indexURL?: string;
}

const DEFAULT_INDEX_URL = 'https://cdn.jsdelivr.net/pyodide/v0.26.2/full/';

/**
 * Dynamically load Pyodide in the browser. Throws outside a browser realm —
 * in production you typically run this inside a Web Worker (pair with
 * `runtime: 'worker'`) so Python executes off the main thread.
 */
export async function loadPyodideRuntime(options: LoadPyodideOptions = {}): Promise<PyodideLike> {
  if (typeof window === 'undefined' && typeof self === 'undefined') {
    throw new Error('Pyodide runs in a browser or Web Worker realm only.');
  }
  const indexURL = options.indexURL ?? DEFAULT_INDEX_URL;
  const mod = (await import(/* @vite-ignore */ `${indexURL}pyodide.mjs`)) as {
    loadPyodide: (opts: { indexURL: string }) => Promise<PyodideLike>;
  };
  return mod.loadPyodide({ indexURL });
}
