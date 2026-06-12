import { describe, expect, it } from 'vitest';
import { planQuestion } from '../src/python/planner';
import { PyodideLike, PythonRunner } from '../src/python/pyodide-runner';

const COLUMNS = ['country', 'revenue', 'units', 'region'];

describe('planQuestion', () => {
  it('counts rows', () => {
    expect(planQuestion('how many rows are there?', COLUMNS).code).toBe('len(df)');
  });

  it('lists columns', () => {
    expect(planQuestion('what columns does this have?', COLUMNS).code).toBe('list(df.columns)');
  });

  it('aggregates a single column', () => {
    expect(planQuestion('what is the average revenue?', COLUMNS).code).toBe('df["revenue"].mean()');
    expect(planQuestion('total units', COLUMNS).code).toBe('df["units"].sum()');
    expect(planQuestion('max revenue', COLUMNS).code).toBe('df["revenue"].max()');
  });

  it('builds a group-by when asked "by <column>"', () => {
    const plan = planQuestion('average revenue by region', COLUMNS);
    expect(plan.code).toBe('df.groupby("region")["revenue"].mean().to_dict()');
    expect(plan.fallback).toBe(false);
  });

  it('handles top-N by a column', () => {
    const plan = planQuestion('top 3 countries by revenue', COLUMNS);
    expect(plan.code).toBe('df.nlargest(3, "revenue").to_dict(orient="records")');
  });

  it('previews head with a default N', () => {
    expect(planQuestion('show me a preview', COLUMNS).code).toBe('df.head(5).to_dict(orient="records")');
  });

  it('falls back to describe for unrecognised questions', () => {
    const plan = planQuestion('what is the meaning of life?', COLUMNS);
    expect(plan.fallback).toBe(true);
    expect(plan.code).toContain('describe');
  });

  it('prefers the longest matching column name', () => {
    const plan = planQuestion('average revenue', ['rev', 'revenue']);
    expect(plan.code).toBe('df["revenue"].mean()');
  });
});

/** Minimal fake Pyodide that records calls and returns canned values. */
class FakePyodide implements PyodideLike {
  readonly ran: string[] = [];
  readonly loadedPackages: string[] = [];
  readonly globalsMap = new Map<string, unknown>();
  private stdoutCb: ((text: string) => void) | null = null;
  private nextValue: unknown = null;
  private throwOnRun = false;

  globals = {
    set: (name: string, value: unknown) => this.globalsMap.set(name, value),
    get: (name: string) => this.globalsMap.get(name),
  };

  setStdout(options: { batched: (text: string) => void }): void {
    this.stdoutCb = options.batched;
  }

  async loadPackage(names: string | string[]): Promise<void> {
    this.loadedPackages.push(...(Array.isArray(names) ? names : [names]));
  }

  willReturn(value: unknown): void {
    this.nextValue = value;
  }

  willThrow(): void {
    this.throwOnRun = true;
  }

  emitStdout(text: string): void {
    this.stdoutCb?.(text);
  }

  async runPythonAsync(code: string): Promise<unknown> {
    this.ran.push(code);
    if (this.throwOnRun) throw new Error('Traceback: ZeroDivisionError');
    return this.nextValue;
  }
}

describe('PythonRunner orchestration', () => {
  it('loads a CSV into a named DataFrame via pandas', async () => {
    const py = new FakePyodide();
    const runner = new PythonRunner(py);
    await runner.loadDataframe('df', 'a,b\n1,2\n');
    expect(py.loadedPackages).toContain('pandas');
    expect(py.globalsMap.get('__webkaya_csv')).toBe('a,b\n1,2\n');
    expect(py.ran.some((c) => c.includes('df = pd.read_csv'))).toBe(true);
  });

  it('rejects an invalid DataFrame name', async () => {
    const runner = new PythonRunner(new FakePyodide());
    await expect(runner.loadDataframe('bad name', 'a\n1\n')).rejects.toThrow(/Invalid DataFrame name/);
  });

  it('returns the value and captured stdout on success', async () => {
    const py = new FakePyodide();
    const runner = new PythonRunner(py);
    py.willReturn(42);
    const resultPromise = runner.run('print("hi"); 42');
    py.emitStdout('hi\n');
    const result = await resultPromise;
    expect(result.ok).toBe(true);
    expect(result.value).toBe(42);
  });

  it('captures Python errors as ok:false', async () => {
    const py = new FakePyodide();
    const runner = new PythonRunner(py);
    py.willThrow();
    const result = await runner.run('1/0');
    expect(result.ok).toBe(false);
    expect(result.error).toMatch(/ZeroDivisionError/);
  });

  it('converts Pyodide proxy values via toJs', async () => {
    const py = new FakePyodide();
    const runner = new PythonRunner(py);
    py.willReturn({ toJs: () => ({ mean: 10 }) });
    const result = await runner.run('df["x"].mean()');
    expect(result.value).toEqual({ mean: 10 });
  });
});
