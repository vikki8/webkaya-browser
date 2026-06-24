import { ExecOutcome, ExecRequest, SandboxExecutor } from '../../sandbox/executor.js';
import { SandboxPolicy } from '../../types/policy.js';

/**
 * Executes guest JavaScript inside QuickJS compiled to WebAssembly — a real
 * realm boundary, not the same-realm `new Function` boundary of inline mode.
 *
 * The guest has no reference to the host: no `window`, `fetch`, `process`, no
 * prototype-walk to host globals. It cannot reach the network, DOM, or
 * filesystem because those objects do not exist in its realm. State and args
 * cross the boundary only as JSON; the return value comes back as JSON. The
 * runtime additionally enforces a hard memory limit and interrupts a guest
 * that exceeds its time budget (e.g. an infinite loop) — neither of which the
 * inline boundary can do.
 *
 * `quickjs-emscripten` is an optional dependency, loaded on first use, so the
 * core package stays dependency-free unless you opt into `runtime: 'wasm'`.
 */

let modulePromise: Promise<any> | null = null;

async function loadQuickJS(): Promise<any> {
  if (!modulePromise) {
    modulePromise = (async () => {
      let mod: any;
      try {
        mod = await import('quickjs-emscripten');
      } catch {
        throw new Error(
          'runtime "wasm" requires the optional "quickjs-emscripten" package. Install it with: npm install quickjs-emscripten'
        );
      }
      return mod.getQuickJS();
    })();
  }
  return modulePromise;
}

function buildWrapper(code: string, state: Record<string, unknown>, args: unknown): string {
  const stateJson = JSON.stringify(state ?? {});
  const argsJson = JSON.stringify(args === undefined ? null : args);
  // The guest body runs as `__guest(ctx)`, matching the inline/worker contract.
  // Everything is wrapped so guest throws, serialization errors, and the final
  // result all come back as a single JSON string.
  return `(function(){"use strict";
var __state=${stateJson};var __args=${argsJson};var __logs=[];
var ctx={state:__state,args:__args,log:function(m){__logs.push(String(m));}};
var __guest=function(ctx){
${code}
};
try{var __v=__guest(ctx);return JSON.stringify({ok:true,value:__v===undefined?null:__v,state:ctx.state,logs:__logs});}
catch(e){return JSON.stringify({ok:false,error:(e&&e.message)?String(e.message):String(e),state:__state,logs:__logs});}
})()`;
}

export class QuickJsExecutor implements SandboxExecutor {
  constructor(private readonly policy: SandboxPolicy) {}

  async execute(request: ExecRequest): Promise<ExecOutcome> {
    let QuickJS: any;
    try {
      QuickJS = await loadQuickJS();
    } catch (error) {
      return { ok: false, error: error instanceof Error ? error.message : String(error), state: request.state, logs: [] };
    }

    const vm = QuickJS.newContext();
    try {
      vm.runtime.setMemoryLimit(Math.max(16, this.policy.memoryBudgetMB) * 1024 * 1024);
      const timeoutMs = Math.max(100, this.policy.timeoutMs);
      const deadline = Date.now() + timeoutMs;
      vm.runtime.setInterruptHandler(() => Date.now() > deadline);

      const result = vm.evalCode(buildWrapper(request.code, request.state, request.args));

      if (result.error) {
        const dumped = vm.dump(result.error);
        result.error.dispose();
        const message = dumped && dumped.message ? String(dumped.message) : String(dumped);
        const timedOut = Date.now() > deadline || /interrupt/i.test(message);
        return {
          ok: false,
          error: timedOut
            ? `Invocation "${request.name}" exceeded its ${timeoutMs}ms time budget and was interrupted.`
            : message,
          state: request.state,
          logs: [],
        };
      }

      const json = vm.getString(result.value);
      result.value.dispose();
      const parsed = JSON.parse(json) as {
        ok: boolean;
        value?: unknown;
        error?: string;
        state?: Record<string, unknown>;
        logs?: string[];
      };
      return {
        ok: parsed.ok,
        value: parsed.value,
        error: parsed.error,
        state: parsed.ok && parsed.state ? parsed.state : request.state,
        logs: parsed.logs ?? [],
      };
    } finally {
      vm.dispose();
    }
  }

  dispose(): void {
    /* contexts are created and disposed per run; the WASM module is shared */
  }
}
