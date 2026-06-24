// Egress monitor: wraps fetch/XHR before anything else loads and records every
// outbound request, checking each body against fingerprints of the loaded
// dataset. Stated factually in the UI as a "Network" panel.

export interface EgressRecord {
  host: string;
  purpose: string;
  method: string;
  bytes: number;
  containedData: boolean;
}

export interface Ledger {
  records: EgressRecord[];
  dataBytesSent: number;
  questionBytes: number;
  datasetBytes: number;
}

const records: EgressRecord[] = [];
const listeners = new Set<(l: Ledger) => void>();
let fingerprints: string[] = [];
let datasetBytes = 0;

function classify(rawUrl: string): { host: string; purpose: string } {
  let host = rawUrl;
  try {
    host = new URL(rawUrl, location.href).host;
  } catch { /* keep raw */ }
  if (/jsdelivr|pyodide|cdn\./i.test(host)) return { host, purpose: 'Python runtime' };
  if (/anthropic/i.test(host)) return { host, purpose: 'Model API (question only)' };
  return { host, purpose: 'other' };
}

function bodyText(body: unknown): string | null {
  if (typeof body === 'string') return body;
  if (body instanceof URLSearchParams) return body.toString();
  return null;
}

function byteLength(text: string | null): number {
  return text ? new Blob([text]).size : 0;
}

function snapshot(): Ledger {
  const leaks = records.filter((r) => r.containedData);
  return {
    records: records.slice(),
    dataBytesSent: leaks.reduce((s, r) => s + r.bytes, 0),
    questionBytes: records.filter((r) => /anthropic/i.test(r.host)).reduce((s, r) => s + r.bytes, 0),
    datasetBytes,
  };
}

function emit(): void {
  const l = snapshot();
  listeners.forEach((fn) => fn(l));
}

function record(rawUrl: string, method: string, body: unknown): void {
  const { host, purpose } = classify(rawUrl);
  const text = bodyText(body);
  records.push({
    host,
    purpose,
    method: method || 'GET',
    bytes: byteLength(text),
    containedData: !!text && fingerprints.some((fp) => fp && text.includes(fp)),
  });
  emit();
}

export function subscribe(fn: (l: Ledger) => void): () => void {
  listeners.add(fn);
  fn(snapshot());
  return () => listeners.delete(fn);
}

export function watchDataset(csvText: string): void {
  datasetBytes = byteLength(csvText);
  const cells = csvText
    .split(/\r?\n/)
    .slice(1, 8)
    .flatMap((line) => line.split(','))
    .map((c) => c.trim())
    .filter((c) => c.length >= 4);
  fingerprints = [...new Set(cells)].sort((a, b) => b.length - a.length).slice(0, 12);
  emit();
}

export function installEgressMonitor(): void {
  const w = window as unknown as { __wkEgress?: boolean };
  if (w.__wkEgress) return;
  w.__wkEgress = true;

  const realFetch = window.fetch?.bind(window);
  if (realFetch) {
    window.fetch = (input: RequestInfo | URL, init: RequestInit = {}) => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.href : (input as Request).url;
      const method = init.method || (input instanceof Request ? input.method : 'GET');
      record(url, method, init.body);
      return realFetch(input, init);
    };
  }

  const XHR = window.XMLHttpRequest;
  if (XHR) {
    const open = XHR.prototype.open;
    const send = XHR.prototype.send;
    XHR.prototype.open = function (this: XMLHttpRequest & { __wk?: { method: string; url: string } }, method: string, url: string | URL, ...rest: unknown[]) {
      this.__wk = { method, url: String(url) };
      // @ts-expect-error variadic passthrough
      return open.call(this, method, url, ...rest);
    };
    XHR.prototype.send = function (this: XMLHttpRequest & { __wk?: { method: string; url: string } }, body?: Document | XMLHttpRequestBodyInit | null) {
      if (this.__wk) record(this.__wk.url, this.__wk.method, body);
      return send.call(this, body as XMLHttpRequestBodyInit);
    };
  }
}
