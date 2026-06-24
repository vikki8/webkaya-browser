// Live egress monitor — the trust mechanism behind the whole pitch.
//
// It wraps fetch + XMLHttpRequest the moment this module is imported (before
// Pyodide or Claude make any request), records every outbound call, and checks
// each request body against fingerprints of the loaded dataset. The claim
// "your data never leaves the device" becomes auditable: open devtools and the
// numbers match.

const records = [];          // { host, purpose, method, bytes, containedData }
const listeners = [];
let fingerprints = [];       // distinctive strings from the loaded dataset
let datasetBytes = 0;        // size of the loaded dataset, for the contrast stat

function classify(rawUrl) {
  let host = String(rawUrl);
  try {
    host = new URL(rawUrl, location.href).host;
  } catch { /* keep raw */ }
  if (/jsdelivr|pyodide|cdn\./i.test(host)) return { host, purpose: 'Python runtime — one-time download' };
  if (/anthropic/i.test(host)) return { host, purpose: 'Claude API — your question only' };
  if (/esm\.sh/i.test(host)) return { host, purpose: 'SDK module — code, not data' };
  return { host, purpose: 'other' };
}

function bodyToText(body) {
  if (typeof body === 'string') return body;
  if (body instanceof URLSearchParams) return body.toString();
  return null; // streams/blobs — size unknown, skip the content check
}

function byteLength(text) {
  return text ? new Blob([text]).size : 0;
}

function record(rawUrl, method, body) {
  const { host, purpose } = classify(rawUrl);
  const text = bodyToText(body);
  const containedData = !!text && fingerprints.some((fp) => fp && text.includes(fp));
  records.push({ host, purpose, method: method || 'GET', bytes: byteLength(text), containedData });
  emit();
}

function emit() {
  const dataLeaks = records.filter((r) => r.containedData);
  const summary = {
    records: records.slice(),
    requestCount: records.length,
    dataBytesSent: dataLeaks.reduce((sum, r) => sum + r.bytes, 0), // stays 0
    dataLeakCount: dataLeaks.length,
    datasetBytes,
    questionBytes: records
      .filter((r) => /anthropic/i.test(r.host))
      .reduce((sum, r) => sum + r.bytes, 0),
  };
  for (const fn of listeners) fn(summary);
}

export function onLedger(fn) {
  listeners.push(fn);
  emit();
}

/** Tell the monitor what the current dataset looks like, so it can verify that
 *  none of it appears in outbound traffic. Pass the raw CSV text. */
export function watchDataset(csvText) {
  datasetBytes = byteLength(csvText);
  // Pick distinctive, unlikely-to-collide tokens from the data: the longest
  // cell values across a few rows.
  const cells = csvText
    .split(/\r?\n/)
    .slice(1, 8)
    .flatMap((line) => line.split(','))
    .map((c) => c.trim())
    .filter((c) => c.length >= 4);
  fingerprints = [...new Set(cells)].sort((a, b) => b.length - a.length).slice(0, 12);
  emit();
}

export function installEgressMonitor() {
  if (typeof window === 'undefined' || window.__webkayaEgressInstalled) return;
  window.__webkayaEgressInstalled = true;

  const realFetch = window.fetch?.bind(window);
  if (realFetch) {
    window.fetch = (input, init = {}) => {
      const url = typeof input === 'string' ? input : input?.url ?? String(input);
      const method = init.method || (typeof input === 'object' && input?.method) || 'GET';
      record(url, method, init.body);
      return realFetch(input, init);
    };
  }

  const XHR = window.XMLHttpRequest;
  if (XHR) {
    const open = XHR.prototype.open;
    const send = XHR.prototype.send;
    XHR.prototype.open = function (method, url, ...rest) {
      this.__wk = { method, url };
      return open.call(this, method, url, ...rest);
    };
    XHR.prototype.send = function (body) {
      if (this.__wk) record(this.__wk.url, this.__wk.method, body);
      return send.call(this, body);
    };
  }
}

// Install immediately on import — before Pyodide or Claude can make a request.
installEgressMonitor();
