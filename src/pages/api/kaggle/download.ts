import type { NextApiRequest, NextApiResponse } from 'next';
import { Readable, Transform } from 'stream';
import { pipeline } from 'stream/promises';
import {
  KaggleAuth,
  kaggleAuthorization,
  readAuthFromRequest,
  toDatasetRef,
  validateKaggleAuth,
} from '../../../server/kaggle';
import { applyRateLimit, enforceSameOrigin } from '../../../server/request-guard';

/** 0 = unlimited. Default unlimited (set e.g. 400MB cap in production if needed). */
function readMaxDownloadBytes(): number {
  const raw = process.env.BROWSER_FIRST_AI_KAGGLE_MAX_DOWNLOAD_BYTES;
  if (raw === undefined || raw === '') return 0;
  const n = Number.parseInt(raw, 10);
  if (!Number.isFinite(n) || n < 0) return 0;
  return n;
}

/** Idle time with no bytes before aborting upstream read. 0 = disabled. Default 5 minutes. */
function readIdleTimeoutMs(): number {
  const raw = process.env.BROWSER_FIRST_AI_KAGGLE_IDLE_TIMEOUT_MS;
  if (raw === undefined || raw === '') return 300_000;
  const n = Number.parseInt(raw, 10);
  if (!Number.isFinite(n) || n < 0) return 300_000;
  return n;
}

export const config = {
  api: {
    bodyParser: {
      sizeLimit: '1mb',
    },
    responseLimit: false,
  },
};

function inferFilename(contentDisposition: string | null, fallback: string): string {
  if (!contentDisposition) return fallback;
  const match = contentDisposition.match(/filename="?([^"]+)"?/i);
  return match?.[1] ?? fallback;
}

class ByteLimitTransform extends Transform {
  private readonly limitBytes: number;
  private totalBytes = 0;

  constructor(limitBytes: number) {
    super();
    this.limitBytes = limitBytes;
  }

  _transform(chunk: Buffer, _encoding: BufferEncoding, callback: (error?: Error | null, data?: Buffer) => void): void {
    this.totalBytes += chunk.length;
    if (this.totalBytes > this.limitBytes) {
      callback(
        new Error(`Kaggle download exceeds ${Math.floor(this.limitBytes / (1024 * 1024))}MB limit (BROWSER_FIRST_AI_KAGGLE_MAX_DOWNLOAD_BYTES).`)
      );
      return;
    }
    callback(null, chunk);
  }
}

/**
 * Abort if no chunk arrives for `idleMs`. Resets on each chunk. 0 = pass-through.
 */
function wrapWebStreamWithIdleTimeout(body: ReadableStream<Uint8Array>, idleMs: number): ReadableStream<Uint8Array> {
  if (idleMs <= 0) return body;

  const reader = body.getReader();
  let idleTimer: ReturnType<typeof setTimeout> | null = null;

  const clearIdle = () => {
    if (idleTimer) {
      clearTimeout(idleTimer);
      idleTimer = null;
    }
  };

  const stallError = () => new Error(`Kaggle download stalled: no data for ${idleMs}ms (BROWSER_FIRST_AI_KAGGLE_IDLE_TIMEOUT_MS).`);

  return new ReadableStream({
    async start(controller) {
      try {
        while (true) {
          idleTimer = setTimeout(() => {
            reader.cancel(stallError()).catch(() => {});
            controller.error(stallError());
          }, idleMs);
          const { done, value } = await reader.read();
          clearIdle();
          if (done) {
            controller.close();
            break;
          }
          if (value) controller.enqueue(value);
        }
      } catch (e) {
        clearIdle();
        controller.error(e);
      }
    },
    cancel(reason) {
      clearIdle();
      return reader.cancel(reason);
    },
  });
}

async function fetchKaggleDownload(url: string, init: RequestInit, idleMs: number): Promise<Response> {
  const response = await fetch(url, init);
  if (!response.ok || !response.body) {
    return response;
  }
  const wrappedBody = wrapWebStreamWithIdleTimeout(response.body, idleMs);
  return new Response(wrappedBody, {
    status: response.status,
    statusText: response.statusText,
    headers: response.headers,
  });
}

async function tryDownload(datasetRef: string, fileName: string | null, auth: KaggleAuth, idleMs: number): Promise<Response | null> {
  const [owner, dataset] = datasetRef.split('/');
  const base = `https://www.kaggle.com/api/v1`;

  const candidates = [
    fileName ? `${base}/datasets/download/${owner}/${dataset}?fileName=${encodeURIComponent(fileName)}` : null,
    fileName ? `${base}/datasets/download/${owner}/${dataset}?file_name=${encodeURIComponent(fileName)}` : null,
    fileName ? `${base}/datasets/download/${owner}/${dataset}/${encodeURIComponent(fileName)}` : null,
    `${base}/datasets/download/${owner}/${dataset}`,
  ].filter(Boolean) as string[];

  for (const url of candidates) {
    const response = await fetchKaggleDownload(
      url,
      {
        headers: {
          Authorization: kaggleAuthorization(auth),
          Accept: '*/*',
          'User-Agent': 'BrowserFirstAI-Platform/0.1.0',
        },
      },
      idleMs
    );
    if (response.ok) return response;
  }
  return null;
}

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    res.setHeader('Allow', 'POST');
    return res.status(405).json({ error: 'Method not allowed' });
  }
  if (!enforceSameOrigin(req, res)) return;
  if (
    !applyRateLimit(req, res, {
      bucket: 'kaggle_download',
      max: 6,
      windowMs: 60_000,
    })
  ) {
    return;
  }

  const auth = readAuthFromRequest(req);
  if (!auth) {
    return res.status(400).json({
      error: 'Kaggle authentication is required (OAuth token or username+API key).',
    });
  }
  const authValidation = await validateKaggleAuth(auth);
  if (!authValidation.ok) {
    return res.status(authValidation.status).json({
      error: 'Kaggle authentication is invalid. Please reconnect OAuth/API key.',
      detail: authValidation.detail,
    });
  }

  const rawRef = String(req.body?.datasetRef ?? '').trim();
  const fileName = req.body?.fileName ? String(req.body.fileName) : null;
  if (!rawRef) return res.status(400).json({ error: 'datasetRef is required.' });

  const maxBytes = readMaxDownloadBytes();
  const idleMs = readIdleTimeoutMs();

  try {
    const datasetRef = toDatasetRef(rawRef);
    const response = await tryDownload(datasetRef, fileName, auth, idleMs);
    if (!response) {
      return res.status(502).json({
        error: 'Kaggle download failed. Verify dataset access and file name.',
      });
    }
    if (!response.body) {
      return res.status(502).json({
        error: 'Kaggle download failed: empty response body.',
      });
    }

    const fallback = fileName || `${datasetRef.split('/')[1]}.zip`;
    const resolvedFileName = inferFilename(response.headers.get('content-disposition'), fallback);
    const headerLength = Number(response.headers.get('content-length') || 0);
    if (maxBytes > 0 && Number.isFinite(headerLength) && headerLength > maxBytes) {
      return res.status(413).json({
        error: `Dataset exceeds configured max download size (${Math.floor(maxBytes / (1024 * 1024))}MB). Set BROWSER_FIRST_AI_KAGGLE_MAX_DOWNLOAD_BYTES=0 for unlimited.`,
      });
    }

    const contentType = response.headers.get('content-type') || 'application/octet-stream';
    const contentLength = response.headers.get('content-length');
    res.setHeader('Content-Type', contentType);
    res.setHeader('X-Browser-First-AI-Filename', resolvedFileName);
    if (contentLength) {
      res.setHeader('X-Browser-First-AI-Content-Length', contentLength);
      res.setHeader('Content-Length', contentLength);
    }
    res.setHeader('Cache-Control', 'no-store');
    res.status(200);

    const body = Readable.fromWeb(response.body as any);
    if (maxBytes > 0) {
      const limiter = new ByteLimitTransform(maxBytes);
      await pipeline(body, limiter, res);
    } else {
      await pipeline(body, res);
    }
    return;
  } catch (error: any) {
    if (!res.headersSent && /exceeds .*MB limit/i.test(error?.message ?? '')) {
      return res.status(413).json({
        error: error.message,
      });
    }
    return res.status(500).json({
      error: 'Unexpected Kaggle download failure.',
      detail: error?.message ?? 'Unknown server error',
    });
  }
}
