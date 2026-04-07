/**
 * Streams a Kaggle proxy response to OPFS to avoid holding the full payload in RAM.
 * Falls back to response.blob() when OPFS is unavailable.
 */

export interface KaggleDownloadStreamProgress {
  bytesReceived: number;
  totalBytes: number | null;
}

export interface KaggleDownloadResult {
  file: File;
  /** Remove the temporary OPFS entry; safe to call after parsing. */
  removeTemp: () => Promise<void>;
}

function randomOpfsPartName(): string {
  const id =
    typeof crypto !== 'undefined' && 'randomUUID' in crypto
      ? crypto.randomUUID()
      : `${Date.now()}-${Math.random().toString(36).slice(2)}`;
  return `.browser-first-ai-kaggle-${id}.part`;
}

/**
 * Reads `fetch('/api/kaggle/download')` response body into a `File`, preferring OPFS.
 */
export async function streamKaggleDownloadToFile(
  response: Response,
  filenameHint: string,
  options?: {
    onProgress?: (p: KaggleDownloadStreamProgress) => void;
  }
): Promise<KaggleDownloadResult> {
  const body = response.body;
  if (!body) {
    throw new Error('Kaggle download failed: empty response body.');
  }

  const rawLen = response.headers.get('x-browser-first-ai-content-length') || response.headers.get('content-length');
  const totalBytes = rawLen ? Number.parseInt(rawLen, 10) : null;
  const total = totalBytes !== null && Number.isFinite(totalBytes) ? totalBytes : null;

  const root = typeof navigator !== 'undefined' && navigator.storage?.getDirectory ? await navigator.storage.getDirectory() : null;

  if (!root) {
    const blob = await response.blob();
    options?.onProgress?.({
      bytesReceived: blob.size,
      totalBytes: total ?? blob.size,
    });
    const file = new File([blob], filenameHint, { type: blob.type || 'application/octet-stream' });
    return {
      file,
      removeTemp: async () => {},
    };
  }

  const entryName = randomOpfsPartName();
  const fileHandle = await root.getFileHandle(entryName, { create: true });
  const writable = await fileHandle.createWritable();

  let received = 0;
  const reader = body.getReader();

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (value && value.byteLength) {
        await writable.write(value);
        received += value.byteLength;
        options?.onProgress?.({
          bytesReceived: received,
          totalBytes: total,
        });
      }
    }
  } finally {
    await writable.close();
  }

  const rawFile = await fileHandle.getFile();
  const mime = response.headers.get('content-type') || 'application/octet-stream';
  const file = new File([rawFile], filenameHint, { type: mime });

  options?.onProgress?.({
    bytesReceived: received,
    totalBytes: total ?? received,
  });

  return {
    file,
    removeTemp: async () => {
      try {
        await root.removeEntry(entryName);
      } catch {
        // Entry may already be removed or unsupported
      }
    },
  };
}
