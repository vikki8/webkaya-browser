import JSZip from 'jszip';
import { DataRow, ParsedDataset } from '../types/data';

const IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.webp', '.gif'];
const STRUCTURED_EXTENSIONS = ['.csv', '.json'];

function hasExtension(filename: string, extensions: string[]): boolean {
  const lower = filename.toLowerCase();
  return extensions.some((ext) => lower.endsWith(ext));
}

function inferDelimiter(headerLine: string): string {
  const delimiters = [',', ';', '\t', '|'];
  let best = ',';
  let bestCount = -1;
  for (const delimiter of delimiters) {
    const count = headerLine.split(delimiter).length;
    if (count > bestCount) {
      best = delimiter;
      bestCount = count;
    }
  }
  return best;
}

function parseCsvRows(text: string): string[][] {
  const rows: string[][] = [];
  const lines = text.replace(/\r\n/g, '\n').replace(/\r/g, '\n').split('\n');
  const nonEmpty = lines.filter((line) => line.trim().length > 0);
  if (!nonEmpty.length) return rows;
  const delimiter = inferDelimiter(nonEmpty[0]);

  for (const line of nonEmpty) {
    const parsed: string[] = [];
    let value = '';
    let inQuotes = false;
    for (let i = 0; i < line.length; i++) {
      const ch = line[i];
      if (ch === '"') {
        if (inQuotes && line[i + 1] === '"') {
          value += '"';
          i++;
        } else {
          inQuotes = !inQuotes;
        }
        continue;
      }
      if (ch === delimiter && !inQuotes) {
        parsed.push(value.trim());
        value = '';
        continue;
      }
      value += ch;
    }
    parsed.push(value.trim());
    rows.push(parsed);
  }

  return rows;
}

function coerceValue(raw: string): string | number | boolean | null {
  const value = raw.trim();
  if (!value) return null;
  if (value.toLowerCase() === 'true') return true;
  if (value.toLowerCase() === 'false') return false;
  const asNumber = Number(value);
  if (!Number.isNaN(asNumber) && value !== '') return asNumber;
  return value;
}

function rowsFromCsv(text: string): DataRow[] {
  const rawRows = parseCsvRows(text);
  if (!rawRows.length) return [];

  const headers = rawRows[0].map((header, idx) => header || `column_${idx + 1}`);
  const rows: DataRow[] = [];

  for (let r = 1; r < rawRows.length; r++) {
    const current = rawRows[r];
    const row: DataRow = {};
    for (let c = 0; c < headers.length; c++) {
      row[headers[c]] = coerceValue(current[c] ?? '');
    }
    rows.push(row);
  }

  return rows;
}

function rowsFromJson(text: string): DataRow[] {
  const parsed = JSON.parse(text);
  const records = Array.isArray(parsed) ? parsed : parsed?.data;
  if (!Array.isArray(records)) {
    throw new Error('JSON must be an array of objects or contain a top-level "data" array.');
  }

  const rows: DataRow[] = [];
  for (const item of records) {
    if (!item || typeof item !== 'object' || Array.isArray(item)) continue;
    const row: DataRow = {};
    for (const [key, value] of Object.entries(item)) {
      if (value === null || value === undefined) row[key] = null;
      else if (typeof value === 'number' || typeof value === 'boolean' || typeof value === 'string') row[key] = value;
      else row[key] = JSON.stringify(value);
    }
    rows.push(row);
  }
  return rows;
}

function extractColumns(rows: DataRow[]): string[] {
  const names = new Set<string>();
  for (const row of rows) {
    for (const key of Object.keys(row)) names.add(key);
  }
  return Array.from(names);
}

export interface DatasetParseProgress {
  stage: 'loading_archive' | 'discovering_files' | 'extracting_images' | 'processing_images' | 'completed';
  processed: number;
  total: number;
  currentFile?: string;
  message: string;
}

export interface DatasetParseOptions {
  onProgress?: (progress: DatasetParseProgress) => void;
  maxImageFiles?: number;
  workerCount?: number;
}

const DEFAULT_MAX_IMAGE_FILES = 3000;
const MIN_WORKERS = 2;
const MAX_WORKERS = 8;

/** Warn in UI when a ZIP is large; parsing still uses a full in-memory buffer (JSZip). */
export const LARGE_ZIP_WARN_BYTES = 512 * 1024 * 1024;

/**
 * Refuse ZIPs larger than this in the browser parser; use Kaggle CLI + “Open folder” instead.
 * (JSZip loads the full archive into memory.)
 */
export const BROWSER_ZIP_PARSE_MAX_BYTES = 6 * 1024 * 1024 * 1024;

interface ImageWorkerRequest {
  type: 'process_image';
  taskId: number;
  entryName: string;
  label: string;
  imageBuffer: ArrayBuffer;
  includePreview?: boolean;
}

interface ImageWorkerSuccess {
  type: 'process_image_success';
  taskId: number;
  row: DataRow;
}

interface ImageWorkerFailure {
  type: 'process_image_error';
  taskId: number;
  error: string;
}

type ImageWorkerResponse = ImageWorkerSuccess | ImageWorkerFailure;
interface DirectoryDataFile {
  path: string;
  file: File;
}

let nextImageTaskId = 1;

function emitParseProgress(
  options: DatasetParseOptions | undefined,
  payload: DatasetParseProgress
): void {
  options?.onProgress?.(payload);
}

function computeImageStats(pixelData: Uint8ClampedArray): Record<string, number> {
  let sumR = 0;
  let sumG = 0;
  let sumB = 0;
  let sqR = 0;
  let sqG = 0;
  let sqB = 0;
  const px = pixelData.length / 4;
  for (let i = 0; i < pixelData.length; i += 4) {
    const r = pixelData[i] / 255;
    const g = pixelData[i + 1] / 255;
    const b = pixelData[i + 2] / 255;
    sumR += r;
    sumG += g;
    sumB += b;
    sqR += r * r;
    sqG += g * g;
    sqB += b * b;
  }
  const meanR = sumR / px;
  const meanG = sumG / px;
  const meanB = sumB / px;
  return {
    mean_r: meanR,
    mean_g: meanG,
    mean_b: meanB,
    std_r: Math.sqrt(Math.max(0, sqR / px - meanR * meanR)),
    std_g: Math.sqrt(Math.max(0, sqG / px - meanG * meanG)),
    std_b: Math.sqrt(Math.max(0, sqB / px - meanB * meanB)),
  };
}

function imageLabelFromEntryName(entryName: string): string {
  const pathParts = entryName.split('/').filter(Boolean);
  return pathParts.length > 1 ? pathParts[0] : 'unknown';
}

async function decodeImageToFeatures(blob: Blob): Promise<Record<string, number>> {
  const bitmap = await createImageBitmap(blob);
  const width = 32;
  const height = 32;

  if (typeof OffscreenCanvas !== 'undefined') {
    const canvas = new OffscreenCanvas(width, height);
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Unable to create 2D context for image preprocessing.');
    ctx.drawImage(bitmap, 0, 0, width, height);
    const { data } = ctx.getImageData(0, 0, width, height);
    bitmap.close();
    return computeImageStats(data);
  }

  if (typeof document === 'undefined') {
    bitmap.close();
    throw new Error('Image preprocessing requires a browser environment.');
  }

  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    bitmap.close();
    throw new Error('Unable to create canvas context for image preprocessing.');
  }
  ctx.drawImage(bitmap, 0, 0, width, height);
  const { data } = ctx.getImageData(0, 0, width, height);
  bitmap.close();
  return computeImageStats(data);
}

function resolveWorkerCount(totalImages: number, requested?: number): number {
  if (totalImages <= 0) return 0;
  const hardwareHint =
    typeof navigator !== 'undefined' && Number.isFinite(navigator.hardwareConcurrency)
      ? Math.max(1, Math.floor(navigator.hardwareConcurrency / 2))
      : 4;
  const desired = requested ?? hardwareHint;
  return Math.max(MIN_WORKERS, Math.min(MAX_WORKERS, desired, totalImages));
}

function canUseWorkerPool(): boolean {
  return typeof window !== 'undefined' && typeof Worker !== 'undefined';
}

function createImageWorker(): Worker {
  return new Worker(new URL('./image-feature-worker.ts', import.meta.url));
}

async function processImageViaWorker(
  worker: Worker,
  entryName: string,
  label: string,
  imageBuffer: ArrayBuffer,
  includePreview: boolean
): Promise<DataRow> {
  const taskId = nextImageTaskId++;
  const payload: ImageWorkerRequest = {
    type: 'process_image',
    taskId,
    entryName,
    label,
    imageBuffer,
    includePreview,
  };

  return new Promise<DataRow>((resolve, reject) => {
    const onMessage = (event: MessageEvent<ImageWorkerResponse>) => {
      if (!event.data || event.data.taskId !== taskId) return;
      cleanup();
      if (event.data.type === 'process_image_success') {
        resolve(event.data.row);
      } else {
        reject(new Error(event.data.error));
      }
    };
    const onError = (event: ErrorEvent) => {
      cleanup();
      reject(new Error(event.message || 'Image worker failed.'));
    };
    const cleanup = () => {
      worker.removeEventListener('message', onMessage as EventListener);
      worker.removeEventListener('error', onError as EventListener);
    };

    worker.addEventListener('message', onMessage as EventListener);
    worker.addEventListener('error', onError as EventListener);
    worker.postMessage(payload, [payload.imageBuffer]);
  });
}

async function rowsFromImageEntriesSequential(
  imageEntries: JSZip.JSZipObject[],
  options?: DatasetParseOptions
): Promise<DataRow[]> {
  const rows: DataRow[] = [];
  const total = imageEntries.length;
  const emitEvery = Math.max(1, Math.floor(total / 100));

  for (let i = 0; i < imageEntries.length; i++) {
    const entry = imageEntries[i];
    emitParseProgress(options, {
      stage: 'extracting_images',
      processed: i,
      total,
      currentFile: entry.name,
      message: `Extracting image ${i + 1}/${total}...`,
    });

    const blob = await entry.async('blob');
    const stats = await decodeImageToFeatures(blob);
    rows.push({
      label: imageLabelFromEntryName(entry.name),
      image_name: entry.name,
      ...stats,
    });

    const processed = i + 1;
    if (processed === total || processed % emitEvery === 0) {
      emitParseProgress(options, {
        stage: 'processing_images',
        processed,
        total,
        currentFile: entry.name,
        message: `Processed ${processed.toLocaleString()} / ${total.toLocaleString()} images`,
      });
    }

    if (processed % 8 === 0) {
      // Keep the main thread responsive when worker pool is unavailable.
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
  }

  return rows;
}

async function rowsFromImageEntriesWithWorkers(
  imageEntries: JSZip.JSZipObject[],
  options?: DatasetParseOptions
): Promise<DataRow[]> {
  const total = imageEntries.length;
  const rows = new Array<DataRow>(total);
  const workerCount = resolveWorkerCount(total, options?.workerCount);
  const workers = Array.from({ length: workerCount }, () => createImageWorker());
  const emitEvery = Math.max(1, Math.floor(total / 100));

  let nextIndex = 0;
  let processed = 0;

  const runWorker = async (worker: Worker) => {
    while (true) {
      const index = nextIndex++;
      if (index >= total) return;

      const entry = imageEntries[index];
      emitParseProgress(options, {
        stage: 'extracting_images',
        processed,
        total,
        currentFile: entry.name,
        message: `Extracting image ${index + 1}/${total}...`,
      });

      const buffer = await entry.async('arraybuffer');
      const row = await processImageViaWorker(
        worker,
        entry.name,
        imageLabelFromEntryName(entry.name),
        buffer,
        index < 256
      );
      rows[index] = row;
      processed += 1;

      if (processed === total || processed % emitEvery === 0) {
        emitParseProgress(options, {
          stage: 'processing_images',
          processed,
          total,
          currentFile: entry.name,
          message: `Processed ${processed.toLocaleString()} / ${total.toLocaleString()} images`,
        });
      }
    }
  };

  try {
    await Promise.all(workers.map((worker) => runWorker(worker)));
  } finally {
    workers.forEach((worker) => worker.terminate());
  }

  return rows.filter(Boolean);
}

async function rowsFromImageEntries(
  imageEntries: JSZip.JSZipObject[],
  options?: DatasetParseOptions
): Promise<DataRow[]> {
  if (!imageEntries.length) {
    throw new Error('No image files found in ZIP.');
  }

  const maxImageFiles = Math.max(1, options?.maxImageFiles ?? DEFAULT_MAX_IMAGE_FILES);
  const cappedEntries = imageEntries.slice(0, maxImageFiles);

  emitParseProgress(options, {
    stage: 'discovering_files',
    processed: 0,
    total: cappedEntries.length,
    message:
      imageEntries.length > cappedEntries.length
        ? `Found ${imageEntries.length.toLocaleString()} images. Processing first ${cappedEntries.length.toLocaleString()} to keep browser stable.`
        : `Found ${cappedEntries.length.toLocaleString()} images. Preparing parallel processing...`,
  });

  if (!canUseWorkerPool()) {
    return rowsFromImageEntriesSequential(cappedEntries, options);
  }

  try {
    return await rowsFromImageEntriesWithWorkers(cappedEntries, options);
  } catch {
    // Fallback when worker processing fails in restrictive environments.
    return rowsFromImageEntriesSequential(cappedEntries, options);
  }
}

async function rowsFromImageDirectoryFilesSequential(
  imageFiles: DirectoryDataFile[],
  options?: DatasetParseOptions
): Promise<DataRow[]> {
  const rows: DataRow[] = [];
  const total = imageFiles.length;
  const emitEvery = Math.max(1, Math.floor(total / 100));

  for (let i = 0; i < imageFiles.length; i++) {
    const item = imageFiles[i];
    emitParseProgress(options, {
      stage: 'extracting_images',
      processed: i,
      total,
      currentFile: item.path,
      message: `Extracting image ${i + 1}/${total}...`,
    });

    const stats = await decodeImageToFeatures(item.file);
    rows.push({
      label: imageLabelFromEntryName(item.path),
      image_name: item.path,
      ...stats,
    });

    const processed = i + 1;
    if (processed === total || processed % emitEvery === 0) {
      emitParseProgress(options, {
        stage: 'processing_images',
        processed,
        total,
        currentFile: item.path,
        message: `Processed ${processed.toLocaleString()} / ${total.toLocaleString()} images`,
      });
    }

    if (processed % 8 === 0) {
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
  }

  return rows;
}

async function rowsFromImageDirectoryFilesWithWorkers(
  imageFiles: DirectoryDataFile[],
  options?: DatasetParseOptions
): Promise<DataRow[]> {
  const total = imageFiles.length;
  const rows = new Array<DataRow>(total);
  const workerCount = resolveWorkerCount(total, options?.workerCount);
  const workers = Array.from({ length: workerCount }, () => createImageWorker());
  const emitEvery = Math.max(1, Math.floor(total / 100));

  let nextIndex = 0;
  let processed = 0;

  const runWorker = async (worker: Worker) => {
    while (true) {
      const index = nextIndex++;
      if (index >= total) return;

      const item = imageFiles[index];
      emitParseProgress(options, {
        stage: 'extracting_images',
        processed,
        total,
        currentFile: item.path,
        message: `Extracting image ${index + 1}/${total}...`,
      });

      const buffer = await item.file.arrayBuffer();
      const row = await processImageViaWorker(
        worker,
        item.path,
        imageLabelFromEntryName(item.path),
        buffer,
        index < 256
      );
      rows[index] = row;
      processed += 1;

      if (processed === total || processed % emitEvery === 0) {
        emitParseProgress(options, {
          stage: 'processing_images',
          processed,
          total,
          currentFile: item.path,
          message: `Processed ${processed.toLocaleString()} / ${total.toLocaleString()} images`,
        });
      }
    }
  };

  try {
    await Promise.all(workers.map((worker) => runWorker(worker)));
  } finally {
    workers.forEach((worker) => worker.terminate());
  }

  return rows.filter(Boolean);
}

async function rowsFromImageDirectoryFiles(
  imageFiles: DirectoryDataFile[],
  options?: DatasetParseOptions
): Promise<DataRow[]> {
  if (!imageFiles.length) {
    throw new Error('No image files found in selected directory.');
  }

  const maxImageFiles = Math.max(1, options?.maxImageFiles ?? DEFAULT_MAX_IMAGE_FILES);
  const cappedFiles = imageFiles.slice(0, maxImageFiles);
  emitParseProgress(options, {
    stage: 'discovering_files',
    processed: 0,
    total: cappedFiles.length,
    message:
      imageFiles.length > cappedFiles.length
        ? `Found ${imageFiles.length.toLocaleString()} images. Processing first ${cappedFiles.length.toLocaleString()} to keep browser stable.`
        : `Found ${cappedFiles.length.toLocaleString()} images. Preparing parallel processing...`,
  });

  if (!canUseWorkerPool()) {
    return rowsFromImageDirectoryFilesSequential(cappedFiles, options);
  }

  try {
    return await rowsFromImageDirectoryFilesWithWorkers(cappedFiles, options);
  } catch {
    return rowsFromImageDirectoryFilesSequential(cappedFiles, options);
  }
}

async function collectDirectoryFiles(
  directoryHandle: any,
  prefix = ''
): Promise<DirectoryDataFile[]> {
  const files: DirectoryDataFile[] = [];
  for await (const [name, entry] of directoryHandle.entries()) {
    const path = prefix ? `${prefix}/${name}` : name;
    if (entry.kind === 'file') {
      const file = await entry.getFile();
      files.push({ path, file });
      continue;
    }
    if (entry.kind === 'directory') {
      const nested = await collectDirectoryFiles(entry, path);
      files.push(...nested);
    }
  }
  return files;
}

async function parseZipAsDataset(
  zipBuffer: ArrayBuffer,
  sourceName: string,
  options?: DatasetParseOptions
): Promise<ParsedDataset> {
  emitParseProgress(options, {
    stage: 'loading_archive',
    processed: 0,
    total: 0,
    message: 'Loading ZIP archive...',
  });

  const zip = await JSZip.loadAsync(zipBuffer);
  const entries = Object.values(zip.files).filter((entry) => !entry.dir);
  const structuredEntry = entries.find((entry) => hasExtension(entry.name, STRUCTURED_EXTENSIONS));

  if (structuredEntry) {
    const lower = structuredEntry.name.toLowerCase();
    if (lower.endsWith('.csv')) {
      const text = await structuredEntry.async('text');
      const rows = rowsFromCsv(text);
      emitParseProgress(options, {
        stage: 'completed',
        processed: rows.length,
        total: rows.length,
        currentFile: structuredEntry.name,
        message: `Parsed ${rows.length.toLocaleString()} rows from ${structuredEntry.name}.`,
      });
      return {
        rows,
        columns: extractColumns(rows),
        source: { type: 'upload', name: sourceName, description: `From ZIP file (${structuredEntry.name})` },
        inferredFormat: 'csv',
      };
    }
    if (lower.endsWith('.json')) {
      const text = await structuredEntry.async('text');
      const rows = rowsFromJson(text);
      emitParseProgress(options, {
        stage: 'completed',
        processed: rows.length,
        total: rows.length,
        currentFile: structuredEntry.name,
        message: `Parsed ${rows.length.toLocaleString()} rows from ${structuredEntry.name}.`,
      });
      return {
        rows,
        columns: extractColumns(rows),
        source: { type: 'upload', name: sourceName, description: `From ZIP file (${structuredEntry.name})` },
        inferredFormat: 'json',
      };
    }
  }

  const imageEntries = entries.filter((entry) => hasExtension(entry.name, IMAGE_EXTENSIONS));
  const rows = await rowsFromImageEntries(imageEntries, options);
  emitParseProgress(options, {
    stage: 'completed',
    processed: rows.length,
    total: rows.length,
    message: `Finished extracting image features for ${rows.length.toLocaleString()} images.`,
  });

  return {
    rows,
    columns: extractColumns(rows),
    source: { type: 'upload', name: sourceName, description: 'Derived image features from ZIP dataset' },
    inferredFormat: 'images_zip',
  };
}

export async function parseDatasetBlob(
  blob: Blob,
  filename: string,
  sourceType: ParsedDataset['source']['type'],
  options?: DatasetParseOptions
): Promise<ParsedDataset> {
  const lower = filename.toLowerCase();
  if (lower.endsWith('.csv')) {
    const text = await blob.text();
    const rows = rowsFromCsv(text);
    emitParseProgress(options, {
      stage: 'completed',
      processed: rows.length,
      total: rows.length,
      message: `Parsed ${rows.length.toLocaleString()} rows from CSV.`,
    });
    return {
      rows,
      columns: extractColumns(rows),
      source: { type: sourceType, name: filename },
      inferredFormat: 'csv',
    };
  }
  if (lower.endsWith('.json')) {
    const text = await blob.text();
    const rows = rowsFromJson(text);
    emitParseProgress(options, {
      stage: 'completed',
      processed: rows.length,
      total: rows.length,
      message: `Parsed ${rows.length.toLocaleString()} rows from JSON.`,
    });
    return {
      rows,
      columns: extractColumns(rows),
      source: { type: sourceType, name: filename },
      inferredFormat: 'json',
    };
  }
  if (lower.endsWith('.zip')) {
    const size = blob.size;
    if (size > BROWSER_ZIP_PARSE_MAX_BYTES) {
      throw new Error(
        `ZIP is about ${(size / (1024 * 1024 * 1024)).toFixed(1)}GB — too large to unpack in the browser. ` +
          'Use the Kaggle CLI (`kaggle datasets download …`), extract on disk, then use “Open folder” in Browser-First AI Platform. See docs/DEPLOYMENT_AND_DATASETS.md.'
      );
    }
    if (size > LARGE_ZIP_WARN_BYTES) {
      emitParseProgress(options, {
        stage: 'loading_archive',
        processed: 0,
        total: 0,
        message: `Large archive (${(size / (1024 * 1024)).toFixed(0)}MB). Unpacking may use a lot of memory; prefer CLI + folder for huge datasets.`,
      });
    }
    const buffer = await blob.arrayBuffer();
    return parseZipAsDataset(buffer, filename, options);
  }
  throw new Error('Unsupported file type. Supported formats: CSV, JSON, ZIP (images or tabular files).');
}

export async function parseDatasetDirectoryHandle(
  directoryHandle: any,
  options?: DatasetParseOptions
): Promise<ParsedDataset> {
  if (!directoryHandle || directoryHandle.kind !== 'directory') {
    throw new Error('Invalid directory handle.');
  }

  emitParseProgress(options, {
    stage: 'discovering_files',
    processed: 0,
    total: 0,
    message: 'Scanning selected directory...',
  });
  const files = await collectDirectoryFiles(directoryHandle);
  if (!files.length) {
    throw new Error('Selected directory does not contain supported dataset files.');
  }

  const structuredFiles = files.filter((item) => hasExtension(item.path, STRUCTURED_EXTENSIONS));
  if (structuredFiles.length > 0) {
    const structured = structuredFiles.sort((a, b) => b.file.size - a.file.size)[0];
    const parsed = await parseDatasetBlob(structured.file, structured.path, 'external', options);
    return {
      ...parsed,
      source: {
        type: 'external',
        name: directoryHandle.name || 'directory',
        description: `Loaded from local directory (${structured.path})`,
      },
    };
  }

  const imageFiles = files.filter((item) => hasExtension(item.path, IMAGE_EXTENSIONS));
  const rows = await rowsFromImageDirectoryFiles(imageFiles, options);
  emitParseProgress(options, {
    stage: 'completed',
    processed: rows.length,
    total: rows.length,
    message: `Finished extracting image features for ${rows.length.toLocaleString()} images.`,
  });

  return {
    rows,
    columns: extractColumns(rows),
    source: {
      type: 'external',
      name: directoryHandle.name || 'directory',
      description: 'Derived image features from local directory',
    },
    inferredFormat: 'images_zip',
  };
}

export async function parseDatasetFile(file: File, options?: DatasetParseOptions): Promise<ParsedDataset> {
  return parseDatasetBlob(file, file.name, 'upload', options);
}
