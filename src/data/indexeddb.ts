import { ParsedDataset, ProcessedDataset } from '../types/data';
import { ModelChoice, ModelMetrics, ResolvedModel, TrainingPreferences } from '../types/training-workflow';

const DB_NAME = 'browser-first-ai-local-state';
const DB_VERSION = 1;
const STORE_KV = 'kv';
const STORE_HISTORY = 'training_history';
const KEY_PREFERENCES = 'training_preferences';
const KEY_PENDING_RUN = 'pending_run';
export type DatasetSplitChoice = '80_20' | '90_10';

export interface TrainingHistoryEntry {
  id?: number;
  runId: string;
  modelChoice: ModelChoice;
  resolvedModel: ResolvedModel | null;
  targetColumn: string;
  problemType: ProcessedDataset['problemType'];
  datasetRows: number;
  metrics: ModelMetrics;
  createdAt: number;
}

export interface PendingRunState {
  runId: string;
  dataset: ParsedDataset;
  processedDataset: ProcessedDataset;
  splitChoice?: DatasetSplitChoice;
  trainingDataset?: ProcessedDataset;
  heldOutDataset?: ProcessedDataset;
  heldOutRows?: ParsedDataset['rows'];
  modelChoice: ModelChoice;
  preferences: TrainingPreferences;
  createdAt: number;
}

function openDb(): Promise<IDBDatabase> {
  if (typeof indexedDB === 'undefined') {
    return Promise.reject(new Error('IndexedDB is unavailable in this environment.'));
  }
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE_KV)) {
        db.createObjectStore(STORE_KV, { keyPath: 'key' });
      }
      if (!db.objectStoreNames.contains(STORE_HISTORY)) {
        db.createObjectStore(STORE_HISTORY, { keyPath: 'id', autoIncrement: true });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error('Failed to open IndexedDB.'));
  });
}

function transactionDone(tx: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onabort = () => reject(tx.error ?? new Error('IndexedDB transaction aborted.'));
    tx.onerror = () => reject(tx.error ?? new Error('IndexedDB transaction failed.'));
  });
}

async function setKv<T>(key: string, value: T): Promise<void> {
  const db = await openDb();
  const tx = db.transaction(STORE_KV, 'readwrite');
  tx.objectStore(STORE_KV).put({ key, value });
  await transactionDone(tx);
  db.close();
}

async function getKv<T>(key: string): Promise<T | null> {
  const db = await openDb();
  const tx = db.transaction(STORE_KV, 'readonly');
  const request = tx.objectStore(STORE_KV).get(key);
  const result = await new Promise<T | null>((resolve, reject) => {
    request.onsuccess = () => resolve(request.result?.value ?? null);
    request.onerror = () => reject(request.error ?? new Error('IndexedDB read failed.'));
  });
  await transactionDone(tx);
  db.close();
  return result;
}

export async function savePreferencesToIndexedDb(preferences: TrainingPreferences): Promise<void> {
  await setKv(KEY_PREFERENCES, preferences);
}

export async function loadPreferencesFromIndexedDb(): Promise<TrainingPreferences | null> {
  return getKv<TrainingPreferences>(KEY_PREFERENCES);
}

export async function savePendingRunState(state: PendingRunState): Promise<void> {
  await setKv(KEY_PENDING_RUN, state);
}

export async function loadPendingRunState(): Promise<PendingRunState | null> {
  return getKv<PendingRunState>(KEY_PENDING_RUN);
}

export async function clearPendingRunState(): Promise<void> {
  const db = await openDb();
  const tx = db.transaction(STORE_KV, 'readwrite');
  tx.objectStore(STORE_KV).delete(KEY_PENDING_RUN);
  await transactionDone(tx);
  db.close();
}

export async function saveTrainingHistoryEntry(entry: TrainingHistoryEntry): Promise<void> {
  const db = await openDb();
  const tx = db.transaction(STORE_HISTORY, 'readwrite');
  tx.objectStore(STORE_HISTORY).add(entry);
  await transactionDone(tx);
  db.close();
}

export async function listTrainingHistory(limit = 25): Promise<TrainingHistoryEntry[]> {
  const db = await openDb();
  const tx = db.transaction(STORE_HISTORY, 'readonly');
  const store = tx.objectStore(STORE_HISTORY);
  const entries = await new Promise<TrainingHistoryEntry[]>((resolve, reject) => {
    const results: TrainingHistoryEntry[] = [];
    const request = store.openCursor(null, 'prev');
    request.onsuccess = () => {
      const cursor = request.result;
      if (!cursor || results.length >= limit) {
        resolve(results);
        return;
      }
      results.push(cursor.value as TrainingHistoryEntry);
      cursor.continue();
    };
    request.onerror = () => reject(request.error ?? new Error('Failed to read training history.'));
  });
  await transactionDone(tx);
  db.close();
  return entries;
}
