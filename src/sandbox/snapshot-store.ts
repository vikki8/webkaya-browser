export interface SandboxSnapshot {
  id: string;
  label?: string;
  createdAt: number;
  /** Sandbox the snapshot was taken from, for fork lineage. */
  sandboxId: string;
  /** Snapshot this sandbox was forked/restored from, if any. */
  parentSnapshotId?: string;
  runCount: number;
  state: Record<string, unknown>;
}

export interface SnapshotStore {
  save(snapshot: SandboxSnapshot): Promise<void>;
  load(id: string): Promise<SandboxSnapshot | null>;
  list(): Promise<SandboxSnapshot[]>;
  remove(id: string): Promise<void>;
}

/** Volatile store for tests and non-persistent hosts. */
export class MemorySnapshotStore implements SnapshotStore {
  private readonly snapshots = new Map<string, SandboxSnapshot>();

  async save(snapshot: SandboxSnapshot): Promise<void> {
    this.snapshots.set(snapshot.id, structuredClone(snapshot));
  }

  async load(id: string): Promise<SandboxSnapshot | null> {
    const found = this.snapshots.get(id);
    return found ? structuredClone(found) : null;
  }

  async list(): Promise<SandboxSnapshot[]> {
    return [...this.snapshots.values()].map((s) => structuredClone(s)).sort((a, b) => a.createdAt - b.createdAt);
  }

  async remove(id: string): Promise<void> {
    this.snapshots.delete(id);
  }
}

const OPFS_DIR = 'webkaya-snapshots';

/**
 * Persists snapshots to the Origin Private File System so sandbox state
 * survives reloads. Snapshot state must be JSON-serializable.
 */
export class OpfsSnapshotStore implements SnapshotStore {
  static isSupported(): boolean {
    try {
      return typeof navigator !== 'undefined' && !!navigator.storage && typeof navigator.storage.getDirectory === 'function';
    } catch {
      return false;
    }
  }

  private async dir(): Promise<FileSystemDirectoryHandle> {
    const root = await navigator.storage.getDirectory();
    return root.getDirectoryHandle(OPFS_DIR, { create: true });
  }

  async save(snapshot: SandboxSnapshot): Promise<void> {
    const dir = await this.dir();
    const file = await dir.getFileHandle(`${snapshot.id}.json`, { create: true });
    const writable = await file.createWritable();
    await writable.write(JSON.stringify(snapshot));
    await writable.close();
  }

  async load(id: string): Promise<SandboxSnapshot | null> {
    try {
      const dir = await this.dir();
      const file = await dir.getFileHandle(`${id}.json`);
      const text = await (await file.getFile()).text();
      return JSON.parse(text) as SandboxSnapshot;
    } catch {
      return null;
    }
  }

  async list(): Promise<SandboxSnapshot[]> {
    const dir = await this.dir();
    const snapshots: SandboxSnapshot[] = [];
    for await (const entry of (dir as any).values()) {
      if (entry.kind !== 'file' || !entry.name.endsWith('.json')) continue;
      try {
        const text = await (await entry.getFile()).text();
        snapshots.push(JSON.parse(text) as SandboxSnapshot);
      } catch {
        // skip unreadable snapshot files
      }
    }
    return snapshots.sort((a, b) => a.createdAt - b.createdAt);
  }

  async remove(id: string): Promise<void> {
    try {
      const dir = await this.dir();
      await dir.removeEntry(`${id}.json`);
    } catch {
      // already gone
    }
  }
}

export function createDefaultSnapshotStore(): SnapshotStore {
  return OpfsSnapshotStore.isSupported() ? new OpfsSnapshotStore() : new MemorySnapshotStore();
}
