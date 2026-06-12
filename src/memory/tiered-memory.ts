/**
 * Redis-shaped key/value interface. The in-browser tiers below implement it
 * synchronously over Maps; a server deployment can back the global tier with
 * a real Redis (ioredis) behind the same method names.
 */
export interface KVStore {
  get(key: string): string | null;
  set(key: string, value: string, opts?: { ttlMs?: number }): void;
  del(key: string): boolean;
  incr(key: string, by?: number): number;
  expire(key: string, ttlMs: number): boolean;
  /** Remaining TTL in ms; -1 if no expiry, -2 if the key is missing. */
  ttl(key: string): number;
  keys(pattern?: string): string[];
  flush(): void;
}

/** Memory binding handed to a sandbox: its private tier plus the shared tier. */
export interface MemoryBinding {
  local: KVStore;
  global: KVStore;
}

interface Entry {
  value: string;
  expiresAt?: number;
}

function globToRegExp(pattern: string): RegExp {
  const escaped = pattern.replace(/[.+^${}()|[\]\\]/g, '\\$&').replace(/\*/g, '.*').replace(/\?/g, '.');
  return new RegExp(`^${escaped}$`);
}

/** One key/value tier with lazy TTL expiry. */
export class MemoryTier implements KVStore {
  private readonly data = new Map<string, Entry>();

  constructor(private readonly now: () => number = () => Date.now()) {}

  private live(key: string): Entry | undefined {
    const entry = this.data.get(key);
    if (!entry) return undefined;
    if (entry.expiresAt !== undefined && entry.expiresAt <= this.now()) {
      this.data.delete(key);
      return undefined;
    }
    return entry;
  }

  get(key: string): string | null {
    return this.live(key)?.value ?? null;
  }

  set(key: string, value: string, opts: { ttlMs?: number } = {}): void {
    const entry: Entry = { value: String(value) };
    if (opts.ttlMs !== undefined && opts.ttlMs > 0) {
      entry.expiresAt = this.now() + opts.ttlMs;
    }
    this.data.set(key, entry);
  }

  del(key: string): boolean {
    return this.data.delete(key);
  }

  incr(key: string, by = 1): number {
    const current = this.live(key);
    const base = current ? Number(current.value) : 0;
    if (!Number.isFinite(base)) {
      throw new Error(`Value at "${key}" is not an integer.`);
    }
    const next = base + by;
    this.data.set(key, { value: String(next), expiresAt: current?.expiresAt });
    return next;
  }

  expire(key: string, ttlMs: number): boolean {
    const entry = this.live(key);
    if (!entry) return false;
    entry.expiresAt = this.now() + ttlMs;
    return true;
  }

  ttl(key: string): number {
    const entry = this.live(key);
    if (!entry) return -2;
    if (entry.expiresAt === undefined) return -1;
    return Math.max(0, entry.expiresAt - this.now());
  }

  keys(pattern?: string): string[] {
    const matcher = pattern && pattern !== '*' ? globToRegExp(pattern) : null;
    const out: string[] = [];
    for (const key of [...this.data.keys()]) {
      if (this.live(key) && (!matcher || matcher.test(key))) out.push(key);
    }
    return out;
  }

  flush(): void {
    this.data.clear();
  }
}

/**
 * Two-tier memory for a sandbox fabric:
 * - `global` is shared by every sandbox (the Redis cluster analogue).
 * - `localFor(id)` is a private, sandbox-scoped tier (the in-process cache).
 *
 * Reads hit local first conceptually, but the tiers are independent stores so
 * a sandbox controls what it promotes to global. Hand `bindingFor(id)` to a
 * sandbox to expose both tiers to guest code as `ctx.local` / `ctx.global`.
 */
export class TieredMemory {
  readonly global: MemoryTier;
  private readonly locals = new Map<string, MemoryTier>();

  constructor(private readonly now: () => number = () => Date.now()) {
    this.global = new MemoryTier(now);
  }

  localFor(id: string): MemoryTier {
    let tier = this.locals.get(id);
    if (!tier) {
      tier = new MemoryTier(this.now);
      this.locals.set(id, tier);
    }
    return tier;
  }

  bindingFor(id: string): MemoryBinding {
    return { local: this.localFor(id), global: this.global };
  }

  dropLocal(id: string): void {
    this.locals.delete(id);
  }
}
