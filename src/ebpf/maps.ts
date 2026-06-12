/**
 * BPF-style map: u64 keys to u64 values. Probes mutate maps via helpers;
 * the host reads them out-of-band — the shared-state channel between
 * eBPF programs and the application, as in kernel eBPF.
 */
export class EbpfMap {
  private readonly data = new Map<bigint, bigint>();

  constructor(public readonly maxEntries = 1024) {}

  get size(): number {
    return this.data.size;
  }

  get(key: bigint): bigint | undefined {
    return this.data.get(BigInt.asUintN(64, key));
  }

  set(key: bigint, value: bigint): boolean {
    const k = BigInt.asUintN(64, key);
    if (!this.data.has(k) && this.data.size >= this.maxEntries) return false;
    this.data.set(k, BigInt.asUintN(64, value));
    return true;
  }

  add(key: bigint, delta: bigint): boolean {
    const k = BigInt.asUintN(64, key);
    const current = this.data.get(k);
    if (current === undefined && this.data.size >= this.maxEntries) return false;
    this.data.set(k, BigInt.asUintN(64, (current ?? 0n) + delta));
    return true;
  }

  delete(key: bigint): boolean {
    return this.data.delete(BigInt.asUintN(64, key));
  }

  entries(): [bigint, bigint][] {
    return [...this.data.entries()];
  }

  clear(): void {
    this.data.clear();
  }
}
