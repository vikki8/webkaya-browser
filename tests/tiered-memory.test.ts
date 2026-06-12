import { describe, expect, it } from 'vitest';
import { MemoryTier, TieredMemory } from '../src/memory/tiered-memory';

describe('MemoryTier', () => {
  it('supports get/set/del', () => {
    const tier = new MemoryTier();
    tier.set('k', 'v');
    expect(tier.get('k')).toBe('v');
    expect(tier.del('k')).toBe(true);
    expect(tier.get('k')).toBeNull();
  });

  it('increments integer counters', () => {
    const tier = new MemoryTier();
    expect(tier.incr('hits')).toBe(1);
    expect(tier.incr('hits', 4)).toBe(5);
    expect(tier.get('hits')).toBe('5');
  });

  it('throws when incrementing a non-integer value', () => {
    const tier = new MemoryTier();
    tier.set('name', 'agent');
    expect(() => tier.incr('name')).toThrow(/not an integer/);
  });

  it('expires keys based on a controllable clock', () => {
    let now = 1_000;
    const tier = new MemoryTier(() => now);
    tier.set('session', 'abc', { ttlMs: 500 });
    expect(tier.get('session')).toBe('abc');
    expect(tier.ttl('session')).toBe(500);
    now = 1_600;
    expect(tier.get('session')).toBeNull();
    expect(tier.ttl('session')).toBe(-2);
  });

  it('reports -1 ttl for keys without expiry', () => {
    const tier = new MemoryTier();
    tier.set('k', 'v');
    expect(tier.ttl('k')).toBe(-1);
  });

  it('matches keys with glob patterns', () => {
    const tier = new MemoryTier();
    tier.set('user:1', 'a');
    tier.set('user:2', 'b');
    tier.set('session:1', 'c');
    expect(tier.keys('user:*').sort()).toEqual(['user:1', 'user:2']);
    expect(tier.keys().length).toBe(3);
  });
});

describe('TieredMemory', () => {
  it('isolates local tiers but shares the global tier', () => {
    const memory = new TieredMemory();
    const a = memory.bindingFor('a');
    const b = memory.bindingFor('b');

    a.local.set('secret', 'a-only');
    b.local.set('secret', 'b-only');
    expect(a.local.get('secret')).toBe('a-only');
    expect(b.local.get('secret')).toBe('b-only');

    a.global.set('shared', 'visible');
    expect(b.global.get('shared')).toBe('visible');
  });

  it('returns the same local tier for a repeated id', () => {
    const memory = new TieredMemory();
    expect(memory.localFor('x')).toBe(memory.localFor('x'));
  });
});
