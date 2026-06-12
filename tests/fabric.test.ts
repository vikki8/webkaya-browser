import { describe, expect, it } from 'vitest';
import { Sandbox } from '../src/sandbox/sandbox';
import { MemorySnapshotStore } from '../src/sandbox/snapshot-store';
import { SandboxFabric } from '../src/net/fabric';
import { LoadBalancer } from '../src/net/load-balancer';
import { TieredMemory } from '../src/memory/tiered-memory';
import { denyEastWestPolicy, hashBalancer } from '../src/net/hooks';

const fastPolicy = { coldStartMs: 0, retryCount: 0, timeoutMs: 2_000 };

function box(memory?: ReturnType<TieredMemory['bindingFor']>) {
  return Sandbox.create({ policy: fastPolicy, store: new MemorySnapshotStore(), memory });
}

describe('SandboxFabric east-west traffic', () => {
  it('delivers requests between sandboxes via their handlers', async () => {
    const fabric = new SandboxFabric();
    const a = await box();
    const b = await box();
    const addrA = fabric.join(a, { name: 'a' });
    const addrB = fabric.join(b, {
      name: 'b',
      handler: 'return { greeting: "hello from b", caller: ctx.args.from };',
    });

    const response = await fabric.request(addrA, addrB, { payload: { ping: true } });
    expect(response.ok).toBe(true);
    expect(response.status).toBe(200);
    expect(response.body).toEqual({ greeting: 'hello from b', caller: addrA });
    expect(fabric.deliveredByDst.get(BigInt(addrB))).toBe(1n);
  });

  it('returns 404 for an unknown destination', async () => {
    const fabric = new SandboxFabric();
    const a = await box();
    const addrA = fabric.join(a);
    const response = await fabric.request(addrA, 999, { payload: {} });
    expect(response.status).toBe(404);
    expect(response.ok).toBe(false);
  });

  it('surfaces a handler failure as a 500', async () => {
    const fabric = new SandboxFabric();
    const a = await box();
    const b = await box();
    const addrA = fabric.join(a);
    const addrB = fabric.join(b, { handler: 'throw new Error("handler boom");' });
    const response = await fabric.request(addrA, addrB, { payload: {} });
    expect(response.status).toBe(500);
    expect(response.error).toMatch(/handler boom/);
  });
});

describe('SandboxFabric network policy', () => {
  it('drops east-west traffic between two sandboxes under deny policy', async () => {
    const fabric = new SandboxFabric({ policyProgram: denyEastWestPolicy() });
    const a = await box();
    const b = await box();
    const addrA = fabric.join(a, { name: 'a' });
    const addrB = fabric.join(b, { name: 'b', handler: 'return "reached b";' });

    const response = await fabric.request(addrA, addrB, { payload: {} });
    expect(response.ok).toBe(false);
    expect(response.status).toBe(403);
    expect(response.denied).toBe(true);
    expect(fabric.droppedBySrc.get(BigInt(addrA))).toBe(1n);
  });

  it('still allows ingress (addr 0) to reach a sandbox under deny policy', async () => {
    const fabric = new SandboxFabric({ policyProgram: denyEastWestPolicy() });
    const b = await box();
    const addrB = fabric.join(b, { handler: 'return "ok";' });
    const response = await fabric.request(0, addrB, { payload: {} });
    expect(response.ok).toBe(true);
    expect(response.body).toBe('ok');
  });
});

describe('LoadBalancer', () => {
  it('round-robins requests across the backend pool', async () => {
    const fabric = new SandboxFabric();
    const lb = new LoadBalancer(fabric);
    const hits: number[] = [0, 0];
    for (let i = 0; i < 2; i++) {
      const b = await box();
      const addr = fabric.join(b, {
        name: `backend-${i}`,
        handler: `return { backend: ${i} };`,
      });
      lb.addBackend(addr);
    }

    for (let i = 0; i < 6; i++) {
      const res = await lb.handle({ path: '/api', payload: { n: i } });
      hits[(res.body as { backend: number }).backend]++;
    }
    expect(hits).toEqual([3, 3]);
  });

  it('sends the same sticky-hashed request to the same backend', async () => {
    const fabric = new SandboxFabric();
    const lb = new LoadBalancer(fabric, { program: hashBalancer() });
    for (let i = 0; i < 3; i++) {
      const b = await box();
      lb.addBackend(fabric.join(b, { handler: `return { backend: ${i} };` }));
    }
    const first = await lb.handle({ path: '/user/42', hash: 42 });
    const again = await lb.handle({ path: '/user/42', hash: 42 });
    expect(again.body).toEqual(first.body);
  });

  it('serves static routes itself without touching a backend', async () => {
    const fabric = new SandboxFabric();
    const lb = new LoadBalancer(fabric);
    lb.serveStatic('/health', { status: 'green' });
    const res = await lb.handle({ path: '/health' });
    expect(res.status).toBe(200);
    expect(res.body).toEqual({ status: 'green' });
  });

  it('returns 503 when no backends are registered', async () => {
    const fabric = new SandboxFabric();
    const lb = new LoadBalancer(fabric);
    const res = await lb.handle({ path: '/api' });
    expect(res.status).toBe(503);
  });
});

describe('fabric with tiered memory', () => {
  it('shares a global counter across sandboxes behind a load balancer', async () => {
    const fabric = new SandboxFabric();
    const memory = new TieredMemory();
    const lb = new LoadBalancer(fabric);
    const handler = 'const n = ctx.global.incr("requests"); ctx.local.incr("local_hits"); return { total: n };';

    for (let i = 0; i < 3; i++) {
      const b = await Sandbox.create({
        policy: fastPolicy,
        store: new MemorySnapshotStore(),
        memory: memory.bindingFor(`backend-${i}`),
      });
      lb.addBackend(fabric.join(b, { name: `backend-${i}`, handler }));
    }

    const totals: number[] = [];
    for (let i = 0; i < 6; i++) {
      const res = await lb.handle({ path: '/api', payload: { i } });
      totals.push((res.body as { total: number }).total);
    }
    // Global counter is shared, so totals are 1..6 regardless of which backend served.
    expect(totals).toEqual([1, 2, 3, 4, 5, 6]);
    expect(memory.global.get('requests')).toBe('6');
    // Local counters are private: 3 backends, 6 round-robin requests => 2 each.
    expect(memory.localFor('backend-0').get('local_hits')).toBe('2');
  });
});
