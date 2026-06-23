import { assemble, op } from '../ebpf/asm.js';
import { HELPERS } from '../ebpf/vm.js';

/**
 * Fabric-level eBPF hooks. Unlike sandbox tracepoints, these run at the
 * network layer: a policy hook returns a verdict (0 = allow, nonzero = drop)
 * on every east-west request, and a load-balancer hook returns a backend
 * index for an incoming request. Both consume a u64 context struct whose
 * field order is fixed below (little-endian, field N at byte offset N * 8).
 */

export const NETWORK_POLICY_LAYOUT = ['srcAddr', 'dstAddr', 'port', 'length', 'protocol'] as const;
export const LOAD_BALANCER_LAYOUT = ['srcAddr', 'srcPort', 'dstPort', 'requestHash', 'backendCount'] as const;

/** Reserved address for ingress / load balancers. Sandboxes get addresses >= 1. */
export const INGRESS_ADDR = 0;

/**
 * Deny east-west traffic: drop a request only when both endpoints are
 * sandboxes (addr != 0). Traffic to/from ingress is always allowed. This is
 * the browser-tier equivalent of a Cilium "default deny between pods" policy.
 */
export function denyEastWestPolicy(): Uint8Array {
  return assemble([
    op.ldxdw(2, 1, 0),       // r2 = srcAddr
    op.ldxdw(3, 1, 8),       // r3 = dstAddr
    op.movImm(0, 0),         // verdict = allow
    op.jeqImm(2, INGRESS_ADDR, 2), // src is ingress -> allow
    op.jeqImm(3, INGRESS_ADDR, 1), // dst is ingress -> allow
    op.movImm(0, 1),         // both sandboxes -> deny
    op.exit(),
  ]);
}

/**
 * Round-robin load balancer. Keeps a counter in map fd 0 and returns
 * counter % backendCount. Requires `maps: [counterMap]` at attach time.
 */
export function roundRobinBalancer(): Uint8Array {
  return assemble([
    op.movReg(9, 1),         // r9 = ctx ptr (preserved across calls)
    op.movImm(1, 0),         // fd 0
    op.movImm(2, 0),         // key 0
    op.call(HELPERS.MAP_GET),
    op.movReg(6, 0),         // r6 = counter (preserved across calls)
    op.movImm(1, 0),
    op.movImm(2, 0),
    op.movReg(3, 6),
    op.addImm(3, 1),
    op.call(HELPERS.MAP_SET), // counter += 1
    op.ldxdw(2, 9, 32),      // r2 = backendCount
    op.movReg(0, 6),
    op.modReg(0, 2),         // index = counter % backendCount
    op.exit(),
  ]);
}

/**
 * Hash (sticky) load balancer: returns requestHash % backendCount, so the same
 * request tuple always lands on the same backend. No maps required.
 */
export function hashBalancer(): Uint8Array {
  return assemble([
    op.ldxdw(0, 1, 24),      // r0 = requestHash
    op.ldxdw(2, 1, 32),      // r2 = backendCount
    op.modReg(0, 2),         // index = requestHash % backendCount
    op.exit(),
  ]);
}
