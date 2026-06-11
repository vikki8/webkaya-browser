export { Sandbox } from './sandbox/sandbox';
export type { SandboxOptions, RunOptions, RunResult, RunRecord, GuestContext } from './sandbox/sandbox';

export {
  MemorySnapshotStore,
  OpfsSnapshotStore,
  createDefaultSnapshotStore,
} from './sandbox/snapshot-store';
export type { SandboxSnapshot, SnapshotStore } from './sandbox/snapshot-store';

export { ProbeRegistry, TRACEPOINT_LAYOUTS } from './sandbox/probes';
export type { SandboxTracepoint, ProbeOptions } from './sandbox/probes';

export { EbpfVm, verifyProgram, HELPERS } from './ebpf/vm';
export type { EbpfEnv, EbpfVmOptions } from './ebpf/vm';
export { EbpfMap } from './ebpf/maps';
export { op, insn, assemble } from './ebpf/asm';

export type { SandboxPolicy, PolicyEditorState, PolicyMode } from './types/policy';
export type { HostToSandboxMessage, SandboxToHostMessage } from './types/protocol';

export {
  DEFAULT_SANDBOX_POLICY,
  DISALLOWED_GUEST_TOKENS,
  normalizePolicy,
  validatePolicy,
  assertGuestCodeSafety,
  generatePolicyCodeFromTemplate,
  normalizePolicyEditorState,
  compilePolicyCode,
  resolvePolicy,
} from './runtime/policy';

export { GuestInvoker } from './runtime/guest-invoker';
export { detectCapabilities } from './runtime/capability-detect';
export type { Capabilities } from './runtime/capability-detect';
export { selectComputeBackend } from './runtime/backends';
export type { ComputeBackendKind, ComputeBackendSelection } from './runtime/backends';
export { HardwareMonitor } from './runtime/hardware-monitor';
export type {
  HardwareMonitorMetrics,
  RuntimeActivitySnapshot,
  ThermalPressureState,
  EnergyImpactLevel,
} from './runtime/hardware-monitor';
