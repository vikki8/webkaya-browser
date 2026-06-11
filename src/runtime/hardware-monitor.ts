export type ThermalPressureState = 'nominal' | 'fair' | 'serious' | 'critical' | 'unknown';
export type EnergyImpactLevel = 'idle' | 'low' | 'moderate' | 'high' | 'heavy';

export interface RuntimeActivitySnapshot {
  busy: boolean;
  progressPercent: number;
  completedOps: number;
  eventCount: number;
  estimatedStateBytes: number;
}

export interface HardwareMonitorMetrics {
  utilizationPercent: number;
  memoryMB: number;
  thermalState: ThermalPressureState;
  energyImpact: EnergyImpactLevel;
  updatedAt: number;
}

type PressureObserverRecordLike = {
  state?: ThermalPressureState;
};

type PressureObserverLike = {
  observe: (source: string, options?: { sampleInterval?: number }) => Promise<void> | void;
  disconnect: () => void;
};

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function toEnergyImpact(utilizationPercent: number, thermalState: ThermalPressureState, busy: boolean): EnergyImpactLevel {
  if (!busy || utilizationPercent <= 0.1) return 'idle';
  const thermalBoost = thermalState === 'critical' ? 15 : thermalState === 'serious' ? 8 : thermalState === 'fair' ? 3 : 0;
  const score = utilizationPercent + thermalBoost;
  if (score >= 85) return 'heavy';
  if (score >= 65) return 'high';
  if (score >= 35) return 'moderate';
  return 'low';
}

/**
 * Samples sandbox activity and device pressure so hosts can throttle, warn,
 * or surface a resource HUD while guest code runs.
 */
export class HardwareMonitor {
  private readonly tickIntervalMs: number;
  private readonly onUpdate: (metrics: HardwareMonitorMetrics) => void;
  private timer: ReturnType<typeof setInterval> | null = null;
  private pressureObserver: PressureObserverLike | null = null;
  private thermalState: ThermalPressureState = 'unknown';
  private lastOps = 0;
  private lastProgress = 0;
  private lastEventCount = 0;
  private lastTickAt = 0;
  private lastEventChangeAt = 0;

  constructor(onUpdate: (metrics: HardwareMonitorMetrics) => void, tickIntervalMs = 800) {
    this.onUpdate = onUpdate;
    this.tickIntervalMs = Math.max(500, tickIntervalMs);
  }

  start(getSnapshot: () => RuntimeActivitySnapshot): void {
    this.stop();
    this.lastTickAt = typeof performance !== 'undefined' ? performance.now() : Date.now();
    this.lastEventChangeAt = this.lastTickAt;
    this.startPressureObserver();
    this.timer = setInterval(() => {
      this.tick(getSnapshot());
    }, this.tickIntervalMs);
    this.tick(getSnapshot());
  }

  stop(): void {
    if (this.timer !== null) {
      clearInterval(this.timer);
      this.timer = null;
    }
    if (this.pressureObserver) {
      try {
        this.pressureObserver.disconnect();
      } catch {
        // ignore disconnect errors
      }
      this.pressureObserver = null;
    }
  }

  private startPressureObserver(): void {
    const PressureObserverCtor = (globalThis as any).PressureObserver;
    if (!PressureObserverCtor) {
      this.thermalState = 'unknown';
      return;
    }
    try {
      this.pressureObserver = new PressureObserverCtor((records: PressureObserverRecordLike[]) => {
        const latest = records?.[records.length - 1];
        const state = latest?.state;
        if (state === 'nominal' || state === 'fair' || state === 'serious' || state === 'critical') {
          this.thermalState = state;
        }
      }) as PressureObserverLike;
      const observed = this.pressureObserver.observe('cpu', { sampleInterval: this.tickIntervalMs });
      if (observed && typeof (observed as Promise<void>).catch === 'function') {
        (observed as Promise<void>).catch(() => {
          this.thermalState = 'unknown';
        });
      }
    } catch {
      this.thermalState = 'unknown';
      this.pressureObserver = null;
    }
  }

  private tick(snapshot: RuntimeActivitySnapshot): void {
    const now = typeof performance !== 'undefined' ? performance.now() : Date.now();
    const elapsedSec = Math.max(0.25, (now - this.lastTickAt) / 1000);
    const eventsChanged = snapshot.eventCount !== this.lastEventCount;
    if (eventsChanged) this.lastEventChangeAt = now;

    const deltaOps = Math.max(0, snapshot.completedOps - this.lastOps);
    const deltaProgress = Math.max(0, snapshot.progressPercent - this.lastProgress);
    const recentActivity = now - this.lastEventChangeAt <= this.tickIntervalMs * 1.8;

    let utilization = 0;
    if (snapshot.busy) {
      const opRate = deltaOps / elapsedSec;
      const progressRate = deltaProgress / elapsedSec;
      const base = opRate * 12 + progressRate * 4;
      utilization = clamp(base, 0, 100);
      if (recentActivity && utilization < 12) utilization = 12;
    }

    const perfMemory = (performance as any)?.memory?.usedJSHeapSize;
    const heapMB = Number.isFinite(perfMemory) ? perfMemory / (1024 * 1024) : 0;
    const stateEstimateMB = Math.max(0, snapshot.estimatedStateBytes) / (1024 * 1024);
    const memoryMB = Math.max(0, heapMB, stateEstimateMB);

    const energyImpact = toEnergyImpact(utilization, this.thermalState, snapshot.busy);
    this.onUpdate({
      utilizationPercent: snapshot.busy ? utilization : 0,
      memoryMB,
      thermalState: this.thermalState,
      energyImpact,
      updatedAt: Date.now(),
    });

    this.lastOps = snapshot.completedOps;
    this.lastProgress = snapshot.progressPercent;
    this.lastEventCount = snapshot.eventCount;
    this.lastTickAt = now;
  }
}
