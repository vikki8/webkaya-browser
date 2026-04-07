export type ThermalPressureState = 'nominal' | 'fair' | 'serious' | 'critical' | 'unknown';
export type EnergyImpactLevel = 'idle' | 'low' | 'moderate' | 'high' | 'heavy';

export interface HardwareMonitorSnapshot {
  trainingActive: boolean;
  progressPercent: number;
  curveStep: number;
  curveLength: number;
  datasetRows: number;
  featureCount: number;
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

function toEnergyImpact(utilizationPercent: number, thermalState: ThermalPressureState, trainingActive: boolean): EnergyImpactLevel {
  if (!trainingActive || utilizationPercent <= 0.1) return 'idle';
  const thermalBoost = thermalState === 'critical' ? 15 : thermalState === 'serious' ? 8 : thermalState === 'fair' ? 3 : 0;
  const score = utilizationPercent + thermalBoost;
  if (score >= 85) return 'heavy';
  if (score >= 65) return 'high';
  if (score >= 35) return 'moderate';
  return 'low';
}

export class HardwareMonitor {
  private readonly tickIntervalMs: number;
  private readonly onUpdate: (metrics: HardwareMonitorMetrics) => void;
  private timer: number | null = null;
  private pressureObserver: PressureObserverLike | null = null;
  private thermalState: ThermalPressureState = 'unknown';
  private lastStep = 0;
  private lastProgress = 0;
  private lastCurveLength = 0;
  private lastTickAt = 0;
  private lastCurveChangeAt = 0;

  constructor(onUpdate: (metrics: HardwareMonitorMetrics) => void, tickIntervalMs = 800) {
    this.onUpdate = onUpdate;
    this.tickIntervalMs = Math.max(500, tickIntervalMs);
  }

  start(getSnapshot: () => HardwareMonitorSnapshot): void {
    this.stop();
    this.lastTickAt = typeof performance !== 'undefined' ? performance.now() : Date.now();
    this.lastCurveChangeAt = this.lastTickAt;
    this.startPressureObserver();
    this.timer = window.setInterval(() => {
      this.tick(getSnapshot());
    }, this.tickIntervalMs);
    this.tick(getSnapshot());
  }

  stop(): void {
    if (this.timer !== null) {
      window.clearInterval(this.timer);
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
    if (typeof window === 'undefined') return;
    const PressureObserverCtor = (window as any).PressureObserver;
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

  private tick(snapshot: HardwareMonitorSnapshot): void {
    const now = typeof performance !== 'undefined' ? performance.now() : Date.now();
    const elapsedSec = Math.max(0.25, (now - this.lastTickAt) / 1000);
    const curveChanged = snapshot.curveLength !== this.lastCurveLength;
    if (curveChanged) this.lastCurveChangeAt = now;

    const deltaStep = Math.max(0, snapshot.curveStep - this.lastStep);
    const deltaProgress = Math.max(0, snapshot.progressPercent - this.lastProgress);
    const recentProgress = now - this.lastCurveChangeAt <= this.tickIntervalMs * 1.8;

    let utilization = 0;
    if (snapshot.trainingActive) {
      const stepRate = deltaStep / elapsedSec;
      const progressRate = deltaProgress / elapsedSec;
      const base = stepRate * 12 + progressRate * 4;
      utilization = clamp(base, 0, 100);
      if (recentProgress && utilization < 12) utilization = 12;
    }

    const perfMemory = (performance as any)?.memory?.usedJSHeapSize;
    const heapMB = Number.isFinite(perfMemory) ? perfMemory / (1024 * 1024) : 0;
    const tensorEstimateMB =
      (Math.max(0, snapshot.datasetRows) * Math.max(0, snapshot.featureCount) * 4) / (1024 * 1024);
    const memoryMB = Math.max(0, heapMB, tensorEstimateMB);

    const energyImpact = toEnergyImpact(utilization, this.thermalState, snapshot.trainingActive);
    this.onUpdate({
      utilizationPercent: snapshot.trainingActive ? utilization : 0,
      memoryMB,
      thermalState: this.thermalState,
      energyImpact,
      updatedAt: Date.now(),
    });

    this.lastStep = snapshot.curveStep;
    this.lastProgress = snapshot.progressPercent;
    this.lastCurveLength = snapshot.curveLength;
    this.lastTickAt = now;
  }
}

