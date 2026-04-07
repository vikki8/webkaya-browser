/**
 * Worker ↔ main thread message contracts (discriminated unions on `type`).
 * Import these types in hooks and workers so new variants are updated in one place.
 *
 * Note: `error` and `log` exist on multiple workers — always narrow using the worker
 * instance (data vs training), not a shared type guard.
 */
export type { MainToDataWorkerMessage, WorkerToDataMainMessage } from './data-worker';
export type { MainToTrainingWorkerMessage, WorkerToTrainingMainMessage } from './training-workflow';
