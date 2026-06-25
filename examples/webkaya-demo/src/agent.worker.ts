// A local worker entry (inside the Vite project root, so the bundler compiles
// it reliably as a module worker). It wires this worker to the SDK's guest
// runtime — the same `runGuestRequest` core used everywhere else.
import { installWorkerHandler } from '@webkaya/sandbox';

installWorkerHandler(self as unknown as {
  onmessage: ((event: MessageEvent) => void) | null;
  postMessage: (message: unknown) => void;
});
