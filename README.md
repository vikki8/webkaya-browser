# WebKaya — Local-First ML Studio

> Train neural networks directly in your browser. No cloud. No uploads. Your data stays on your machine.

WebKaya is a browser-native machine learning studio that runs CNN and dense network training entirely client-side using a custom tensor engine with autograd. Build model architectures visually, train on synthetic datasets, and export trained weights — all without leaving the browser.

---

## Quick Start

### Prerequisites

- **Node.js** 18+ (recommended: 20 LTS)
- **npm** 9+ (comes with Node.js)
- A modern browser: **Chrome 113+**, **Edge 113+**, **Firefox 120+**, or **Safari 17+**

### Install & Run

```bash
# Clone the repository
git clone https://github.com/vikki8/webkaya.git
cd webkaya

# Install dependencies
npm install

# Start the development server
npm run dev
```

Open **http://localhost:3000** in your browser.

### Production Build

```bash
# Build for production
npm run build

# Start the production server
npm start
```

The app will be available at **http://localhost:3000**.

---

## How to Use

### 1. Build Your Model

- **Presets**: Click "Simple CNN", "Deep CNN", or "Linear Net" in the left sidebar to load a preset architecture.
- **Custom**: Click any layer in the Layer Palette (left sidebar) to add it to your model. Drag layers to reorder them.
- **Configure**: Edit layer parameters directly in the pipeline canvas (center).

### 2. Configure Training

In the right sidebar:
- **Dataset**: Choose Synthetic MNIST (28x28 grayscale), Synthetic CIFAR-10 (32x32 RGB), or Synthetic Tabular.
- **Hyperparameters**: Set learning rate, batch size, epochs, optimizer (Adam/SGD), and loss function.
- The **Model Info** panel shows parameter count and estimated memory usage.

### 3. Train

- Click **Train** in the header toolbar.
- Watch real-time loss and accuracy curves in the bottom-left panel.
- Monitor training logs in the bottom-right console.
- **Pause/Resume/Stop** training at any time.

### 4. Export

- After training completes (or while paused), click **Export .kaya** to download your trained model.
- The `.kaya` bundle contains the model architecture and trained weights.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Browser Tab                                                 │
│                                                              │
│  ┌──────────────┐   ┌──────────────┐   ┌─────────────────┐ │
│  │  Main Thread  │   │  Training     │   │  Right Sidebar  │ │
│  │              │   │  Web Worker   │   │                 │ │
│  │  Next.js UI  │◄──┤              │   │  Config Panel   │ │
│  │  - Pipeline  │   │  Tensor       │   │  Hardware       │ │
│  │    Builder   │──►│  Autograd     │   │  Monitor        │ │
│  │  - Dashboard │msg│  Engine       │   │                 │ │
│  │  - Charts    │   │              │   │                 │ │
│  └──────────────┘   └──────────────┘   └─────────────────┘ │
│                                                              │
│  ┌──────────────────────────────────────────────────────────┐│
│  │  Bottom Panel: Loss/Accuracy Charts + Training Console   ││
│  └──────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

### Compute Engine (`src/engine/`)

- **`tensor.ts`** — Tensor class with full autograd support (backpropagation). Implements conv2d, matmul, relu, sigmoid, tanh, softmax, cross-entropy loss, MSE loss, dropout, pooling, and more. All operations track gradients for automatic differentiation.
- **`nn.ts`** — Neural network layer abstractions (Conv2d, Linear, BatchNorm2d, MaxPool2d, etc.) with a `Sequential` container.
- **`optimizer.ts`** — SGD and Adam optimizers.
- **`datasets.ts`** — Synthetic dataset generators (MNIST-like patterns, CIFAR-like RGB, tabular data) for immediate training without external data.
- **`model-builder.ts`** — Translates the UI's model graph configuration into a runnable `Sequential` model.
- **`checkpoint.ts`** — Serialize/restore model state for training resumability.
- **`trainer.ts`** — Web Worker entry point. Runs the training loop, reports progress via `postMessage`, handles pause/resume/stop/export.
- **`capability-detect.ts`** — Detects browser capabilities (WebGPU, WebGL2, WASM SIMD, OPFS, SharedArrayBuffer) and assigns a hardware tier.

### UI (`src/ui/`)

- **Zustand store** (`store.ts`) — Centralized state for model graph, training config, training state, and logs.
- **Components** — LayerPalette, PipelineCanvas, TrainingDashboard, ConfigPanel, HardwareMonitor, LogConsole.
- **Hooks** — `useTrainingWorker` (worker lifecycle), `useCapabilities` (hardware detection).

---

## Available Layers

| Layer | Parameters | Description |
|-------|-----------|-------------|
| Conv2d | inChannels, outChannels, kernelSize, stride, padding | 2D convolution |
| Linear | inFeatures, outFeatures | Fully connected / dense |
| ReLU | — | Rectified linear activation |
| Sigmoid | — | Sigmoid activation |
| Tanh | — | Hyperbolic tangent activation |
| Softmax | — | Softmax activation |
| MaxPool2d | kernelSize, stride | Max pooling |
| AvgPool2d | kernelSize, stride | Average pooling |
| BatchNorm2d | numFeatures | Batch normalization |
| Dropout | p (drop probability) | Dropout regularization |
| Flatten | — | Flatten spatial dims to 1D |

---

## Synthetic Datasets

Since this is a local-first app, it ships with synthetic datasets that generate structured patterns for the model to learn:

- **Synthetic MNIST**: 28x28 grayscale, 10 classes (circles, lines, stripes, corners, checkerboard, etc.)
- **Synthetic CIFAR-10**: 32x32 RGB, 10 classes (colored patterns)
- **Synthetic Tabular**: 16 numerical features, 4 classes

---

## Key Design Decisions

1. **No LibTorch/WASM** — Instead of compiling the 150MB+ LibTorch to WASM (which produces slow, bloated binaries), we built a minimal custom tensor engine in TypeScript with full autograd support.

2. **Web Worker isolation** — Training runs in a dedicated Web Worker so the UI stays responsive during computation.

3. **Cross-Origin Isolation** — The `next.config.js` sets `Cross-Origin-Opener-Policy` and `Cross-Origin-Embedder-Policy` headers to enable `SharedArrayBuffer` (needed for future WebGPU integration).

4. **Tiered hardware detection** — The app detects WebGPU, WebGL2, WASM SIMD availability and displays the hardware tier.

5. **NaN/Inf protection** — Training automatically halts if loss becomes NaN or Infinity, with a descriptive error message.

---

## Project Structure

```
webkaya/
├── src/
│   ├── engine/           # Core compute engine
│   │   ├── tensor.ts     # Tensor + autograd + all operations
│   │   ├── nn.ts         # Layer implementations + Sequential
│   │   ├── optimizer.ts  # SGD, Adam
│   │   ├── datasets.ts   # Synthetic data generators
│   │   ├── model-builder.ts
│   │   ├── checkpoint.ts
│   │   ├── trainer.ts    # Web Worker entry
│   │   └── capability-detect.ts
│   ├── ui/
│   │   ├── store.ts      # Zustand state management
│   │   ├── components/   # React components
│   │   └── hooks/        # Custom hooks
│   ├── types/            # Shared TypeScript types
│   ├── styles/           # Global CSS
│   └── pages/            # Next.js pages
├── docs/                 # Design documents
├── package.json
├── tsconfig.json
├── next.config.js
└── .eslintrc.json
```

---

## Troubleshooting

**"Worker error" on training start**
- Make sure you're using a browser that supports Web Workers with module syntax (Chrome 80+, Firefox 114+).
- Check the browser console (F12) for detailed error messages.

**Training is slow**
- Reduce batch size (try 8 or 16).
- Use a simpler model (Linear Net preset).
- Reduce epochs.
- The synthetic datasets are designed for quick iteration; a Simple CNN on MNIST should converge within 5-10 epochs.

**"NaN detected" error**
- Lower the learning rate (try 0.0001).
- Use Adam optimizer instead of SGD.
- Check that your model architecture is valid (e.g., Linear layer input features match the previous layer's output).

---

## License

MIT
