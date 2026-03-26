# Design Review: Local-First WASM Training Architecture

**Document**: WebKaya Local-First WASM Architecture Review  
**Status**: Draft  
**Branch**: `cursor/local-first-wasm-design-212e`

---

## Executive Summary

The WebKaya "Zero-Cloud ML Studio" vision is ambitious and directionally sound — browser-native ML training addresses real pain points around privacy, latency, and cost. However, the current design has several **critical feasibility gaps**, particularly around the LibTorch-to-WASM compilation pipeline, WebGPU integration, and memory constraints. This review identifies 12 issues across three severity tiers and proposes concrete improvements.

---

## Table of Contents

1. [Critical Issues (Blockers)](#1-critical-issues-blockers)
2. [Significant Issues (High Risk)](#2-significant-issues-high-risk)
3. [Minor Issues (Improvements)](#3-minor-issues-improvements)
4. [Suggested Improvements](#4-suggested-improvements)
5. [Revised Architecture Proposal](#5-revised-architecture-proposal)
6. [Revised File Structure](#6-revised-file-structure)
7. [Risk Matrix](#7-risk-matrix)

---

## 1. Critical Issues (Blockers)

### 1.1 LibTorch → Emscripten Compilation Is Not Production-Viable

**The Problem**: The design assumes LibTorch can be compiled to WASM via Emscripten and used for training. While a proof-of-concept exists ([patrikohlsson/libtorch-wasm](https://github.com/patrikohlsson/libtorch-wasm)), it explicitly warns:

> *"WASM builds are slow, and optimization libraries are disabled, so native performance shouldn't be expected."*

LibTorch is ~150MB+ as a native library. Key issues:

- **Autograd engine**: LibTorch's autograd relies on dynamic dispatch, exception handling, and RTTI — all of which are problematic under Emscripten. `aten` operators use TH/THC backends that assume native threading (`std::thread`, OpenMP), which WASM doesn't support natively.
- **Binary size**: Even with dead-code elimination, a WASM build of LibTorch will be **40-80MB+**, resulting in unacceptable initial load times.
- **Missing operators**: Many ATen operators depend on BLAS/LAPACK (via MKL or OpenBLAS). Emscripten builds disable these, meaning operations like `torch::linalg::solve` or efficient `mm` will fall back to naive implementations.

**Recommendation**: Do not use LibTorch as the WASM compilation target. See [Section 4.1](#41-replace-libtorch-with-a-purpose-built-wasm-compute-kernel) for alternatives.

### 1.2 No Viable LibTorch → WebGPU Backend Exists

**The Problem**: The design states "WebGPU (via the wgpu or WebGPU backend in LibTorch)." **No such backend exists.** LibTorch backends are:

- CPU (default)
- CUDA (NVIDIA GPUs — not available in browser)
- ROCm (AMD GPUs — not available in browser)
- MPS (Apple Metal — not available in browser)
- Vulkan (experimental, not in WASM)
- XLA (for TPUs)

WebGPU is a *browser* API. LibTorch has no integration with it. Creating one would require:

1. Writing a full ATen dispatch backend mapping ~1,200+ operators to WGSL compute shaders
2. Handling memory management between WASM linear memory and GPU buffers
3. Managing async GPU dispatch from synchronous C++ code

This is a multi-year engineering effort for a dedicated team.

**Recommendation**: Use a framework that already targets WebGPU natively. See [Section 4.1](#41-replace-libtorch-with-a-purpose-built-wasm-compute-kernel).

### 1.3 WASM Memory Limits Constrain Model + Data Size

**The Problem**: Standard WASM has a **4GB memory ceiling** (32-bit linear memory). While the Memory64 proposal has reached Phase 4 and is implemented in Chrome and Firefox, practical constraints remain:

- A ResNet-50 model alone uses ~100MB of weights. With optimizer states (Adam stores 2x parameter copies), activations for backprop, and a batch of images, you can easily need **2-4GB** just for a medium CNN.
- For VAEs on image data (the stated use case), training on 256x256 images with batch size 32 requires storing all intermediate activations for the backward pass.
- Browser tabs are typically limited to 1-4GB even on capable machines; the OS and browser itself consume memory.

**Recommendation**: 
- Design with strict memory budgets. Use gradient checkpointing to trade compute for memory.
- Implement model-parallel sharding if needed (split layers across Web Workers, though this adds latency).
- Target the Memory64 spec but have a graceful fallback.

---

## 2. Significant Issues (High Risk)

### 2.1 OPFS Cannot Reliably Handle "GBs of Images"

**The Problem**: The design claims OPFS provides "near-native disk speed" for "GBs of images." Reality:

- **Quota**: Browsers dynamically allocate OPFS quota. Practical limits are **300MB to a few GB** depending on the device, browser, and available disk space. There is no API to request a specific quota.
- **No streaming reads**: OPFS `read()` copies data into ArrayBuffers. For a 5GB dataset, you'd need to implement your own chunked streaming layer.
- **Sync API only in Workers**: The performant synchronous `read()`/`write()` methods are only available inside Web Workers — not the main thread, iframes, or SharedWorkers.
- **No directory watching**: Unlike native FS, OPFS has no `inotify`-style events for changed files.

**Recommendation**: 
- Use OPFS for the **model weights cache** and preprocessed tensor batches, not raw image storage.
- For raw data ingestion, use the File System Access API (user grants access to a local directory) with lazy loading.
- Implement a **DataLoader** abstraction that prefetches the next N batches into OPFS while the current batch trains.

### 2.2 "Canvas API for Image Preprocessing" Won't Scale

**The Problem**: Using `<canvas>` for image resizing/normalization has issues:

- Canvas operations run on the **main thread** (unless using OffscreenCanvas) and will block the UI during batch preprocessing.
- Canvas uses premultiplied alpha, which corrupts pixel values for ML preprocessing.
- No EXIF-aware rotation handling by default.
- Resizing quality varies across browsers (different interpolation algorithms).

**Recommendation**: Use `createImageBitmap()` + `OffscreenCanvas` in a Web Worker, or better yet, do the resize/normalize in WASM/WebGPU compute shaders directly. The preprocessing pipeline should live in the same Worker as the training engine.

### 2.3 The ".kaya Bundle" Format Needs More Thought

**The Problem**: A custom `.kaya` format (Zip/Protobuf of `.pth` + architecture JSON) has portability issues:

- `.pth` files are PyTorch pickle format — they require Python/LibTorch to deserialize. If WebKaya doesn't use LibTorch internally (per our recommendation), the format should change.
- Protobuf adds a build dependency (protoc) and requires schema management.
- "Guaranteed to run in any WASM environment" is only true if you also ship the inference runtime.

**Recommendation**:
- Use **ONNX** as the export format. It's the industry standard for portable model interchange and has runtimes for every platform (ONNX Runtime Web, ONNX.js, TFLite via ONNX converter).
- The `.kaya` bundle should contain: `model.onnx` (weights + graph), `metadata.json` (hyperparameters, training history, input/output schema), and `config.json` (preprocessing pipeline definition).
- Alternatively, use **SafeTensors** format for weights (no arbitrary code execution risk, unlike pickle).

### 2.4 Training Loop Error Handling and Resumability

**The Problem**: The design doesn't address what happens when:

- The user closes the tab mid-training
- The browser runs out of memory (OOM)
- A NaN/Inf propagates through the loss
- WebGPU device is lost (GPU reset, driver crash)

Browser environments are inherently less stable than a server. Users will lose hours of training progress without checkpointing.

**Recommendation**:
- Implement **automatic checkpointing** to OPFS every N epochs (configurable).
- On page load, detect incomplete training runs and offer to resume.
- Add NaN/Inf detection in the loss with automatic learning rate reduction.
- Handle WebGPU device loss events (`device.lost` promise) gracefully.

---

## 3. Minor Issues (Improvements)

### 3.1 DuckDB-WASM for CSV Cleaning Is Overkill

DuckDB-WASM adds ~10MB to the bundle and boots slowly. For CSV cleaning (filtering rows, type coercion, null handling), consider:

- **Papa Parse** (~14KB) for streaming CSV parsing
- **Arquero** (~150KB) for DataFrame-like transformations  
- Custom WASM preprocessing if you already have the engine loaded

Reserve DuckDB-WASM for genuinely SQL-heavy workloads (joins across multiple CSVs, complex aggregations).

### 3.2 "60fps Loss Curves" Is Unnecessary and Wasteful

Rendering loss curves at 60fps means updating every 16.6ms. Training steps (forward + backward) on even a small CNN take 50-500ms. You're rendering the same data point multiple times. This wastes main-thread cycles and can cause jank.

**Recommendation**: Update charts on a **per-step** or **per-epoch** basis. Use `requestAnimationFrame` but only render when new data arrives. Libraries like `uPlot` (lightweight, ~35KB) are better than heavy options like Chart.js for real-time streaming data.

### 3.3 File Structure Missing Key Components

The proposed structure lacks:

- Build configuration (CMakeLists.txt or Makefile for the C++ → WASM pipeline)
- Test infrastructure
- Type definitions for the WASM ↔ JS bridge
- Worker entry point
- Data pipeline code

See [Section 6](#6-revised-file-structure) for a complete structure.

### 3.4 Thread Model Needs Clarification

The design says "the Nadi lives in a Web Worker" (singular). For "heavy-duty" CNN/VAE training, you likely need:

- **1 Worker** for the compute engine (WASM + WebGPU)
- **1 Worker** for data loading/preprocessing (to keep the pipeline fed)
- **Main thread** for UI only

These workers need a communication protocol (SharedArrayBuffer for zero-copy tensor transfer, or `postMessage` with Transferable objects).

---

## 4. Suggested Improvements

### 4.1 Replace LibTorch with a Purpose-Built WASM Compute Kernel

Instead of compiling the monolithic LibTorch, adopt one of these strategies:

| Approach | Pros | Cons |
|----------|------|------|
| **torch.js** (TypeScript + WebGPU) | 70%+ PyTorch API compatibility, actively maintained, native WebGPU | TypeScript not C++, smaller op coverage |
| **Custom WGSL kernels** (hand-written) | Maximum control, minimal binary size | Huge engineering effort, error-prone |
| **TVM/Apache TVM Web** (compiler approach) | Compiles models to optimized WASM+WebGPU, production-proven (WebLLM uses this) | Compile step needed, less flexible for dynamic graphs |
| **ONNX Runtime Web** (WebGPU backend) | Industry standard, Microsoft-backed, inference + training support | Training support is limited compared to inference |
| **webgpu-torch** | PyTorch-inspired API with autograd, WebGPU tensors | Less mature ecosystem |

**Recommended path**: Use **torch.js** or **webgpu-torch** for the compute layer. These provide PyTorch-like APIs with native WebGPU acceleration, eliminating the LibTorch-to-WASM compilation problem entirely. For the custom model builder, write a thin abstraction layer that maps the drag-and-drop UI to the framework's API.

If you must use C++, consider writing a **minimal custom engine** in C++ compiled to WASM that handles only the operations you need (Conv2d, Linear, ReLU, BatchNorm, etc.) with hand-written WGSL kernels dispatched via WebGPU. This keeps the WASM binary under 1MB.

### 4.2 Implement a Tiered Compute Strategy

Not every user has a GPU. Design for graceful degradation:

```
Tier 1: WebGPU available     → Full GPU-accelerated training
Tier 2: WebGL2 only          → Use WebGL compute (limited but works for small models)
Tier 3: CPU only (WASM SIMD) → Use WASM SIMD (128-bit) for vectorized ops; warn user about speed
Tier 4: Unsupported browser  → Show "please upgrade" or offer a cloud fallback
```

Detect capabilities at startup and adjust the batch size, model size limits, and expected training times accordingly.

### 4.3 Add a Model Complexity Estimator

Before training starts, estimate:

- **Memory required**: Based on model parameters, batch size, and activation sizes
- **Time per epoch**: Based on a quick benchmark kernel run
- **Feasibility score**: "Your model needs ~2.1GB RAM and ~45min/epoch on your hardware"

This prevents users from attempting to train a model that will OOM or take days.

### 4.4 Progressive Training with Transfer Learning

For "heavy-duty" models like CNNs, support **transfer learning**:

1. Ship pre-trained backbone weights (e.g., MobileNetV2 features) as part of the app.
2. Users only train the classification head → 10-100x faster.
3. Offer fine-tuning with frozen layers as an intermediate option.

This makes "heavy-duty" training practical even on modest hardware.

### 4.5 Federated Learning Extension (Future)

The local-first architecture naturally extends to federated learning:

1. Multiple users train on their local data.
2. Only model weight **deltas** (gradients) are shared via WebRTC or a lightweight relay.
3. A coordinator aggregates updates.

This maintains the privacy guarantee while enabling distributed training.

### 4.6 Use Web Workers with SharedArrayBuffer

For zero-copy data transfer between the data-loading worker and the training worker:

```
Main Thread              DataLoader Worker         Training Worker
    │                         │                         │
    ├─ UI events ────────────>│                         │
    │                         ├─ Load batch to SAB ────>│
    │                         │                         ├─ Train step
    │<─ Progress update ──────┼─────────────────────────┤
    │                         │<─ Request next batch ───┤
```

This requires `Cross-Origin-Isolation` headers (`Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp`), which must be set on the Next.js server.

---

## 5. Revised Architecture Proposal

```
┌─────────────────────────────────────────────────────────────────┐
│  Browser Tab                                                     │
│                                                                  │
│  ┌──────────────┐   ┌──────────────────┐   ┌─────────────────┐ │
│  │  Main Thread  │   │  Data Worker      │   │ Training Worker │ │
│  │              │   │                  │   │                 │ │
│  │  Next.js UI  │◄──┤  File System     │   │  WASM Engine    │ │
│  │  - Pipeline  │   │  Access API      │──►│  (torch.js or   │ │
│  │    Builder   │   │  - Lazy loading  │SAB│   custom C++)   │ │
│  │  - Charts    │   │  - Preprocessing │   │                 │ │
│  │  - Controls  │   │  OPFS Cache      │   │  WebGPU Context │ │
│  │              │   │  - Batch buffers │   │  - Compute      │ │
│  │              │   │  - Checkpoints   │   │    Shaders      │ │
│  └──────┬───────┘   └──────────────────┘   └────────┬────────┘ │
│         │                                            │          │
│         └────────────── postMessage ─────────────────┘          │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  Storage Layer                                               │ │
│  │  - OPFS: checkpoints, preprocessed batches, model cache     │ │
│  │  - IndexedDB: training history, user preferences            │ │
│  │  - File System Access API: raw dataset (user directory)     │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘

Export: .kaya bundle = model.onnx + metadata.json + config.json
```

---

## 6. Revised File Structure

```
/webkaya
├── README.md
├── package.json
├── next.config.js                   # Cross-Origin-Isolation headers
├── tsconfig.json
│
├── /src
│   ├── /engine                      # Compute Engine
│   │   ├── /wasm                    # If using custom C++ kernel
│   │   │   ├── CMakeLists.txt       # Emscripten build config
│   │   │   ├── ops.cpp              # Custom operators (Conv2d, Linear, etc.)
│   │   │   ├── autograd.cpp         # Minimal autograd engine
│   │   │   └── bindings.cpp         # Embind/Emscripten glue
│   │   │
│   │   ├── /kernels                 # WebGPU compute shaders
│   │   │   ├── matmul.wgsl
│   │   │   ├── conv2d.wgsl
│   │   │   ├── relu.wgsl
│   │   │   ├── batchnorm.wgsl
│   │   │   └── softmax.wgsl
│   │   │
│   │   ├── model-builder.ts         # Translates UI graph → compute graph
│   │   ├── trainer.ts               # Training loop orchestration
│   │   ├── optimizer.ts             # SGD, Adam implementations
│   │   ├── checkpoint.ts            # Save/restore to OPFS
│   │   └── capability-detect.ts     # WebGPU/WebGL/SIMD feature detection
│   │
│   ├── /data                        # Data Pipeline
│   │   ├── data-worker.ts           # Web Worker entry point
│   │   ├── loader.ts                # Batch loading + prefetching
│   │   ├── preprocessor.ts          # Resize, normalize, augment
│   │   ├── opfs-cache.ts            # OPFS read/write abstraction
│   │   └── csv-parser.ts            # Lightweight CSV ingestion
│   │
│   ├── /ui                          # Next.js Frontend
│   │   ├── /components
│   │   │   ├── PipelineCanvas.tsx   # Drag-and-drop model builder
│   │   │   ├── LayerPalette.tsx     # Available layers sidebar
│   │   │   ├── TrainingDashboard.tsx# Loss curves, metrics, progress
│   │   │   ├── DatasetPanel.tsx     # File upload / directory select
│   │   │   ├── HardwareMonitor.tsx  # GPU/memory usage display
│   │   │   └── ExportDialog.tsx     # .kaya bundle export
│   │   │
│   │   ├── /hooks
│   │   │   ├── useTrainingWorker.ts # Manages training Web Worker lifecycle
│   │   │   ├── useDataWorker.ts     # Manages data loading worker
│   │   │   ├── useModelGraph.ts     # Pipeline builder state
│   │   │   └── useCapabilities.ts   # Hardware capability detection
│   │   │
│   │   └── /pages                   # Next.js pages
│   │       ├── index.tsx            # Landing / Studio
│   │       └── _app.tsx             # App wrapper
│   │
│   ├── /export                      # Model Export
│   │   ├── onnx-exporter.ts         # Convert trained model → ONNX
│   │   ├── kaya-bundle.ts           # Package .kaya archive
│   │   └── metadata.ts              # Training metadata serialization
│   │
│   └── /types                       # Shared Type Definitions
│       ├── model.ts                 # Layer, Graph, Model types
│       ├── training.ts              # TrainingConfig, Metrics types
│       ├── worker-messages.ts       # Worker ↔ Main thread protocol
│       └── engine.ts                # WASM/WebGPU engine interface
│
├── /public
│   ├── kaya_engine.wasm             # Compiled WASM binary (if using C++)
│   └── /pretrained                  # Pre-trained backbones for transfer learning
│       └── mobilenetv2-features.onnx
│
├── /tests
│   ├── engine.test.ts               # Compute engine unit tests
│   ├── trainer.test.ts              # Training loop tests
│   ├── data-pipeline.test.ts        # Data loading tests
│   └── export.test.ts               # Export format tests
│
└── /scripts
    ├── build-wasm.sh                # Emscripten build script
    └── benchmark.ts                 # Performance benchmarking
```

---

## 7. Risk Matrix

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| LibTorch WASM compilation fails or is too slow | **High** | **Critical** | Use torch.js/webgpu-torch instead |
| WebGPU not available on target browsers | **Medium** | **High** | Tiered compute strategy (WebGL/SIMD fallback) |
| WASM OOM on large models | **High** | **High** | Memory estimator, gradient checkpointing, Memory64 |
| OPFS quota exceeded | **Medium** | **Medium** | Lazy loading from FS Access API, batch caching only |
| Training interrupted (tab close/crash) | **High** | **Medium** | Auto-checkpointing to OPFS |
| WebGPU device lost mid-training | **Low** | **High** | Device loss handler, resume from checkpoint |
| Cross-Origin-Isolation breaks third-party scripts | **Medium** | **Low** | Careful header configuration, test integrations |
| User hardware too weak for meaningful training | **Medium** | **Medium** | Complexity estimator, transfer learning defaults |

---

## Summary of Recommendations

1. **Drop LibTorch/Emscripten** — use torch.js, webgpu-torch, or a custom minimal engine
2. **Don't assume WebGPU** — implement tiered compute (WebGPU → WebGL → WASM SIMD)
3. **Fix the data pipeline** — OPFS for caching, File System Access API for raw data, OffscreenCanvas for preprocessing
4. **Use ONNX for export** — not `.pth` pickle format
5. **Add checkpointing** — auto-save to OPFS, resume on page reload
6. **Add a complexity estimator** — prevent users from attempting infeasible training runs
7. **Support transfer learning** — ship pre-trained backbones for practical "heavy-duty" training
8. **Design the worker protocol** — SharedArrayBuffer for zero-copy transfer, typed message protocol
9. **Use lightweight charting** — per-step updates, not 60fps rendering
10. **Plan for memory** — gradient checkpointing, strict budgets, Memory64 where available
