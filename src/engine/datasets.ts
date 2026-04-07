import { Tensor } from './tensor';

export interface DataBatch {
  inputs: Tensor;
  targets: Int32Array;
}

export interface Dataset {
  getBatch(batchIdx: number, batchSize: number): DataBatch;
  readonly numSamples: number;
  readonly inputShape: number[];
  readonly numClasses: number;
}

/**
 * Synthetic MNIST-like dataset: 28x28 grayscale images, 10 classes.
 * Each class is a distinct pattern (stripes, circles, corners, etc.)
 * so the CNN has actual structure to learn.
 */
export class SyntheticMNIST implements Dataset {
  private images: Float32Array;
  private labels: Int32Array;
  readonly numSamples: number;
  readonly inputShape = [1, 28, 28];
  readonly numClasses = 10;

  constructor(numSamples = 1000) {
    this.numSamples = numSamples;
    const imgSize = 28 * 28;
    this.images = new Float32Array(numSamples * imgSize);
    this.labels = new Int32Array(numSamples);

    for (let i = 0; i < numSamples; i++) {
      const label = i % 10;
      this.labels[i] = label;
      const offset = i * imgSize;
      this.generatePattern(this.images, offset, label);
      for (let j = 0; j < imgSize; j++) {
        this.images[offset + j] += (Math.random() - 0.5) * 0.2;
      }
    }
  }

  private generatePattern(buf: Float32Array, offset: number, label: number) {
    const W = 28;
    for (let y = 0; y < W; y++) {
      for (let x = 0; x < W; x++) {
        const idx = offset + y * W + x;
        const cx = x - 14, cy = y - 14;
        const dist = Math.sqrt(cx * cx + cy * cy);
        switch (label) {
          case 0: buf[idx] = dist < 10 ? 1 : 0; break; // circle
          case 1: buf[idx] = Math.abs(x - 14) < 3 ? 1 : 0; break; // vertical line
          case 2: buf[idx] = Math.abs(y - 14) < 3 ? 1 : 0; break; // horizontal line
          case 3: buf[idx] = (x + y) % 6 < 3 ? 1 : 0; break; // diagonal stripes
          case 4: buf[idx] = x < 14 && y < 14 ? 1 : 0; break; // top-left
          case 5: buf[idx] = x >= 14 && y >= 14 ? 1 : 0; break; // bottom-right
          case 6: buf[idx] = (x % 4 < 2 && y % 4 < 2) ? 1 : 0; break; // checkerboard
          case 7: buf[idx] = dist > 8 && dist < 12 ? 1 : 0; break; // ring
          case 8: buf[idx] = (Math.abs(cx) + Math.abs(cy)) < 10 ? 1 : 0; break; // diamond
          case 9: buf[idx] = (x > 6 && x < 22 && y > 6 && y < 22 &&
                              !(x > 10 && x < 18 && y > 10 && y < 18)) ? 1 : 0; break; // frame
        }
      }
    }
  }

  getBatch(batchIdx: number, batchSize: number): DataBatch {
    const start = (batchIdx * batchSize) % this.numSamples;
    const actualSize = Math.min(batchSize, this.numSamples - start);
    const imgSize = 28 * 28;
    const inputData = new Float32Array(actualSize * imgSize);
    const targetData = new Int32Array(actualSize);

    for (let i = 0; i < actualSize; i++) {
      const srcIdx = start + i;
      inputData.set(this.images.subarray(srcIdx * imgSize, (srcIdx + 1) * imgSize), i * imgSize);
      targetData[i] = this.labels[srcIdx];
    }

    return {
      inputs: new Tensor(inputData, [actualSize, 1, 28, 28]),
      targets: targetData,
    };
  }
}

/**
 * Synthetic CIFAR-like dataset: 32x32 RGB images, 10 classes.
 */
export class SyntheticCIFAR implements Dataset {
  private images: Float32Array;
  private labels: Int32Array;
  readonly numSamples: number;
  readonly inputShape = [3, 32, 32];
  readonly numClasses = 10;

  constructor(numSamples = 1000) {
    this.numSamples = numSamples;
    const imgSize = 3 * 32 * 32;
    this.images = new Float32Array(numSamples * imgSize);
    this.labels = new Int32Array(numSamples);

    for (let i = 0; i < numSamples; i++) {
      const label = i % 10;
      this.labels[i] = label;
      const offset = i * imgSize;
      this.generateColorPattern(this.images, offset, label);
    }
  }

  private generateColorPattern(buf: Float32Array, offset: number, label: number) {
    const W = 32;
    const chSize = W * W;
    const colors = [
      [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1],
      [0, 1, 1], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5], [1, 1, 1]
    ];
    const [r, g, b] = colors[label];

    for (let y = 0; y < W; y++) {
      for (let x = 0; x < W; x++) {
        const cx = x - 16, cy = y - 16;
        const dist = Math.sqrt(cx * cx + cy * cy);
        const pattern = label < 5
          ? (dist < 12 ? 1 : 0)
          : ((x + y) % (label + 2) < (label / 2 + 1) ? 1 : 0);
        const noise = (Math.random() - 0.5) * 0.15;
        buf[offset + 0 * chSize + y * W + x] = pattern * r + noise;
        buf[offset + 1 * chSize + y * W + x] = pattern * g + noise;
        buf[offset + 2 * chSize + y * W + x] = pattern * b + noise;
      }
    }
  }

  getBatch(batchIdx: number, batchSize: number): DataBatch {
    const start = (batchIdx * batchSize) % this.numSamples;
    const actualSize = Math.min(batchSize, this.numSamples - start);
    const imgSize = 3 * 32 * 32;
    const inputData = new Float32Array(actualSize * imgSize);
    const targetData = new Int32Array(actualSize);

    for (let i = 0; i < actualSize; i++) {
      const srcIdx = start + i;
      inputData.set(this.images.subarray(srcIdx * imgSize, (srcIdx + 1) * imgSize), i * imgSize);
      targetData[i] = this.labels[srcIdx];
    }

    return {
      inputs: new Tensor(inputData, [actualSize, 3, 32, 32]),
      targets: targetData,
    };
  }
}

/** Simple tabular dataset for linear models */
export class SyntheticTabular implements Dataset {
  private features: Float32Array;
  private labels: Int32Array;
  readonly numSamples: number;
  readonly inputShape: number[];
  readonly numClasses = 4;

  constructor(numSamples = 500, numFeatures = 16) {
    this.numSamples = numSamples;
    this.inputShape = [numFeatures];
    this.features = new Float32Array(numSamples * numFeatures);
    this.labels = new Int32Array(numSamples);

    const weights = new Float32Array(numFeatures);
    for (let i = 0; i < numFeatures; i++) weights[i] = (Math.random() - 0.5) * 2;

    for (let i = 0; i < numSamples; i++) {
      let sum = 0;
      for (let f = 0; f < numFeatures; f++) {
        const val = (Math.random() - 0.5) * 2;
        this.features[i * numFeatures + f] = val;
        sum += val * weights[f];
      }
      this.labels[i] = Math.min(3, Math.max(0, Math.floor((sum + 4) / 2)));
    }
  }

  getBatch(batchIdx: number, batchSize: number): DataBatch {
    const numFeatures = this.inputShape[0];
    const start = (batchIdx * batchSize) % this.numSamples;
    const actualSize = Math.min(batchSize, this.numSamples - start);
    const inputData = new Float32Array(actualSize * numFeatures);
    const targetData = new Int32Array(actualSize);

    for (let i = 0; i < actualSize; i++) {
      const srcIdx = start + i;
      inputData.set(
        this.features.subarray(srcIdx * numFeatures, (srcIdx + 1) * numFeatures),
        i * numFeatures
      );
      targetData[i] = this.labels[srcIdx];
    }

    return {
      inputs: new Tensor(inputData, [actualSize, numFeatures]),
      targets: targetData,
    };
  }
}
