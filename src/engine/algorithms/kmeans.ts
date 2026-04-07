export interface KMeansTrainOptions {
  clusters: number;
  maxIterations: number;
  tolerance?: number;
  shouldStop?: () => boolean;
}

export interface KMeansEpochStat {
  iteration: number;
  inertia: number;
  movedFraction: number;
}

export interface KMeansModel {
  featureCount: number;
  clusterCount: number;
  centroids: number[][];
  assignments: number[];
  clusterToLabel: number[];
  classCount: number;
  fallbackLabel: number;
  lossCurve: KMeansEpochStat[];
}

function squaredDistance(a: number[], b: number[]): number {
  let sum = 0;
  const size = Math.min(a.length, b.length);
  for (let i = 0; i < size; i++) {
    const diff = (a[i] ?? 0) - (b[i] ?? 0);
    sum += diff * diff;
  }
  return sum;
}

function nearestCentroidIndex(sample: number[], centroids: number[][]): number {
  let bestIndex = 0;
  let bestDistance = Infinity;
  for (let i = 0; i < centroids.length; i++) {
    const distance = squaredDistance(sample, centroids[i]);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = i;
    }
  }
  return bestIndex;
}

function initializeCentroids(features: number[][], clusterCount: number): number[][] {
  const sampleCount = features.length;
  const centroids: number[][] = [];
  const picked = new Set<number>();

  centroids.push(features[0].slice());
  picked.add(0);

  while (centroids.length < clusterCount && picked.size < sampleCount) {
    let bestIndex = -1;
    let bestDistance = -1;
    for (let sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++) {
      if (picked.has(sampleIndex)) continue;
      let minDistance = Infinity;
      for (const centroid of centroids) {
        minDistance = Math.min(minDistance, squaredDistance(features[sampleIndex], centroid));
      }
      if (minDistance > bestDistance) {
        bestDistance = minDistance;
        bestIndex = sampleIndex;
      }
    }
    if (bestIndex < 0) break;
    picked.add(bestIndex);
    centroids.push(features[bestIndex].slice());
  }

  let cursor = 0;
  while (centroids.length < clusterCount) {
    centroids.push(features[cursor % sampleCount].slice());
    cursor++;
  }
  return centroids;
}

function mapClustersToLabels(
  assignments: number[],
  labels: number[],
  clusterCount: number,
  classCount: number
): { clusterToLabel: number[]; fallbackLabel: number } {
  const globalCounts = new Array(classCount).fill(0);
  for (const label of labels) {
    if (label >= 0 && label < classCount) globalCounts[label]++;
  }
  let fallbackLabel = 0;
  for (let label = 1; label < classCount; label++) {
    if (globalCounts[label] > globalCounts[fallbackLabel]) fallbackLabel = label;
  }

  const clusterToLabel = new Array(clusterCount).fill(fallbackLabel);
  const countsByCluster = Array.from({ length: clusterCount }, () => new Array(classCount).fill(0));
  for (let i = 0; i < assignments.length; i++) {
    const cluster = assignments[i] ?? -1;
    const label = labels[i] ?? fallbackLabel;
    if (cluster >= 0 && cluster < clusterCount && label >= 0 && label < classCount) {
      countsByCluster[cluster][label]++;
    }
  }
  for (let cluster = 0; cluster < clusterCount; cluster++) {
    let bestLabel = fallbackLabel;
    let bestCount = -1;
    for (let label = 0; label < classCount; label++) {
      if (countsByCluster[cluster][label] > bestCount) {
        bestCount = countsByCluster[cluster][label];
        bestLabel = label;
      }
    }
    clusterToLabel[cluster] = bestLabel;
  }
  return { clusterToLabel, fallbackLabel };
}

export function trainKMeansClassifier(
  features: number[][],
  labels: number[],
  classCount: number,
  options: KMeansTrainOptions,
  onIteration?: (stat: KMeansEpochStat) => void
): KMeansModel {
  if (!features.length) throw new Error('Cannot train K-Means on empty dataset.');
  const featureCount = features[0].length;
  const sampleCount = features.length;
  const clusterCount = Math.max(2, Math.min(options.clusters, sampleCount));
  const maxIterations = Math.max(1, Math.floor(options.maxIterations));
  const tolerance = Math.max(1e-7, options.tolerance ?? 1e-4);

  const centroids = initializeCentroids(features, clusterCount);

  const assignments = new Array(sampleCount).fill(0);
  const previousAssignments = new Array(sampleCount).fill(-1);
  const lossCurve: KMeansEpochStat[] = [];

  for (let iteration = 0; iteration < maxIterations; iteration++) {
    if (options.shouldStop?.()) break;

    let moved = 0;
    let inertia = 0;
    for (let i = 0; i < sampleCount; i++) {
      const cluster = nearestCentroidIndex(features[i], centroids);
      assignments[i] = cluster;
      if (previousAssignments[i] !== cluster) moved++;
      previousAssignments[i] = cluster;
      inertia += squaredDistance(features[i], centroids[cluster]);
    }

    const sums = Array.from({ length: clusterCount }, () => new Float64Array(featureCount));
    const counts = new Uint32Array(clusterCount);
    for (let i = 0; i < sampleCount; i++) {
      const cluster = assignments[i];
      counts[cluster]++;
      const row = features[i];
      for (let f = 0; f < featureCount; f++) {
        sums[cluster][f] += row[f] ?? 0;
      }
    }

    let maxCentroidShift = 0;
    for (let cluster = 0; cluster < clusterCount; cluster++) {
      if (!counts[cluster]) {
        const fallbackRow = features[(cluster * 17 + iteration * 31) % sampleCount];
        const shift = squaredDistance(centroids[cluster], fallbackRow);
        maxCentroidShift = Math.max(maxCentroidShift, shift);
        centroids[cluster] = fallbackRow.slice();
        continue;
      }
      const updated = new Array<number>(featureCount);
      for (let f = 0; f < featureCount; f++) {
        updated[f] = sums[cluster][f] / counts[cluster];
      }
      const shift = squaredDistance(centroids[cluster], updated);
      maxCentroidShift = Math.max(maxCentroidShift, shift);
      centroids[cluster] = updated;
    }

    const stat: KMeansEpochStat = {
      iteration: iteration + 1,
      inertia: inertia / Math.max(1, sampleCount),
      movedFraction: moved / Math.max(1, sampleCount),
    };
    lossCurve.push(stat);
    onIteration?.(stat);

    // Max shift across centroids (not sum) — summed shift scales with k×dims and can prevent convergence.
    if (maxCentroidShift <= tolerance) break;
  }

  const { clusterToLabel, fallbackLabel } = mapClustersToLabels(assignments, labels, clusterCount, classCount);

  return {
    featureCount,
    clusterCount,
    centroids,
    assignments,
    clusterToLabel,
    classCount,
    fallbackLabel,
    lossCurve,
  };
}

export function predictKMeansClassifier(model: KMeansModel, features: number[][]): number[] {
  return features.map((sample) => {
    const cluster = nearestCentroidIndex(sample, model.centroids);
    return model.clusterToLabel[cluster] ?? model.fallbackLabel;
  });
}

/**
 * Same as {@link predictKMeansClassifier} but yields between chunks so the worker stays responsive
 * and progress can be logged on large datasets.
 */
export async function predictKMeansClassifierAsync(
  model: KMeansModel,
  features: number[][],
  options?: { chunkSize?: number; onProgress?: (processed: number, total: number) => void }
): Promise<number[]> {
  const chunkSize = Math.max(512, Math.floor(options?.chunkSize ?? 4096));
  const total = features.length;
  const out: number[] = new Array(total);
  for (let start = 0; start < total; start += chunkSize) {
    const end = Math.min(total, start + chunkSize);
    for (let i = start; i < end; i++) {
      const cluster = nearestCentroidIndex(features[i], model.centroids);
      out[i] = model.clusterToLabel[cluster] ?? model.fallbackLabel;
    }
    options?.onProgress?.(end, total);
    if (end < total) {
      await new Promise<void>((resolve) => setTimeout(resolve, 0));
    }
  }
  return out;
}

export function kmeansFeatureImportance(model: KMeansModel): number[] {
  const scores = new Array(model.featureCount).fill(0);
  if (!model.centroids.length) return scores;

  for (let featureIndex = 0; featureIndex < model.featureCount; featureIndex++) {
    const values = model.centroids.map((centroid) => centroid[featureIndex] ?? 0);
    const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
    let variance = 0;
    for (const value of values) {
      const diff = value - mean;
      variance += diff * diff;
    }
    scores[featureIndex] = variance / values.length;
  }

  const maxScore = scores.reduce((max, value) => Math.max(max, value), 0);
  if (maxScore > 0) {
    for (let i = 0; i < scores.length; i++) {
      scores[i] /= maxScore;
    }
  }
  return scores;
}

