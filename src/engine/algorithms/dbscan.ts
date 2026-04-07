export interface DbscanTrainOptions {
  epsilon: number;
  minSamples: number;
  shouldStop?: () => boolean;
}

export interface DbscanEpochStat {
  processedPoints: number;
  clusterCount: number;
  noisePoints: number;
}

export interface DbscanModel {
  featureCount: number;
  epsilon: number;
  minSamples: number;
  clusterCount: number;
  noiseCount: number;
  trainFeatures: number[][];
  clusterLabels: number[];
  corePointIndices: number[];
  clusterToLabel: number[];
  classCount: number;
  fallbackLabel: number;
  lossCurve: DbscanEpochStat[];
}

const UNVISITED = -2;
const NOISE = -1;

function squaredDistance(a: number[], b: number[]): number {
  let sum = 0;
  const length = Math.min(a.length, b.length);
  for (let i = 0; i < length; i++) {
    const diff = (a[i] ?? 0) - (b[i] ?? 0);
    sum += diff * diff;
  }
  return sum;
}

function regionQuery(features: number[][], index: number, epsilonSquared: number): number[] {
  const neighbors: number[] = [];
  const point = features[index];
  for (let i = 0; i < features.length; i++) {
    if (squaredDistance(point, features[i]) <= epsilonSquared) {
      neighbors.push(i);
    }
  }
  return neighbors;
}

function mapClustersToLabels(
  clusterLabels: number[],
  targetLabels: number[],
  clusterCount: number,
  classCount: number
): { clusterToLabel: number[]; fallbackLabel: number } {
  const globalCounts = new Array(classCount).fill(0);
  for (const label of targetLabels) {
    if (label >= 0 && label < classCount) globalCounts[label]++;
  }
  let fallbackLabel = 0;
  for (let label = 1; label < classCount; label++) {
    if (globalCounts[label] > globalCounts[fallbackLabel]) fallbackLabel = label;
  }

  const clusterToLabel = new Array(clusterCount).fill(fallbackLabel);
  const countsByCluster = Array.from({ length: clusterCount }, () => new Array(classCount).fill(0));
  for (let i = 0; i < clusterLabels.length; i++) {
    const cluster = clusterLabels[i];
    const label = targetLabels[i] ?? fallbackLabel;
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

export function trainDbscanClassifier(
  features: number[][],
  labels: number[],
  classCount: number,
  options: DbscanTrainOptions,
  onProgress?: (stat: DbscanEpochStat) => void
): DbscanModel {
  if (!features.length) throw new Error('Cannot train DBSCAN on empty dataset.');
  const featureCount = features[0].length;
  const epsilon = Math.max(1e-6, options.epsilon);
  const epsilonSquared = epsilon * epsilon;
  const minSamples = Math.max(2, Math.floor(options.minSamples));
  const clusterLabels = new Array(features.length).fill(UNVISITED);
  const neighborCache = new Map<number, number[]>();
  const corePointSet = new Set<number>();
  const lossCurve: DbscanEpochStat[] = [];

  const getNeighbors = (index: number): number[] => {
    const cached = neighborCache.get(index);
    if (cached) return cached;
    const computed = regionQuery(features, index, epsilonSquared);
    neighborCache.set(index, computed);
    return computed;
  };

  let clusterId = 0;
  for (let pointIndex = 0; pointIndex < features.length; pointIndex++) {
    if (options.shouldStop?.()) break;
    if (clusterLabels[pointIndex] !== UNVISITED) continue;

    const neighbors = getNeighbors(pointIndex);
    if (neighbors.length < minSamples) {
      clusterLabels[pointIndex] = NOISE;
      onProgress?.({
        processedPoints: pointIndex + 1,
        clusterCount: clusterId,
        noisePoints: clusterLabels.filter((label) => label === NOISE).length,
      });
      continue;
    }

    corePointSet.add(pointIndex);
    clusterLabels[pointIndex] = clusterId;
    const queue = neighbors.slice();
    while (queue.length) {
      const neighborIndex = queue.shift()!;
      if (clusterLabels[neighborIndex] === NOISE) {
        clusterLabels[neighborIndex] = clusterId;
      }
      if (clusterLabels[neighborIndex] !== UNVISITED) continue;
      clusterLabels[neighborIndex] = clusterId;

      const neighborNeighbors = getNeighbors(neighborIndex);
      if (neighborNeighbors.length >= minSamples) {
        corePointSet.add(neighborIndex);
        for (const candidate of neighborNeighbors) {
          if (clusterLabels[candidate] === UNVISITED || clusterLabels[candidate] === NOISE) {
            queue.push(candidate);
          }
        }
      }
    }

    clusterId++;
    const noisePoints = clusterLabels.filter((label) => label === NOISE).length;
    const stat: DbscanEpochStat = {
      processedPoints: pointIndex + 1,
      clusterCount: clusterId,
      noisePoints,
    };
    lossCurve.push(stat);
    onProgress?.(stat);
  }

  const noiseCount = clusterLabels.filter((label) => label === NOISE).length;
  const { clusterToLabel, fallbackLabel } = mapClustersToLabels(clusterLabels, labels, clusterId, classCount);

  return {
    featureCount,
    epsilon,
    minSamples,
    clusterCount: clusterId,
    noiseCount,
    trainFeatures: features.map((row) => row.slice()),
    clusterLabels,
    corePointIndices: Array.from(corePointSet),
    clusterToLabel,
    classCount,
    fallbackLabel,
    lossCurve,
  };
}

function predictOneDbscan(model: DbscanModel, sample: number[]): number {
  const epsilonSquared = model.epsilon * model.epsilon;
  let bestIndex = -1;
  let bestDistance = Infinity;

  for (const coreIndex of model.corePointIndices) {
    const distance = squaredDistance(sample, model.trainFeatures[coreIndex]);
    if (distance <= epsilonSquared && distance < bestDistance) {
      bestDistance = distance;
      bestIndex = coreIndex;
    }
  }

  if (bestIndex === -1) return model.fallbackLabel;
  const cluster = model.clusterLabels[bestIndex];
  if (cluster < 0 || cluster >= model.clusterToLabel.length) return model.fallbackLabel;
  return model.clusterToLabel[cluster] ?? model.fallbackLabel;
}

export function predictDbscanClassifier(model: DbscanModel, features: number[][]): number[] {
  return features.map((sample) => predictOneDbscan(model, sample));
}

export async function predictDbscanClassifierAsync(
  model: DbscanModel,
  features: number[][],
  options?: { chunkSize?: number; onProgress?: (processed: number, total: number) => void }
): Promise<number[]> {
  const chunkSize = Math.max(256, Math.floor(options?.chunkSize ?? 2048));
  const total = features.length;
  const out: number[] = new Array(total);
  for (let start = 0; start < total; start += chunkSize) {
    const end = Math.min(total, start + chunkSize);
    for (let i = start; i < end; i++) {
      out[i] = predictOneDbscan(model, features[i]);
    }
    options?.onProgress?.(end, total);
    if (end < total) {
      await new Promise<void>((resolve) => setTimeout(resolve, 0));
    }
  }
  return out;
}

export function dbscanFeatureImportance(model: DbscanModel): number[] {
  const scores = new Array(model.featureCount).fill(0);
  if (model.clusterCount <= 1) return scores;

  const centroids = Array.from({ length: model.clusterCount }, () => new Float64Array(model.featureCount));
  const counts = new Uint32Array(model.clusterCount);

  for (let i = 0; i < model.trainFeatures.length; i++) {
    const cluster = model.clusterLabels[i];
    if (cluster < 0 || cluster >= model.clusterCount) continue;
    counts[cluster]++;
    const sample = model.trainFeatures[i];
    for (let f = 0; f < model.featureCount; f++) {
      centroids[cluster][f] += sample[f] ?? 0;
    }
  }

  for (let cluster = 0; cluster < model.clusterCount; cluster++) {
    if (!counts[cluster]) continue;
    for (let f = 0; f < model.featureCount; f++) {
      centroids[cluster][f] /= counts[cluster];
    }
  }

  for (let f = 0; f < model.featureCount; f++) {
    const values = centroids.map((centroid) => centroid[f]);
    const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
    let variance = 0;
    for (const value of values) {
      const diff = value - mean;
      variance += diff * diff;
    }
    scores[f] = variance / values.length;
  }

  const maxScore = scores.reduce((max, value) => Math.max(max, value), 0);
  if (maxScore > 0) {
    for (let i = 0; i < scores.length; i++) {
      scores[i] /= maxScore;
    }
  }
  return scores;
}

