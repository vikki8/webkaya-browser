export type KnnDistanceMetric = 'euclidean' | 'manhattan' | 'cosine';

export interface KnnTrainOptions {
  neighbors: number;
  distanceMetric: KnnDistanceMetric;
}

export interface KnnModel {
  k: number;
  distanceMetric: KnnDistanceMetric;
  classCount: number;
  featureCount: number;
  trainFeatures: number[][];
  trainLabels: number[];
}

function euclideanDistance(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const delta = (a[i] ?? 0) - (b[i] ?? 0);
    sum += delta * delta;
  }
  return Math.sqrt(sum);
}

function manhattanDistance(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += Math.abs((a[i] ?? 0) - (b[i] ?? 0));
  }
  return sum;
}

function cosineDistance(a: number[], b: number[]): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    const av = a[i] ?? 0;
    const bv = b[i] ?? 0;
    dot += av * bv;
    normA += av * av;
    normB += bv * bv;
  }
  if (normA === 0 || normB === 0) return 1;
  return 1 - dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

function distance(metric: KnnDistanceMetric, a: number[], b: number[]): number {
  if (metric === 'manhattan') return manhattanDistance(a, b);
  if (metric === 'cosine') return cosineDistance(a, b);
  return euclideanDistance(a, b);
}

export function trainKnnClassifier(
  features: number[][],
  labels: number[],
  classCount: number,
  options: KnnTrainOptions
): KnnModel {
  if (!features.length) throw new Error('Cannot train KNN on empty dataset.');
  if (features.length !== labels.length) throw new Error('KNN features and labels length mismatch.');
  const featureCount = features[0]?.length ?? 0;
  if (!featureCount) throw new Error('KNN requires at least one feature.');

  return {
    k: Math.max(1, Math.floor(options.neighbors || 5)),
    distanceMetric: options.distanceMetric,
    classCount: Math.max(2, Math.floor(classCount || 2)),
    featureCount,
    trainFeatures: features,
    trainLabels: labels,
  };
}

export function predictKnnClassifier(model: KnnModel, samples: number[][]): number[] {
  const predictions: number[] = [];
  for (const sample of samples) {
    const ranking = model.trainFeatures
      .map((trainRow, idx) => ({
        idx,
        distance: distance(model.distanceMetric, sample, trainRow),
      }))
      .sort((a, b) => a.distance - b.distance);

    const votes = new Array(model.classCount).fill(0);
    const neighbors = Math.min(model.k, ranking.length);
    for (let i = 0; i < neighbors; i++) {
      const label = model.trainLabels[ranking[i].idx] ?? 0;
      const weight = ranking[i].distance === 0 ? 1 : 1 / ranking[i].distance;
      votes[label] += weight;
    }

    let bestLabel = 0;
    let bestVote = -Infinity;
    for (let c = 0; c < votes.length; c++) {
      if (votes[c] > bestVote) {
        bestVote = votes[c];
        bestLabel = c;
      }
    }
    predictions.push(bestLabel);
  }
  return predictions;
}

export function knnFeatureImportance(model: KnnModel): number[] {
  const importance = new Array(model.featureCount).fill(0);
  if (!model.trainFeatures.length) return importance;

  for (let f = 0; f < model.featureCount; f++) {
    let mean = 0;
    for (const row of model.trainFeatures) mean += row[f] ?? 0;
    mean /= model.trainFeatures.length;
    let variance = 0;
    for (const row of model.trainFeatures) {
      const delta = (row[f] ?? 0) - mean;
      variance += delta * delta;
    }
    importance[f] = variance / model.trainFeatures.length;
  }
  return importance;
}
