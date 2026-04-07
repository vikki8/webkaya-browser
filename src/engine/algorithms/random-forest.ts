export interface RandomForestTrainOptions {
  trees: number;
  maxDepth: number;
  minSamplesSplit: number;
  featureSampleRate: number;
  bootstrap?: boolean;
}

export interface DecisionTreeNode {
  prediction: number;
  featureIndex: number | null;
  threshold: number;
  left: DecisionTreeNode | null;
  right: DecisionTreeNode | null;
}

export interface RandomForestModel {
  trees: DecisionTreeNode[];
  classes: number[];
  featureImportance: number[];
}

function gini(labels: number[]): number {
  if (!labels.length) return 0;
  const counts = new Map<number, number>();
  for (const label of labels) counts.set(label, (counts.get(label) ?? 0) + 1);
  let impurity = 1;
  for (const count of counts.values()) {
    const p = count / labels.length;
    impurity -= p * p;
  }
  return impurity;
}

function majorityLabel(labels: number[]): number {
  const counts = new Map<number, number>();
  for (const label of labels) counts.set(label, (counts.get(label) ?? 0) + 1);
  let bestLabel = labels[0] ?? 0;
  let bestCount = -1;
  for (const [label, count] of counts.entries()) {
    if (count > bestCount) {
      bestLabel = label;
      bestCount = count;
    }
  }
  return bestLabel;
}

function uniqueSorted(values: number[]): number[] {
  return Array.from(new Set(values)).sort((a, b) => a - b);
}

function candidateThresholds(values: number[]): number[] {
  const unique = uniqueSorted(values);
  if (unique.length <= 1) return [];
  if (unique.length <= 32) {
    const thresholds: number[] = [];
    for (let i = 1; i < unique.length; i++) thresholds.push((unique[i - 1] + unique[i]) / 2);
    return thresholds;
  }
  const sampleCount = 16;
  const thresholds: number[] = [];
  for (let i = 1; i <= sampleCount; i++) {
    const idx = Math.floor((i / (sampleCount + 1)) * (unique.length - 1));
    const nextIdx = Math.min(unique.length - 1, idx + 1);
    thresholds.push((unique[idx] + unique[nextIdx]) / 2);
  }
  return Array.from(new Set(thresholds));
}

function sampleFeatureIndices(totalFeatures: number, featureSampleRate: number): number[] {
  const count = Math.max(1, Math.floor(totalFeatures * featureSampleRate));
  const indices = Array.from({ length: totalFeatures }, (_, i) => i);
  for (let i = indices.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }
  return indices.slice(0, count);
}

function bootstrapIndices(totalRows: number): number[] {
  const out = new Array<number>(totalRows);
  for (let i = 0; i < totalRows; i++) out[i] = Math.floor(Math.random() * totalRows);
  return out;
}

function buildTree(
  features: number[][],
  labels: number[],
  rowIndices: number[],
  depth: number,
  options: RandomForestTrainOptions,
  featureImportance: number[]
): DecisionTreeNode {
  const nodeLabels = rowIndices.map((idx) => labels[idx]);
  const prediction = majorityLabel(nodeLabels);
  const parentImpurity = gini(nodeLabels);

  if (
    depth >= options.maxDepth ||
    rowIndices.length < options.minSamplesSplit ||
    parentImpurity <= 1e-6
  ) {
    return { prediction, featureIndex: null, threshold: 0, left: null, right: null };
  }

  const featureCandidates = sampleFeatureIndices(features[0].length, options.featureSampleRate);

  let bestFeature: number | null = null;
  let bestThreshold = 0;
  let bestGain = 0;
  let bestLeft: number[] = [];
  let bestRight: number[] = [];

  for (const featureIdx of featureCandidates) {
    const values = rowIndices.map((idx) => features[idx][featureIdx]);
    const thresholds = candidateThresholds(values);
    for (const threshold of thresholds) {
      const left: number[] = [];
      const right: number[] = [];
      for (const idx of rowIndices) {
        if (features[idx][featureIdx] <= threshold) left.push(idx);
        else right.push(idx);
      }
      if (!left.length || !right.length) continue;
      const leftGini = gini(left.map((idx) => labels[idx]));
      const rightGini = gini(right.map((idx) => labels[idx]));
      const weighted = (left.length / rowIndices.length) * leftGini + (right.length / rowIndices.length) * rightGini;
      const gain = parentImpurity - weighted;
      if (gain > bestGain) {
        bestGain = gain;
        bestFeature = featureIdx;
        bestThreshold = threshold;
        bestLeft = left;
        bestRight = right;
      }
    }
  }

  if (bestFeature === null || !bestLeft.length || !bestRight.length) {
    return { prediction, featureIndex: null, threshold: 0, left: null, right: null };
  }

  featureImportance[bestFeature] += bestGain;

  return {
    prediction,
    featureIndex: bestFeature,
    threshold: bestThreshold,
    left: buildTree(features, labels, bestLeft, depth + 1, options, featureImportance),
    right: buildTree(features, labels, bestRight, depth + 1, options, featureImportance),
  };
}

export function trainRandomForest(
  features: number[][],
  labels: number[],
  options: RandomForestTrainOptions,
  onProgress?: (progress: number) => void
): RandomForestModel {
  if (!features.length) throw new Error('Cannot train Random Forest on empty dataset.');
  const classes = Array.from(new Set(labels)).sort((a, b) => a - b);
  const featureImportance = new Array(features[0].length).fill(0);
  const trees: DecisionTreeNode[] = [];

  for (let treeIndex = 0; treeIndex < options.trees; treeIndex++) {
    const rows =
      options.bootstrap === false
        ? Array.from({ length: features.length }, (_, idx) => idx)
        : bootstrapIndices(features.length);
    const tree = buildTree(features, labels, rows, 0, options, featureImportance);
    trees.push(tree);
    onProgress?.((treeIndex + 1) / options.trees);
  }

  const totalImportance = featureImportance.reduce((sum, value) => sum + value, 0) || 1;
  const normalized = featureImportance.map((value) => value / totalImportance);

  return { trees, classes, featureImportance: normalized };
}

function predictTree(node: DecisionTreeNode, sample: number[]): number {
  if (node.featureIndex === null || !node.left || !node.right) return node.prediction;
  if (sample[node.featureIndex] <= node.threshold) return predictTree(node.left, sample);
  return predictTree(node.right, sample);
}

export function predictRandomForest(model: RandomForestModel, features: number[][]): number[] {
  return features.map((sample) => {
    const votes = new Map<number, number>();
    for (const tree of model.trees) {
      const prediction = predictTree(tree, sample);
      votes.set(prediction, (votes.get(prediction) ?? 0) + 1);
    }
    return Array.from(votes.entries()).sort((a, b) => b[1] - a[1])[0]?.[0] ?? 0;
  });
}

export function trainRandomForestAlgorithm(
  features: number[][],
  labels: number[],
  options: RandomForestTrainOptions,
  onProgress?: (progress: number) => void
): RandomForestModel {
  return trainRandomForest(features, labels, options, onProgress);
}

export function predictRandomForestAlgorithm(model: RandomForestModel, features: number[][]): number[] {
  return predictRandomForest(model, features);
}

