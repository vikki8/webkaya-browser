import { AlgorithmId, AutoInsights, DataColumnProfile, ParsedDataset, PreprocessingConfig } from '../types/data';

const TREE_ALGORITHMS = new Set<AlgorithmId>(['decision_tree', 'random_forest']);

const ALGORITHM_LABEL: Partial<Record<AlgorithmId, string>> = {
  decision_tree: 'Decision Tree',
  random_forest: 'Random Forest',
  logistic_regression: 'Logistic Regression',
  svm: 'SVM',
  knn: 'k-NN',
  neural_network: 'Neural Network',
  linear_regression: 'Linear Regression',
  kmeans: 'k-means',
  dbscan: 'DBSCAN',
  cnn: 'CNN',
  rnn: 'RNN',
  lstm: 'LSTM',
  gru: 'GRU',
};

function algorithmLabel(id: AlgorithmId): string {
  return ALGORITHM_LABEL[id] ?? id;
}

export function shouldNormalizeForAlgorithm(algorithm: AlgorithmId, disableForPerformance: boolean): boolean {
  if (disableForPerformance) return false;
  return !TREE_ALGORITHMS.has(algorithm);
}

function normalizePlanLine(algorithm: AlgorithmId, normalize: boolean, performanceDisabled: boolean): string {
  const label = algorithmLabel(algorithm);
  if (performanceDisabled) {
    return `Normalization: off — large or image-heavy dataset; skipping scaling for faster preprocessing.`;
  }
  if (normalize) {
    return `Normalization: on — ${label} works better with similarly scaled numeric features.`;
  }
  if (TREE_ALGORITHMS.has(algorithm)) {
    return `Normalization: off — ${label} is tree-based and scale-invariant.`;
  }
  return `Normalization: off — optional for ${label}; enable under Advanced if you want scaled features.`;
}

export function refreshNormalizationPlanLine(
  plan: string[],
  algorithm: AlgorithmId,
  normalize: boolean,
  performanceDisabled: boolean
): string[] {
  const line = normalizePlanLine(algorithm, normalize, performanceDisabled);
  const filtered = plan.filter((entry) => !entry.startsWith('Normalization:'));
  return [...filtered, line];
}

export interface IntelligentPreprocessOptions {
  inferredFormat: ParsedDataset['inferredFormat'];
  rowCount: number;
  disableNormalizationForPerformance: boolean;
  /** Used for scaling rules (recommended model at import time). */
  algorithmHint: AlgorithmId | null;
}

export interface IntelligentPreprocessResult {
  config: PreprocessingConfig;
  plan: string[];
}

export function buildIntelligentPreprocessingPlan(
  insights: AutoInsights,
  profiles: DataColumnProfile[],
  options: IntelligentPreprocessOptions
): IntelligentPreprocessResult {
  const target = insights.detectedTarget;
  const rowCount = Math.max(1, options.rowCount);
  const algorithm = options.algorithmHint ?? insights.recommendedAlgorithm ?? 'random_forest';

  const droppedColumns: string[] = [];
  for (const profile of profiles) {
    if (target && profile.name === target) continue;
    const rate = profile.missingCount / rowCount;
    if (rate > 0.3) droppedColumns.push(profile.name);
  }

  const dropped = new Set(droppedColumns);
  const featureProfiles = profiles.filter((p) => p.name !== target && !dropped.has(p.name));

  let hasMissingInFeatures = false;
  let hasMissingInTarget = false;
  if (target) {
    const tp = profiles.find((p) => p.name === target);
    if (tp && tp.missingCount > 0) hasMissingInTarget = true;
  }
  for (const p of featureProfiles) {
    if (p.missingCount > 0) {
      hasMissingInFeatures = true;
      break;
    }
  }

  const fixMissingValues = hasMissingInFeatures || hasMissingInTarget;

  const encodeCategories = featureProfiles.some(
    (p) => p.type === 'categorical' || p.type === 'boolean' || p.type === 'text'
  );

  const normalizeData = shouldNormalizeForAlgorithm(algorithm, options.disableNormalizationForPerformance);

  const augmentImageData = false;

  const config: PreprocessingConfig = {
    fixMissingValues,
    encodeCategories,
    normalizeData,
    augmentImageData,
    imageAugmentationFactor: 1,
    imageAugmentationNoise: 0.06,
    droppedColumns,
    targetColumn: target,
  };

  const plan: string[] = [];

  plan.push(
    `Dataset: ${rowCount.toLocaleString()} rows · ${profiles.length} columns · ${insights.learningMode} · ${insights.taskType} (${insights.modality}).`
  );

  if (droppedColumns.length) {
    plan.push(
      `Dropped ${droppedColumns.length} column(s) with >30% missing values: ${droppedColumns.slice(0, 8).join(', ')}${
        droppedColumns.length > 8 ? '…' : ''
      }.`
    );
  }

  if (fixMissingValues) {
    const colsWithMissing = profiles.filter((p) => p.missingCount > 0).map((p) => p.name);
    plan.push(
      `Missing values: will impute (${colsWithMissing.length} column(s)) using median/mode heuristics in the preprocessing engine.`
    );
  } else {
    plan.push('Missing values: none detected — imputation disabled.');
  }

  if (encodeCategories) {
    const highCard = featureProfiles.filter(
      (p) => (p.type === 'categorical' || p.type === 'boolean' || p.type === 'text') && p.uniqueCount > 50
    );
    if (highCard.length) {
      plan.push(
        `Categorical / text features: label encoding on (${highCard.length} high-cardinality column(s) use integer codes).`
      );
    } else {
      plan.push('Categorical / text features: label encoding enabled.');
    }
  } else {
    plan.push('Encoding: skipped — no categorical or text feature columns after drops.');
  }

  plan.push(normalizePlanLine(algorithm, normalizeData, options.disableNormalizationForPerformance));

  if (options.inferredFormat === 'images_zip' || insights.modality === 'image') {
    plan.push('Image augmentation: off by default — enable under Advanced if you want rotation/flip/zoom.');
  }

  return { config, plan };
}
