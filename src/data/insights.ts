import {
  AlgorithmId,
  AlgorithmSuggestion,
  AutoInsights,
  DataColumnProfile,
  DataRow,
  DatasetModality,
  DatasetSizeBand,
  InsightTaskType,
  UploadedDataFormat,
} from '../types/data';

function isMissing(value: unknown): boolean {
  return value === null || value === undefined || value === '' || (typeof value === 'number' && Number.isNaN(value));
}

function toNumber(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (typeof value !== 'string') return null;
  const parsed = Number(value.trim());
  return Number.isFinite(parsed) ? parsed : null;
}

function inferColumnType(values: unknown[]): DataColumnProfile['type'] {
  const nonMissing = values.filter((v) => !isMissing(v));
  if (!nonMissing.length) return 'text';

  const numericCount = nonMissing.filter((v) => toNumber(v) !== null).length;
  const booleanCount = nonMissing.filter((v) => typeof v === 'boolean' || v === 'true' || v === 'false').length;
  const unique = new Set(nonMissing.map((v) => String(v)));

  if (booleanCount === nonMissing.length) return 'boolean';
  if (numericCount === nonMissing.length) return 'numeric';
  if (unique.size <= Math.max(20, Math.floor(nonMissing.length * 0.2))) return 'categorical';
  return 'text';
}

export function buildColumnProfiles(rows: DataRow[], columns: string[]): DataColumnProfile[] {
  return columns.map((column) => {
    const values = rows.map((row) => row[column]);
    const missingCount = values.filter(isMissing).length;
    const uniqueValues = new Set(values.filter((v) => !isMissing(v)).map((v) => String(v)));
    const sampleValues = Array.from(uniqueValues).slice(0, 5);
    return {
      name: column,
      type: inferColumnType(values),
      missingCount,
      uniqueCount: uniqueValues.size,
      sampleValues,
    };
  });
}

const ALGORITHM_CATALOG: Record<
  AlgorithmId,
  Omit<AlgorithmSuggestion, 'reason'> & {
    defaultReason: string;
  }
> = {
  decision_tree: {
    id: 'decision_tree',
    label: 'Decision Tree',
    family: 'supervised',
    useCaseHint: 'Interpretable baseline for structured features.',
    speed: 'fast',
    expectedQuality: 'baseline',
    runtimeSupport: 'native',
    defaultReason: 'Handles mixed tabular patterns with low tuning overhead.',
  },
  random_forest: {
    id: 'random_forest',
    label: 'Random Forest',
    family: 'supervised',
    useCaseHint: 'Strong tabular baseline with robust performance.',
    speed: 'balanced',
    expectedQuality: 'strong',
    runtimeSupport: 'native',
    defaultReason: 'Good accuracy/speed trade-off for structured datasets.',
  },
  logistic_regression: {
    id: 'logistic_regression',
    label: 'Logistic Regression',
    family: 'supervised',
    useCaseHint: 'Probabilistic linear classifier for tabular classification.',
    speed: 'fast',
    expectedQuality: 'baseline',
    runtimeSupport: 'native',
    defaultReason: 'Strong and interpretable baseline for categorical target tasks.',
  },
  svm: {
    id: 'svm',
    label: 'SVM',
    family: 'supervised',
    useCaseHint: 'Works well for smaller high-signal tabular datasets.',
    speed: 'balanced',
    expectedQuality: 'strong',
    runtimeSupport: 'native',
    defaultReason: 'Useful when data is small and class boundaries are crisp.',
  },
  knn: {
    id: 'knn',
    label: 'KNN',
    family: 'supervised',
    useCaseHint: 'Simple local-neighborhood classifier/regressor for small datasets.',
    speed: 'fast',
    expectedQuality: 'baseline',
    runtimeSupport: 'native',
    defaultReason: 'Good quick baseline for compact datasets.',
  },
  neural_network: {
    id: 'neural_network',
    label: 'Neural Network',
    family: 'supervised',
    useCaseHint: 'Flexible model for larger datasets and nonlinear interactions.',
    speed: 'slow',
    expectedQuality: 'high',
    runtimeSupport: 'native',
    defaultReason: 'Scales to larger datasets and complex feature interactions.',
  },
  linear_regression: {
    id: 'linear_regression',
    label: 'Linear Regression',
    family: 'supervised',
    useCaseHint: 'Fast numeric baseline for regression.',
    speed: 'fast',
    expectedQuality: 'baseline',
    runtimeSupport: 'native',
    defaultReason: 'Strong baseline when numeric target follows smoother trends.',
  },
  kmeans: {
    id: 'kmeans',
    label: 'K-Means',
    family: 'unsupervised',
    useCaseHint: 'Cluster tabular data into segments.',
    speed: 'fast',
    expectedQuality: 'baseline',
    runtimeSupport: 'native',
    defaultReason: 'Common first-pass clustering method for unlabeled tabular data.',
  },
  dbscan: {
    id: 'dbscan',
    label: 'DBSCAN',
    family: 'unsupervised',
    useCaseHint: 'Density-based clustering with noise handling.',
    speed: 'balanced',
    expectedQuality: 'strong',
    runtimeSupport: 'native',
    defaultReason: 'Useful when clusters are irregular and noise points exist.',
  },
  cnn: {
    id: 'cnn',
    label: 'CNN',
    family: 'supervised',
    useCaseHint: 'Image-specialized neural network.',
    speed: 'slow',
    expectedQuality: 'high',
    runtimeSupport: 'mapped',
    defaultReason: 'Best default for image-like inputs.',
  },
  rnn: {
    id: 'rnn',
    label: 'RNN',
    family: 'supervised',
    useCaseHint: 'Sequence-aware neural architecture.',
    speed: 'slow',
    expectedQuality: 'strong',
    runtimeSupport: 'native',
    defaultReason: 'Designed for ordered sequence dependencies.',
  },
  lstm: {
    id: 'lstm',
    label: 'LSTM',
    family: 'supervised',
    useCaseHint: 'Long-range sequence modeling.',
    speed: 'slow',
    expectedQuality: 'high',
    runtimeSupport: 'native',
    defaultReason: 'Stronger long-context sequence modeling than plain RNN.',
  },
  gru: {
    id: 'gru',
    label: 'GRU',
    family: 'supervised',
    useCaseHint: 'Efficient gated recurrent model for sequence data.',
    speed: 'balanced',
    expectedQuality: 'strong',
    runtimeSupport: 'native',
    defaultReason: 'Captures sequence context with fewer parameters than LSTM.',
  },
};

function makeSuggestion(id: AlgorithmId, reason?: string): AlgorithmSuggestion {
  const base = ALGORITHM_CATALOG[id];
  return {
    ...base,
    reason: reason ?? base.defaultReason,
  };
}

export function getAlgorithmCatalogSuggestions(): AlgorithmSuggestion[] {
  return (Object.keys(ALGORITHM_CATALOG) as AlgorithmId[]).map((id) => makeSuggestion(id));
}

function inferDatasetSize(rowCount: number): DatasetSizeBand {
  if (rowCount <= 5_000) return 'small';
  if (rowCount <= 50_000) return 'medium';
  return 'large';
}

function detectModality(
  rows: DataRow[],
  columns: string[],
  profiles: DataColumnProfile[],
  inferredFormat?: UploadedDataFormat | 'kaggle'
): DatasetModality {
  if (inferredFormat === 'images_zip') return 'image';

  const lowerColumns = columns.map((c) => c.toLowerCase());
  const imageSignals = ['image', 'pixel', 'mean_r', 'mean_g', 'mean_b', 'std_r', 'std_g', 'std_b'];
  if (lowerColumns.some((column) => imageSignals.some((signal) => column.includes(signal)))) {
    return 'image';
  }

  const numericProfiles = profiles.filter((profile) => profile.type === 'numeric');
  const numericRatio = numericProfiles.length / Math.max(1, profiles.length);
  const digitLikeColumns = lowerColumns.filter((column) =>
    /^pixel_?\d+$/i.test(column) || /^x_?\d+$/i.test(column) || /^\d+$/.test(column)
  ).length;
  const digitLikeRatio = digitLikeColumns / Math.max(1, columns.length);
  const hasLabelSignal = lowerColumns.some((column) =>
    column === 'label' || column === 'class' || column === 'target' || column.includes('label')
  );
  const highDimensional = columns.length >= 196;

  if (
    highDimensional &&
    numericRatio >= 0.9 &&
    (digitLikeRatio >= 0.35 || (hasLabelSignal && columns.length >= 256))
  ) {
    return 'image';
  }

  const timeSignals = ['time', 'date', 'timestamp', 'ts'];
  const timeColumn = columns.find((column) => timeSignals.some((signal) => column.toLowerCase().includes(signal)));
  if (timeColumn) {
    const sample = rows.slice(0, 300).map((row) => row[timeColumn]).filter((value) => !isMissing(value));
    const parseable = sample.filter((value) => !Number.isNaN(Date.parse(String(value)))).length;
    if (sample.length > 0 && parseable / sample.length >= 0.6) return 'time_series';
  }

  const textColumns = profiles.filter((profile) => profile.type === 'text').map((profile) => profile.name);
  if (textColumns.length) {
    const sequenceSignals = ['text', 'sequence', 'sentence', 'message', 'content', 'prompt', 'token'];
    if (textColumns.some((column) => sequenceSignals.some((signal) => column.toLowerCase().includes(signal)))) {
      return 'sequence';
    }
    const sampleValues = rows.slice(0, 250).flatMap((row) => textColumns.map((column) => String(row[column] ?? '')));
    const avgWords =
      sampleValues.reduce((sum, value) => sum + value.trim().split(/\s+/).filter(Boolean).length, 0) /
      Math.max(1, sampleValues.length);
    if (avgWords >= 4) return 'sequence';
  }

  return 'tabular';
}

function detectTargetColumn(
  columns: string[],
  profiles: DataColumnProfile[],
  rowCount: number,
  selectedTarget?: string | null
): string | null {
  if (selectedTarget && columns.includes(selectedTarget)) return selectedTarget;

  const preferred = ['target', 'label', 'class', 'y', 'price', 'outcome', 'prediction'];
  for (const p of preferred) {
    const exact = columns.find((col) => col.toLowerCase() === p);
    if (exact) return exact;
  }
  for (const p of preferred) {
    const partial = columns.find((col) => col.toLowerCase().includes(p));
    if (partial) return partial;
  }

  const heuristicCandidates = profiles.filter((profile) => {
    if (profile.type !== 'categorical' && profile.type !== 'boolean') return false;
    if (profile.uniqueCount <= 1) return false;
    return profile.uniqueCount <= Math.max(12, Math.floor(rowCount * 0.15));
  });
  if (heuristicCandidates.length === 1) return heuristicCandidates[0].name;

  return null;
}

function inferProblemTypeFromTarget(
  rows: DataRow[],
  target: string | null,
  profiles: DataColumnProfile[]
): AutoInsights['problemType'] {
  if (!target) return 'unknown';
  const profile = profiles.find((item) => item.name === target);
  if (profile && (profile.type === 'categorical' || profile.type === 'boolean' || profile.type === 'text')) {
    return 'classification';
  }
  const values = rows.map((row) => row[target]).filter((v) => !isMissing(v));
  if (!values.length) return 'unknown';

  const numericValues = values.map(toNumber);
  const numericCount = numericValues.filter((v): v is number => v !== null).length;
  const uniqueCount = new Set(values.map((v) => String(v))).size;

  if (numericCount === values.length && uniqueCount > Math.max(20, Math.floor(values.length * 0.12))) {
    return 'regression';
  }
  return 'classification';
}

function inferTaskType(learningMode: AutoInsights['learningMode'], problemType: AutoInsights['problemType'], modality: DatasetModality): InsightTaskType {
  if (learningMode === 'supervised') {
    if (problemType === 'classification' || problemType === 'regression') return problemType;
    return 'unknown';
  }
  if (modality === 'image' || modality === 'sequence' || modality === 'time_series') {
    return 'representation_generation';
  }
  return 'clustering';
}

function buildSuggestedAlgorithms(
  learningMode: AutoInsights['learningMode'],
  taskType: InsightTaskType,
  modality: DatasetModality,
  datasetSize: DatasetSizeBand
): AlgorithmId[] {
  if (learningMode === 'supervised') {
    if (taskType === 'classification') {
      if (modality === 'image') return ['cnn', 'neural_network', 'random_forest', 'decision_tree', 'logistic_regression'];
      if (modality === 'time_series' || modality === 'sequence') {
        return ['lstm', 'gru', 'rnn', 'neural_network', 'random_forest', 'logistic_regression'];
      }
      if (datasetSize === 'small') return ['logistic_regression', 'svm', 'knn', 'random_forest', 'decision_tree', 'neural_network'];
      if (datasetSize === 'medium') return ['random_forest', 'decision_tree', 'logistic_regression', 'svm', 'knn', 'neural_network'];
      return ['neural_network', 'random_forest', 'logistic_regression', 'svm', 'knn', 'decision_tree'];
    }
    if (taskType === 'regression') {
      if (modality === 'time_series' || modality === 'sequence') return ['lstm', 'gru', 'rnn', 'neural_network', 'linear_regression'];
      if (datasetSize === 'small') return ['linear_regression', 'random_forest', 'decision_tree', 'neural_network'];
      if (datasetSize === 'medium') return ['random_forest', 'decision_tree', 'linear_regression', 'neural_network'];
      return ['neural_network', 'random_forest', 'linear_regression', 'decision_tree'];
    }
  }

  if (taskType === 'representation_generation') {
    return modality === 'image' ? ['kmeans', 'dbscan'] : ['kmeans', 'dbscan'];
  }
  return ['kmeans', 'dbscan'];
}

function chooseRecommendedAlgorithm(
  learningMode: AutoInsights['learningMode'],
  taskType: InsightTaskType,
  modality: DatasetModality,
  datasetSize: DatasetSizeBand
): AlgorithmId {
  if (learningMode === 'supervised') {
    if (modality === 'image') return 'cnn';
    if (modality === 'time_series' || modality === 'sequence') {
      if (datasetSize === 'large') return 'lstm';
      if (datasetSize === 'medium') return 'gru';
      return 'rnn';
    }
    if (taskType === 'regression') {
      if (datasetSize === 'small') return 'linear_regression';
      if (datasetSize === 'medium') return 'random_forest';
      return 'neural_network';
    }
    if (datasetSize === 'small') return 'logistic_regression';
    if (datasetSize === 'medium') return 'random_forest';
    return 'neural_network';
  }
  if (taskType === 'representation_generation') return 'kmeans';
  return datasetSize === 'small' ? 'kmeans' : 'dbscan';
}

function recommendationReason(
  recommended: AlgorithmId,
  learningMode: AutoInsights['learningMode'],
  taskType: InsightTaskType,
  modality: DatasetModality,
  rows: number
): string {
  const label = ALGORITHM_CATALOG[recommended]?.label ?? recommended;
  return `${learningMode} ${taskType} dataset (${modality}, ${rows.toLocaleString()} rows) -> ${label} is recommended.`;
}

export function analyzeDataset(
  rows: DataRow[],
  columns: string[],
  options?: { inferredFormat?: UploadedDataFormat | 'kaggle'; selectedTarget?: string | null }
): { insights: AutoInsights; profiles: DataColumnProfile[] } {
  const profiles = buildColumnProfiles(rows, columns);
  const rowCount = rows.length;
  const target = detectTargetColumn(columns, profiles, rowCount, options?.selectedTarget);
  const targetColumnExists = Boolean(target);
  const learningMode: AutoInsights['learningMode'] = targetColumnExists ? 'supervised' : 'unsupervised';
  const problemType = inferProblemTypeFromTarget(rows, target, profiles);
  const modality = detectModality(rows, columns, profiles, options?.inferredFormat);
  const datasetSize = inferDatasetSize(rowCount);
  const taskType = inferTaskType(learningMode, problemType, modality);
  const recommendation = chooseRecommendedAlgorithm(learningMode, taskType, modality, datasetSize);
  const recommendationText = recommendationReason(recommendation, learningMode, taskType, modality, rowCount);
  const suggestedAlgorithmIds = buildSuggestedAlgorithms(learningMode, taskType, modality, datasetSize);

  const missingColumns = profiles.filter((p) => p.missingCount > 0).map((p) => p.name);
  const categoricalColumns = profiles.filter((p) => p.type === 'categorical' || p.type === 'boolean').map((p) => p.name);
  const numericColumns = profiles.filter((p) => p.type === 'numeric').map((p) => p.name);

  const warnings: string[] = [];
  if (!rows.length) warnings.push('Dataset appears empty.');
  if (!target) warnings.push('No clear target column detected. Treating this as unsupervised for recommendations.');
  if (problemType === 'regression') {
    warnings.push('Regression target detected. Browser trainer currently evaluates bucketed labels for consistent in-app metrics.');
  }
  if (taskType === 'clustering' || taskType === 'representation_generation') {
    warnings.push('Unsupervised clustering models can use lower metric confidence on overlapping feature distributions.');
  }
  if (rows.length < 50) warnings.push('Very small dataset detected. Model quality may be unstable.');

  const suggestedAlgorithms: AlgorithmSuggestion[] = suggestedAlgorithmIds.map((id) =>
    makeSuggestion(id, id === recommendation ? recommendationText : undefined)
  );

  return {
    insights: {
      problemType,
      detectedTarget: target,
      learningMode,
      taskType,
      modality,
      datasetSize,
      rowCount,
      featureCount: columns.length,
      targetColumnExists,
      recommendedAlgorithm: recommendation,
      recommendationReason: recommendationText,
      suggestedAlgorithms,
      missingColumns,
      categoricalColumns,
      numericColumns,
      warnings,
    },
    profiles,
  };
}
