import { analyzeDataset, buildColumnProfiles } from './insights';
import { AutoInsights, DataRow, PreprocessingConfig, PreprocessingStats, ProcessedDataset } from '../types/data';

type PreprocessStage = 'analyzing' | 'fix_missing' | 'encode' | 'normalize' | 'augment' | 'build_features' | 'complete';

/**
 * Above this row×column count, serializing `rows` as JSON for `/api/dataset/preprocess` can throw
 * `Invalid string length` in the browser. Use {@link preprocessDataset} in-process instead.
 */
export const MAX_CLIENT_JSON_PREPROCESS_CELLS = 2_000_000;

export interface PreprocessDatasetOptions {
  previewOnly?: boolean;
  allowInPlace?: boolean;
  onProgress?: (progress: { stage: PreprocessStage; message: string; percent: number }) => void;
}

function isMissing(value: unknown): boolean {
  return value === null || value === undefined || value === '' || (typeof value === 'number' && Number.isNaN(value));
}

function toFiniteNumber(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (typeof value !== 'string') return null;
  const parsed = Number(value.trim());
  return Number.isFinite(parsed) ? parsed : null;
}

function deepCopyRows(rows: DataRow[]): DataRow[] {
  return rows.map((row) => ({ ...row }));
}

function emitProgress(
  options: PreprocessDatasetOptions | undefined,
  stage: PreprocessStage,
  message: string,
  percent: number
) {
  options?.onProgress?.({
    stage,
    message,
    percent: Math.max(0, Math.min(100, Math.floor(percent))),
  });
}

function countMissing(rows: DataRow[]): number {
  let total = 0;
  for (const row of rows) {
    for (const value of Object.values(row)) if (isMissing(value)) total++;
  }
  return total;
}

function fillMissingValues(rows: DataRow[], columns: string[], options?: PreprocessDatasetOptions): void {
  const totalColumns = Math.max(1, columns.length);
  const progressEvery = Math.max(1, Math.floor(columns.length / 20));

  for (let columnIndex = 0; columnIndex < columns.length; columnIndex++) {
    const column = columns[columnIndex];
    let missingCount = 0;
    let nonMissingCount = 0;
    const numericValues: number[] = [];
    const counts = new Map<string, number>();

    for (let i = 0; i < rows.length; i++) {
      const value = rows[i][column];
      if (isMissing(value)) {
        missingCount++;
        continue;
      }
      nonMissingCount++;
      const asNumber = toFiniteNumber(value);
      if (asNumber !== null) {
        numericValues.push(asNumber);
      }
      const key = String(value);
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }

    if (missingCount > 0) {
      if (numericValues.length >= Math.floor(nonMissingCount * 0.7) && numericValues.length > 0) {
        numericValues.sort((a, b) => a - b);
        const mid = Math.floor(numericValues.length / 2);
        const replacement =
          numericValues.length % 2 ? numericValues[mid] : (numericValues[mid - 1] + numericValues[mid]) / 2;
        for (const row of rows) if (isMissing(row[column])) row[column] = replacement;
      } else {
        const mode = Array.from(counts.entries()).sort((a, b) => b[1] - a[1])[0]?.[0] ?? 'unknown';
        for (const row of rows) if (isMissing(row[column])) row[column] = mode;
      }
    }

    if ((columnIndex + 1) % progressEvery === 0 || columnIndex === columns.length - 1) {
      emitProgress(
        options,
        'fix_missing',
        `Fixing missing values (${columnIndex + 1}/${columns.length} columns)`,
        ((columnIndex + 1) / totalColumns) * 100
      );
    }
  }
}

function encodeCategoricalColumns(rows: DataRow[], columns: string[], options?: PreprocessDatasetOptions): string[] {
  const encoded: string[] = [];
  const profiles = buildColumnProfiles(rows, columns);
  const categoricalProfiles = profiles.filter(
    (profile) => profile.type === 'categorical' || profile.type === 'boolean' || profile.type === 'text'
  );
  const total = Math.max(1, categoricalProfiles.length);
  const progressEvery = Math.max(1, Math.floor(categoricalProfiles.length / 20));

  for (let profileIndex = 0; profileIndex < categoricalProfiles.length; profileIndex++) {
    const profile = categoricalProfiles[profileIndex];
    const map = new Map<string, number>();
    let idx = 0;
    for (const row of rows) {
      const key = String(row[profile.name] ?? '__missing__');
      if (!map.has(key)) map.set(key, idx++);
      row[profile.name] = map.get(key) ?? 0;
    }
    encoded.push(profile.name);

    if ((profileIndex + 1) % progressEvery === 0 || profileIndex === categoricalProfiles.length - 1) {
      emitProgress(
        options,
        'encode',
        `Encoding categorical columns (${profileIndex + 1}/${categoricalProfiles.length})`,
        ((profileIndex + 1) / total) * 100
      );
    }
  }
  return encoded;
}

function normalizeNumericColumns(rows: DataRow[], columns: string[], options?: PreprocessDatasetOptions): string[] {
  const normalized: string[] = [];
  const profiles = buildColumnProfiles(rows, columns);
  const numericProfiles = profiles.filter((profile) => profile.type === 'numeric');
  const total = Math.max(1, numericProfiles.length);
  const progressEvery = Math.max(1, Math.floor(numericProfiles.length / 20));

  for (let profileIndex = 0; profileIndex < numericProfiles.length; profileIndex++) {
    const profile = numericProfiles[profileIndex];

    let min = Infinity;
    let max = -Infinity;
    let validCount = 0;
    for (const row of rows) {
      const value = toFiniteNumber(row[profile.name]);
      if (value === null) continue;
      validCount++;
      if (value < min) min = value;
      if (value > max) max = value;
    }
    if (!validCount) continue;
    const range = max - min;
    if (range === 0) continue;

    for (const row of rows) {
      const value = toFiniteNumber(row[profile.name]);
      if (value === null) continue;
      row[profile.name] = (value - min) / range;
    }
    normalized.push(profile.name);

    if ((profileIndex + 1) % progressEvery === 0 || profileIndex === numericProfiles.length - 1) {
      emitProgress(
        options,
        'normalize',
        `Normalizing numeric columns (${profileIndex + 1}/${numericProfiles.length})`,
        ((profileIndex + 1) / total) * 100
      );
    }
  }
  return normalized;
}

function shouldSkipCategoricalEncodingForHighDimNumeric(rows: DataRow[], columns: string[]): boolean {
  if (columns.length < 128) return false;
  const sampleRows = rows.slice(0, Math.min(rows.length, 200));
  if (!sampleRows.length) return false;

  let mostlyNumericColumns = 0;
  let inspectedColumns = 0;

  for (const column of columns) {
    let nonMissing = 0;
    let numeric = 0;
    for (const row of sampleRows) {
      const value = row[column];
      if (isMissing(value)) continue;
      nonMissing++;
      if (toFiniteNumber(value) !== null) numeric++;
    }
    if (!nonMissing) continue;
    inspectedColumns++;
    if (numeric / nonMissing >= 0.98) mostlyNumericColumns++;
  }

  if (!inspectedColumns) return false;
  return mostlyNumericColumns / inspectedColumns >= 0.95;
}

function isLikelyImageDataset(columns: string[]): boolean {
  const lower = columns.map((column) => column.toLowerCase());
  const hasImageStats =
    lower.includes('mean_r') &&
    lower.includes('mean_g') &&
    lower.includes('mean_b') &&
    (lower.includes('std_r') || lower.includes('std_g') || lower.includes('std_b'));
  if (hasImageStats) return true;
  if (lower.includes('image_name') || lower.includes('image_preview')) return true;
  const pixelLikeColumns = lower.filter(
    (column) => /^pixel_?\d+$/i.test(column) || /^\d+$/.test(column) || /^x_?\d+$/i.test(column)
  );
  return columns.length >= 128 && pixelLikeColumns.length >= Math.floor(columns.length * 0.3);
}

function clampAugmentedValue(original: number, value: number): number {
  if (Number.isNaN(value) || !Number.isFinite(value)) return original;
  if (original >= 0 && original <= 1.2) {
    return Math.min(1, Math.max(0, value));
  }
  if (original >= 0 && original <= 255) {
    return Math.min(255, Math.max(0, value));
  }
  return value;
}

function shouldAugmentColumn(columnName: string): boolean {
  const lower = columnName.toLowerCase();
  return (
    lower.startsWith('mean_') ||
    lower.startsWith('std_') ||
    /^pixel_?\d+$/i.test(lower) ||
    /^x_?\d+$/i.test(lower) ||
    /^\d+$/.test(lower)
  );
}

function augmentImageRows(
  rows: DataRow[],
  featureColumns: string[],
  config: PreprocessingConfig,
  options?: PreprocessDatasetOptions
): number {
  if (!rows.length) return 0;
  if (!isLikelyImageDataset(featureColumns)) return 0;
  if (!config.augmentImageData) return 0;
  const factor = Math.max(1, Math.min(3, config.imageAugmentationFactor || 1));
  if (factor <= 1) return 0;

  const numericAugColumns = featureColumns.filter((column) => shouldAugmentColumn(column));
  if (!numericAugColumns.length) return 0;

  const baseRows = rows.slice();
  const noiseScale = Math.max(0, Math.min(0.5, config.imageAugmentationNoise || 0.06));
  const totalCopies = baseRows.length * (factor - 1);
  let created = 0;
  const progressEvery = Math.max(1, Math.floor(totalCopies / 20));

  for (const baseRow of baseRows) {
    for (let copy = 1; copy < factor; copy++) {
      const augmented: DataRow = { ...baseRow };
      for (const column of numericAugColumns) {
        const value = toFiniteNumber(baseRow[column]);
        if (value === null) continue;
        const noise = (Math.random() * 2 - 1) * noiseScale * (1 + copy * 0.15);
        const adjusted = value + noise;
        augmented[column] = clampAugmentedValue(value, adjusted);
      }
      if (typeof baseRow.image_name === 'string') {
        augmented.image_name = `${baseRow.image_name}__aug_${copy}`;
      }
      rows.push(augmented);
      created++;

      if (created % progressEvery === 0 || created === totalCopies) {
        emitProgress(
          options,
          'augment',
          `Applying image augmentation (${created}/${totalCopies} synthetic rows)`,
          72 + (created / Math.max(1, totalCopies)) * 8
        );
      }
    }
  }
  return created;
}

function buildLabels(
  rows: DataRow[],
  targetColumn: string
): {
  labels: number[];
  labelNames: string[];
  targetProblem: 'classification' | 'regression';
  regressionTargets?: number[];
} {
  const targetValues = rows.map((row) => row[targetColumn]);
  const numeric = targetValues.map(toFiniteNumber);
  const allNumeric = numeric.every((v) => v !== null);
  const uniqueCount = new Set(targetValues.map((v) => String(v))).size;

  if (allNumeric && uniqueCount > 50) {
    const numValues = numeric as number[];
    const min = Math.min(...numValues);
    const max = Math.max(...numValues);
    const bins = 10;
    const labels = numValues.map((value) => {
      if (max === min) return 0;
      return Math.min(bins - 1, Math.floor(((value - min) / (max - min)) * bins));
    });
    const labelNames = Array.from({ length: bins }, (_, i) => `bin_${i}`);
    return { labels, labelNames, targetProblem: 'regression', regressionTargets: numValues };
  }

  const labelMap = new Map<string, number>();
  const labelNames: string[] = [];
  const labels = targetValues.map((raw) => {
    const key = String(raw ?? '__missing__');
    if (!labelMap.has(key)) {
      labelMap.set(key, labelMap.size);
      labelNames.push(key);
    }
    return labelMap.get(key) ?? 0;
  });

  return {
    labels,
    labelNames,
    targetProblem: allNumeric && uniqueCount > 20 ? 'regression' : 'classification',
    regressionTargets: allNumeric ? (numeric as number[]) : undefined,
  };
}

function buildFeatureMatrix(
  rows: DataRow[],
  featureNames: string[],
  options?: PreprocessDatasetOptions
): number[][] {
  const features: number[][] = [];
  const totalRows = Math.max(1, rows.length);
  const progressEvery = Math.max(1, Math.floor(rows.length / 25));
  for (let rowIndex = 0; rowIndex < rows.length; rowIndex++) {
    const row = rows[rowIndex];
    const featureRow = new Array<number>(featureNames.length);
    for (let featureIndex = 0; featureIndex < featureNames.length; featureIndex++) {
      const value = row[featureNames[featureIndex]];
      const asNumber = toFiniteNumber(value);
      featureRow[featureIndex] = asNumber !== null ? asNumber : Number(String(value).length) || 0;
    }
    features.push(featureRow);

    if ((rowIndex + 1) % progressEvery === 0 || rowIndex === rows.length - 1) {
      emitProgress(
        options,
        'build_features',
        `Building feature matrix (${rowIndex + 1}/${rows.length} rows)`,
        80 + ((rowIndex + 1) / totalRows) * 18
      );
    }
  }
  return features;
}

export function preprocessDataset(
  inputRows: DataRow[],
  columns: string[],
  config: PreprocessingConfig,
  options?: PreprocessDatasetOptions
): {
  insights: AutoInsights;
  previewRows: DataRow[];
  processed: ProcessedDataset;
} {
  emitProgress(options, 'analyzing', 'Preparing preprocessing pipeline...', 2);
  const rows = options?.allowInPlace ? inputRows : deepCopyRows(inputRows);
  const beforeMissing = countMissing(rows);
  const targetColumn = config.targetColumn;

  if (!targetColumn) {
    throw new Error('Please select a target column before continuing.');
  }
  if (!columns.includes(targetColumn)) {
    throw new Error(`Target column "${targetColumn}" is not present in the dataset.`);
  }

  const workingColumns = columns.filter(
    (column) => !config.droppedColumns.includes(column) && column !== targetColumn
  );
  if (!workingColumns.length) {
    throw new Error('All feature columns were dropped. Keep at least one feature column.');
  }

  if (config.fixMissingValues) {
    emitProgress(options, 'fix_missing', 'Fixing missing values...', 8);
    fillMissingValues(rows, [...workingColumns, targetColumn], options);
  }

  const skipEncoding = config.encodeCategories && shouldSkipCategoricalEncodingForHighDimNumeric(rows, workingColumns);
  if (skipEncoding) {
    emitProgress(options, 'encode', 'Skipping categorical encoding for high-dimensional numeric features.', 56);
  }
  const encodedColumns = config.encodeCategories && !skipEncoding ? encodeCategoricalColumns(rows, workingColumns, options) : [];
  const normalizedColumns = config.normalizeData ? normalizeNumericColumns(rows, workingColumns, options) : [];
  const augmentedRows = options?.previewOnly ? 0 : augmentImageRows(rows, workingColumns, config, options);
  const featureNames = [...workingColumns];

  const stats: PreprocessingStats = {
    beforeMissing,
    afterMissing: countMissing(rows),
    encodedColumns,
    normalizedColumns,
    droppedColumns: [...config.droppedColumns],
  };

  if (augmentedRows > 0) {
    emitProgress(
      options,
      'augment',
      `Image augmentation added ${augmentedRows.toLocaleString()} synthetic samples.`,
      80
    );
  }

  const previewRows = rows.slice(0, 50);
  const { insights } = analyzeDataset(rows, [...featureNames, targetColumn], {
    selectedTarget: targetColumn,
  });

  if (options?.previewOnly) {
    emitProgress(options, 'complete', 'Preview ready.', 100);
    const lightweightProcessed: ProcessedDataset = {
      featureNames,
      features: [],
      labels: [],
      regressionTargets: [],
      labelNames: [],
      targetColumn,
      problemType: insights.problemType === 'regression' ? 'regression' : 'classification',
      sampleRows: previewRows,
      preprocessing: config,
      stats,
    };
    return { insights, previewRows, processed: lightweightProcessed };
  }

  emitProgress(options, 'build_features', 'Building training tensors...', 80);
  const { labels, labelNames, targetProblem, regressionTargets } = buildLabels(rows, targetColumn);
  const features = buildFeatureMatrix(rows, featureNames, options);

  const processed: ProcessedDataset = {
    featureNames,
    features,
    labels,
    regressionTargets,
    labelNames,
    targetColumn,
    problemType: targetProblem,
    sampleRows: previewRows,
    preprocessing: config,
    stats,
  };

  emitProgress(options, 'complete', 'Preprocessing complete.', 100);
  return { insights, previewRows, processed };
}
