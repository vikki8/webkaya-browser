export type DataSourceType = 'upload' | 'kaggle' | 'external';
export type UploadedDataFormat = 'csv' | 'json' | 'images_zip';
export type ColumnType = 'numeric' | 'categorical' | 'text' | 'boolean';
export type LearningMode = 'supervised' | 'unsupervised';
export type InsightTaskType = 'classification' | 'regression' | 'clustering' | 'representation_generation' | 'unknown';
export type DatasetModality = 'tabular' | 'image' | 'time_series' | 'sequence';
export type DatasetSizeBand = 'small' | 'medium' | 'large';
export type AlgorithmId =
  | 'decision_tree'
  | 'random_forest'
  | 'logistic_regression'
  | 'svm'
  | 'knn'
  | 'neural_network'
  | 'linear_regression'
  | 'kmeans'
  | 'dbscan'
  | 'cnn'
  | 'rnn'
  | 'lstm'
  | 'gru';

export interface AlgorithmSuggestion {
  id: AlgorithmId;
  label: string;
  family: 'supervised' | 'unsupervised';
  useCaseHint: string;
  reason: string;
  speed: 'fast' | 'balanced' | 'slow';
  expectedQuality: 'baseline' | 'strong' | 'high';
  runtimeSupport: 'native' | 'mapped' | 'planned';
}

export interface DataRow {
  [key: string]: string | number | boolean | null;
}

export interface DataColumnProfile {
  name: string;
  type: ColumnType;
  missingCount: number;
  uniqueCount: number;
  sampleValues: string[];
}

export interface ParsedDataset {
  rows: DataRow[];
  columns: string[];
  source: {
    type: DataSourceType;
    name: string;
    description?: string;
  };
  inferredFormat: UploadedDataFormat | 'kaggle';
}

export interface AutoInsights {
  problemType: 'classification' | 'regression' | 'unknown';
  detectedTarget: string | null;
  learningMode: LearningMode;
  taskType: InsightTaskType;
  modality: DatasetModality;
  datasetSize: DatasetSizeBand;
  rowCount: number;
  featureCount: number;
  targetColumnExists: boolean;
  recommendedAlgorithm: AlgorithmId | null;
  recommendationReason: string;
  suggestedAlgorithms: AlgorithmSuggestion[];
  missingColumns: string[];
  categoricalColumns: string[];
  numericColumns: string[];
  warnings: string[];
}

export interface PreprocessingConfig {
  fixMissingValues: boolean;
  encodeCategories: boolean;
  normalizeData: boolean;
  augmentImageData: boolean;
  imageAugmentationFactor: 1 | 2 | 3;
  imageAugmentationNoise: number;
  droppedColumns: string[];
  targetColumn: string | null;
}

export interface PreprocessingStats {
  beforeMissing: number;
  afterMissing: number;
  encodedColumns: string[];
  normalizedColumns: string[];
  droppedColumns: string[];
}

export interface ProcessedDataset {
  featureNames: string[];
  features: number[][];
  labels: number[];
  regressionTargets?: number[];
  labelNames: string[];
  targetColumn: string;
  problemType: 'classification' | 'regression';
  sampleRows: DataRow[];
  preprocessing: PreprocessingConfig;
  stats: PreprocessingStats;
}

export interface KaggleDatasetResult {
  ref: string;
  title: string;
  subtitle?: string;
  owner?: string;
  totalBytes?: number;
  downloadCount?: number;
  voteCount?: number;
  lastUpdated?: string;
  usableFiles?: string[];
}

export interface KaggleCredentials {
  username: string;
  apiKey: string;
}
