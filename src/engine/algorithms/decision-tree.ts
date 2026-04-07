import { RandomForestModel, trainRandomForest } from './random-forest';

export interface DecisionTreeTrainOptions {
  maxDepth: number;
  minSamplesSplit: number;
}

export function trainDecisionTree(
  features: number[][],
  labels: number[],
  options: DecisionTreeTrainOptions,
  onProgress?: (progress: number) => void
): RandomForestModel {
  return trainRandomForest(
    features,
    labels,
    {
      trees: 1,
      maxDepth: Math.max(2, options.maxDepth),
      minSamplesSplit: Math.max(2, options.minSamplesSplit),
      featureSampleRate: 1,
      bootstrap: false,
    },
    (progress) => onProgress?.(progress)
  );
}

