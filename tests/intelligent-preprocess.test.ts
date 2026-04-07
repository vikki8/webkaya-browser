import { describe, expect, it } from 'vitest';
import { buildIntelligentPreprocessingPlan, refreshNormalizationPlanLine, shouldNormalizeForAlgorithm } from '../src/data/intelligent-preprocess';
import { AutoInsights, DataColumnProfile } from '../src/types/data';

function baseInsights(over: Partial<AutoInsights> = {}): AutoInsights {
  return {
    problemType: 'classification',
    detectedTarget: 'y',
    learningMode: 'supervised',
    taskType: 'classification',
    modality: 'tabular',
    datasetSize: 'small',
    rowCount: 100,
    featureCount: 3,
    targetColumnExists: true,
    recommendedAlgorithm: 'random_forest',
    recommendationReason: 'test',
    suggestedAlgorithms: [],
    missingColumns: [],
    categoricalColumns: [],
    numericColumns: [],
    warnings: [],
    ...over,
  };
}

describe('shouldNormalizeForAlgorithm', () => {
  it('skips trees when performance allows', () => {
    expect(shouldNormalizeForAlgorithm('random_forest', false)).toBe(false);
    expect(shouldNormalizeForAlgorithm('decision_tree', false)).toBe(false);
  });
  it('enables for neural nets and distance models when performance allows', () => {
    expect(shouldNormalizeForAlgorithm('neural_network', false)).toBe(true);
    expect(shouldNormalizeForAlgorithm('knn', false)).toBe(true);
    expect(shouldNormalizeForAlgorithm('kmeans', false)).toBe(true);
  });
  it('disables when performance flag is set', () => {
    expect(shouldNormalizeForAlgorithm('neural_network', true)).toBe(false);
  });
});

describe('buildIntelligentPreprocessingPlan', () => {
  it('drops columns with >30% missing and disables imputation when nothing left to fix', () => {
    const profiles: DataColumnProfile[] = [
      { name: 'a', type: 'numeric', missingCount: 40, uniqueCount: 60, sampleValues: [] },
      { name: 'y', type: 'categorical', missingCount: 0, uniqueCount: 2, sampleValues: [] },
    ];
    const insights = baseInsights({ detectedTarget: 'y', rowCount: 100 });
    const { config, plan } = buildIntelligentPreprocessingPlan(insights, profiles, {
      inferredFormat: 'csv',
      rowCount: 100,
      disableNormalizationForPerformance: false,
      algorithmHint: 'random_forest',
    });
    expect(config.droppedColumns).toContain('a');
    expect(config.fixMissingValues).toBe(false);
    expect(plan.some((line) => line.includes('Dropped'))).toBe(true);
  });

  it('enables imputation when feature columns have moderate missingness', () => {
    const profiles: DataColumnProfile[] = [
      { name: 'a', type: 'numeric', missingCount: 5, uniqueCount: 90, sampleValues: [] },
      { name: 'y', type: 'categorical', missingCount: 0, uniqueCount: 2, sampleValues: [] },
    ];
    const insights = baseInsights({ detectedTarget: 'y', rowCount: 100 });
    const { config } = buildIntelligentPreprocessingPlan(insights, profiles, {
      inferredFormat: 'csv',
      rowCount: 100,
      disableNormalizationForPerformance: false,
      algorithmHint: 'logistic_regression',
    });
    expect(config.fixMissingValues).toBe(true);
    expect(config.encodeCategories).toBe(false);
    expect(config.normalizeData).toBe(true);
  });
});

describe('refreshNormalizationPlanLine', () => {
  it('replaces the normalization bullet', () => {
    const plan = ['Dataset: …', 'Normalization: old'];
    const rf = refreshNormalizationPlanLine(plan, 'random_forest', false, false);
    expect(rf.find((l) => l.startsWith('Normalization:'))).toContain('tree');
    const nn = refreshNormalizationPlanLine(plan, 'neural_network', true, false);
    expect(nn.find((l) => l.startsWith('Normalization:'))).toContain('on');
  });
});
