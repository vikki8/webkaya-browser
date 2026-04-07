import { describe, expect, test } from 'vitest';
import { predictRandomForest, trainRandomForest } from '../src/engine/algorithms/random-forest';

describe('random forest', () => {
  test('learns a separable dataset', () => {
    const features = [
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1],
      [8, 8],
      [9, 8],
      [8, 9],
      [9, 9],
    ];
    const labels = [0, 0, 0, 0, 1, 1, 1, 1];

    const model = trainRandomForest(features, labels, {
      trees: 25,
      maxDepth: 5,
      minSamplesSplit: 2,
      featureSampleRate: 1,
    });

    const predictions = predictRandomForest(model, features);
    const accuracy =
      predictions.filter((prediction, index) => prediction === labels[index]).length / labels.length;

    expect(accuracy).toBeGreaterThanOrEqual(0.9);
  });
});
