import { describe, expect, test } from 'vitest';
import {
  predictDbscanClassifier,
  predictKnnClassifier,
  predictKMeansClassifier,
  predictLogisticRegressionClassifier,
  predictLinearRegressionClassifier,
  predictRecurrentClassifier,
  predictRandomForestAlgorithm,
  predictSvmClassifier,
  trainDbscanClassifier,
  trainKnnClassifier,
  trainKMeansClassifier,
  trainDecisionTree,
  trainLogisticRegressionClassifier,
  trainLinearRegressionClassifier,
  trainRecurrentClassifier,
  trainSvmClassifier,
} from '../src/engine/algorithms';

describe('algorithm implementations', () => {
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

  test('trains decision tree classifier', () => {
    const treeModel = trainDecisionTree(features, labels, {
      maxDepth: 5,
      minSamplesSplit: 2,
    });
    const predictions = predictRandomForestAlgorithm(treeModel, features);
    const accuracy =
      predictions.filter((prediction, index) => prediction === labels[index]).length / labels.length;
    expect(accuracy).toBeGreaterThanOrEqual(0.9);
  });

  test('trains linear regression classifier', () => {
    const model = trainLinearRegressionClassifier(
      features,
      labels,
      2,
      {
        epochs: 120,
        learningRate: 0.02,
        l2: 0.001,
      },
      {}
    );
    const predictions = predictLinearRegressionClassifier(model, features);
    const accuracy =
      predictions.filter((prediction, index) => prediction === labels[index]).length / labels.length;
    expect(accuracy).toBeGreaterThanOrEqual(0.75);
  });

  test('trains logistic regression classifier', () => {
    const model = trainLogisticRegressionClassifier(features, labels, 2, {
      epochs: 120,
      learningRate: 0.05,
      l2: 0.001,
    });
    const predictions = predictLogisticRegressionClassifier(model, features);
    const accuracy =
      predictions.filter((prediction, index) => prediction === labels[index]).length / labels.length;
    expect(accuracy).toBeGreaterThanOrEqual(0.9);
  });

  test('trains SVM classifier', () => {
    const model = trainSvmClassifier(
      features,
      labels,
      2,
      {
        epochs: 80,
        learningRate: 0.02,
        regularization: 0.001,
        kernel: 'linear',
      }
    );
    const predictions = predictSvmClassifier(model, features);
    const accuracy =
      predictions.filter((prediction, index) => prediction === labels[index]).length / labels.length;
    expect(accuracy).toBeGreaterThanOrEqual(0.8);
  });

  test('builds KNN classifier', () => {
    const model = trainKnnClassifier(features, labels, 2, {
      neighbors: 3,
      distanceMetric: 'euclidean',
    });
    const predictions = predictKnnClassifier(model, features);
    const accuracy =
      predictions.filter((prediction, index) => prediction === labels[index]).length / labels.length;
    expect(accuracy).toBeGreaterThanOrEqual(0.9);
  });

  test('trains K-Means clustering classifier mapping', () => {
    const model = trainKMeansClassifier(
      features,
      labels,
      2,
      {
        clusters: 2,
        maxIterations: 40,
      }
    );
    const predictions = predictKMeansClassifier(model, features);
    const accuracy =
      predictions.filter((prediction, index) => prediction === labels[index]).length / labels.length;
    expect(accuracy).toBeGreaterThanOrEqual(0.9);
  });

  test('trains DBSCAN clustering classifier mapping', () => {
    const model = trainDbscanClassifier(
      features,
      labels,
      2,
      {
        epsilon: 2.2,
        minSamples: 2,
      }
    );
    const predictions = predictDbscanClassifier(model, features);
    const accuracy =
      predictions.filter((prediction, index) => prediction === labels[index]).length / labels.length;
    expect(accuracy).toBeGreaterThanOrEqual(0.75);
  });

  test('trains recurrent classifier variants', () => {
    for (const variant of ['rnn', 'lstm', 'gru'] as const) {
      const model = trainRecurrentClassifier(
        features,
        labels,
        2,
        {
          variant,
          epochs: 80,
          learningRate: 0.03,
          hiddenSize: 24,
          inputSize: 2,
        }
      );
      const predictions = predictRecurrentClassifier(model, features);
      const accuracy =
        predictions.filter((prediction, index) => prediction === labels[index]).length / labels.length;
      expect(accuracy).toBeGreaterThanOrEqual(0.7);
    }
  });
});

