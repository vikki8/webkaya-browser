import { describe, expect, test } from 'vitest';
import { analyzeDataset } from '../src/data/insights';
import { DataRow } from '../src/types/data';

describe('analyzeDataset', () => {
  test('detects supervised classification and recommends small-data algorithm', () => {
    const rows: DataRow[] = [
      { age: 20, income: 30000, label: 'A' },
      { age: 25, income: 32000, label: 'A' },
      { age: 40, income: 68000, label: 'B' },
      { age: 41, income: 70000, label: 'B' },
    ];

    const { insights } = analyzeDataset(rows, ['age', 'income', 'label']);

    expect(insights.learningMode).toBe('supervised');
    expect(insights.taskType).toBe('classification');
    expect(insights.problemType).toBe('classification');
    expect(insights.recommendedAlgorithm).toBeTruthy();
    expect(insights.suggestedAlgorithms.some((algorithm) => algorithm.id === 'random_forest')).toBe(true);
  });

  test('detects unsupervised tabular when no target exists', () => {
    const rows: DataRow[] = [
      { f1: 0.2, f2: 1.1, f3: 4.2 },
      { f1: 0.3, f2: 1.4, f3: 4.0 },
      { f1: 2.5, f2: 3.8, f3: 8.1 },
      { f1: 2.7, f2: 3.5, f3: 7.9 },
    ];

    const { insights } = analyzeDataset(rows, ['f1', 'f2', 'f3']);

    expect(insights.learningMode).toBe('unsupervised');
    expect(insights.taskType).toBe('clustering');
    expect(insights.detectedTarget).toBeNull();
    expect(insights.suggestedAlgorithms.some((algorithm) => algorithm.id === 'kmeans')).toBe(true);
  });

  test('detects image modality and image-specialized recommendation', () => {
    const rows: DataRow[] = [
      { label: 'cat', image_name: 'cat_1.png', mean_r: 0.4, mean_g: 0.3, mean_b: 0.2 },
      { label: 'dog', image_name: 'dog_1.png', mean_r: 0.3, mean_g: 0.4, mean_b: 0.5 },
    ];

    const { insights } = analyzeDataset(rows, ['label', 'image_name', 'mean_r', 'mean_g', 'mean_b'], {
      inferredFormat: 'images_zip',
    });

    expect(insights.modality).toBe('image');
    expect(insights.recommendedAlgorithm).toBe('cnn');
    expect(insights.suggestedAlgorithms.some((algorithm) => algorithm.id === 'cnn')).toBe(true);
  });

  test('detects MNIST-style high-dimensional pixel columns as image modality', () => {
    const columns = ['label', ...Array.from({ length: 784 }, (_, idx) => String(idx))];
    const makeRow = (label: string, base: number): DataRow => {
      const row: DataRow = { label };
      for (let i = 0; i < 784; i++) {
        row[String(i)] = (base + i) % 255;
      }
      return row;
    };
    const rows: DataRow[] = [makeRow('0', 1), makeRow('1', 7), makeRow('2', 13)];

    const { insights } = analyzeDataset(rows, columns);

    expect(insights.modality).toBe('image');
    expect(insights.taskType).toBe('classification');
    expect(insights.recommendedAlgorithm).toBe('cnn');
  });
});

