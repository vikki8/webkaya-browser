import { describe, expect, test } from 'vitest';
import { preprocessDataset } from '../src/data/preprocess';
import { DataRow, PreprocessingConfig } from '../src/types/data';

describe('preprocessDataset', () => {
  test('fills missing values, encodes categories, and normalizes numeric columns', () => {
    const rows: DataRow[] = [
      { age: 20, city: 'A', income: 1000, label: 'yes' },
      { age: 40, city: 'B', income: 3000, label: 'no' },
      { age: null, city: 'A', income: 2000, label: 'yes' },
    ];
    const config: PreprocessingConfig = {
      fixMissingValues: true,
      encodeCategories: true,
      normalizeData: true,
      droppedColumns: [],
      targetColumn: 'label',
    };

    const result = preprocessDataset(rows, ['age', 'city', 'income', 'label'], config);

    expect(result.processed.features.length).toBe(3);
    expect(result.processed.featureNames).toEqual(['age', 'city', 'income']);
    expect(result.processed.stats.beforeMissing).toBeGreaterThan(0);
    expect(result.processed.stats.afterMissing).toBe(0);
    expect(result.processed.labelNames.sort()).toEqual(['no', 'yes']);
  });

  test('skips categorical encoding for high-dimensional numeric features', () => {
    const featureColumns = Array.from({ length: 256 }, (_, idx) => `f_${idx}`);
    const columns = [...featureColumns, 'label'];
    const rows: DataRow[] = Array.from({ length: 40 }, (_, rowIdx) => {
      const row: DataRow = { label: rowIdx % 2 === 0 ? 'cat' : 'dog' };
      for (let c = 0; c < featureColumns.length; c++) {
        row[featureColumns[c]] = (rowIdx + c) % 255;
      }
      return row;
    });
    const config: PreprocessingConfig = {
      fixMissingValues: true,
      encodeCategories: true,
      normalizeData: false,
      droppedColumns: [],
      targetColumn: 'label',
    };

    const result = preprocessDataset(rows, columns, config, { allowInPlace: true });

    expect(result.processed.featureNames.length).toBe(256);
    expect(result.processed.stats.encodedColumns).toEqual([]);
    expect(result.processed.features.length).toBe(40);
  });
});
