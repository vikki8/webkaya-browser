import { describe, expect, test } from 'vitest';
import { zipSync } from 'fflate';
import { parseDatasetBlob } from '../src/data/parsers';

describe('parseDatasetBlob streaming (Fashion-MNIST–class ZIP/CSV)', () => {
  test('parses CSV inside ZIP via streamed ZIP + streamed CSV (no full-archive string)', async () => {
    const csv = 'label,pixel_0,pixel_1\n0,10,20\n9,30,40\n';
    const zipped = zipSync({ 'fashion-mnist_train.csv': new TextEncoder().encode(csv) });
    const blob = new Blob([zipped], { type: 'application/zip' });
    const parsed = await parseDatasetBlob(blob, 'fashion.zip', 'upload');
    expect(parsed.inferredFormat).toBe('csv');
    expect(parsed.rows).toHaveLength(2);
    expect(parsed.rows[0].label).toBe(0);
    expect(parsed.rows[1].label).toBe(9);
  });

  test('prefers train CSV when both train and test exist in the archive', async () => {
    const train = 'a,b\n1,2\n';
    const testCsv = 'a,b\n9,9\n';
    const zipped = zipSync({
      'fashion-mnist_test.csv': new TextEncoder().encode(testCsv),
      'fashion-mnist_train.csv': new TextEncoder().encode(train),
    });
    const blob = new Blob([zipped], { type: 'application/zip' });
    const parsed = await parseDatasetBlob(blob, 'fashion.zip', 'upload');
    expect(parsed.rows).toHaveLength(1);
    expect(parsed.rows[0].a).toBe(1);
  });
});
