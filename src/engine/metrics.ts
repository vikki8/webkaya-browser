export interface ClassificationMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  confusionMatrix: number[][];
}

export interface RegressionMetrics {
  mae: number;
  rmse: number;
  r2: number;
}

export function computeClassificationMetrics(
  trueLabels: number[],
  predictedLabels: number[],
  numClasses: number
): ClassificationMetrics {
  if (trueLabels.length !== predictedLabels.length) {
    throw new Error('Label and prediction arrays must have same length.');
  }
  const nClass = Math.max(1, Math.floor(numClasses));
  const matrix = Array.from({ length: nClass }, () => new Array(nClass).fill(0));
  let correct = 0;

  for (let i = 0; i < trueLabels.length; i++) {
    let actual = Math.floor(Number(trueLabels[i]));
    let predicted = Math.floor(Number(predictedLabels[i]));
    if (!Number.isFinite(actual) || actual < 0 || actual >= nClass) actual = 0;
    if (!Number.isFinite(predicted) || predicted < 0 || predicted >= nClass) predicted = 0;
    if (actual === predicted) correct++;
    matrix[actual][predicted] += 1;
  }

  const accuracy = trueLabels.length ? correct / trueLabels.length : 0;

  let precisionSum = 0;
  let recallSum = 0;
  for (let cls = 0; cls < nClass; cls++) {
    const tp = matrix[cls][cls];
    let fp = 0;
    let fn = 0;
    for (let i = 0; i < nClass; i++) {
      if (i !== cls) {
        fp += matrix[i][cls];
        fn += matrix[cls][i];
      }
    }
    const precision = tp + fp === 0 ? 0 : tp / (tp + fp);
    const recall = tp + fn === 0 ? 0 : tp / (tp + fn);
    precisionSum += precision;
    recallSum += recall;
  }

  const precision = nClass ? precisionSum / nClass : 0;
  const recall = nClass ? recallSum / nClass : 0;
  const f1 = precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);

  return { accuracy, precision, recall, f1, confusionMatrix: matrix };
}

export function rankFeatureImportance(
  featureNames: string[],
  scores: number[]
): Array<{ feature: string; score: number }> {
  return featureNames
    .map((feature, idx) => ({ feature, score: scores[idx] ?? 0 }))
    .sort((a, b) => b.score - a.score);
}

export function computeRegressionMetrics(
  trueValues: number[],
  predictedValues: number[]
): RegressionMetrics {
  if (trueValues.length !== predictedValues.length) {
    throw new Error('Regression target and prediction arrays must have same length.');
  }
  if (!trueValues.length) {
    return { mae: 0, rmse: 0, r2: 0 };
  }

  let absoluteErrorSum = 0;
  let squaredErrorSum = 0;
  let mean = 0;
  for (const value of trueValues) mean += value;
  mean /= trueValues.length;

  let totalVariance = 0;
  for (let i = 0; i < trueValues.length; i++) {
    const actual = trueValues[i];
    const predicted = predictedValues[i];
    const error = predicted - actual;
    absoluteErrorSum += Math.abs(error);
    squaredErrorSum += error * error;
    const centered = actual - mean;
    totalVariance += centered * centered;
  }

  const mae = absoluteErrorSum / trueValues.length;
  const mse = squaredErrorSum / trueValues.length;
  const rmse = Math.sqrt(mse);
  const r2 = totalVariance <= 1e-12 ? 1 : 1 - squaredErrorSum / totalVariance;
  return { mae, rmse, r2 };
}
