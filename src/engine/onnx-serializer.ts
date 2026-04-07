import { onnx } from 'onnx-proto';
import { ProcessedDataset } from '../types/data';
import { ModelMetrics, TrainedModelArtifact, TrainingPreferences } from '../types/training-workflow';

export interface OnnxExportContext {
  runId: string;
  artifact: TrainedModelArtifact;
  metrics: ModelMetrics;
  dataset: ProcessedDataset;
  preferences: TrainingPreferences;
}

type ForestNode = {
  prediction: number;
  featureIndex: number | null;
  threshold: number;
  left: ForestNode | null;
  right: ForestNode | null;
};

const TEXT_ENCODER = new TextEncoder();

function encodeString(value: string): Uint8Array {
  return TEXT_ENCODER.encode(value);
}

function tensorShape(dims: Array<number | string>): onnx.ITensorShapeProto {
  return {
    dim: dims.map((value) =>
      typeof value === 'number'
        ? { dimValue: value }
        : { dimParam: value }
    ),
  };
}

function valueInfo(name: string, dataType: number, dims: Array<number | string>): onnx.IValueInfoProto {
  return {
    name,
    type: {
      tensorType: {
        elemType: dataType,
        shape: tensorShape(dims),
      },
    },
  };
}

function floatTensor(name: string, dims: number[], values: number[]): onnx.ITensorProto {
  const rawData = new Uint8Array(new Float32Array(values).buffer);
  return {
    name,
    dims,
    dataType: onnx.TensorProto.DataType.FLOAT,
    rawData,
  };
}

function attributeInts(name: string, values: number[]): onnx.IAttributeProto {
  return {
    name,
    type: onnx.AttributeProto.AttributeType.INTS,
    ints: values,
  };
}

function attributeFloats(name: string, values: number[]): onnx.IAttributeProto {
  return {
    name,
    type: onnx.AttributeProto.AttributeType.FLOATS,
    floats: values,
  };
}

function attributeStrings(name: string, values: string[]): onnx.IAttributeProto {
  return {
    name,
    type: onnx.AttributeProto.AttributeType.STRINGS,
    strings: values.map(encodeString),
  };
}

function attributeString(name: string, value: string): onnx.IAttributeProto {
  return {
    name,
    type: onnx.AttributeProto.AttributeType.STRING,
    s: encodeString(value),
  };
}

function attributeInt(name: string, value: number): onnx.IAttributeProto {
  return {
    name,
    type: onnx.AttributeProto.AttributeType.INT,
    i: value,
  };
}

function baseModel(graph: onnx.IGraphProto, opsets: onnx.IOperatorSetIdProto[], metadata: Record<string, string>): Uint8Array {
  const metadataProps = Object.entries(metadata).map(([key, value]) => ({ key, value }));
  const model: onnx.IModelProto = {
    irVersion: onnx.Version.IR_VERSION,
    producerName: 'browser-first-ai',
    producerVersion: '0.1.0',
    domain: 'browser-first-ai',
    modelVersion: 1,
    graph,
    opsetImport: opsets,
    metadataProps,
  };

  const verificationError = onnx.ModelProto.verify(model as any);
  if (verificationError) {
    throw new Error(`ONNX verification failed: ${verificationError}`);
  }

  return onnx.ModelProto.encode(model).finish();
}

type DenseLayerSnapshot = {
  inputSize: number;
  outputSize: number;
  weights: number[];
  bias: number[];
};

type BatchNormLayerSnapshot = {
  featureCount: number;
  gamma: number[];
  beta: number[];
  runningMean: number[];
  runningVar: number[];
  epsilon?: number;
};

function activationOpType(value: unknown): 'Relu' | 'Tanh' | 'Sigmoid' {
  if (value === 'tanh') return 'Tanh';
  if (value === 'sigmoid') return 'Sigmoid';
  return 'Relu';
}

function resolveDenseLayers(modelData: Record<string, unknown>, featureCount: number, classCount: number): DenseLayerSnapshot[] {
  const linearLayers = modelData.linearLayers as DenseLayerSnapshot[] | undefined;
  if (Array.isArray(linearLayers) && linearLayers.length > 0) {
    return linearLayers;
  }

  const legacyLayers = modelData.layers as number[][] | undefined;
  if (!legacyLayers || legacyLayers.length < 4) {
    throw new Error('Neural network export requires serialized dense layers.');
  }
  const [w1, b1, w2, b2] = legacyLayers;
  return [
    {
      inputSize: featureCount,
      outputSize: b1.length,
      weights: w1,
      bias: b1,
    },
    {
      inputSize: b1.length,
      outputSize: classCount,
      weights: w2,
      bias: b2,
    },
  ];
}

function buildNeuralNetworkOnnx(context: OnnxExportContext): Uint8Array {
  const featureCount = context.dataset.featureNames.length;
  const classCount = context.dataset.labelNames.length;
  const denseLayers = resolveDenseLayers(context.artifact.modelData, featureCount, classCount);
  if (!denseLayers.length) throw new Error('Neural network export requires at least one dense layer.');
  if (denseLayers[0].inputSize !== featureCount) {
    throw new Error('Input feature size mismatch while exporting neural network to ONNX.');
  }
  if (denseLayers[denseLayers.length - 1].outputSize !== classCount) {
    throw new Error('Output class size mismatch while exporting neural network to ONNX.');
  }

  const activation = activationOpType(context.artifact.modelData.activation);
  const batchNormLayers = (context.artifact.modelData.batchNormLayers as BatchNormLayerSnapshot[] | undefined) ?? [];
  const valueInfos: onnx.IValueInfoProto[] = [];
  const initializers: onnx.ITensorProto[] = [];
  const nodes: onnx.INodeProto[] = [];

  let currentInput = 'X';
  denseLayers.forEach((layer, index) => {
    if (layer.weights.length !== layer.inputSize * layer.outputSize) {
      throw new Error(`Dense layer ${index + 1} weight tensor shape mismatch for ONNX export.`);
    }
    if (layer.bias.length !== layer.outputSize) {
      throw new Error(`Dense layer ${index + 1} bias tensor shape mismatch for ONNX export.`);
    }

    const wName = `W${index + 1}`;
    const bName = `B${index + 1}`;
    const denseOut = index === denseLayers.length - 1 ? 'Logits' : `H${index + 1}`;
    initializers.push(floatTensor(wName, [layer.inputSize, layer.outputSize], layer.weights));
    initializers.push(floatTensor(bName, [layer.outputSize], layer.bias));
    valueInfos.push(valueInfo(denseOut, onnx.TensorProto.DataType.FLOAT, ['N', layer.outputSize]));

    nodes.push({
      name: `dense_${index + 1}`,
      opType: 'Gemm',
      input: [currentInput, wName, bName],
      output: [denseOut],
      attribute: [
        attributeFloatGemm('alpha', 1),
        attributeFloatGemm('beta', 1),
        attributeInt('transA', 0),
        attributeInt('transB', 0),
      ],
    });

    if (index < denseLayers.length - 1) {
      let activationInput = denseOut;
      const bnLayer = batchNormLayers[index];
      if (bnLayer) {
        if (
          bnLayer.featureCount !== layer.outputSize ||
          bnLayer.gamma.length !== layer.outputSize ||
          bnLayer.beta.length !== layer.outputSize ||
          bnLayer.runningMean.length !== layer.outputSize ||
          bnLayer.runningVar.length !== layer.outputSize
        ) {
          throw new Error(`BatchNorm layer ${index + 1} tensor shape mismatch for ONNX export.`);
        }
        const bnOut = `${denseOut}_bn`;
        const bnScale = `BN${index + 1}_scale`;
        const bnBias = `BN${index + 1}_bias`;
        const bnMean = `BN${index + 1}_mean`;
        const bnVar = `BN${index + 1}_var`;
        initializers.push(floatTensor(bnScale, [layer.outputSize], bnLayer.gamma));
        initializers.push(floatTensor(bnBias, [layer.outputSize], bnLayer.beta));
        initializers.push(floatTensor(bnMean, [layer.outputSize], bnLayer.runningMean));
        initializers.push(floatTensor(bnVar, [layer.outputSize], bnLayer.runningVar));
        valueInfos.push(valueInfo(bnOut, onnx.TensorProto.DataType.FLOAT, ['N', layer.outputSize]));
        nodes.push({
          name: `batchnorm_${index + 1}`,
          opType: 'BatchNormalization',
          input: [denseOut, bnScale, bnBias, bnMean, bnVar],
          output: [bnOut],
          attribute: [attributeFloatGemm('epsilon', bnLayer.epsilon ?? 1e-5)],
        });
        activationInput = bnOut;
      }

      const activationOut = `${denseOut}_act`;
      valueInfos.push(valueInfo(activationOut, onnx.TensorProto.DataType.FLOAT, ['N', layer.outputSize]));
      nodes.push({
        name: `${activation.toLowerCase()}_${index + 1}`,
        opType: activation,
        input: [activationInput],
        output: [activationOut],
      });
      currentInput = activationOut;
    } else {
      currentInput = denseOut;
    }
  });

  nodes.push({
    name: 'probabilities',
    opType: 'Softmax',
    input: [currentInput],
    output: ['Y'],
    attribute: [attributeInt('axis', 1)],
  });

  const graph: onnx.IGraphProto = {
    name: `${context.runId}_nn`,
    input: [valueInfo('X', onnx.TensorProto.DataType.FLOAT, ['N', featureCount])],
    output: [valueInfo('Y', onnx.TensorProto.DataType.FLOAT, ['N', classCount])],
    valueInfo: valueInfos,
    initializer: initializers,
    node: nodes,
  };

  return baseModel(
    graph,
    [{ domain: '', version: 13 }],
    {
      bfai_model_type: context.artifact.modelType,
      bfai_backend: context.artifact.backend,
      bfai_target_column: context.dataset.targetColumn,
    }
  );
}

function buildLinearRegressionOnnx(context: OnnxExportContext): Uint8Array {
  const featureCount = context.dataset.featureNames.length;
  const classCount = Number(context.artifact.modelData.classCount ?? context.dataset.labelNames.length);
  const mode = String(context.artifact.modelData.mode ?? 'classifier');
  const isRegressor = mode === 'regressor' || classCount === 1;
  const weights = context.artifact.modelData.weights as number[] | undefined;
  const bias = context.artifact.modelData.bias as number[] | undefined;

  if (!weights || !bias) {
    throw new Error('Linear classifier export requires "weights" and "bias" tensors.');
  }
  if (weights.length !== featureCount * classCount) {
    throw new Error('Linear regression weight tensor shape mismatch for ONNX export.');
  }
  if (bias.length !== classCount) {
    throw new Error('Linear regression bias tensor shape mismatch for ONNX export.');
  }

  const graph: onnx.IGraphProto = {
    name: `${context.runId}_${context.artifact.modelType}`,
    input: [valueInfo('X', onnx.TensorProto.DataType.FLOAT, ['N', featureCount])],
    output: [valueInfo('Y', onnx.TensorProto.DataType.FLOAT, ['N', classCount])],
    valueInfo: isRegressor ? [] : [valueInfo('Logits', onnx.TensorProto.DataType.FLOAT, ['N', classCount])],
    initializer: [
      floatTensor('W_lr', [featureCount, classCount], weights),
      floatTensor('B_lr', [classCount], bias),
    ],
    node: isRegressor
      ? [
          {
            name: 'linear_projection',
            opType: 'Gemm',
            input: ['X', 'W_lr', 'B_lr'],
            output: ['Y'],
            attribute: [
              attributeFloatGemm('alpha', 1),
              attributeFloatGemm('beta', 1),
              attributeInt('transA', 0),
              attributeInt('transB', 0),
            ],
          },
        ]
      : [
          {
            name: 'linear_projection',
            opType: 'Gemm',
            input: ['X', 'W_lr', 'B_lr'],
            output: ['Logits'],
            attribute: [
              attributeFloatGemm('alpha', 1),
              attributeFloatGemm('beta', 1),
              attributeInt('transA', 0),
              attributeInt('transB', 0),
            ],
          },
          {
            name: 'linear_probabilities',
            opType: 'Softmax',
            input: ['Logits'],
            output: ['Y'],
            attribute: [attributeInt('axis', 1)],
          },
        ],
  };

  return baseModel(
    graph,
    [{ domain: '', version: 13 }],
    {
      bfai_model_type: context.artifact.modelType,
      bfai_backend: context.artifact.backend,
      bfai_target_column: context.dataset.targetColumn,
    }
  );
}

function attributeFloatGemm(name: string, value: number): onnx.IAttributeProto {
  return {
    name,
    type: onnx.AttributeProto.AttributeType.FLOAT,
    f: value,
  };
}

type TreeEncoding = {
  nodesTreeIds: number[];
  nodesNodeIds: number[];
  nodesFeatureIds: number[];
  nodesModes: string[];
  nodesValues: number[];
  nodesTrueNodeIds: number[];
  nodesFalseNodeIds: number[];
  nodesMissingTracksTrue: number[];
  classTreeIds: number[];
  classNodeIds: number[];
  classIds: number[];
  classWeights: number[];
};

function createEmptyTreeEncoding(): TreeEncoding {
  return {
    nodesTreeIds: [],
    nodesNodeIds: [],
    nodesFeatureIds: [],
    nodesModes: [],
    nodesValues: [],
    nodesTrueNodeIds: [],
    nodesFalseNodeIds: [],
    nodesMissingTracksTrue: [],
    classTreeIds: [],
    classNodeIds: [],
    classIds: [],
    classWeights: [],
  };
}

function encodeTreeNodes(
  node: ForestNode,
  treeId: number,
  classCount: number,
  encoding: TreeEncoding,
  nextNodeId: { value: number }
): number {
  const nodeId = nextNodeId.value++;

  if (node.featureIndex === null || !node.left || !node.right) {
    encoding.nodesTreeIds.push(treeId);
    encoding.nodesNodeIds.push(nodeId);
    encoding.nodesFeatureIds.push(0);
    encoding.nodesModes.push('LEAF');
    encoding.nodesValues.push(0);
    encoding.nodesTrueNodeIds.push(0);
    encoding.nodesFalseNodeIds.push(0);
    encoding.nodesMissingTracksTrue.push(0);

    const classId = Math.max(0, Math.min(classCount - 1, Math.round(node.prediction)));
    encoding.classTreeIds.push(treeId);
    encoding.classNodeIds.push(nodeId);
    encoding.classIds.push(classId);
    encoding.classWeights.push(1);
    return nodeId;
  }

  const trueNodeId = encodeTreeNodes(node.left, treeId, classCount, encoding, nextNodeId);
  const falseNodeId = encodeTreeNodes(node.right, treeId, classCount, encoding, nextNodeId);

  encoding.nodesTreeIds.push(treeId);
  encoding.nodesNodeIds.push(nodeId);
  encoding.nodesFeatureIds.push(node.featureIndex);
  encoding.nodesModes.push('BRANCH_LEQ');
  encoding.nodesValues.push(node.threshold);
  encoding.nodesTrueNodeIds.push(trueNodeId);
  encoding.nodesFalseNodeIds.push(falseNodeId);
  encoding.nodesMissingTracksTrue.push(0);

  return nodeId;
}

function buildRandomForestOnnx(context: OnnxExportContext): Uint8Array {
  const trees = (context.artifact.modelData.trees as ForestNode[] | undefined) ?? [];
  if (!trees.length) {
    throw new Error('Random forest ONNX export requires serialized tree structures.');
  }

  const featureCount = context.dataset.featureNames.length;
  const classCount = context.dataset.labelNames.length;
  const encoding = createEmptyTreeEncoding();

  for (let treeIndex = 0; treeIndex < trees.length; treeIndex++) {
    encodeTreeNodes(trees[treeIndex], treeIndex, classCount, encoding, { value: 0 });
  }

  const graph: onnx.IGraphProto = {
    name: `${context.runId}_rf`,
    input: [valueInfo('X', onnx.TensorProto.DataType.FLOAT, ['N', featureCount])],
    output: [
      valueInfo('Y', onnx.TensorProto.DataType.INT64, ['N']),
      valueInfo('Z', onnx.TensorProto.DataType.FLOAT, ['N', classCount]),
    ],
    node: [
      {
        name: 'tree_ensemble_classifier',
        opType: 'TreeEnsembleClassifier',
        domain: 'ai.onnx.ml',
        input: ['X'],
        output: ['Y', 'Z'],
        attribute: [
          attributeInts('classlabels_int64s', Array.from({ length: classCount }, (_, i) => i)),
          attributeInts('nodes_treeids', encoding.nodesTreeIds),
          attributeInts('nodes_nodeids', encoding.nodesNodeIds),
          attributeInts('nodes_featureids', encoding.nodesFeatureIds),
          attributeStrings('nodes_modes', encoding.nodesModes),
          attributeFloats('nodes_values', encoding.nodesValues),
          attributeInts('nodes_truenodeids', encoding.nodesTrueNodeIds),
          attributeInts('nodes_falsenodeids', encoding.nodesFalseNodeIds),
          attributeInts('nodes_missing_value_tracks_true', encoding.nodesMissingTracksTrue),
          attributeInts('class_treeids', encoding.classTreeIds),
          attributeInts('class_nodeids', encoding.classNodeIds),
          attributeInts('class_ids', encoding.classIds),
          attributeFloats('class_weights', encoding.classWeights),
          attributeString('post_transform', 'NONE'),
        ],
      },
    ],
  };

  return baseModel(
    graph,
    [
      { domain: '', version: 13 },
      { domain: 'ai.onnx.ml', version: 3 },
    ],
    {
      bfai_model_type: context.artifact.modelType,
      bfai_backend: context.artifact.backend,
      bfai_target_column: context.dataset.targetColumn,
    }
  );
}

export function serializeModelToOnnx(context: OnnxExportContext): Uint8Array {
  if (context.artifact.modelType === 'neural_network') {
    return buildNeuralNetworkOnnx(context);
  }
  if (context.artifact.modelType === 'random_forest' || context.artifact.modelType === 'decision_tree') {
    return buildRandomForestOnnx(context);
  }
  if (
    context.artifact.modelType === 'linear_regression' ||
    context.artifact.modelType === 'logistic_regression' ||
    context.artifact.modelType === 'svm'
  ) {
    return buildLinearRegressionOnnx(context);
  }
  if (context.artifact.modelType === 'knn') {
    throw new Error('KNN ONNX export is not supported yet. Use .pth or .kaya export instead.');
  }
  throw new Error(`Unsupported model type for ONNX export: ${context.artifact.modelType}`);
}
