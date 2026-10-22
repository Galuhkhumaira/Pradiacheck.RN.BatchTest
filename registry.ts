// AUTO-GENERATED oleh scripts/gen-registry.js - jangan diedit tangan.
export type Format = 'tflite' | 'onnx' | 'pte';
export type ModelEntry = { id: string; model: string; format: Format; source: number };

export const MODELS: ModelEntry[] = [
  { id: 'best11n.onnx', model: 'best11n', format: 'onnx', source: require('./assets/models/best11n.onnx') },
  { id: 'best11n.pte', model: 'best11n', format: 'pte', source: require('./assets/models/best11n.pte') },
  { id: 'best11n.tflite', model: 'best11n', format: 'tflite', source: require('./assets/models/best11n.tflite') },
  { id: 'best11s.onnx', model: 'best11s', format: 'onnx', source: require('./assets/models/best11s.onnx') },
  { id: 'best11s.pte', model: 'best11s', format: 'pte', source: require('./assets/models/best11s.pte') },
  { id: 'best11s.tflite', model: 'best11s', format: 'tflite', source: require('./assets/models/best11s.tflite') },
  { id: 'best8n.onnx', model: 'best8n', format: 'onnx', source: require('./assets/models/best8n.onnx') },
  { id: 'best8n.pte', model: 'best8n', format: 'pte', source: require('./assets/models/best8n.pte') },
  { id: 'best8n.tflite', model: 'best8n', format: 'tflite', source: require('./assets/models/best8n.tflite') },
];

