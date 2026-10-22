const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);

// Izinkan file model dibundel sebagai asset.
config.resolver.assetExts = [
  ...config.resolver.assetExts.filter((e) => !['tflite', 'onnx', 'pte'].includes(e)),
  'tflite',
  'onnx',
  'pte',
];

module.exports = config;
