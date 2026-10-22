// Jalankan dari root project:  node scripts/gen-registry.js
// Memindai assets/models/*.tflite|onnx|pte lalu menulis registry.ts.
// require() di Metro harus teks literal, jadi daftar ini digenerate, bukan ditulis tangan.
// Dataset tes TIDAK dibundel: di-push lewat adb ke folder app di HP (lihat engine.ts).
//
//   assets/models/best8n.tflite, best8n.onnx, best11n.tflite, best11s.onnx, best11n.pte, ...
const fs = require('fs');
const path = require('path');

const root = path.join(__dirname, '..');
const modelsDir = path.join(root, 'assets', 'models');

const list = (dir, re) =>
  fs.existsSync(dir) ? fs.readdirSync(dir).filter((f) => re.test(f)).sort() : [];

let out = '// AUTO-GENERATED oleh scripts/gen-registry.js - jangan diedit tangan.\n';
out += "export type Format = 'tflite' | 'onnx' | 'pte';\n";
out += 'export type ModelEntry = { id: string; model: string; format: Format; source: number };\n';
out += '\n';

out += 'export const MODELS: ModelEntry[] = [\n';
for (const f of list(modelsDir, /\.(tflite|onnx|pte)$/i)) {
  const ext = path.extname(f).slice(1).toLowerCase();
  const model = path.basename(f, path.extname(f));
  out += `  { id: '${f}', model: '${model}', format: '${ext}', source: require('./assets/models/${f}') },\n`;
}
out += '];\n\n';

fs.writeFileSync(path.join(root, 'registry.ts'), out);
console.log('registry.ts ditulis.');
