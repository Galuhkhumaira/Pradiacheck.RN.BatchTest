import { loadTensorflowModel } from 'react-native-fast-tflite';
import { InferenceSession, Tensor } from 'onnxruntime-react-native';
import { ImageManipulator, SaveFormat } from 'expo-image-manipulator';
import { ExecutorchModule, ScalarType } from 'react-native-executorch/legacy';
import { Asset } from 'expo-asset';
import * as Application from 'expo-application';
import { Directory, File, Paths } from 'expo-file-system';
import UPNG from 'upng-js';
import type { ModelEntry } from './registry';

// ===== Letakkan di: C:\Galuh\PradiacheckRN\engine.ts (sejajar dengan App.tsx) =====
// Model ditaruh di assets/models lalu jalankan `node scripts/gen-registry.js`;
// model/format dipilih dari UI aplikasi. Dataset tes & hasil ada di penyimpanan internal app
// (lihat bagian "Dataset tes & hasil" di bawah).
// Catatan: di build release, react-native-fast-tflite butuh patch kecil di
// HybridAssetLoader.kt supaya bisa membaca model dari resource APK.

export const INPUT_SIZE = 640;

// Urutan HARUS sama dengan model.names: {0: 'Diabetes', 1: 'Nondiabetes', 2: 'Pradiabetes'}
export const LABELS = ['Diabetes', 'Non-diabetes', 'Pradiabetes'];
export const SHORT = ['Dia', 'Non', 'Pra'];

// Skor kelas di bawah ini dianggap "tidak terdeteksi".
export const CONF_THRESHOLD = 0.25;

export type Prediction = {
  label: string;
  index: number; // -1 = tidak terdeteksi
  confidence: number;
  scores: number[];
  detections: number; // jumlah kotak setelah ambang skor + NMS
};

export type BenchResult = {
  engine: string;
  model: string;
  preprocessMs: number;
  firstRunMs: number;
  avgMs: number;
  medianMs: number;
  minMs: number;
  maxMs: number;
  inputInfo: string;
  outputShape: string;
  prediction: Prediction;
};

const now = (): number => {
  const p = (globalThis as any).performance;
  return p && typeof p.now === 'function' ? p.now() : Date.now();
};

// ---------- Backend (tflite / onnx / pte) ----------
export type Backend = {
  entry: ModelEntry;
  engineName: string;
  nchw: boolean;
  asUint8: boolean;
  inputInfo: string;
  outShape?: readonly number[];
  run: (buffer: ArrayBuffer) => Promise<Float32Array>;
  dispose?: () => void; // melepas memori native saat ganti model
  loadMs?: number; // waktu memuat model ke engine native
  modelMb?: number; // ukuran file model (MB)
  coldStartMs?: number; // inferensi pertama setelah model dimuat
};

async function loadTflite(entry: ModelEntry): Promise<Backend> {
  const tl = now();
  const model = await loadTensorflowModel(entry.source, []);
  const loadMs = now() - tl;
  const inInfo = (model as any).inputs?.[0];
  const inShape = inInfo?.shape as number[] | undefined;
  const inType = String(inInfo?.dataType ?? 'float32');
  const nchw = !!inShape && inShape.length === 4 && inShape[1] === 3 && inShape[3] !== 3;
  const asUint8 = inType.toLowerCase().includes('uint8');
  return {
    entry,
    engineName: 'TFLite (fast-tflite)',
    nchw,
    asUint8,
    inputInfo:
      `${inShape ? `[${inShape.join(', ')}]` : '?'} ${inType}` +
      ` -> dikirim ${nchw ? 'NCHW' : 'NHWC'} ${asUint8 ? 'uint8' : 'float32'}`,
    outShape: (model as any).outputs?.[0]?.shape as number[] | undefined,
    loadMs,
    run: async (buffer) => new Float32Array((await model.run([buffer]))[0] as ArrayBuffer),
  };
}

async function loadOnnx(entry: ModelEntry): Promise<Backend> {
  // onnxruntime butuh path file, jadi aset disalin dulu ke penyimpanan lokal oleh expo-asset.
  const [asset] = await Asset.loadAsync(entry.source);
  const tl = now();
  const session = await InferenceSession.create(asset.localUri ?? asset.uri);
  const loadMs = now() - tl;
  const inName = session.inputNames[0];
  const outName = session.outputNames[0];
  const b: Backend = {
    entry,
    engineName: 'ONNX Runtime',
    nchw: true,
    asUint8: false,
    inputInfo: `[1, 3, ${INPUT_SIZE}, ${INPUT_SIZE}] float32 -> dikirim NCHW float32`,
    outShape: undefined,
    loadMs,
    run: async (buffer) => {
      const t = new Tensor('float32', new Float32Array(buffer), [1, 3, INPUT_SIZE, INPUT_SIZE]);
      const res = await session.run({ [inName]: t });
      const o = res[outName];
      b.outShape = o.dims;
      return o.data as Float32Array;
    },
    dispose: () => {
      session.release().catch(() => {});
    },
  };
  return b;
}

async function loadPte(entry: ModelEntry): Promise<Backend> {
  const m = new ExecutorchModule();
  const tl = now();
  await m.load(entry.source);
  const loadMs = now() - tl;
  const b: Backend = {
    entry,
    engineName: 'ExecuTorch (PyTorch)',
    nchw: true,
    asUint8: false,
    inputInfo: `[1, 3, ${INPUT_SIZE}, ${INPUT_SIZE}] float32 -> dikirim NCHW float32`,
    outShape: undefined,
    loadMs,
    run: async (buffer) => {
      const out = await m.forward([
        {
          dataPtr: new Float32Array(buffer),
          sizes: [1, 3, INPUT_SIZE, INPUT_SIZE],
          scalarType: ScalarType.FLOAT,
        },
      ]);
      const o = out[0] as any;
      b.outShape = o.sizes;
      const d = o.dataPtr;
      if (d instanceof Float32Array) return d;
      if (ArrayBuffer.isView(d)) return new Float32Array(d.buffer, d.byteOffset, d.byteLength / 4);
      return new Float32Array(d as ArrayBuffer);
    },
    dispose: () => {
      try {
        m.delete();
      } catch {
        /* sudah dilepas */
      }
    },
  };
  return b;
}

/** Ukuran file model dalam MB (bytes / 1048576). Mengembalikan undefined kalau tidak terbaca. */
async function modelSizeMb(source: number): Promise<number | undefined> {
  try {
    const [asset] = await Asset.loadAsync(source);
    const uri = asset.localUri ?? asset.uri;
    const bytes = new File(uri).size ?? 0;
    return bytes > 0 ? bytes / 1048576 : undefined;
  } catch {
    return undefined;
  }
}

let cache: { id: string; p: Promise<Backend> } | null = null;

export function loadBackend(entry: ModelEntry): Promise<Backend> {
  if (cache && cache.id === entry.id) return cache.p;
  const load =
    entry.format === 'onnx' ? loadOnnx : entry.format === 'pte' ? loadPte : loadTflite;
  const p = (async () => {
    const b = await load(entry);
    b.modelMb = await modelSizeMb(entry.source);
    return b;
  })().catch((e) => {
    if (cache && cache.id === entry.id) cache = null;
    throw e;
  });
  const prev = cache;
  cache = { id: entry.id, p };
  prev?.p.then((b) => b.dispose?.()).catch(() => {});
  return p;
}

// ---------- Dataset tes & hasil di penyimpanan INTERNAL app ----------
// Lokasi di HP: /data/user/0/<applicationId>/files/   (= Paths.document)
//   test/Diabetes/*.jpg, test/Nondiabetes/*.jpg, test/Pradiabetes/*.jpg
//     <- adb push ke /data/local/tmp, lalu `run-as <pkg> cp -r ... files/test`
//   results/<stem>/<stem>.txt, <stem>_detail.csv, <stem>_ringkasan.csv
//     <- tarik dengan `adb exec-out run-as <pkg> cat files/results/...`
// (Folder /sdcard/Android/data/... tidak terbaca oleh app di perangkat ini, makanya pindah.)
export type TestImage = { cls: number; name: string; uri: string };
const CLASS_FOLDERS = ['Diabetes', 'Nondiabetes', 'Pradiabetes']; // urutan = model.names

const root = () => Paths.document;
export const deviceDir = () => `/data/user/0/${Application.applicationId}/files`;

let listDebug = '';
/** Catatan diagnosis dari pemanggilan listTestImages() terakhir (untuk ditampilkan/di-log). */
export const getListDebug = () => listDebug;

export function listTestImages(): TestImage[] {
  const logs: string[] = [];
  const log = (s: string) => {
    logs.push(s);
    console.warn('LTI ' + s);
  };
  log(`applicationId=${Application.applicationId} root=${root().uri}`);

  const out: TestImage[] = [];
  CLASS_FOLDERS.forEach((folder, cls) => {
    try {
      const dir = new Directory(root(), 'test', folder);
      const exists = dir.exists;
      log(`${folder} uri=${dir.uri} exists=${exists}`);
      if (!exists) return;
      const entries = dir.list();
      const files = entries
        .filter((e) => e instanceof File && /\.(jpe?g|png)$/i.test(e.name))
        .sort((a, b) => a.name.localeCompare(b.name));
      log(`${folder} entries=${entries.length} gambar=${files.length}`);
      for (const f of files) out.push({ cls, name: `${folder}/${f.name}`, uri: f.uri });
    } catch (e) {
      log(`GAGAL ${folder}: ${String(e)}`);
    }
  });
  // Urutan sama dengan CSV Flutter/Kotlin: diurutkan global menurut nama file (ASCII).
  const base = (t: TestImage) => t.name.slice(t.name.indexOf('/') + 1);
  out.sort((a, b) => (base(a) < base(b) ? -1 : base(a) > base(b) ? 1 : 0));
  listDebug = logs.join('\n');
  return out;
}

const stampNow = () => new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);

function resultsDir(): Directory {
  const dir = new Directory(root(), 'results');
  if (!dir.exists) dir.create({ intermediates: true });
  return dir;
}

function writeText(dir: Directory, name: string, text: string) {
  const f = new File(dir, name);
  if (f.exists) f.delete();
  f.create();
  f.write(text);
}

/**
 * Menulis 3 file ke results/<stem>/ : <stem>.txt, <stem>_detail.csv, <stem>_ringkasan.csv
 * (satu folder per kombinasi, format sama dengan Flutter). Isi .txt dibuat oleh batchToTxt(),
 * jadi parameter `_summary` (teks untuk layar) tidak dipakai di file.
 * Mengembalikan path detail.csv di HP.
 */
export function saveResults(r: BatchResult, _summary: string): string {
  const stem = `reactnative_${r.modelKey}_${r.format}`;
  const dir = new Directory(resultsDir(), stem);
  if (!dir.exists) dir.create({ intermediates: true });
  writeText(dir, `${stem}.txt`, batchToTxt(r));
  writeText(dir, `${stem}_detail.csv`, batchToCsv(r));
  writeText(dir, `${stem}_ringkasan.csv`, batchToSummaryCsv(r));
  return `${deviceDir()}/results/${stem}/${stem}_detail.csv`;
}

/** Menulis ringkasan semua kombinasi ke results/ringkasan_<waktu>.csv. */
export function saveSummary(csv: string): string {
  const name = `ringkasan_${stampNow()}.csv`;
  writeText(resultsDir(), name, csv);
  return `${deviceDir()}/results/${name}`;
}

// ---------- Preprocessing: gambar -> buffer input model ----------
function base64ToBytes(b64: string): Uint8Array {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

/**
 * nchw = true  -> [1, 3, H, W], false -> [1, H, W, 3]
 * asUint8 = true -> nilai 0..255 (model terkuantisasi), false -> float32 0..1
 */
export async function preprocess(
  uri: string,
  nchw: boolean,
  asUint8: boolean
): Promise<ArrayBuffer> {
  const ctx = ImageManipulator.manipulate(uri);
  ctx.resize({ width: INPUT_SIZE, height: INPUT_SIZE });
  const ref = await ctx.renderAsync();
  const saved = await ref.saveAsync({ format: SaveFormat.PNG, base64: true });
  if (!saved.base64) throw new Error('Gagal membaca piksel gambar (base64 kosong).');

  const bytes = base64ToBytes(saved.base64);
  const png = UPNG.decode(bytes.buffer as ArrayBuffer);
  if (png.width !== INPUT_SIZE || png.height !== INPUT_SIZE) {
    throw new Error(`Ukuran gambar ${png.width}x${png.height}, harusnya ${INPUT_SIZE}x${INPUT_SIZE}.`);
  }
  const rgba = new Uint8Array(UPNG.toRGBA8(png)[0]);

  const n = INPUT_SIZE * INPUT_SIZE;
  const out: Float32Array | Uint8Array = asUint8 ? new Uint8Array(n * 3) : new Float32Array(n * 3);
  const k = asUint8 ? 1 : 1 / 255;
  for (let i = 0; i < n; i++) {
    const r = rgba[i * 4] * k;
    const g = rgba[i * 4 + 1] * k;
    const b = rgba[i * 4 + 2] * k;
    if (nchw) {
      out[i] = r;
      out[n + i] = g;
      out[2 * n + i] = b;
    } else {
      out[i * 3] = r;
      out[i * 3 + 1] = g;
      out[i * 3 + 2] = b;
    }
  }
  return out.buffer as ArrayBuffer;
}

// ---------- Postprocess: output mentah -> label + confidence ----------
function classPrediction(out: Float32Array): Prediction {
  const scores = Array.from(out);
  const sum = scores.reduce((a, b) => a + b, 0);
  const isProb = scores.every((v) => v >= 0 && v <= 1) && Math.abs(sum - 1) < 0.01;
  let probs = scores;
  if (!isProb) {
    const max = Math.max(...scores);
    const e = scores.map((v) => Math.exp(v - max));
    const total = e.reduce((a, b) => a + b, 0);
    probs = e.map((v) => v / total);
  }
  let idx = 0;
  for (let c = 1; c < probs.length; c++) if (probs[c] > probs[idx]) idx = c;
  return {
    label: LABELS[idx] ?? `kelas-${idx}`,
    index: idx,
    confidence: probs[idx],
    scores: probs,
    detections: 1,
  };
}

export const IOU_THRESHOLD = 0.45; // NMS (per kelas)

type Det = { x1: number; y1: number; x2: number; y2: number; score: number; cls: number };

function iou(a: Det, b: Det): number {
  const w = Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1);
  const h = Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1);
  if (w <= 0 || h <= 0) return 0;
  const inter = w * h;
  const ua = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
  return ua > 0 ? inter / ua : 0;
}

/**
 * Output model deteksi YOLO: [4 + jumlahKelas, N] (channels-first) atau [N, 4 + jumlahKelas].
 * 4 nilai pertama = kotak (cx, cy, w, h), sisanya = skor tiap kelas. Kandidat dengan skor
 * >= CONF_THRESHOLD disaring NMS per kelas (IoU IOU_THRESHOLD); prediksi = kotak teratas
 * (asumsi: satu lidah per foto), "Jumlah Deteksi" = kotak yang tersisa setelah NMS.
 */
export function toPrediction(out: Float32Array, shape?: readonly number[]): Prediction {
  const nc = LABELS.length;
  if (out.length === nc) return classPrediction(out);

  const channels = 4 + nc;
  if (out.length % channels !== 0) {
    throw new Error(
      `Ukuran output ${out.length} tidak cocok untuk ${nc} kelas (harus kelipatan ${channels}).`
    );
  }
  const n = out.length / channels;

  let channelsFirst = true;
  if (shape && shape.length >= 2) {
    const a = shape[shape.length - 2];
    const b = shape[shape.length - 1];
    if (a !== channels && b === channels) channelsFirst = false;
  }
  const at = (ch: number, i: number) => (channelsFirst ? out[ch * n + i] : out[i * channels + ch]);

  const best = new Array<number>(nc).fill(0); // skor tertinggi tiap kelas
  const dets: Det[] = [];
  let bestScore = 0;
  for (let i = 0; i < n; i++) {
    let c = 0;
    let sc = at(4, i);
    if (sc > best[0]) best[0] = sc;
    for (let k = 1; k < nc; k++) {
      const v = at(4 + k, i);
      if (v > best[k]) best[k] = v;
      if (v > sc) {
        sc = v;
        c = k;
      }
    }
    if (sc > bestScore) bestScore = sc;
    if (sc >= CONF_THRESHOLD) {
      const cx = at(0, i);
      const cy = at(1, i);
      const w = at(2, i);
      const h = at(3, i);
      dets.push({ x1: cx - w / 2, y1: cy - h / 2, x2: cx + w / 2, y2: cy + h / 2, score: sc, cls: c });
    }
  }

  dets.sort((a, b) => b.score - a.score);
  const kept: Det[] = [];
  for (const d of dets) {
    let ok = true;
    for (const k of kept) {
      if (k.cls === d.cls && iou(k, d) > IOU_THRESHOLD) {
        ok = false;
        break;
      }
    }
    if (ok) kept.push(d);
  }

  if (kept.length === 0) {
    return { label: 'tidak terdeteksi', index: -1, confidence: bestScore, scores: best, detections: 0 };
  }
  const top = kept[0];
  return {
    label: LABELS[top.cls],
    index: top.cls,
    confidence: top.score,
    scores: best,
    detections: kept.length,
  };
}

// ---------- Mode satu gambar: pipeline lengkap + benchmark ----------
export async function runPipeline(
  b: Backend,
  uri: string,
  warmup = 5,
  runs = 50
): Promise<BenchResult> {
  const t0 = now();
  const buffer = await preprocess(uri, b.nchw, b.asUint8);
  const preprocessMs = now() - t0;

  const t1 = now();
  let last = await b.run(buffer);
  const firstRunMs = now() - t1;

  for (let i = 0; i < warmup; i++) last = await b.run(buffer);

  const times: number[] = [];
  for (let i = 0; i < runs; i++) {
    const s = now();
    last = await b.run(buffer);
    times.push(now() - s);
  }
  const sorted = [...times].sort((a, c) => a - c);
  const avgMs = times.reduce((a, c) => a + c, 0) / times.length;

  return {
    engine: b.engineName,
    model: `${b.entry.model}.${b.entry.format}`,
    preprocessMs,
    firstRunMs,
    avgMs,
    medianMs: sorted[Math.floor(sorted.length / 2)],
    minMs: sorted[0],
    maxMs: sorted[sorted.length - 1],
    inputInfo: b.inputInfo,
    outputShape: b.outShape ? `[${b.outShape.join(', ')}]` : `${last.length} nilai`,
    prediction: toPrediction(last, b.outShape),
  };
}

// ---------- Mode batch: seluruh gambar tes -> confusion matrix ----------
export type BatchRow = {
  round: number; // putaran ke-n (0-based)
  name: string; // <Kelas>/<nama file>
  truth: number;
  pred: number; // -1 = tidak terdeteksi
  confidence: number;
  detections: number;
  preprocessMs: number;
  inferMs: number;
  postMs: number;
  e2eMs: number; // dari awal memuat gambar sampai prediksi jadi
};

export type BatchResult = {
  engine: string;
  model: string; // mis. "best8n.tflite"
  modelKey: string; // mis. "best8n"
  format: string; // tflite | onnx | pte
  startedAt: string; // "YYYY-MM-DD HH:mm:ss" (waktu lokal)
  total: number;
  correct: number;
  accuracy: number;
  matrix: number[][]; // [asli][prediksi], kolom terakhir = tidak terdeteksi
  perClass: { precision: number; recall: number; f1: number }[];
  macroF1: number;
  avgInferMs: number;
  medianInferMs: number;
  avgPreprocessMs: number;
  repeats: number; // jumlah putaran seluruh gambar
  stdInferMs: number; // simpangan baku waktu inferensi (semua putaran)
  roundAvgMs: number[]; // rata-rata inferensi per putaran
  rows: BatchRow[];
  // ---- tambahan untuk format ringkasan Flutter ----
  avgPostMs: number;
  avgTotalMs: number; // pre + inf + post
  medianTotalMs: number;
  stdTotalMs: number; // simpangan baku (sampel) total, semua pengukuran
  minTotalMs: number;
  maxTotalMs: number;
  roundTotalAvgMs: number[]; // rata-rata total per putaran
  stdBetweenRounds: number; // simpangan baku (sampel) antar rata-rata putaran
  avgE2eMs: number;
  coldStartMs?: number;
  modelLoadMs?: number;
  modelSizeMb?: number;
  tensorInfo: string;
};

const pad2 = (v: number) => String(v).padStart(2, '0');
const localStamp = (d: Date) =>
  `${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())} ` +
  `${pad2(d.getHours())}:${pad2(d.getMinutes())}:${pad2(d.getSeconds())}`;

const meanOf = (a: number[]) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0);
/** Simpangan baku sampel (n-1), sama dengan angka Flutter (mis. 446.15/443.80/451.12 -> 3.74). */
const sampleStd = (a: number[]) => {
  if (a.length < 2) return 0;
  const m = meanOf(a);
  return Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / (a.length - 1));
};
const medianOf = (a: number[]) => {
  if (!a.length) return 0;
  const s = [...a].sort((x, y) => x - y);
  return s[Math.floor(s.length / 2)];
};

export async function runBatch(
  b: Backend,
  images: TestImage[],
  onProgress?: (done: number, total: number) => void,
  repeats = 3
): Promise<BatchResult> {
  const nc = LABELS.length;
  const matrix = Array.from({ length: nc }, () => new Array<number>(nc + 1).fill(0));
  const rows: BatchRow[] = [];
  const startedAt = localStamp(new Date());

  // Warm-up sekali di awal supaya gambar pertama tidak mengotori rata-rata waktu.
  // Inferensi pertama setelah model dimuat dicatat sebagai "cold start" (hanya sekali per model).
  const wbuf = await preprocess(images[0].uri, b.nchw, b.asUint8);
  if (b.coldStartMs === undefined) {
    const tc = now();
    await b.run(wbuf);
    b.coldStartMs = now() - tc;
    for (let i = 0; i < 2; i++) await b.run(wbuf);
  } else {
    for (let i = 0; i < 3; i++) await b.run(wbuf);
  }

  // Seluruh gambar dijalankan `repeats` putaran (mis. 94 x 3). Tiap putaran memuat ulang gambar
  // dan menjalankan preprocessing + inferensi + postprocess penuh. Confusion matrix & akurasi
  // dari putaran pertama saja (prediksi deterministik), waktu dari semua putaran.
  const totalSteps = images.length * repeats;
  let done = 0;
  for (let round = 0; round < repeats; round++) {
    for (let i = 0; i < images.length; i++) {
      const img = images[i];
      const tIter = now();
      const buf = await preprocess(img.uri, b.nchw, b.asUint8);
      const preprocessMs = now() - tIter;
      const t1 = now();
      const out = await b.run(buf);
      const inferMs = now() - t1;
      const t2 = now();
      const pred = toPrediction(out, b.outShape);
      const postMs = now() - t2;
      const e2eMs = now() - tIter;
      if (round === 0) matrix[img.cls][pred.index < 0 ? nc : pred.index]++;
      rows.push({
        round,
        name: img.name,
        truth: img.cls,
        pred: pred.index,
        confidence: pred.confidence,
        detections: pred.detections,
        preprocessMs,
        inferMs,
        postMs,
        e2eMs,
      });
      onProgress?.(++done, totalSteps);
    }
  }

  const first = rows.filter((r) => r.round === 0);
  const total = first.length;
  const correct = first.filter((r) => r.truth === r.pred).length;
  const perClass = LABELS.map((_, c) => {
    const tp = matrix[c][c];
    const rowSum = matrix[c].reduce((a, v) => a + v, 0);
    const colSum = matrix.reduce((a, r) => a + r[c], 0);
    const precision = colSum ? tp / colSum : 0;
    const recall = rowSum ? tp / rowSum : 0;
    const f1 = precision + recall ? (2 * precision * recall) / (precision + recall) : 0;
    return { precision, recall, f1 };
  });

  const inferAll = rows.map((r) => r.inferMs);
  const avgInfer = meanOf(inferAll);
  const stdInfer = Math.sqrt(meanOf(inferAll.map((x) => (x - avgInfer) ** 2)));
  const roundAvgMs = Array.from({ length: repeats }, (_, k) =>
    meanOf(rows.filter((r) => r.round === k).map((r) => r.inferMs))
  );

  // Statistik "total" (pre + inf + post) seperti di Flutter.
  const totals = rows.map((r) => r.preprocessMs + r.inferMs + r.postMs);
  const roundTotalAvgMs = Array.from({ length: repeats }, (_, k) =>
    meanOf(rows.filter((r) => r.round === k).map((r) => r.preprocessMs + r.inferMs + r.postMs))
  );

  const inDims = b.nchw ? [1, 3, INPUT_SIZE, INPUT_SIZE] : [1, INPUT_SIZE, INPUT_SIZE, 3];
  const tensorInfo =
    `Input [${inDims.join(', ')}] ${b.asUint8 ? 'uint8' : 'float32'} | ` +
    `Output ${b.outShape ? `[${b.outShape.join(', ')}]` : '?'} float32`;

  return {
    engine: b.engineName,
    model: `${b.entry.model}.${b.entry.format}`,
    modelKey: b.entry.model,
    format: b.entry.format,
    startedAt,
    total,
    correct,
    accuracy: total ? correct / total : 0,
    matrix,
    perClass,
    macroF1: perClass.reduce((a, p) => a + p.f1, 0) / nc,
    avgInferMs: avgInfer,
    medianInferMs: medianOf(inferAll),
    avgPreprocessMs: meanOf(rows.map((r) => r.preprocessMs)),
    repeats,
    stdInferMs: stdInfer,
    roundAvgMs,
    rows,
    avgPostMs: meanOf(rows.map((r) => r.postMs)),
    avgTotalMs: meanOf(totals),
    medianTotalMs: medianOf(totals),
    stdTotalMs: sampleStd(totals),
    minTotalMs: Math.min(...totals),
    maxTotalMs: Math.max(...totals),
    roundTotalAvgMs,
    stdBetweenRounds: sampleStd(roundTotalAvgMs),
    avgE2eMs: meanOf(rows.map((r) => r.e2eMs)),
    coldStartMs: b.coldStartMs,
    modelLoadMs: b.loadMs,
    modelSizeMb: b.modelMb,
    tensorInfo,
  };
}

// Nama tampilan & engine di CSV mengikuti file contoh Flutter/Kotlin.
const MODEL_DISPLAY: Record<string, string> = {
  best8n: 'YOLOv8n',
  best11n: 'YOLO11n',
  best11s: 'YOLO11s',
};
const CSV_ENGINE: Record<string, string> = {
  onnx: 'ONNX Runtime',
  pte: 'ExecuTorch (PyTorch)',
  tflite: 'TensorFlow Lite',
};

const csvCell = (v: string | number) => {
  const t = String(v);
  return /[",\n]/.test(t) ? `"${t.replace(/"/g, '""')}"` : t;
};

/**
 * CSV detail: SATU baris per gambar, kolom sama dengan Flutter/Kotlin. Waktu = rata-rata dari
 * semua putaran; prediksi, confidence, dan jumlah deteksi dari putaran pertama.
 */
export function batchToCsv(r: BatchResult): string {
  const head =
    'Model,Format,Gambar,Label Asli,Kelas Prediksi,Cocok,Jumlah Deteksi,Confidence,' +
    'Waktu Preprocess (ms),Waktu Inference (ms),Waktu Postprocess (ms),Waktu Total (ms),' +
    'Waktu End-to-End (ms),Waktu Tes,Framework,Engine';
  const groups = new Map<string, BatchRow[]>();
  for (const x of r.rows) {
    const g = groups.get(x.name);
    if (g) g.push(x);
    else groups.set(x.name, [x]);
  }
  const f2 = (v: number) => v.toFixed(2);
  const lines = Array.from(groups.values()).map((g) => {
    const x = g[0];
    const pre = meanOf(g.map((v) => v.preprocessMs));
    const inf = meanOf(g.map((v) => v.inferMs));
    const post = meanOf(g.map((v) => v.postMs));
    const e2e = meanOf(g.map((v) => v.e2eMs));
    return [
      MODEL_DISPLAY[r.modelKey] ?? r.modelKey,
      r.format,
      x.name.slice(x.name.indexOf('/') + 1),
      LABELS[x.truth],
      x.pred < 0 ? 'Tidak terdeteksi' : LABELS[x.pred],
      x.truth === x.pred ? 'Ya' : 'Tidak',
      x.detections,
      x.confidence.toFixed(4),
      f2(pre),
      f2(inf),
      f2(post),
      f2(pre + inf + post),
      f2(e2e),
      r.startedAt,
      'React Native',
      CSV_ENGINE[r.format] ?? r.engine,
    ]
      .map(csvCell)
      .join(',');
  });
  return [head, ...lines].join('\n');
}

/**
 * Ringkasan satu kombinasi, 29 kolom persis seperti flutter_*_ringkasan.csv.
 * Ukuran APK dan RAM tidak bisa diukur dari JS, jadi sel dibiarkan kosong.
 */
export function batchToSummaryCsv(r: BatchResult): string {
  const head =
    'Framework,Engine,Model,Format,Jumlah Gambar,Ulangan,Accuracy (%),Precision Macro,Recall Macro,' +
    'F1 Macro,Preprocess Avg (ms),Inference Avg (ms),Postprocess Avg (ms),Total Avg (ms),' +
    'Total Median (ms),Total Std (ms),Std Antar Ulangan (ms),Total Min (ms),Total Max (ms),' +
    'End-to-End Avg (ms),Cold Start (ms),Model Load (ms),Ukuran Model (MB),Ukuran APK (MB),' +
    'RAM Sebelum Load (MB),RAM Setelah Load (MB),RAM Puncak (MB),Tambahan RAM Puncak (MB),Waktu Tes';
  const f2 = (v: number | undefined) => (v === undefined || !isFinite(v) ? '' : v.toFixed(2));
  const f4 = (v: number) => v.toFixed(4);
  const macroP = meanOf(r.perClass.map((c) => c.precision));
  const macroR = meanOf(r.perClass.map((c) => c.recall));
  const row = [
    'React Native',
    CSV_ENGINE[r.format] ?? r.engine,
    MODEL_DISPLAY[r.modelKey] ?? r.modelKey,
    r.format,
    r.total,
    r.repeats,
    f2(r.accuracy * 100),
    f4(macroP),
    f4(macroR),
    f4(r.macroF1),
    f2(r.avgPreprocessMs),
    f2(r.avgInferMs),
    f2(r.avgPostMs),
    f2(r.avgTotalMs),
    f2(r.medianTotalMs),
    f2(r.stdTotalMs),
    f2(r.stdBetweenRounds),
    f2(r.minTotalMs),
    f2(r.maxTotalMs),
    f2(r.avgE2eMs),
    f2(r.coldStartMs),
    f2(r.modelLoadMs),
    f2(r.modelSizeMb),
    '', // Ukuran APK (MB)
    '', // RAM Sebelum Load (MB)
    '', // RAM Setelah Load (MB)
    '', // RAM Puncak (MB)
    '', // Tambahan RAM Puncak (MB)
    r.startedAt,
  ]
    .map(csvCell)
    .join(',');
  return `${head}\n${row}`;
}

/** Teks ringkasan dengan susunan baris yang sama seperti flutter_*.txt. */
export function batchToTxt(r: BatchResult): string {
  const f2 = (v: number | undefined) => (v === undefined || !isFinite(v) ? '-' : v.toFixed(2));
  const f3 = (v: number) => v.toFixed(3);
  const display = MODEL_DISPLAY[r.modelKey] ?? r.modelKey;
  const macroP = meanOf(r.perClass.map((c) => c.precision));
  const macroR = meanOf(r.perClass.map((c) => c.recall));
  const matrixLines = r.matrix.map((row, i) => `${LABELS[i]}: ${row.join(', ')}`).join('\n');
  const classLines = r.perClass
    .map(
      (p, i) =>
        `${LABELS[i]}: P=${f3(p.precision)} R=${f3(p.recall)} F1=${f3(p.f1)}`
    )
    .join('\n');
  return (
    `Model: ${r.modelKey} (${display}) | Format: ${r.format} | ` +
    `Engine: ${CSV_ENGINE[r.format] ?? r.engine} | Framework: React Native\n` +
    `Tensor: ${r.tensorInfo} | Threads -\n` +
    `Gambar: ${r.total} | Ulangan: ${r.repeats} | Pengukuran: ${r.rows.length}\n\n` +
    `Accuracy: ${r.correct}/${r.total} = ${f2(r.accuracy * 100)}%\n` +
    `Precision/Recall/F1 (macro): ${f3(macroP)} / ${f3(macroR)} / ${f3(r.macroF1)}\n\n` +
    `Waktu rata-rata (ms): pre=${f2(r.avgPreprocessMs)} inf=${f2(r.avgInferMs)} ` +
    `post=${f2(r.avgPostMs)} total=${f2(r.avgTotalMs)}\n` +
    `Total: median=${f2(r.medianTotalMs)} std=${f2(r.stdTotalMs)} ` +
    `min=${f2(r.minTotalMs)} max=${f2(r.maxTotalMs)}\n` +
    `Rata-rata tiap ulangan (ms): ${r.roundTotalAvgMs.map((v) => f2(v)).join(' / ')}\n` +
    `Std antar ulangan: ${f2(r.stdBetweenRounds)} | End-to-end avg: ${f2(r.avgE2eMs)}\n` +
    `Cold start: ${f2(r.coldStartMs)} ms | Model load: ${f2(r.modelLoadMs)} ms\n\n` +
    `Ukuran model: ${f2(r.modelSizeMb)} MB | Ukuran APK: - MB\n` +
    `RAM (PSS): sebelum load=- setelah load=- puncak=- MB\n\n` +
    `Confusion matrix (baris=truth, kolom=pred: ${LABELS.join(', ')}, tidak terdeteksi):\n` +
    `${matrixLines}\n\n` +
    `Per kelas:\n${classLines}\n`
  );
}

/** CSV mentah: satu baris per run (gambar x putaran), untuk analisis variasi waktu. */
export function batchToRunsCsv(r: BatchResult): string {
  const head =
    'Model,Format,Putaran,Gambar,Label Asli,Kelas Prediksi,Jumlah Deteksi,Confidence,' +
    'Preprocess (ms),Inference (ms),Postprocess (ms),End-to-End (ms)';
  const lines = r.rows.map((x) =>
    [
      MODEL_DISPLAY[r.modelKey] ?? r.modelKey,
      r.format,
      x.round + 1,
      x.name.slice(x.name.indexOf('/') + 1),
      LABELS[x.truth],
      x.pred < 0 ? 'Tidak terdeteksi' : LABELS[x.pred],
      x.detections,
      x.confidence.toFixed(4),
      x.preprocessMs.toFixed(2),
      x.inferMs.toFixed(2),
      x.postMs.toFixed(2),
      x.e2eMs.toFixed(2),
    ]
      .map(csvCell)
      .join(',')
  );
  return [head, ...lines].join('\n');
}