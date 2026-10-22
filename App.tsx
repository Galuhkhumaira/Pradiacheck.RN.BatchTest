import React, { useEffect, useMemo, useState } from 'react';
import {
  Alert,
  Image,
  ImageBackground,
  Pressable,
  ScrollView,
  Share,
  StatusBar,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import * as ImagePicker from 'expo-image-picker';
import { ImageManipulator } from 'expo-image-manipulator';
import {
  Backend,
  BatchResult,
  BenchResult,
  LABELS,
  SHORT,
  batchToCsv,
  listTestImages,
  loadBackend,
  runBatch,
  saveResults,
  saveSummary,
  deviceDir,
  getListDebug,
  runPipeline,
} from './engine';
import { MODELS } from './registry';

// Warna sama dengan versi Kotlin (perkiraan dari screenshot Flutter).
const C = {
  bg: '#FCF0EE',
  textDark: '#3B1A1F',
  textMuted: '#8A6A6A',
  textHint: '#B09A9A',
  cardBorder: '#F0DCDA',
  accent: '#7A4A4F',
  disabledBg: '#E0D6D4',
  disabledText: '#9A8F8F',
};

const fmt = (x: number) => x.toFixed(1);
const pct = (x: number) => `${(x * 100).toFixed(1)}%`;
const uniq = (a: string[]) => Array.from(new Set(a));

function formatSingle(r: BenchResult): string {
  const skor = r.prediction.scores
    .map((s, i) => `${LABELS[i]} ${(s * 100).toFixed(1)}%`)
    .join(' | ');
  return (
    `${r.prediction.label} (${pct(r.prediction.confidence)})\n` +
    `Skor kelas: ${skor}\n` +
    `Engine: ${r.engine}\n` +
    `Model: ${r.model}\n` +
    `Waktu inferensi: rata-rata ${fmt(r.avgMs)} ms, median ${fmt(r.medianMs)} ms\n` +
    `Run pertama: ${fmt(r.firstRunMs)} ms\n` +
    `Preprocessing: ${fmt(r.preprocessMs)} ms\n` +
    `Input: ${r.inputInfo}\n` +
    `Output shape: ${r.outputShape}`
  );
}

function formatBatch(r: BatchResult): string {
  const pad = (s: string | number, w: number) => String(s).padStart(w, ' ');
  const header = `${pad('', 13)}${SHORT.map((s) => pad(s, 5)).join('')}${pad('Tdk', 5)}`;
  const matrix = r.matrix
    .map((row, i) => `${LABELS[i].padEnd(13, ' ')}${row.map((v) => pad(v, 5)).join('')}`)
    .join('\n');
  const perClass = r.perClass
    .map(
      (p, i) =>
        `${LABELS[i].padEnd(13, ' ')}P ${pct(p.precision)}  R ${pct(p.recall)}  F1 ${pct(p.f1)}`
    )
    .join('\n');
  return (
    `Model: ${r.model}\n` +
    `Engine: ${r.engine}\n` +
    `Akurasi: ${r.correct}/${r.total} = ${pct(r.accuracy)}  (macro-F1 ${pct(r.macroF1)})\n` +
    `Putaran: ${r.repeats}x ${r.total} gambar = ${r.rows.length} run\n` +
    `Inferensi: rata-rata ${fmt(r.avgInferMs)} ms (±${fmt(r.stdInferMs)}), median ${fmt(r.medianInferMs)} ms\n` +
    `Rata-rata per putaran: ${r.roundAvgMs.map(fmt).join(' | ')} ms\n` +
    `Preprocessing: rata-rata ${fmt(r.avgPreprocessMs)} ms\n\n` +
    `Confusion matrix (baris = asli, kolom = prediksi, Tdk = tidak terdeteksi)\n` +
    `${header}\n${matrix}\n\n${perClass}`
  );
}

export default function App() {
  const modelNames = useMemo(() => uniq(MODELS.map((m) => m.model)), []);
  const formats = useMemo(() => uniq(MODELS.map((m) => m.format)), []);

  const [modelName, setModelName] = useState(modelNames[0] ?? '');
  const [format, setFormat] = useState(formats[0] ?? '');
  const entry = MODELS.find((m) => m.model === modelName && m.format === format) ?? null;

  const [mode, setMode] = useState<'single' | 'batch'>('single');
  const [imageUri, setImageUri] = useState<string | null>(null);
  const [result, setResult] = useState<string | null>(null);
  const [batch, setBatch] = useState<BatchResult | null>(null);
  const [testCount, setTestCount] = useState(0);
  const [mono, setMono] = useState(false);
  const [repeats, setRepeats] = useState(3);
  const [progress, setProgress] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [backend, setBackend] = useState<Backend | null>(null);
  const [engineError, setEngineError] = useState<string | null>(null);

  // Muat ulang model setiap kali pilihan model/format berubah.
  useEffect(() => {
    let cancelled = false;
    setBackend(null);
    setEngineError(null);
    setResult(null);
    setBatch(null);
    if (!entry) return;
    loadBackend(entry)
      .then((b) => !cancelled && setBackend(b))
      .catch((e) => !cancelled && setEngineError(`Gagal memuat model: ${e?.message ?? e}`));
    return () => {
      cancelled = true;
    };
  }, [entry?.id]);

  // Hitung ulang gambar tes setiap masuk mode batch (jadi tidak perlu restart app setelah adb push).
  useEffect(() => {
    if (mode === 'batch') setTestCount(listTestImages().length);
  }, [mode]);

  // ===== TES SEMENTARA: apakah app bisa membaca gambar hasil adb push? (hapus setelah dicoba) =====
  useEffect(() => {
    const uri =
      'file:///storage/emulated/0/Android/data/com.anonymous.PradiacheckRN/files/test/Diabetes/1717188223926_jpg.rf.4351f37394bccd3e58e9817307481f3f.jpg';
    ImageManipulator.manipulate(uri)
      .resize({ width: 64, height: 64 })
      .renderAsync()
      .then(() => Alert.alert('BISA DIBACA'))
      .catch((e) => Alert.alert('GAGAL', String(e)));
  }, []);
  // ===== akhir tes sementara =====

  const pickFromGallery = async () => {
    const res = await ImagePicker.launchImageLibraryAsync({ mediaTypes: ['images'], quality: 1 });
    if (!res.canceled) {
      setImageUri(res.assets[0].uri);
      setResult(null);
    }
  };

  const takePhoto = async () => {
    const perm = await ImagePicker.requestCameraPermissionsAsync();
    if (!perm.granted) {
      Alert.alert('Izin kamera diperlukan', 'Aktifkan izin kamera di pengaturan HP.');
      return;
    }
    const res = await ImagePicker.launchCameraAsync({ quality: 1 });
    if (!res.canceled) {
      setImageUri(res.assets[0].uri);
      setResult(null);
    }
  };

  const chooseSource = () =>
    Alert.alert('Pilih sumber gambar', undefined, [
      { text: 'Kamera', onPress: takePhoto },
      { text: 'Galeri', onPress: pickFromGallery },
      { text: 'Batal', style: 'cancel' },
    ]);

  const onCheck = async () => {
    if (!imageUri || !backend) return;
    setMono(false);
    setRunning(true);
    try {
      setResult(formatSingle(await runPipeline(backend, imageUri)));
    } catch (e: any) {
      setResult(`Error inferensi: ${e?.message ?? e}`);
    } finally {
      setRunning(false);
    }
  };

  const onBatch = async () => {
    if (!backend) return;
    const images = listTestImages();
    setTestCount(images.length);
    if (images.length === 0) {
      setResult(`Belum ada gambar tes. Push dataset ke:\n${deviceDir()}/test/<Kelas>/\n\n${getListDebug()}`);
      return;
    }
    setRunning(true);
    setBatch(null);
    setResult(null);
    try {
      const r = await runBatch(backend, images, (d, t) => setProgress(`Memproses ${d}/${t}...`), repeats);
      const summary = formatBatch(r);
      let saved = '';
      try {
        saved = `\n\nTersimpan: ${saveResults(r, summary)}`;
      } catch (e: any) {
        saved = `\n\nGagal menyimpan file: ${e?.message ?? e}`;
      }
      setBatch(r);
      setResult(summary + saved);
    } catch (e: any) {
      setResult(`Error batch: ${e?.message ?? e}`);
    } finally {
      setProgress(null);
      setRunning(false);
    }
  };

  const onBatchAll = async () => {
    const images = listTestImages();
    setTestCount(images.length);
    if (images.length === 0) {
      setResult(`Belum ada gambar tes. Push dataset ke:\n${deviceDir()}/test/<Kelas>/\n\n${getListDebug()}`);
      return;
    }
    setRunning(true);
    setBatch(null);
    setMono(true);
    setResult(null);
    const csv = ['model,format,putaran,akurasi,macro_f1,avg_infer_ms,std_infer_ms,median_infer_ms,avg_preprocess_ms,status'];
    const view: string[] = [];
    try {
      for (let i = 0; i < MODELS.length; i++) {
        const e = MODELS[i];
        const tag = `[${i + 1}/${MODELS.length}] ${e.model}.${e.format}`;
        setProgress(`${tag} memuat...`);
        try {
          const b = await loadBackend(e);
          const r = await runBatch(b, images, (d, t) => setProgress(`${tag} ${d}/${t}`), repeats);
          try {
            saveResults(r, formatBatch(r));
          } catch {
            /* ringkasan tetap jalan walau file per kombinasi gagal ditulis */
          }
          csv.push(
            [e.model, e.format, r.repeats, r.accuracy.toFixed(4), r.macroF1.toFixed(4), fmt(r.avgInferMs), fmt(r.stdInferMs), fmt(r.medianInferMs), fmt(r.avgPreprocessMs), 'ok'].join(',')
          );
          view.push(`${(e.model + '.' + e.format).padEnd(16, ' ')}acc ${pct(r.accuracy)}  F1 ${pct(r.macroF1)}  ${fmt(r.avgInferMs)} ms`);
        } catch (err: any) {
          const msg = String(err?.message ?? err).replace(/[\r\n,]+/g, ' ');
          csv.push(`${e.model},${e.format},,,,,,,,ERROR: ${msg}`);
          view.push(`${(e.model + '.' + e.format).padEnd(16, ' ')}GAGAL: ${msg}`);
        }
        setResult(view.join('\n'));
      }
      let saved = '';
      try {
        saved = `\n\nTersimpan: ${saveSummary(csv.join('\n'))}`;
      } catch (e: any) {
        saved = `\n\nGagal menyimpan ringkasan: ${e?.message ?? e}`;
      }
      setResult(view.join('\n') + saved);
    } finally {
      setProgress(null);
      setRunning(false);
      // Model terakhir sudah dilepas saat ganti model; muat ulang pilihan di chip.
      setBackend(null);
      if (entry) loadBackend(entry).then(setBackend).catch(() => {});
    }
  };

  const onShare = () => {
    if (batch) Share.share({ message: batchToCsv(batch), title: `batch_${batch.model}.csv` });
  };

  const canCheck = imageUri !== null && backend !== null && !running;
  const canBatch = backend !== null && !running;

  const Chip = ({
    label,
    active,
    disabled,
    onPress,
  }: {
    label: string;
    active: boolean;
    disabled?: boolean;
    onPress: () => void;
  }) => (
    <Pressable
      onPress={onPress}
      disabled={disabled || running}
      style={[
        styles.chip,
        active && { backgroundColor: C.accent, borderColor: C.accent },
        disabled && { opacity: 0.35 },
      ]}
    >
      <Text style={[styles.chipText, active && { color: '#fff' }]}>{label}</Text>
    </Pressable>
  );

  return (
    <View style={styles.root}>
      <StatusBar barStyle="dark-content" backgroundColor={C.bg} />
      <ScrollView contentContainerStyle={styles.content}>
        {/* Kartu info */}
        <ImageBackground
          source={require('./assets/lidah.png')}
          resizeMode="cover"
          style={styles.header}
        >
          <LinearGradient
            colors={['rgba(0,0,0,0.8)', 'rgba(0,0,0,0.4)']}
            style={StyleSheet.absoluteFill}
          />
          <View style={styles.headerContent}>
            <View style={styles.headerRow}>
              <View style={styles.infoCircle}>
                <Ionicons name="information-circle-outline" size={26} color="#fff" />
              </View>
              <Text style={styles.headerLabel}>Tata Cara Pemakaian</Text>
            </View>
            <Text style={styles.headerTitle}>Deteksi Diabetes dari Citra Lidah</Text>
            <Text style={styles.headerSteps}>
              {'1. Ambil atau unggah gambar lidahmu.\n' +
                '2. Pastikan lidah terlihat jelas dan pencahayaan cukup.\n' +
                '3. Tekan tombol "Cek" untuk melihat hasilnya.'}
            </Text>
          </View>
        </ImageBackground>

        <Text style={styles.sectionTitle}>Deteksi Diabetes</Text>

        {/* Pemilih model, format, dan mode */}
        <Text style={styles.groupLabel}>Model</Text>
        <View style={styles.chipRow}>
          {modelNames.map((n) => (
            <Chip key={n} label={n} active={n === modelName} onPress={() => setModelName(n)} />
          ))}
        </View>
        <Text style={styles.groupLabel}>Format</Text>
        <View style={styles.chipRow}>
          {formats.map((f) => (
            <Chip
              key={f}
              label={f}
              active={f === format}
              disabled={!MODELS.some((m) => m.model === modelName && m.format === f)}
              onPress={() => setFormat(f)}
            />
          ))}
        </View>
        <Text style={styles.groupLabel}>Mode</Text>
        <View style={styles.chipRow}>
          <Chip label="Satu gambar" active={mode === 'single'} onPress={() => setMode('single')} />
          <Chip
            label={`Batch (${testCount})`}
            active={mode === 'batch'}
            onPress={() => setMode('batch')}
          />
        </View>
        <Text style={styles.engineLabel}>
          {backend
            ? `${backend.engineName} · ${backend.entry.model}.${backend.entry.format}`
            : entry
            ? 'Memuat model...'
            : 'Kombinasi model/format ini tidak tersedia'}
        </Text>

        {mode === 'single' ? (
          <>
            <Pressable style={styles.pickerCard} onPress={chooseSource}>
              {imageUri ? (
                <Image source={{ uri: imageUri }} style={styles.pickedImage} resizeMode="cover" />
              ) : (
                <View style={styles.placeholder}>
                  <Ionicons name="image-outline" size={40} color={C.accent} />
                  <Text style={styles.placeholderTitle}>Belum ada gambar dipilih</Text>
                  <Text style={styles.placeholderHint}>Ketuk untuk ambil foto atau unggah</Text>
                </View>
              )}
            </Pressable>
            <Pressable
              onPress={onCheck}
              disabled={!canCheck}
              style={[styles.button, { backgroundColor: canCheck ? C.accent : C.disabledBg }]}
            >
              <Text style={[styles.buttonText, { color: canCheck ? '#fff' : C.disabledText }]}>
                {running ? 'Memproses...' : 'Cek'}
              </Text>
            </Pressable>
          </>
        ) : (
          <>
            <Text style={styles.groupLabel}>Ulangan per gambar</Text>
            <View style={styles.chipRow}>
              {[1, 3, 5].map((n) => (
                <Chip key={n} label={`${n}x`} active={repeats === n} onPress={() => setRepeats(n)} />
              ))}
            </View>
            <Pressable
              onPress={onBatch}
              disabled={!canBatch}
              style={[styles.button, { marginTop: 0, backgroundColor: canBatch ? C.accent : C.disabledBg }]}
            >
              <Text style={[styles.buttonText, { color: canBatch ? '#fff' : C.disabledText }]}>
                {running ? progress ?? 'Memproses...' : 'Jalankan Batch'}
              </Text>
            </Pressable>
            <Pressable
              onPress={onBatchAll}
              disabled={!canBatch}
              style={[styles.button, styles.buttonOutline, !canBatch && { opacity: 0.5 }]}
            >
              <Text style={[styles.buttonText, { color: C.accent }]}>
                {`Jalankan semua kombinasi (${MODELS.length})`}
              </Text>
            </Pressable>
            {testCount === 0 && (
              <Text style={styles.engineLabel}>
                Belum ada gambar tes di {deviceDir()}/test/
              </Text>
            )}
            {batch && !running && (
              <Pressable onPress={onShare} style={[styles.button, styles.buttonOutline]}>
                <Text style={[styles.buttonText, { color: C.accent }]}>Bagikan CSV</Text>
              </Pressable>
            )}
          </>
        )}

        <Text style={styles.sectionTitle}>Hasil</Text>
        <View style={styles.resultCard}>
          <Text style={[styles.resultText, (batch || mono) && styles.mono]}>
            {result ?? engineError ?? 'Hasil akan muncul di sini setelah kamu menekan tombol.'}
          </Text>
        </View>

        <Text style={styles.disclaimer}>
          Alat bantu skrining awal — bukan pengganti diagnosis medis.
        </Text>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: C.bg },
  content: {
    padding: 16,
    paddingTop: (StatusBar.currentHeight ?? 24) + 8,
    paddingBottom: 32,
  },

  header: { borderRadius: 28, overflow: 'hidden' },
  headerContent: { padding: 24 },
  headerRow: { flexDirection: 'row', alignItems: 'center' },
  infoCircle: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: 'rgba(255,255,255,0.2)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  headerLabel: { marginLeft: 14, fontSize: 18, fontWeight: '500', color: '#fff' },
  headerTitle: { marginTop: 20, fontSize: 28, lineHeight: 36, fontWeight: 'bold', color: '#fff' },
  headerSteps: { marginTop: 14, fontSize: 16, lineHeight: 24, color: 'rgba(255,255,255,0.9)' },

  sectionTitle: {
    marginTop: 28,
    marginBottom: 16,
    fontSize: 24,
    fontWeight: 'bold',
    color: C.textDark,
  },

  groupLabel: { marginBottom: 6, fontSize: 13, color: C.textMuted },
  chipRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 8, marginBottom: 12 },
  chip: {
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: C.cardBorder,
    backgroundColor: '#fff',
  },
  chipText: { fontSize: 14, color: C.textDark },
  engineLabel: { marginBottom: 14, fontSize: 14, color: C.accent },

  pickerCard: {
    height: 210,
    borderRadius: 28,
    backgroundColor: '#fff',
    borderWidth: 1,
    borderColor: C.cardBorder,
    overflow: 'hidden',
    alignItems: 'center',
    justifyContent: 'center',
  },
  pickedImage: { width: '100%', height: '100%' },
  placeholder: { alignItems: 'center' },
  placeholderTitle: { marginTop: 16, fontSize: 18, color: C.textMuted },
  placeholderHint: { marginTop: 6, fontSize: 15, color: C.textHint },

  button: {
    marginTop: 16,
    height: 54,
    borderRadius: 20,
    alignItems: 'center',
    justifyContent: 'center',
  },
  buttonOutline: { backgroundColor: '#fff', borderWidth: 1, borderColor: C.accent },
  buttonText: { fontSize: 18, fontWeight: 'bold' },

  resultCard: {
    borderRadius: 24,
    backgroundColor: '#fff',
    borderWidth: 1,
    borderColor: C.cardBorder,
    paddingHorizontal: 20,
    paddingVertical: 22,
  },
  resultText: { fontSize: 17, lineHeight: 24, color: C.textMuted },
  mono: { fontFamily: 'monospace', fontSize: 12, lineHeight: 18 },

  disclaimer: {
    marginTop: 24,
    paddingHorizontal: 12,
    fontSize: 14,
    textAlign: 'center',
    color: C.textMuted,
  },
});
