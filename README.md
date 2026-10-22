# Pradiacheck.RN.BatchTest

Aplikasi React Native (Expo) untuk **batch testing** deteksi dini diabetes dari citra lidah menggunakan model YOLO yang berjalan **offline di Android**. Repo ini adalah bagian dari Tugas Akhir (S1 Teknik Biomedis, Universitas Telkom) dan menjadi salah satu dari tiga aplikasi pembanding (Flutter, Kotlin, React Native).

Tujuan: membandingkan **3 model** x **3 konfigurasi engine** langsung di HP, tanpa mengubah kode di antara kombinasi, lalu memilih satu kombinasi terbaik untuk aplikasi final.

## Kombinasi yang diuji

| Model | ONNX | ExecuTorch (`.pte`) | TFLite |
|-------|:----:|:-------------------:|:------:|
| YOLOv8n (`best8n`)   | ✓ | ✓ | ✓ |
| YOLO11n (`best11n`)  | ✓ | ✓ | ✓ |
| YOLO11s (`best11s`)  | ✓ | ✓ | ✓ |

Total 9 kombinasi per aplikasi. File model ada di `assets/models/`.

## Kelas

Urutan kelas mengikuti `model.names`:

| Indeks | Kelas |
|:------:|-------|
| 0 | Diabetes |
| 1 | Nondiabetes |
| 2 | Pradiabetes |

## Dataset uji

94 gambar (format YOLO, dikonversi ke folder per kelas):

- Pradiabetes: 35
- Diabetes: 31
- Nondiabetes: 28

Dataset **tidak** disertakan di APK dan tidak ada di repo ini. Dataset dikirim ke HP lewat `adb push`.

## Alur pengujian

1. Pilih model dan engine dari UI aplikasi (tanpa edit kode).
2. Jalankan batch test pada seluruh gambar di folder dataset di HP.
3. Protokol: 94 gambar x 3 kali pengulangan, sama dengan aplikasi Kotlin dan Flutter.
4. Hasil ditulis ke file CSV, lalu diambil ke laptop dengan `adb pull`.

Kolom CSV per gambar mencakup jumlah deteksi, confidence, dan waktu (ms) untuk tahap preprocess, inferensi, postprocess, dan end-to-end.

### Perintah adb

```bash
# kirim dataset ke HP
adb push <folder-dataset-lokal> <folder-tujuan-di-HP>

# ambil hasil ke laptop
adb pull <folder-hasil-di-HP> <folder-tujuan-lokal>
```

Sesuaikan path dengan yang dipakai aplikasi.

## Menjalankan

Prasyarat: Node.js, Android SDK, dan HP Android dengan USB debugging aktif.

```bash
npm install
npx expo run:android            # build debug
npx expo run:android --variant release   # build release
```

Karena memakai native module (ONNX Runtime, ExecuTorch, TFLite), aplikasi **tidak bisa** dijalankan lewat Expo Go. Gunakan development build seperti perintah di atas.

## Struktur singkat

- `App.tsx` : UI dan alur batch test
- `engine.ts` : pemuatan model, inferensi, dan penulisan CSV hasil
- `registry.ts` : daftar model dan engine
- `assets/models/` : file model (`.onnx`, `.pte`, `.tflite`)
- `patches/`, `scripts/` : patch dependensi dan skrip bantu build

## Perangkat uji

Samsung Galaxy A56 (Android).

## Repo terkait

- Pradiacheck-Flutter
- Pradiacheck-Kotlin

## Catatan

Benchmark ini hanya untuk memilih kombinasi terbaik. Aplikasi final akan memakai satu model dengan satu format saja.
