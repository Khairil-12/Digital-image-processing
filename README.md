# Digital Image Processing — Multi-Format Image Analyzer & Converter

Program pengolahan citra digital untuk tugas **Pengolahan Citra Digital (PCD)** — mengkonversi berkas citra `.bmp`, `.jpg`, dan `.png` menjadi tiga representasi: **Biner**, **Keabu-abuan 16 bit**, dan **RGB 8 bit**, lengkap dengan ekspor matriks, laporan, dan visualisasi.

## 📸 Fitur Utama

| Fitur | Deskripsi |
|---|---|
| **Multi-format input** | Mendukung `.bmp`, `.jpg`, `.jpeg`, `.png` (termasuk 16-bit PNG) |
| **3 Representasi output** | Biner (threshold), Keabu-abuan 16-bit (BT.601 luminance), RGB 8-bit |
| **Ekspor matriks** | `.npy` (NumPy) + `.txt` (human-readable 5×5 preview) |
| **Visualisasi** | Figure matplotlib 2×2 (original, Biner, Grey16, RGB8) |
| **Laporan otomatis** | `report.txt` + `summary.csv` (UTF-8 BOM, semicolon-delimited) |
| **BMP 1-bit** | Berkas biner diekspor sebagai BMP 1-bit per piksel asli |
| **Mode interaktif** | Jalankan tanpa argumen → prompt path di terminal |
| **Threshold Otsu** | Pilihan `'auto'` untuk threshold adaptif Otsu |

## 🚀 Cara Penggunaan

### Mode Interaktif (Tanpa Argumen)

```bash
python main.py
# → Program meminta path citra secara interaktif
# → Re-prompts jika path salah
# → Ketik 'q' untuk keluar
```

### CLI dengan Argumen

```bash
# Proses satu berkas
python main.py --input foto.png --threshold 150

# Proses seluruh folder
python main.py --input samples/

# Threshold otomatis (Otsu)
python main.py --input foto.png --threshold auto

# Tanpa tampilan visual (headless/CI)
python main.py --input foto.png --no-display

# Simpan figur ke file
python main.py --input foto.png --save-figure

# Ubah folder keluaran
python main.py --input foto.png --output-dir hasil/

# Ubah ukuran preview matriks
python main.py --input foto.png --preview 10
```

### Automation via Pipe

```bash
echo "foto.png" | python main.py --no-display
```

## 📁 Struktur Project

```
P2/
├── main.py                      # Entry point CLI (argparse + orchestrator)
├── requirements.txt             # Dependencies
├── README.md                    # Dokumentasi ini
│
├── doc/
│   └── digital_image_processing_prd.md   # PRD lengkap (FR, AC, NFR)
│
├── src/
│   ├── __init__.py              # Package marker
│   ├── image_io.py              # Loading (.bmp/.jpg/.png), validasi, normalisasi float32
│   ├── transforms.py            # Biner, Greyscale16 (BT.601), RGB8, Otsu
│   ├── report.py                # Ekspor matriks (.npy/.txt), CSV, report.txt
│   ├── matrix_view.py           # Format slice 5×5, assert shape/range
│   └── display.py               # Visualisasi matplotlib 2×2
│
├── tests/
│   ├── __init__.py              # Package marker
│   ├── make_samples.py          # Generator sampel uji (3 format)
│   ├── verify.py                # Suite verifikasi (63 pemeriksaan)
│   └── samples/                 # Berkas sampel
│       ├── test.bmp
│       ├── test.jpg
│       └── test_16bit.png
│
└── output/                      # Hasil proses (di-generate saat runtime)
    ├── matrices/                # .npy + .txt
    ├── images/                  # .png + .bmp 1-bit
    ├── report.txt
    └── summary.csv
```

## 📋 Functional Requirements (PRD)

| FR | Deskripsi | Status |
|---|---|---|
| **FR-1** | Input citra multi-format (CLI argumen / interaktif terminal) | ✅ |
| **FR-2** | Konversi ke 3 representasi (Biner, Grey16, RGB8) | ✅ |
| **FR-3** | Resolusi terjaga, nilai pada rentang yang dijanjikan | ✅ |
| **FR-4** | Ekspor matriks + laporan (report.txt, summary.csv) | ✅ |
| **FR-5** | Tampilan visual + BMP 1-bit asli | ✅ |

## 🧪 Verifikasi

```bash
# Jalankan seluruh suite verifikasi (63 pemeriksaan)
python tests/verify.py

# Generate sampel uji (jika belum ada)
python tests/make_samples.py
```

**Coverage:** AC-1 sampai AC-12 terverifikasi.

## 🔧 Dependencies

```
opencv-python>=4.8
numpy>=1.24
matplotlib>=3.7
```

Install:

```bash
pip install -r requirements.txt
```

## 📐 Algoritma

### Keabu-abuan 16-bit (BT.601 Luminance)
$$Y = \lfloor 0.299R + 0.587G + 0.114B \rfloor_{16\text{-bit}}$$

### Biner (Threshold)
$$B(x,y) = \begin{cases} 255 & \text{ jika } Y(x,y) \geq T \\ 0 & \text{ jika } Y(x,y) < T \end{cases}$$

### Threshold Otsu
Menghitung $T$ optimal yang meminimalkan *weighted within-class variance* dari histogram greyscale.

## 👤 Author

Dibuat untuk tugas **Pengolahan Citra Digital** — Smt 5.
