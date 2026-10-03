# Digital Image Processing — Multi-Format Image Analyzer & Converter

Program pengolahan citra digital untuk tugas **Pengolahan Citra Digital (PCD)** — mengkonversi berkas citra `.bmp`, `.jpg`, dan `.png` menjadi tiga representasi: **Biner**, **Keabu-abuan 16 bit**, dan **RGB 8 bit**, lengkap dengan ekspor matriks, laporan, dan visualisasi.

## Features

| Feature | Description |
|---|---|
| **Multi-format input** | Supports `.bmp`, `.jpg`, `.jpeg`, `.png` (including 16-bit PNG) |
| **3 Output representations** | Binary (threshold), 16-bit Greyscale (BT.601 luminance), RGB 8-bit |
| **Matrix export** | `.npy` (NumPy) + `.txt` (human-readable 5x5 preview) |
| **Visualization** | Matplotlib 2x2 figure (original, Binary, Grey16, RGB8) |
| **Automatic report** | `report.txt` + `summary.csv` (UTF-8 BOM, semicolon-delimited) |
| **BMP 1-bit** | Binary image exported as native 1-bit-per-pixel BMP |
| **Interactive mode** | Run without arguments for interactive terminal prompt |
| **Otsu threshold** | Use `'auto'` option for adaptive Otsu thresholding |

## Usage

### Interactive Mode (No Arguments)

```bash
python main.py
```

The program will prompt for an image path interactively. Re-prompts on invalid path. Type `'q'` to exit.

### CLI with Arguments

```bash
# Process a single file
python main.py --input photo.png --threshold 150

# Process an entire folder
python main.py --input samples/

# Automatic threshold (Otsu)
python main.py --input photo.png --threshold auto

# No visual display (headless/CI)
python main.py --input photo.png --no-display

# Save figure to file
python main.py --input photo.png --save-figure

# Change output directory
python main.py --input photo.png --output-dir results/

# Change matrix preview size
python main.py --input photo.png --preview 10
```

### Automation via Pipe

```bash
echo "photo.png" | python main.py --no-display
```

## Project Structure

```
P2/
|-- main.py                      # Entry point CLI (argparse + orchestrator)
|-- requirements.txt             # Dependencies
|-- README.md                    # Documentation
|
|-- doc/
|   |-- digital_image_processing_prd.md   # Full PRD (FR, AC, NFR)
|
|-- src/
|   |-- __init__.py              # Package marker
|   |-- image_io.py              # Loading (.bmp/.jpg/.png), validation, float32 normalization
|   |-- transforms.py            # Binary, Greyscale16 (BT.601), RGB8, Otsu
|   |-- report.py                # Export matrices (.npy/.txt), CSV, report.txt
|   |-- matrix_view.py           # Format slice 5x5, assert shape/range
|   |-- display.py               # Matplotlib 2x2 visualization
|
|-- tests/
|   |-- __init__.py              # Package marker
|   |-- make_samples.py          # Test sample generator (3 formats)
|   |-- verify.py                # Verification suite (63 checks)
|   |-- samples/                 # Sample files
|       |-- test.bmp
|       |-- test.jpg
|       |-- test_16bit.png
|
|-- output/                      # Generated at runtime
    |-- matrices/                # .npy + .txt
    |-- images/                  # .png + .bmp 1-bit
    |-- report.txt
    |-- summary.csv
```

## Functional Requirements (PRD)

| FR | Description | Status |
|---|---|---|
| **FR-1** | Multi-format image input (CLI arguments / interactive terminal) | Done |
| **FR-2** | Conversion to 3 representations (Binary, Grey16, RGB8) | Done |
| **FR-3** | Resolution preserved, values within specified ranges | Done |
| **FR-4** | Matrix export + report (report.txt, summary.csv) | Done |
| **FR-5** | Visual display + native BMP 1-bit | Done |

## Verification

```bash
# Run full verification suite (63 checks)
python tests/verify.py

# Generate test samples (if not already present)
python tests/make_samples.py
```

Coverage: AC-1 through AC-12 verified.

## Dependencies

```
opencv-python>=4.8
numpy>=1.24
matplotlib>=3.7
```

Install:

```bash
pip install -r requirements.txt
```

## Algorithms

### Greyscale 16-bit (BT.601 Luminance)

$$Y = \lfloor 0.299R + 0.587G + 0.114B \rfloor_{16\text{-bit}}$$

### Binary (Threshold)

$$B(x,y) = \begin{cases} 255 & \text{if } Y(x,y) \geq T \\ 0 & \text{if } Y(x,y) < T \end{cases}$$

### Otsu Threshold

Computes optimal threshold $T$ that minimizes weighted within-class variance of the greyscale histogram.

## Author

Created for **Digital Image Processing** assignment — Semester 5.
