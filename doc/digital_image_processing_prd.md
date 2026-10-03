# Product Requirements Document (PRD)

## Digital Image Processing Assignment: Multi-Format Image Analyzer & Converter

## 1. Project Overview

This project is a software program designed for a Digital Image Processing assignment. Its primary objective is to load images in the standard raster formats (`.bmp`, `.jpg`, `.png`), convert them into three specified representation types — **Binary (Citra Biner)**, **16-bit Greyscale (Keabu-abuan 16 bit)**, and **8-bit RGB** — while maintaining the original resolution and providing a mathematical matrix representation for inspection.

The program supports **two input modes**:
1. **CLI argument mode** — pass image paths via `--input` flag (suitable for scripting/batch processing).
2. **Interactive terminal mode** — run `python main.py` with no arguments and the program prompts for a file/folder path via stdin (suitable for casual/one-off use).

---

## 2. Objectives & Academic Scope

* **Multi-Format Ingestion:** Load `.bmp`, `.jpg`, and `.png` images of any source bit-depth (8-bit and 16-bit per channel) without failure; normalise them into a working 8-bit BGR baseline.
* **Dual Input Modes:** Accept image paths via CLI arguments (`--input`) OR prompt the user interactively via terminal stdin when no arguments are provided.
* **Format & Bit-Depth Transformations** — three mandatory output representations:
  * **Binary Image (Citra Biner):** 1-bit logic stored as $0$ (Black) / $255$ (White) via adjustable threshold $T$ (default $127$, optional Otsu).
  * **Greyscale 16-bit (Keabu-abuan 16 bit):** 1 channel, `uint16`, $0$–$65535$.
  * **RGB 8-bit:** 3 channels, `uint8`, $0$–$255$ per channel.
* **Resolution & Matrix Preservation:** Every output keeps the exact input spatial dimensions ($W \times H$); no resize, no crop, no padding.
* **Data Inspection:** Expose matrix properties (shape, dtype, min, max, mean, 5$\times$5 sample slice) for **all three** outputs to prove the processing logic.
* **Deliverable Report:** Emit machine-readable matrices (`.npy` + readable `.txt`) and a summary table so the work can be attached to the assignment report.

---

## 3. Detailed Functional Requirements

### FR-1: File Input & Compatibility
* **Supported Formats:** Bitmap (`.bmp`), JPEG (`.jpg`, `.jpeg`), Portable Network Graphics (`.png`).
* **Input Method A — CLI Arguments:** `--input PATH [PATH ...]` accepts one or more file paths or directory paths. Directories are scanned non-recursively for supported image files.
* **Input Method B — Interactive Terminal Mode:** When `--input` is omitted entirely, the program prints a banner ("MODE INTERAKTIF") and calls Python's built-in `input()` to prompt the user to type a file or folder path. The program loops until a valid path is entered or the user quits.
* **Interactive Quit Commands:** The user may type `q`, `quit`, `exit`, or `keluar` to exit cleanly (exit code 0). Pressing Ctrl+C also exits cleanly (exit code 130).
* **Validation:** Program must check file existence and confirm valid image headers for both input modes.
* **Error Handling (both modes):** Graceful error messages in terminal for missing files or unsupported formats; non-zero exit code, no raw traceback. In interactive mode, an invalid path prints the error message and re-prompts (does not crash).
* **Bit-Depth Safe Read:** Images are decoded with `IMREAD_UNCHANGED` and normalised to float32 in $[0,1]$ so a natively 16-bit-per-channel PNG/BMP is never silently truncated to 8 bits. The detected `source_bit_depth` (8 or 16) is recorded in the summary report.

### FR-2: Image Processing & Bit-Depth Pipeline

All three outputs are produced from a single working baseline, so results stay directly comparable. The baseline is 8-bit BGR, used for RGB 8-bit output and as the luminance input. 16-bit-native sources still feed the 16-bit greyscale from their full float32 precision instead of the up-scaled 8-bit path.

| ID | Output Type (Indo / EN) | Channels | Bit-Depth | Value Range | Method / Transformation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| FR-2.1 | **Biner / Binary** | 1 | 1 bit, stored as `uint8` | $\{0, 255\}$ | Global thresholding: $G(x,y) = 255$ if $f(x,y) \ge T$ else $0$; default $T = 127$, or Otsu when $T = \text{auto}$ |
| FR-2.2 | **Keabu-abuan 16 bit / Greyscale 16** | 1 | `uint16` | $0 - 65535$ | Luminance $Y = 0.299R + 0.587G + 0.114B$ on the float32 $[0,1]$ samples, scaled by $65535$ and rounded |
| FR-2.3 | **RGB 8 bit** | 3 | `uint8` | $0 - 255$ / channel | Reorder BGR $\rightarrow$ RGB from the 8-bit baseline |

Notes:
* FR-2.2 is the **primary** greyscale representation; no separate 8-bit greyscale output is produced. Luminance is evaluated in float so the 16-bit result keeps full precision instead of passing through an 8-bit intermediate.
* Binarization thresholds the 16-bit greyscale plane at the equivalent 16-bit threshold $T_{16} = T \times 257$, so a threshold expressed in $[0,255]$ behaves the same as the textbook $f(x,y) \ge T$ rule.
* For a 16-bit-native source the greyscale keeps its original precision; for an 8-bit source the 16-bit values are the exact $\times 257$ replication of the 8-bit luminance. `source_bit_depth` in the summary records which path ran.

### FR-3: Spatial Resolution & Matrix Representation
* **Resolution Maintenance:** Output dimensions must exactly equal input dimensions ($W \times H$) for **all three** outputs.
* **Matrix Data Structure Requirements:**

| ID | Output | Matrix Shape (NumPy) | dtype | Element Values |
| :--- | :--- | :--- | :--- | :--- |
| FR-3.1 | Biner | $(H, W)$ | `uint8` | $0$ or $255$ (`bool` view also acceptable) |
| FR-3.2 | Keabu-abuan 16 bit | $(H, W)$ | `uint16` | $0 - 65535$ |
| FR-3.3 | RGB 8 bit | $(H, W, 3)$ | `uint8` | $0 - 255$ / channel |

* **Matrix Preview:** Print a $5 \times 5$ (configurable) slice from the top-left corner of every output matrix to console, with dtype and shape headers, for manual verification against the source image.
* **Shape Assertion:** Program asserts `out.shape[:2] == (H, W)` for every output; failure aborts with an error (FR-1 error handling).

### FR-4: Visual Output & Display
* **Interactive Display:** Show the original image alongside all three variants — Biner, Keabu-abuan 16 bit, RGB 8 bit — as a single $2 \times 2$ figure or sequential windows via `matplotlib` / `cv2.imshow`.
* **16-bit Display Rule:** The 16-bit greyscale plane (`uint16`) must be normalised for display only (`/ 65535.0`) — the array itself is never divided in place.
* **Window Controls:** Implement a blocking key press (`cv2.waitKey(0)`) or `plt.show()` so windows do not close immediately.

### FR-5: Deliverable Artifacts & Reporting
* **Matrix Export:** Save each of the three output matrices to `output/matrices/<name>_matrix.npy` (NumPy binary) and `<name>_matrix.txt` (human-readable, full matrix or head/tail window for large images).
* **Image Export:** Save each rendered variant to `output/images/<name>.png`. The 16-bit greyscale is written as a 16-bit PNG so the file on disk is genuinely 16 bits per pixel.
* **Summary Report:** Write `output/report.txt` and `output/summary.csv` (semicolon-delimited, UTF-8 with BOM so it opens as columns in Excel) containing per-output: filename, representation, shape, dtype, min, max, mean, and processing time in seconds.
* **Binary Image Export:** Biner output additionally saved as a 1-bit-packed `.bmp` to demonstrate true 1-bit-per-pixel storage.

---

## 4. Non-Functional Requirements

* **Correctness:** Every FR has at least one automated assertion or printed numeric proof (matrix slice, shape/dtype check, bit-depth range check) — verifiable without a GUI.
* **Performance:** Full pipeline over all three outputs under $2.0$ seconds for images up to 4K ($3840 \times 2160$), measured and written into the summary report.
* **Portability:** Pure Python on `opencv-python`, `numpy`, `matplotlib`; runnable on Windows, Linux, macOS. No hardcoded absolute paths — output dirs created relative to the script.
* **Code Clarity:** One function per task (`load_image`, `to_biner`, `to_greyscale16`, `to_rgb8`, `print_matrix`, `save_outputs`, `_prompt_for_input`), Indonesian-friendly docstrings, type hints.
* **Dependency Footprint:** `numpy`, `opencv-python`, `matplotlib` only. Add nothing else for what these three already do.
* **Interactive UX:** When running without arguments, the program must print a clear banner explaining the interactive mode, show a prompt, accept quit commands gracefully, and never crash on invalid input (re-prompt instead).

---

## 5. Execution Flow Chart

### Path A: CLI Argument Mode (`python main.py --input <path> [...]`)

1. **Parse Arguments:** Read `--input`, `--threshold`, `--preview`, `--output-dir`, `--no-display`/`--save-figure`.
2. **Input Stage:** Load image file(s) → decode with `IMREAD_UNCHANGED` to float32 $[0,1]$ (16-bit safe) → record source bit depth → build 8-bit BGR baseline → read dimensions $(H, W, C)$.
3. **Greyscale 16-bit Conversion:** Baseline BGR $\rightarrow$ float luminance $Y$ $\rightarrow$ $\times 65535$ rounded → 2D `uint16` matrix $(H, W)$.
4. **Binarization (Citra Biner):** Apply threshold $T_{16} = T \times 257$ on the 16-bit greyscale matrix → 2D `uint8` matrix $\{0, 255\}$.
5. **RGB 8-bit Conversion:** BGR $\rightarrow$ RGB reorder → 3D `uint8` matrix $(H, W, 3)$.
6. **Verification:** Assert shape and dtype of all three matrices → print $5 \times 5$ slices and min/max/mean stats.
7. **Output:** Write `.npy` + `.txt` matrices, variant images, 1-bit `.bmp`, `report.txt`, `summary.csv` → render display figure (original + 3 variants).

### Path B: Interactive Terminal Mode (`python main.py`)

1. **Detect Missing Input:** `--input` not provided → enter interactive mode.
2. **Print Banner:** Display "MODE INTERAKTIF" header listing supported formats, instructions, and quit options.
3. **Prompt Loop:** Call `input()` to read a path from the user:
   * If empty → re-prompt with warning.
   * If `q`/`quit`/`exit`/`keluar` → exit cleanly (code 0).
   * If Ctrl+C / EOF → exit cleanly (code 130).
   * If valid path → proceed to **Path A, step 2** (Input Stage) with this path.
   * If invalid path → print error message (from FR-1 error handling) and **re-prompt** (loop continues).
4. **Processing:** Execute Path A steps 2–7 identically with the entered path.
5. **After Processing:** If multiple files were processed (e.g., user entered a directory), offer to process another path or quit. (Optional enhancement: for v1, single-prompt is sufficient.)

---

## 6. Acceptance Criteria (Checklist)

* [x] AC-1: `.bmp`, `.jpg`/`.jpeg`, `.png` files each load without error (verified per format).
* [x] AC-2: 16-bit-per-channel PNG and BMP sources load without truncation; `source_bit_depth` recorded as 16.
* [x] AC-3: Biner matrix values are exactly $\{0, 255\}$ and change when $T$ changes.
* [x] AC-4: Greyscale 16-bit matrix dtype is `uint16`, shape $(H, W)$, min $\ge 0$, max $\le 65535$.
* [x] AC-5: RGB 8-bit matrix dtype is `uint8`, shape $(H, W, 3)$, channel order R, G, B.
* [x] AC-6: All three outputs have `shape[:2] == input.shape[:2]` — resolution preserved.
* [x] AC-7: A $5 \times 5$ slice of every output prints to console with dtype/shape header.
* [x] AC-8: Display figure shows original + all 3 variants with the 16-bit plane normalised (not black).
* [x] AC-9: `output/summary.csv` opens in Excel as proper columns (semicolon + BOM) with 3 output rows, and `output/report.txt` lists all three outputs.
* [x] AC-10: A missing or corrupt file produces a clear error message and non-zero exit code, not a traceback crash.
* [ ] **AC-11: Interactive mode activates when `--input` is omitted.** Program prints "MODE INTERAKTIF" banner, prompts for path via `input()`, accepts valid file/directory paths, rejects invalid ones with a clear message and re-prompts, and exits cleanly on `q`/`quit`/`exit`/`keluar`/Ctrl+C.
* [ ] **AC-12: Piped stdin works for automation.** Running `echo "foto.png" | python main.py --no-display` processes the piped path and produces correct output artifacts (exit code 0).

---

## 7. Implementation Plan (Build Order)

| Step | Task | Files | Done when |
| :--- | :--- | :--- | :--- |
| 1 | **Setup + ingestion** — install `opencv-python numpy matplotlib`; write `load_image(path)` with existence check, extension whitelist (`.bmp/.jpg/.jpeg/.png`), `IMREAD_UNCHANGED`, float32 $[0,1]$ decode, source-bit-depth detection, 8-bit BGR baseline, readable error messages. | `requirements.txt`, `src/image_io.py` | Each of the 3 formats loads; a bad path raises a readable error |
| 2 | **Conversions** — implement `to_greyscale16` (float luminance $\times 65535$), `to_biner(y16, T)`, `to_rgb8`. One file, three small functions. | `src/transforms.py` | Each returns correct dtype and range; biner thresholds at $T \times 257$ |
| 3 | **Matrix inspection** — `print_matrix(name, arr, n=5)` printing dtype, shape, min/max/mean, and the top-left $n \times n$ slice; `assert_shape(arr, H, W)`. | `src/matrix_view.py` | Slices print correctly for all 3 outputs; shape assertion holds |
| 4 | **Export & report** — `save_outputs()` writes `.npy` + `.txt` matrices, variant `.png`s (16-bit PNG for greyscale), 1-bit `.bmp`, `report.txt`, and `summary.csv` (sep `;`, `utf-8-sig`). Times each stage. | `src/report.py` | Running it produces every file listed in FR-5 |
| 5 | **CLI entry** — argparse: `--input` (one or more paths or a directory, **optional** — defaults to interactive mode), `--threshold`, `--preview N`, `--no-display`, `--output-dir`. Orchestrates steps 1–4. Includes `_prompt_for_input()` function for interactive terminal mode. | `main.py` | `python main.py --input img.png --threshold 150` completes end-to-end; `python main.py` (no args) enters interactive mode |
| 6 | **Display** — `display_all(original, variants)`: matplotlib $2 \times 2$ figure, imshow with per-bit-depth `vmin/vmax`, grey colormap for biner and greyscale, title per variant. | `src/display.py` | Figure shows 4 panels; the 16-bit panel is not all-black |
| 7 | **Sample assets** — generate small test images (`test.bmp` 8-bit, `test.png` 16-bit, `test.jpg`) via `cv2.imwrite`, committed so the run is reproducible without external files. | `tests/make_samples.py` | `python tests/make_samples.py` creates the three files |
| 8 | **Verification suite** — one script of `assert`s covering AC-3…AC-7 (dtype, range, shapes, binarization at $T \times 257$, BGR→RGB order, three-format coverage, 16-bit source depth). No test framework. | `tests/verify.py` | `python tests/verify.py` prints `ALL CHECKS PASSED` with exit 0 |
| 9 | **Interactive input mode** — make `--input` optional (`nargs="*"`, `required` removed); implement `_prompt_for_input()` with `input()` loop, quit commands (`q`/`quit`/`exit`/`keluar`), Ctrl+C handling, empty-input re-prompt, and invalid-path-reprompts inside the loop. Wire into `main()` after argparse. Update help text and epilog examples. | `main.py` | `python main.py` shows interactive banner and prompts; typing a valid path processes successfully; typing `q` exits 0; piping a path via stdin works |
| 10 | **Interactive mode tests** — add `check_interactive_mode()` to verification suite: subprocess run with no `--input`, pipe a valid BMP path via `stdin`, verify exit code 0, verify "INTERAKTIF" appears in stdout, verify output artifacts exist. Also test `echo "q" | python main.py` exits 0 cleanly. | `tests/verify.py` | `python tests/verify.py` includes new checks, total count increases, still prints `ALL CHECKS PASSED` |

**Definition of Done:** steps 1–10 complete, `python tests/verify.py` exits 0, `python main.py --input <any of the 3 formats>` produces all FR-5 artifacts, `python main.py` (no args) enters interactive mode and processes a typed/piped path correctly, and `output/summary.csv` opens cleanly in Excel with 3 output rows.

---

## 8. Out of Scope

* 8-bit greyscale and 16-bit RGB outputs — the assignment targets 16-bit greyscale and 8-bit RGB only.
* Filters beyond thresholding (blur, edge, histogram equalisation) — not requested by the assignment checklist.
* GUI application (Tkinter/Qt) — CLI plus a matplotlib figure is sufficient.
* Training data / deep learning — outside this assignment's scope.
* Color spaces other than BGR/RGB/greyscale.
* **Persistent config files or history** — the interactive mode does not remember previous paths between runs.
* **Tab-completion or fuzzy path matching** in interactive mode — the user types the full path manually (or pastes it).
* **Multi-file interactive selection** — in interactive mode, the user types one path (file or directory); batch multi-file selection is done via `--input file1.png file2.png folder/` in CLI mode.
