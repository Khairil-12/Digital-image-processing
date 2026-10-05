"""Suite verifikasi untuk PRD PCD (Langkah 8).
Memakai assert biasa tanpa framework. Menjalankan AC-1 sampai AC-10 dan
mencetak ALL CHECKS PASSED dengan kode keluar 0 bila semuanya benar.
Jalankan: python tests/verify.py
"""

from __future__ import annotations
import subprocess
import sys
import tempfile
from pathlib import Path
import cv2
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import transforms  # noqa: E402
from src.image_io import ImageLoadError, load_image  # noqa: E402
from src.matrix_view import assert_shape, format_slice, verify_range  # noqa: E402
from src.report import _write_binary_bmp  # noqa: E402
from tests.make_samples import make_samples  # noqa: E402

SCALE = transforms.SCALE_8_TO_16
PASSED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    """Catat satu pemeriksaan; galat saat kondisi tidak terpenuhi."""
    if not condition:
        raise AssertionError(f"GAGAL: {name}" + (f" ({detail})" if detail else ""))
    PASSED.append(name)
    print(f"  [OK] {name}")


def check_three_formats(samples: dict[str, Path]) -> None:
    """AC-1 dan AC-2: ketiga format termuat, dan kedalaman bit terdeteksi."""
    print("\nAC-1/AC-2: pembacaan tiga format + kedalaman bit")
    for key, expected_depth in (("bmp", 8), ("jpg", 8), ("png16", 16)):
        loaded = load_image(samples[key])
        check(f"{samples[key].name} termuat", loaded.rgb_f32.shape[2] == 3, str(loaded.rgb_f32.shape))
        check(
            f"{samples[key].name} kedalaman bit = {expected_depth}",
            loaded.source_bit_depth == expected_depth,
            f"terbaca {loaded.source_bit_depth}",
        )
        check(
            f"{samples[key].name} rentang float [0,1]",
            0.0 <= float(loaded.rgb_f32.min()) and float(loaded.rgb_f32.max()) <= 1.0,
        )

def check_greyscale16(loaded) -> None:
    """AC-4: keabu-abuan 16-bit bertipe benar, rentang benar, resolusi terjaga."""
    print("\nAC-4: Keabu-abuan 16 bit")
    y16 = transforms.to_greyscale16(loaded.rgb_f32)
    check("dtype uint16", y16.dtype == np.uint16, str(y16.dtype))
    check("bentuk (H, W)", y16.ndim == 2 and y16.shape == (loaded.height, loaded.width), str(y16.shape))
    check("rentang 0..65535", int(y16.min()) >= 0 and int(y16.max()) <= 65535, f"{y16.min()}..{y16.max()}")
    assert_shape(y16, loaded.height, loaded.width, "Keabu-abuan 16 bit")
    verify_range(y16, 0, 65535, "Keabu-abuan 16 bit")

    manual = np.rint(
        (
            0.299 * loaded.rgb_f32[:, :, 0]
            + 0.587 * loaded.rgb_f32[:, :, 1]
            + 0.114 * loaded.rgb_f32[:, :, 2]
        )
        * 65535
    )
    check("cocok dengan luminansi BT.601 manual", np.array_equal(y16, manual.astype(np.uint16)))

    luma8 = np.rint(
        (
            0.299 * loaded.rgb_f32[:, :, 0]
            + 0.587 * loaded.rgb_f32[:, :, 1]
            + 0.114 * loaded.rgb_f32[:, :, 2]
        )
        * 255.0
    ).astype(np.uint8)
    for t in (60, 127, 200):
        via16 = transforms.to_biner(y16, t)
        via8 = np.where(luma8 >= t, np.uint8(255), np.uint8(0))
        agree = float((via16 == via8).mean()) * 100
        check(f"biner T={t} cocok dengan ambang pada skala 8-bit (>=99%)", agree >= 99.0, f"{agree:.2f}%")


def check_biner(loaded) -> None:
    """AC-3: biner berisi tepat {0, 255} dan berubah mengikuti ambang."""
    print("\nAC-3: Biner")
    y16 = transforms.to_greyscale16(loaded.rgb_f32)
    low = transforms.to_biner(y16, 60)
    high = transforms.to_biner(y16, 200)

    check("dtype uint8", low.dtype == np.uint8, str(low.dtype))
    check("nilai tepat {0, 255}", set(np.unique(low).tolist()) <= {0, 255}, str(np.unique(low).tolist()))
    check("berubah saat ambang dinaikkan", not np.array_equal(low, high))
    check("ambang tinggi tak lebih banyak piksel putih", int(high.sum()) <= int(low.sum()))
    check(
        "aturan f(x,y) >= T pada skala 16-bit terpenuhi",
        np.array_equal(high == 255, y16 >= 200 * SCALE),
    )
    assert_shape(low, loaded.height, loaded.width, "Biner")


def check_rgb8(loaded) -> None:
    """AC-5: RGB 8-bit bertipe benar dan urutan kanal R, G, B."""
    print("\nAC-5: RGB 8 bit")
    rgb8 = transforms.to_rgb8(loaded.bgr8)
    check("dtype uint8", rgb8.dtype == np.uint8, str(rgb8.dtype))
    check("bentuk (H, W, 3)", rgb8.shape == (loaded.height, loaded.width, 3), str(rgb8.shape))
    check("rentang 0..255", int(rgb8.min()) >= 0 and int(rgb8.max()) <= 255)
    check(
        "kanal 0 = R (bukan B)",
        np.array_equal(rgb8[:, :, 0], loaded.bgr8[:, :, 2]),
    )
    assert_shape(rgb8, loaded.height, loaded.width, "RGB 8 bit")

def check_resolution_preserved(loaded) -> None:
    """AC-6: ketiga keluaran mempertahankan resolusi masukan."""
    print("\nAC-6: resolusi terjaga untuk ketiga keluaran")
    y16 = transforms.to_greyscale16(loaded.rgb_f32)
    variants = {
        "Biner": transforms.to_biner(y16, 127),
        "Keabu-abuan 16 bit": y16,
        "RGB 8 bit": transforms.to_rgb8(loaded.bgr8),
    }
    for label, array in variants.items():
        check(
            f"{label} (H, W) sama dengan masukan",
            array.shape[:2] == (loaded.height, loaded.width),
            str(array.shape),
        )

def check_preview(loaded) -> None:
    """AC-7: potongan matriks 5x5 tercetak dengan header dtype dan shape."""
    print("\nAC-7: potongan matriks")
    y16 = transforms.to_greyscale16(loaded.rgb_f32)
    variants = {
        "Biner": transforms.to_biner(y16, 127),
        "Keabu-abuan 16 bit": y16,
        "RGB 8 bit": transforms.to_rgb8(loaded.bgr8),
    }
    for label, array in variants.items():
        text = format_slice(array, 5)
        check(f"{label} memuat baris r0..r4", all(f"r{i}" in text for i in range(5)), text[:40])
        check(f"{label} memuat label kanal bila RGB", True if array.ndim == 2 else "kanal R" in text)

def check_error_handling() -> None:
    """AC-10: berkas salah memberi pesan jelas, bukan traceback."""
    print("\nAC-10: penanganan galat")
    try:
        load_image(Path(tempfile.gettempdir()) / "tidak_ada_berkas_12345.png")
    except ImageLoadError as error:
        check("berkas hilang ditolak", "tidak ditemukan" in str(error).lower(), str(error))
    else:
        raise AssertionError("GAGAL: berkas hilang tidak ditolak")

    bad = Path(tempfile.gettempdir()) / "bukan_citra.png"
    bad.write_bytes(b"ini bukan data citra")
    try:
        load_image(bad)
    except ImageLoadError as error:
        check("berkas rusak ditolak", "gagal membaca" in str(error).lower(), str(error))
    finally:
        bad.unlink(missing_ok=True)

def check_bmp_1bit(tmp_dir: Path) -> None:
    """FR-5: berkas BMP biner benar-benar 1 bit per piksel dan terbaca utuh."""
    print("\nFR-5: BMP 1-bit")
    biner = np.zeros((16, 24), dtype=np.uint8)
    biner[4:12, 8:16] = 255
    path = Path(_write_binary_bmp(tmp_dir, biner))

    raw = path.read_bytes()
    check("header BMP 'BM'", raw[:2] == b"BM", raw[:2].decode("latin-1"))
    check("bits per pixel = 1", int.from_bytes(raw[28:30], "little") == 1)
    check("ukuran berkas sesuai header", int.from_bytes(raw[2:6], "little") == len(raw))

    decoded = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    check("BMP terbaca tanpa kehilangan", decoded is not None and decoded.shape == biner.shape)
    if decoded is not None:
        check("piksel BMP cocok dengan matriks biner", np.array_equal(decoded, biner))

def check_cli_end_to_end(tmp_dir: Path) -> None:
    """FR-5/AC-9: CLI berjalan penuh dan menghasilkan seluruh artefak."""
    print("\nFR-5/AC-9: CLI end-to-end")
    samples_dir = ROOT / "tests" / "samples"
    out_dir = tmp_dir / "out"
    completed = subprocess.run(
        [
            sys.executable, str(ROOT / "main.py"),
            "--input", str(samples_dir),
            "--threshold", "150",
            "--no-display",
            "--output-dir", str(out_dir),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    check("CLI kode keluar 0", completed.returncode == 0, completed.stderr[-400:])
    for relative in (
        "report.txt",
        "summary.csv",
        "matrices/biner_matrix.npy",
        "matrices/greyscale16_matrix.npy",
        "matrices/rgb8_matrix.npy",
        "matrices/biner_matrix.txt",
        "matrices/greyscale16_matrix.txt",
        "matrices/rgb8_matrix.txt",
        "images/biner.png",
        "images/greyscale16.png",
        "images/rgb8.png",
        "images/biner_1bit.bmp",
    ):
        check(f"artefak {relative} ada", (out_dir / relative).exists())

    csv_text = (out_dir / "summary.csv").read_text(encoding="utf-8-sig")
    check("summary.csv memakai pemisah titik koma", ";" in csv_text.splitlines()[0])
    check("summary.csv punya 3 baris data per berkas", csv_text.count("Biner") >= 3, str(csv_text.count("Biner")))
    check("report.txt menyebut ambang 150", "150" in (out_dir / "report.txt").read_text(encoding="utf-8"))

    grey_png = out_dir / "images" / "greyscale16.png"
    depth = cv2.imread(str(grey_png), cv2.IMREAD_UNCHANGED)
    check("PNG keabu-abuan benar-benar 16-bit", depth is not None and depth.dtype == np.uint16, str(depth.dtype))


def check_interactive_mode(tmp_dir: Path) -> None:
    """AC-11 & AC-12: mode interaktif tanpa --input, path dari stdin."""
    print("\nAC-11/AC-12: Mode interaktif (stdin)")
    samples_dir = ROOT / "tests" / "samples"
    bmp_sample = next(p for p in samples_dir.iterdir() if p.suffix == ".bmp")
    interactive_out = tmp_dir / "interactive_out"
    completed = subprocess.run(
        [
            sys.executable, str(ROOT / "main.py"),
            "--no-display",
            "--output-dir", str(interactive_out),
        ],
        input=str(bmp_sample) + "\n",
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=30,
    )
    check(
        "mode interaktif exit code 0",
        completed.returncode == 0,
        f"stdout: {completed.stdout[-300:]}\nstderr: {completed.stderr[-300:]}",
    )
    check(
        "mode interaktif mencetak banner INTERAKTIF",
        "INTERAKTIF" in completed.stdout,
        f"stdout: {completed.stdout[:500]}",
    )
    check(
        "artefak report.txt dari mode interaktif ada",
        (interactive_out / "report.txt").exists(),
    )
    check(
        "artefak summary.csv dari mode interaktif ada",
        (interactive_out / "summary.csv").exists(),
    )
    completed_q = subprocess.run(
        [sys.executable, str(ROOT / "main.py")],
        input="q\n",
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=10,
    )
    check(
        "'q' di mode interaktif -> exit 0",
        completed_q.returncode == 0,
        f"stdout: {completed_q.stdout[-200:]}\nstderr: {completed_q.stderr[-200:]}",
    )

def main() -> int:
    """Jalankan seluruh pemeriksaan; kembalikan 0 bila semua lulus."""
    with tempfile.TemporaryDirectory() as temp:
        tmp_dir = Path(temp)
        created = make_samples()
        samples = {
            "bmp": next(p for p in created if p.suffix == ".bmp"),
            "png16": next(p for p in created if p.suffix == ".png"),
            "jpg": next(p for p in created if p.suffix == ".jpg"),
        }
        print(f"Berkas contoh: {', '.join(p.name for p in created)}")

        loaded = load_image(samples["bmp"])
        check_three_formats(samples)
        check_greyscale16(loaded)
        check_biner(loaded)
        check_rgb8(loaded)
        check_resolution_preserved(loaded)
        check_preview(loaded)
        check_error_handling()
        check_bmp_1bit(tmp_dir)
        check_cli_end_to_end(tmp_dir)
        check_interactive_mode(tmp_dir)

    print("\n" + "=" * 78)
    print(f"ALL CHECKS PASSED ({len(PASSED)} pemeriksaan)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())