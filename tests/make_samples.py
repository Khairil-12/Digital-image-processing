"""Membuat berkas citra contoh untuk pengujian (Langkah 7).
Menghasilkan samples/test.bmp (8-bit), samples/test.png (16-bit), dan
samples/test.jpg (8-bit) berisi pola bergradien yang sama, sehingga hasil
ketiganya bisa dibandingkan langsung.
Jalankan: python tests/make_samples.py
"""

from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np

SAMPLES_DIR = Path(__file__).resolve().parent / "samples"
SIZE = 32

def _base_rgb() -> np.ndarray:
    """Buat pola RGB 8-bit dengan warnaلياً, abu-abu, dan area gelap."""
    rgb = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    rgb[:, : SIZE // 3] = (220, 40, 30)  # merah
    rgb[:, SIZE // 3 : 2 * SIZE // 3] = (40, 180, 60)  # hijau
    rgb[:, 2 * SIZE // 3 :] = (30, 60, 200)  # biru
    rgb[: SIZE // 4, :] = (30, 30, 30)  # baris gelap untuk menguji ambang
    rgb[SIZE // 2 :, :] = np.clip(rgb[SIZE // 2 :, :].astype(np.int16) + 60, 0, 255).astype(np.uint8)
    return rgb

def _to_bgr(rgb: np.ndarray) -> np.ndarray:
    """Ubah RGB ke BGR sesuai urutan kanal OpenCV."""
    return np.ascontiguousarray(rgb[:, :, ::-1])

def make_samples(directory: Path = SAMPLES_DIR) -> list[Path]:
    """Tulis tiga berkas citra contoh.
    Args:
        directory: Folder tujuan berkas contoh.
    Returns:
        Daftar Path berkas yang ditulis.
    Raises:
        OSError: Penulisan berkas gagal.
    """
    directory.mkdir(parents=True, exist_ok=True)
    rgb = _base_rgb()
    bgr = _to_bgr(rgb)
    written: list[Path] = []

    # BMP 8-bit
    bmp_path = directory / "test.bmp"
    if not cv2.imwrite(str(bmp_path), bgr):
        raise OSError(f"Gagal menulis {bmp_path}")
    written.append(bmp_path)

    # PNG 16-bit per kanal
    bgr16 = (bgr.astype(np.uint16) * 257).astype(np.uint16)
    png16_path = directory / "test_16bit.png"
    if not cv2.imwrite(str(png16_path), bgr16):
        raise OSError(f"Gagal menulis {png16_path}")
    written.append(png16_path)

    # JPEG 8-bit
    jpg_path = directory / "test.jpg"
    if not cv2.imwrite(str(jpg_path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
        raise OSError(f"Gagal menulis {jpg_path}")
    written.append(jpg_path)
    return written

if __name__ == "__main__":
    for item in make_samples():
        print(f"dibuat: {item}")