"""Inspeksi matriks citra (FR-3): pencetak potongan matriks dan pemeriksa bentuk.
Setiap keluaran diperiksa bentuk (H, W) dan rentang nilainya sehingga kebenaran
pipeline dapat dibuktikan lewat angka di terminal, tanpa perlu GUI.
"""

from __future__ import annotations
import numpy as np
from .image_io import CHANNEL_LABELS

DEFAULT_PREVIEW = 5

_COLUMN_WIDTH = 6

_LIMITS = {"uint8": (0, 255), "uint16": (0, 65535)}


class MatrixCheckError(AssertionError):
    """Pelanggaran terhadap jaminan bentuk atau kedalaman bit (FR-3)."""


def assert_shape(array: np.ndarray, height: int, width: int, label: str) -> None:
    """Pastikan dimensi spasial matriks sama dengan citra masukan (FR-3).

    Args:
        array: Matriks hasil, 2D atau 3D.
        height: Nilai H citra masukan.
        width: Nilai W citra masukan.
        label: Nama keluaran untuk pesan galat.

    Raises:
        MatrixCheckError: Dimensi matriks tidak sama dengan (H, W).
    """
    if array.ndim < 2:
        raise MatrixCheckError(f"{label}: matriks harus minimal 2 dimensi, diterima {array.shape}")
    actual_h, actual_w = int(array.shape[0]), int(array.shape[1])
    if (actual_h, actual_w) != (height, width):
        raise MatrixCheckError(
            f"{label}: resolusi tidak terjaga. Masukan (H, W)=({height}, {width}), "
            f"keluaran (H, W)=({actual_h}, {actual_w})"
        )


def matrix_summary(array: np.ndarray, label: str) -> dict[str, object]:
    """Ringkas sifat matriks untuk laporan dan CSV.

    Args:
        array: Matriks hasil.
        label: Nama representasi.

    Returns:
        Dict berisi label, shape, dtype, min, max, mean, dan nilai unik (maksimal 5).
    """
    unique_values = np.unique(array)
    preview_unique = unique_values[:5].tolist()
    if unique_values.size > 5:
        preview_unique.append("...")
    return {
        "representation": label,
        "shape": "x".join(str(dim) for dim in array.shape),
        "dtype": str(array.dtype),
        "min": int(array.min()),
        "max": int(array.max()),
        "mean": round(float(array.mean()), 4),
        "unique_values": preview_unique,
        "total_pixels": int(array.size),
    }


def format_slice(array: np.ndarray, preview: int = DEFAULT_PREVIEW) -> str:
    """Format potongan n x n dari sudut kiri atas sebagai teks tabel.

    Args:
        array: Matriks 2D atau 3D.
        preview: Sisi potongan (potongan dibatasi ukuran matriks).

    Returns:
        String berisi baris-baris nilai piksel yang disejikan kolom.
    """
    size = min(preview, array.shape[0], array.shape[1])
    block = array[:size, :size]

    if block.ndim == 2:
        rows = [block]
        labels = [None]
    else:
        rows = [block[:, :, c] for c in range(block.shape[2])]
        labels = list(CHANNEL_LABELS[: block.shape[2]])

    lines: list[str] = []
    for channel, plane in zip(labels, rows):
        if channel is not None:
            lines.append(f"  kanal {channel}:")
        header = " ".join(f"{'r' + str(r):>{_COLUMN_WIDTH}}" for r in range(size))
        lines.append(f"    {'':<6}{header}")
        for r in range(size):
            cells = " ".join(f"{int(v):>{_COLUMN_WIDTH}d}" for v in plane[r])
            lines.append(f"    {('r' + str(r)):<6}{cells}")
    return "\n".join(lines)


def print_matrix(array: np.ndarray, label: str, preview: int = DEFAULT_PREVIEW) -> dict[str, object]:
    """Cetak sifat matriks dan potongan ke terminal, lalu kembalikan ringkasannya.

    Args:
        array: Matriks hasil yang akan diperiksa.
        label: Nama representasi, mis. "Biner".
        preview: Sisi potongan yang dicetak.

    Returns:
        Dict ringkasan dari :func:`matrix_summary`.
    """
    summary = matrix_summary(array, label)
    low, high = _LIMITS.get(summary["dtype"], (None, None))

    print(f"\n--- {label} ---")
    print(f"bentuk (shape) : {summary['shape']}")
    print(f"tipe data      : {summary['dtype']}")
    print(f"nilai min/max  : {summary['min']} / {summary['max']}", end="")
    print(f"  (rentang {low}..{high})" if low is not None else "")
    print(f"nilai rata-rata: {summary['mean']}")
    print(f"total piksel   : {summary['total_pixels']}")
    print(f"nilai unik     : {summary['unique_values']}")
    print(f"matriks {preview}x{preview} kiri atas:")
    print(format_slice(array, preview))
    return summary


def verify_range(array: np.ndarray, low: int, high: int, label: str) -> None:
    """Pastikan seluruh nilai matriks berada pada rentang yang diperbolehkan.

    Args:
        array: Matriks hasil.
        low: Nilai minimum yang sah.
        high: Nilai maksimum yang sah.
        label: Nama keluaran untuk pesan galat.

    Raises:
        MatrixCheckError: Ada nilai di luar rentang.
    """
    actual_min, actual_max = int(array.min()), int(array.max())
    if actual_min < low or actual_max > high:
        raise MatrixCheckError(
            f"{label}: nilai di luar rentang {low}..{high} "
            f"(min={actual_min}, max={actual_max})"
        )