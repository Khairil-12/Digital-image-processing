"""Penyimpanan keluaran dan laporan (FR-5).
Menulis matriks (.npy + .txt), gambar varian, BMP 1-bit, report.txt, dan summary.csv.
summary.csv memakai pemisah titik koma dan BOM UTF-8 agar langsung tersusun kolom di Excel.
"""

from __future__ import annotations
import csv
from dataclasses import dataclass, field
from pathlib import Path
import cv2
import numpy as np
from .image_io import CHANNEL_LABELS
from .matrix_view import format_slice, matrix_summary

VARIANT_FILES = {
    "Biner": "biner",
    "Keabu-abuan 16 bit": "greyscale16",
    "RGB 8 bit": "rgb8",
}

_MAX_TXT_ELEMENTS = 64 * 64

_CSV_DELIMITER = ";"
_CSV_ENCODING = "utf-8-sig"

_CSV_FIELDS = (
    "file_input",
    "source_bit_depth",
    "representation",
    "shape",
    "dtype",
    "min",
    "max",
    "mean",
    "waktu_proses_detik",
)

_LINE = "=" * 78


@dataclass
class OutputBundle:
    """Kumpulan matriks keluaran beserta waktu prosesnya (FR-5)."""

    variants: dict[str, np.ndarray] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)

    def add(self, label: str, array: np.ndarray, seconds: float) -> None:
        """Daftarkan satu matriks keluaran beserta durasi prosesnya."""
        self.variants[label] = array
        self.timings[label] = seconds


def _write_matrix_txt(path: Path, array: np.ndarray) -> None:
    """Tulis matriks ke .txt; matriks besar dipangkas agar berkas tetap ringkas."""
    if array.ndim == 3:
        # savetxt hanya menerima 1D/2D, jadi tiap kanal RGB ditulis berurutan.
        with path.open("w", encoding="utf-8") as handle:
            for index, label in enumerate(CHANNEL_LABELS[: array.shape[2]]):
                handle.write(f"# kanal {label}\n")
                handle.write(np.array2string(array[:, :, index], separator="\t", threshold=10**9, formatter={"all": lambda v: f"{int(v):d}"}))
                handle.write("\n")
        return
    if array.size <= _MAX_TXT_ELEMENTS:
        np.savetxt(path, array, fmt="%d", delimiter="\t")
        return
    preview = int(_MAX_TXT_ELEMENTS**0.5)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            f"# Matriks dipangkas karena berukuran {array.size} sel (> {_MAX_TXT_ELEMENTS}).\n"
            f"# Bentuk penuh: {array.shape}, dtype: {array.dtype}\n"
            f"# Potongan kiri atas {preview}x{preview}:\n"
        )
        handle.write(format_slice(array, preview))
        handle.write("\n")


def _write_variant_images(output_dir: Path, variants: dict[str, np.ndarray]) -> list[str]:
    """Tulis gambar varian; keabu-abuan 16-bit disimpan sebagai PNG 16-bit."""
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    for label, array in variants.items():
        stem = VARIANT_FILES[label]
        if array.ndim == 2:
            # uint16 2D ditulis apa adanya supaya PNG benar-benar 16 bit per kanal.
            cv2.imwrite(str(images_dir / f"{stem}.png"), array)
            written.append(str(images_dir / f"{stem}.png"))
        else:
            cv2.imwrite(str(images_dir / f"{stem}.png"), array[:, :, ::-1])
            written.append(str(images_dir / f"{stem}.png"))

    return written


def _write_binary_bmp(output_dir: Path, biner: np.ndarray) -> str:
    """Tulis citra biner sebagai BMP 1-bit-per-piksel (FR-5).

    OpenCV tidak menulis BMP 1-bit langsung, jadi bit dikemas manual dengan
    setiap byte menyimpan 8 piksel (bit tertinggi dipakai lebih dulu) dan baris
    disimpan dari bawah ke atas sesuai konvensi BMP.

    Args:
        output_dir: Folder keluaran utama.
        biner: (H, W) uint8 berisi 0 atau 255.

    Returns:
        Path berkas BMP yang ditulis.
    """
    height, width = biner.shape
    bits_per_row = ((width + 31) // 32) * 32  # baris BMP 1-bit harus kelipatan 32 bit
    packed_row_bytes = bits_per_row // 8
    padding = bytes(packed_row_bytes - (width + 7) // 8)
    pixels = biner > 127  # True = putih (bit 1)

    body = bytearray()
    for row in range(height - 1, -1, -1):  # BMP menyimpan baris dari bawah ke atas
        packed = bytearray()
        for start in range(0, width, 8):
            chunk = pixels[row, start : start + 8]
            byte = 0
            for index, value in enumerate(chunk):
                if value:
                    byte |= 1 << (7 - index)  # MSB diisi lebih dulu
            packed.append(byte)
        body.extend(packed)
        body.extend(padding)

    palette = bytes((0, 0, 0, 0, 255, 255, 255, 255))  # 2 entri BGRA, masing-masing 4 byte
    offset = 14 + 40 + len(palette)
    file_size = offset + len(body)

    dib_size = 40
    dib = (
        dib_size.to_bytes(4, "little")  # biSize
        + width.to_bytes(4, "little", signed=True)  # biWidth
        + height.to_bytes(4, "little", signed=True)  # biHeight
        + (1).to_bytes(2, "little")  # biPlanes
        + (1).to_bytes(2, "little")  # biBitCount = 1
        + (0).to_bytes(4, "little")  # biCompression = BI_RGB
        + len(body).to_bytes(4, "little")  # biSizeImage = ukuran data piksel
        + (2835).to_bytes(4, "little")  # biXPelsPerMeter (72 DPI)
        + (2835).to_bytes(4, "little")  # biYPelsPerMeter (72 DPI)
        + (2).to_bytes(4, "little")  # biClrUsed = 2 entri palet
        + (2).to_bytes(4, "little")  # biClrImportant = 2 entri palet
    )
    assert len(dib) == dib_size, f"DIB header harus {dib_size} byte, bukan {len(dib)}"
    header = b"BM" + file_size.to_bytes(4, "little") + b"\x00\x00\x00\x00" + offset.to_bytes(4, "little")

    path = output_dir / "images" / "biner_1bit.bmp"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + dib + palette + bytes(body))
    return str(path)


def _append_summary_csv(
    path: Path,
    file_name: str,
    source_bit_depth: int,
    variants: dict[str, np.ndarray],
    timings: dict[str, float],
) -> None:
    """Tambahkan baris satu berkas ke summary.csv dengan pemisah titik koma."""
    write_header = not path.exists()
    with path.open("a", newline="", encoding=_CSV_ENCODING) as handle:
        writer = csv.writer(handle, delimiter=_CSV_DELIMITER)
        if write_header:
            writer.writerow(_CSV_FIELDS)
        for label, array in variants.items():
            stats = matrix_summary(array, label)
            writer.writerow(
                [
                    file_name,
                    source_bit_depth,
                    label,
                    stats["shape"],
                    stats["dtype"],
                    stats["min"],
                    stats["max"],
                    stats["mean"],
                    round(timings.get(label, 0.0), 6),
                ]
            )


def save_outputs(
    output_dir: Path,
    file_name: str,
    source_bit_depth: int,
    height: int,
    width: int,
    bundle: OutputBundle,
    threshold: int,
    preview: int,
) -> dict[str, object]:
    """Tulis seluruh artefak keluaran yang diminta FR-5.

    Args:
        output_dir: Folder keluaran, dibuat bila belum ada.
        file_name: Nama berkas citra masukan.
        source_bit_depth: Kedalaman bit berkas asal (8 atau 16).
        height: Nilai H citra masukan.
        width: Nilai W citra masukan.
        bundle: Matriks keluaran dan waktu prosesnya.
        threshold: Ambang biner yang dipakai, pada skala 0..255.
        preview: Sisi potongan matriks pada laporan.

    Returns:
        Dict berisi daftar berkas yang ditulis dan ringkasan tiap keluaran.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    matrices_dir = output_dir / "matrices"
    matrices_dir.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    summaries: dict[str, object] = {}

    for label, array in bundle.variants.items():
        stem = VARIANT_FILES[label]
        npy_path = matrices_dir / f"{stem}_matrix.npy"
        txt_path = matrices_dir / f"{stem}_matrix.txt"
        np.save(npy_path, array)
        _write_matrix_txt(txt_path, array)
        written.extend([str(npy_path), str(txt_path)])
        summaries[label] = matrix_summary(array, label)

    written.extend(_write_variant_images(output_dir, bundle.variants))
    written.append(_write_binary_bmp(output_dir, bundle.variants["Biner"]))

    csv_path = output_dir / "summary.csv"
    _append_summary_csv(csv_path, file_name, source_bit_depth, bundle.variants, bundle.timings)
    written.append(str(csv_path))

    report_path = output_dir / "report.txt"
    report_path.write_text(
        _build_report(file_name, source_bit_depth, height, width, threshold, preview, summaries, bundle, written),
        encoding="utf-8",
    )
    written.append(str(report_path))

    return {"written": written, "summaries": summaries}


def _build_report(
    file_name: str,
    source_bit_depth: int,
    height: int,
    width: int,
    threshold: int,
    preview: int,
    summaries: dict[str, object],
    bundle: OutputBundle,
    written: list[str],
) -> str:
    """Susun isi report.txt."""
    lines = [
        _LINE,
        "Laporan Analisis Citra - Praktikum Pengolahan Citra Digital",
        _LINE,
        "",
        f"Berkas masukan        : {file_name}",
        f"Kedalaman bit asal    : {source_bit_depth} bit per kanal",
        f"Resolusi (H x W)      : {height} x {width}",
        f"Ambang biner (T)      : {threshold} (ekuivalen 16-bit: {threshold * 257})",
        "",
        "KELUARAN:",
        _LINE,
        f"{'Representasi':<24}{'Bentuk':<16}{'dtype':<10}{'min':>8}{'max':>8}{'mean':>12}{'waktu(s)':>11}",
    ]
    for label in VARIANT_FILES:
        stats = summaries[label]
        lines.append(
            f"{label:<24}{stats['shape']:<16}{stats['dtype']:<10}"
            f"{stats['min']:>8}{stats['max']:>8}{stats['mean']:>12.4f}"
            f"{bundle.timings.get(label, 0.0):>11.6f}"
        )

    lines += ["", "DETAIL MATRIKS:", _LINE]
    for label, array in bundle.variants.items():
        stats = summaries[label]
        lines += [
            "",
            f"[{label}]",
            f"  bentuk    : {stats['shape']}",
            f"  dtype     : {stats['dtype']}",
            f"  min/max   : {stats['min']} / {stats['max']}",
            f"  rata-rata : {stats['mean']}",
            f"  piksel    : {stats['total_pixels']}",
            f"  nilai unik: {stats['unique_values']}",
            f"  matriks {preview}x{preview} kiri atas:",
        ]
        lines.append(format_slice(array, preview))

    lines += ["", "BERKAS KELUARAN:", _LINE]
    lines += [f"  {path}" for path in written]
    lines += ["", ""]
    return "\n".join(lines)