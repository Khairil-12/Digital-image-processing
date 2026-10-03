"""Inputan citra (FR-1): validasi berkas, deteksi kedalaman bit, dan baseline BGR 8-bit.

Format yang didukung: .bmp, .jpg/.jpeg, .png.
Pembacaan memakai IMREAD_UNCHANGED lalu dinormalisasi ke float32 [0,1] supaya citra
16-bit per kanal tidak terpotong jadi 8-bit secara diam-diam.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

SUPPORTED_EXTENSIONS = (".bmp", ".jpg", ".jpeg", ".png")

#: Nama kanal RGB untuk pencetakan matriks (FR-3.3).
CHANNEL_LABELS = ("R", "G", "B")


class ImageLoadError(Exception):
    """Galat yang aman ditampilkan ke pengguna tanpa traceback mentah (FR-1)."""


@dataclass(frozen=True)
class LoadedImage:
    """Hasil pembacaan satu berkas citra.

    Attributes:
        path: Lokasi berkas asal.
        name: Nama berkas untuk laporan.
        rgb_f32: (H, W, 3) float32 dalam rentang [0,1], urutan kanal R, G, B.
        bgr8: (H, W, 3) uint8, baseline kerja 8-bit urutan B, G, R.
        source_bit_depth: Kedalaman bit per kanal berkas asal (8 atau 16).
        source_channels: Jumlah kanal berkas asal sebelum channel alpha dibuang.
    """

    path: Path
    name: str
    rgb_f32: np.ndarray
    bgr8: np.ndarray
    source_bit_depth: int
    source_channels: int

    @property
    def height(self) -> int:
        """Jumlah baris matriks (H)."""
        return int(self.rgb_f32.shape[0])

    @property
    def width(self) -> int:
        """Jumlah kolom matriks (W)."""
        return int(self.rgb_f32.shape[1])


def _expand_to_three_channels(raw: np.ndarray) -> np.ndarray:
    """Buang kanal alpha dan kembangkan citra abu-abu jadi 3 kanal, dtype dipertahankan."""
    if raw.ndim == 2:
        return cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    if raw.shape[2] == 4:  # BGRA -> BGR
        return np.ascontiguousarray(raw[:, :, :3])
    return raw


def _to_unit_float(raw: np.ndarray) -> np.ndarray:
    """Normalisasi kanal integer ke float32 [0,1] sesuai kedalaman bit asalnya."""
    if raw.dtype == np.uint8:
        return raw.astype(np.float32) / 255.0
    if raw.dtype == np.uint16:
        return raw.astype(np.float32) / 65535.0
    if raw.dtype in (np.float32, np.float64):
        unit = raw.astype(np.float32)
        if unit.size and float(unit.max()) > 1.5:
            raise ImageLoadError(
                f"Citra berformat pecahan (float) punya nilai {float(unit.max()):.3f}; "
                "rentang [0,1] diharapkan, file mungkin rusak."
            )
        return np.clip(unit, 0.0, 1.0)
    raise ImageLoadError(f"Tipe data citra tidak didukung: {raw.dtype}")


def load_image(path: str | Path) -> LoadedImage:
    """Baca satu berkas citra dan siapkan baseline kerja.

    Args:
        path: Lokasi berkas .bmp, .jpg, .jpeg, atau .png.

    Returns:
        LoadedImage berisi array float32 [0,1], baseline BGR 8-bit, dan metadata.

    Raises:
        ImageLoadError: Berkas tidak ada, format tidak didukung, header rusak,
            atau tipe data piksel tidak dikenali.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise ImageLoadError(f"Berkas tidak ditemukan: {file_path}")
    if not file_path.is_file():
        raise ImageLoadError(f"Bukan berkas: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        shown = suffix if suffix else "(tanpa ekstensi)"
        raise ImageLoadError(
            f"Format tidak didukung: {shown}. "
            f"Hanya yang diizinkan: {', '.join(SUPPORTED_EXTENSIONS)}."
        )

    raw = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ImageLoadError(
            f"Gagal membaca citra: {file_path.name}. Header rusak atau bukan berkas citra."
        )

    source_channels = 1 if raw.ndim == 2 else int(raw.shape[2])
    raw = _expand_to_three_channels(raw)
    source_bit_depth = 16 if raw.dtype == np.uint16 else 8

    unit = _to_unit_float(raw)  # BGR float32 [0,1]
    rgb_f32 = np.ascontiguousarray(unit[:, :, ::-1])  # BGR -> RGB
    bgr8 = np.clip(np.rint(unit * 255.0), 0, 255).astype(np.uint8)

    return LoadedImage(
        path=file_path,
        name=file_path.name,
        rgb_f32=rgb_f32,
        bgr8=bgr8,
        source_bit_depth=source_bit_depth,
        source_channels=source_channels,
    )


def collect_inputs(targets: list[str]) -> list[Path]:
    """Kumpulkan berkas citra dari daftar path atau folder (FR-1).

    Args:
        targets: Path berkas atau folder. Folder dipindai isinya (tanpa rekursi).

    Returns:
        Daftar Path citra yang sudah ada, unik, dan terurut.

    Raises:
        ImageLoadError: Sebuah target tidak ada, atau tidak ada citra yang cocok.
    """
    found: list[Path] = []
    for target in targets:
        candidate = Path(target)
        if not candidate.exists():
            raise ImageLoadError(f"Input tidak ditemukan: {candidate}")
        if candidate.is_dir():
            found.extend(
                sorted(
                    entry
                    for entry in candidate.iterdir()
                    if entry.is_file() and entry.suffix.lower() in SUPPORTED_EXTENSIONS
                )
            )
        else:
            found.append(candidate)

    unique: list[Path] = []
    seen: set[Path] = set()
    for item in found:
        resolved = item.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(item)

    if not unique:
        raise ImageLoadError(
            "Tidak ada berkas citra yang cocok "
            f"({', '.join(SUPPORTED_EXTENSIONS)}) pada input yang diberikan."
        )
    return unique