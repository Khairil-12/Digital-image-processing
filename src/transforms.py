"""Transformasi representasi citra (FR-2).

Tiga keluaran wajib:
  * Biner (Citra Biner)          -> (H, W)     uint8   {0, 255}
  * Keabu-abuan 16 bit            -> (H, W)     uint16  0..65535
  * RGB 8 bit                     -> (H, W, 3)  uint8   0..255 per kanal

Luminansi dihitung di float32 [0,1] lalu dikali 65535, jadi presisi 16-bit
benar-benar terjaga tanpa perantara 8-bit.
"""

from __future__ import annotations

import cv2
import numpy as np

#: Ambang penskalaan 8-bit ke 16-bit: 255 * 257 = 65535 (FR-2.2, catatan).
SCALE_8_TO_16 = 257

#: Bobot luminansi ITU-R BT.601.
LUMA_B, LUMA_G, LUMA_R = 0.114, 0.587, 0.299

#: Nilai piksel untuk putih dan hitam pada citra biner (FR-2.1).
WHITE = 255
BLACK = 0

DEFAULT_THRESHOLD = 127

#: Batas nilai yang dicetak pada matriks 16-bit (FR-3.2).
UINT16_MAX = 65535


def to_greyscale16(rgb_f32: np.ndarray) -> np.ndarray:
    """Ubah citra RGB float32 [0,1] menjadi keabu-abuan 16-bit (FR-2.2).

    Args:
        rgb_f32: (H, W, 3) float32 dalam rentang [0,1], urutan kanal R, G, B.

    Returns:
        (H, W) uint16 dalam rentang 0..65535.

    Raises:
        ValueError: Bentuk atau rentang masukan tidak sesuai.
    """
    if rgb_f32.ndim != 3 or rgb_f32.shape[2] != 3:
        raise ValueError(f"rgb_f32 harus (H, W, 3), diterima {rgb_f32.shape}")
    if rgb_f32.dtype != np.float32:
        raise ValueError(f"rgb_f32 harus float32, diterima {rgb_f32.dtype}")
    if rgb_f32.size and (float(rgb_f32.min()) < -1e-6 or float(rgb_f32.max()) > 1.0 + 1e-6):
        raise ValueError(f"rgb_f32 harus pada rentang [0,1], ditemukan min={rgb_f32.min()}, max={rgb_f32.max()}")

    luma = (
        LUMA_R * rgb_f32[:, :, 0]
        + LUMA_G * rgb_f32[:, :, 1]
        + LUMA_B * rgb_f32[:, :, 2]
    )
    return np.clip(np.rint(luma * UINT16_MAX), 0, UINT16_MAX).astype(np.uint16)


def to_biner(greyscale16: np.ndarray, threshold: int = DEFAULT_THRESHOLD) -> np.ndarray:
    """Ubah keabu-abuan 16-bit menjadi citra biner (FR-2.1).

    Ambang T dinyatakan pada skala 0..255 lalu dikonversi ke ekuivalen 16-bit
    sebagai ``T * 257``, sehingga aturan ``f(x,y) >= T`` tetap berlaku.

    Args:
        greyscale16: (H, W) uint16 hasil :func:`to_greyscale16`.
        threshold: Ambang T pada skala 0..255. Gunakan "auto" melalui
            :func:`otsu_threshold_8`.

    Returns:
        (H, W) uint8 dengan nilai tepat 0 atau 255.

    Raises:
        ValueError: Masukan bukan matriks 2D uint16 atau ambang di luar 0..255.
    """
    if greyscale16.ndim != 2:
        raise ValueError(f"greyscale16 harus (H, W), diterima {greyscale16.shape}")
    if greyscale16.dtype != np.uint16:
        raise ValueError(f"greyscale16 harus uint16, diterima {greyscale16.dtype}")
    if not 0 <= threshold <= 255:
        raise ValueError(f"threshold harus di rentang 0..255, diterima {threshold}")

    threshold16 = int(threshold) * SCALE_8_TO_16
    return np.where(greyscale16 >= threshold16, np.uint8(WHITE), np.uint8(BLACK))


def to_rgb8(bgr8: np.ndarray) -> np.ndarray:
    """Ubah baseline BGR 8-bit menjadi RGB 8-bit (FR-2.3).

    Args:
        bgr8: (H, W, 3) uint8 urutan kanal B, G, R.

    Returns:
        (H, W, 3) uint8 urutan kanal R, G, B.

    Raises:
        ValueError: Bentuk atau dtype masukan tidak sesuai.
    """
    if bgr8.ndim != 3 or bgr8.shape[2] != 3:
        raise ValueError(f"bgr8 harus (H, W, 3), diterima {bgr8.shape}")
    if bgr8.dtype != np.uint8:
        raise ValueError(f"bgr8 harus uint8, diterima {bgr8.dtype}")
    return np.ascontiguousarray(bgr8[:, :, ::-1])


def otsu_threshold_8(greyscale16: np.ndarray) -> int:
    """Hitung ambang Otsu pada skala 0..255 dari matriks keabu-abuan 16-bit.

    Args:
        greyscale16: (H, W) uint16.

    Returns:
        Ambang T bulat pada rentang 0..255 hasil Otsu.
    """
    luma8 = np.clip(np.rint(greyscale16.astype(np.float32) / SCALE_8_TO_16), 0, 255).astype(np.uint8)
    _threshold, biner = cv2.threshold(luma8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.threshold mengembalikan ambang sebagai float64 dalam skala citra.
    return int(np.clip(round(float(_threshold)), 0, 255))


def resolve_threshold(greyscale16: np.ndarray, threshold: str | int) -> int:
    """Terjemahkan ambang dari CLI menjadi angka pada skala 0..255.

    Args:
        greyscale16: (H, W) uint16 untuk perhitungan Otsu.
        threshold: Integer 0..255, atau string "auto" untuk Otsu.

    Returns:
        Ambang final pada rentang 0..255.

    Raises:
        ValueError: Nilai ambang tidak valid.
    """
    if isinstance(threshold, int):
        if not 0 <= threshold <= 255:
            raise ValueError(f"threshold harus di rentang 0..255, diterima {threshold}")
        return threshold
    if threshold.strip().lower() == "auto":
        return otsu_threshold_8(greyscale16)
    raise ValueError(f"threshold harus integer 0..255 atau 'auto', diterima {threshold!r}")