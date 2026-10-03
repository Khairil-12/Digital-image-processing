"""Tampilan visual keluaran (FR-4).

Menampilkan citra asal bersama tiga varian dalam figur 2x2. Array uint16 hanya
dinormalisasi untuk keperluan tampil; array aslinya tidak pernah diubah.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .report import VARIANT_FILES

#: Batas atas tampilan per dtype supaya citra 16-bit tidak tampak hitam.
_DISPLAY_VMAX = {"uint8": 255, "uint16": 65535}


def _panel(ax: "plt.Axes", array: np.ndarray, title: str, gray: bool) -> None:
    """Gambar satu panel: array uint16 dinormalisasi tanpa mengubah aslinya."""
    vmax = _DISPLAY_VMAX.get(str(array.dtype))
    shown = array.astype(np.float32) / vmax if vmax else array.astype(np.float32)
    kwargs = {"cmap": "gray", "vmin": 0} if gray else {"vmin": 0}
    if vmax:
        kwargs["vmax"] = vmax
    ax.imshow(shown, **kwargs)
    ax.set_title(f"{title}\n{array.dtype} {array.shape}", fontsize=9)
    ax.axis("off")


def display_all(
    original: np.ndarray,
    variants: dict[str, np.ndarray],
    save_path: str | None = None,
    block: bool = False,
) -> str | None:
    """Bangun figur 2x2 berisi citra asal dan ketiga varian (FR-4).

    Args:
        original: (H, W, 3) uint8 RGB untuk panel pertama.
        variants: Matriks keluaran berlabel "Biner", "Keabu-abuan 16 bit", "RGB 8 bit".
        save_path: Bila diisi, figur disimpan ke path ini sebagai PNG.
        block: Bila True, jendela tunggu sampai ditutup (FR-4 window controls).

    Returns:
        Path figur yang disimpan, atau None bila tidak ada.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.ravel()

    _panel(axes[0], original, "Asli (RGB 8 bit)", gray=False)

    order = [label for label in VARIANT_FILES if label in variants]
    for ax, label in zip(axes[1:], order):
        array = variants[label]
        _panel(ax, array, f"{label}", gray=array.ndim == 2)
    for ax in axes[len(order) + 1 :]:
        ax.axis("off")

    fig.suptitle("Analisis Citra: Biner, Keabu-abuan 16 bit, RGB 8 bit", fontsize=12)
    fig.tight_layout()

    if save_path is None and not block:
        return None
    if save_path is not None:
        fig.savefig(save_path, dpi=110)
    if block:
        plt.show()  # jendela menahan diri sampai pengguna menutupnya (FR-4)
    plt.close(fig)
    return save_path