"""Titik masuk program: analisis citra multi-format.

Menghasilkan tiga representasi (Biner, Keabu-abuan 16 bit, RGB 8 bit) dari berkas
.bmp, .jpg, dan .png, lengkap dengan matriks, laporan, dan tampilan visual.

Contoh:
    python main.py --input samples/
    python main.py --input foto.png --threshold 150
    python main.py --input foto.png --threshold auto --no-display
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from src import transforms
from src.display import display_all
from src.image_io import ImageLoadError, collect_inputs, load_image
from src.matrix_view import (
    DEFAULT_PREVIEW,
    MatrixCheckError,
    assert_shape,
    print_matrix,
    verify_range,
)
from src.report import OutputBundle, save_outputs

#: Label keluaran dalam urutan tampil; kunci ini sama dengan report.VARIANT_FILES.
LABELS = ("Biner", "Keabu-abuan 16 bit", "RGB 8 bit")


def build_parser() -> argparse.ArgumentParser:
    """Susun parser argumen baris perintah."""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Konversi citra .bmp/.jpg/.png menjadi Biner, Keabu-abuan 16 bit, dan RGB 8 bit.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Contoh:\n"
            "  python main.py                                    # mode interaktif\n"
            "  python main.py --input samples/\n"
            "  python main.py --input foto.png --threshold 150\n"
            "  python main.py --input foto.png --threshold auto --no-display"
        ),
    )
    parser.add_argument(
        "--input",
        nargs="*",
        metavar="PATH",
        help="Berkas citra atau folder berisi citra (.bmp, .jpg, .jpeg, .png). "
             "Jika dikosongkan, program akan meminta input interaktif.",
    )
    parser.add_argument(
        "--threshold",
        default=str(transforms.DEFAULT_THRESHOLD),
        help="Ambang biner T pada skala 0..255, atau 'auto' untuk Otsu (default: 127)",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=DEFAULT_PREVIEW,
        help=f"Sisi potongan matriks yang dicetak (default: {DEFAULT_PREVIEW})",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Folder keluaran untuk matriks, gambar, dan laporan (default: output)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--no-display",
        action="store_true",
        help="Lewati tampilan visual sepenuhnya",
    )
    group.add_argument(
        "--save-figure",
        action="store_true",
        help="Simpan figur tampilan ke <output>/figure_<nama>.png tanpa membuka jendela",
    )
    return parser


def _parse_threshold(raw: str) -> str | int:
    """Ubah ambang dari CLI menjadi int atau string 'auto'."""
    text = raw.strip().lower()
    if text == "auto":
        return "auto"
    try:
        return int(text)
    except ValueError:
        raise ValueError(f"--threshold harus integer 0..255 atau 'auto', diterima {raw!r}") from None


def _prompt_for_input() -> list[str]:
    """Minta pengguna mengetik path berkas/folder secara interaktif (FR-1, Metode B).

    Terus meminta hingga path yang diberikan valid (ada dan mengandung
    citra yang didukung), atau pengguna memilih keluar.
    Validasi dilakukan di dalam loop sehingga path salah hanya menyebabkan
    re-prompt, bukan crash.

    Returns:
        Daftar string path yang sudah divalidasi oleh collect_inputs().
    """
    from src.image_io import collect_inputs

    print("=" * 78)
    print("MODE INTERAKTIF — Ketik path citra yang ingin diproses.")
    print("Format yang didukung: .bmp, .jpg, .jpeg, .png")
    print("Ketik path folder untuk memproses semua citra di dalamnya.")
    print("Ketik 'q' / 'quit' / 'exit' / 'keluar' untuk keluar.")
    print("Tekan Ctrl+C untuk keluar.")
    print("=" * 78)

    while True:
        try:
            raw = input("\nPath berkas/folder (atau 'q' untuk keluar): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nDibatalkan oleh pengguna.")
            raise SystemExit(130)  # 128 + SIGINT(2)

        if raw.lower() in ("q", "quit", "exit", "keluar"):
            print("Keluar dari program.")
            raise SystemExit(0)

        if not raw:
            print("  [!] Path tidak boleh kosong. Coba lagi.")
            continue

        # Validasi di dalam loop — path salah → pesan galat → re-prompt (FR-1).
        try:
            paths = collect_inputs([raw])
            return [str(p) for p in paths]
        except ImageLoadError as exc:
            print(f"  [!] {exc} Coba lagi.")


def process_one(
    path: Path,
    threshold_raw: str | int,
    preview: int,
    output_dir: Path,
    show: bool,
    save_figure: bool,
) -> dict[str, object]:
    """Proses satu berkas citra dari pembacaan sampai seluruh artefak tertulis.

    Args:
        path: Lokasi berkas citra.
        threshold_raw: Ambang sebagai integer 0..255 atau "auto".
        preview: Sisi potongan matriks untuk dicetak dan dilaporkan.
        output_dir: Folder keluaran utama.
        show: Bila True, figur ditampilkan di jendela.
        save_figure: Bila True, figur disimpan ke PNG.

    Returns:
        Dict berisi ringkasan keluaran, daftar berkas tertulis, dan figur.

    Raises:
        ImageLoadError: Berkas bermasalah.
        MatrixCheckError: Jaminan bentuk atau rentang matriks dilanggar.
        ValueError: Nilai ambang tidak valid.
    """
    loaded = load_image(path)
    height, width = loaded.height, loaded.width

    print("=" * 78)
    print(f"BERKAS  : {loaded.name}")
    print(f"RESOLUSI: {width} x {height} piksel")
    print(f"BIT ASAL: {loaded.source_bit_depth} bit per kanal, {loaded.source_channels} kanal")
    print(f"FORMAT  : {loaded.path.suffix.lower()}")
    print("=" * 78)

    started = time.perf_counter()
    greyscale16 = transforms.to_greyscale16(loaded.rgb_f32)
    t_grey = time.perf_counter() - started

    threshold = transforms.resolve_threshold(greyscale16, threshold_raw)
    print(f"Ambang biner dipakai: T = {threshold} (ekuivalen 16-bit: {threshold * transforms.SCALE_8_TO_16})")

    started = time.perf_counter()
    biner = transforms.to_biner(greyscale16, threshold)
    t_biner = time.perf_counter() - started

    started = time.perf_counter()
    rgb8 = transforms.to_rgb8(loaded.bgr8)
    t_rgb = time.perf_counter() - started

    bundle = OutputBundle()
    bundle.add("Biner", biner, t_biner)
    bundle.add("Keabu-abuan 16 bit", greyscale16, t_grey)
    bundle.add("RGB 8 bit", rgb8, t_rgb)

    # FR-3: resolusi terjaga dan nilai berada pada rentang yang dijanjikan.
    for label in LABELS:
        assert_shape(bundle.variants[label], height, width, label)
    verify_range(biner, 0, 255, "Biner")
    verify_range(greyscale16, 0, 65535, "Keabu-abuan 16 bit")
    verify_range(rgb8, 0, 255, "RGB 8 bit")

    for label in LABELS:
        print_matrix(bundle.variants[label], label, preview)

    result = save_outputs(
        output_dir=output_dir,
        file_name=loaded.name,
        source_bit_depth=loaded.source_bit_depth,
        height=height,
        width=width,
        bundle=bundle,
        threshold=threshold,
        preview=preview,
    )

    print(f"\nBerkas keluaran ({loaded.name}):")
    for item in result["written"]:
        print(f"  {item}")

    if show or save_figure:
        figure_path = output_dir / f"figure_{path.stem}.png" if save_figure else None
        saved = display_all(loaded.bgr8[:, :, ::-1], bundle.variants, figure_path, block=show)
        if saved:
            print(f"\nFigure tampilan disimpan: {saved}")

    return result


def main(argv: list[str] | None = None) -> int:
    """Jalankan program; kembalikan 0 bila sukses, selain itu kode galat."""
    args = build_parser().parse_args(argv)

    if args.preview < 1:
        print("GALAT: --preview harus minimal 1.", file=sys.stderr)
        return 2

    try:
        threshold_raw = _parse_threshold(args.threshold)

        # Jika --input tidak diberikan, minta secara interaktif (FR-1 Metode B).
        if not args.input:
            raw_inputs = _prompt_for_input()
        else:
            raw_inputs = args.input
        paths = collect_inputs(raw_inputs)
    except (ImageLoadError, ValueError) as error:
        print(f"GALAT: {error}", file=sys.stderr)
        return 1
    except SystemExit:
        raise  # biar _prompt_for_input bisa keluar bersih

    output_dir = Path(args.output_dir)
    failures = 0

    for path in paths:
        try:
            process_one(
                path=path,
                threshold_raw=threshold_raw,
                preview=args.preview,
                output_dir=output_dir,
                show=not args.no_display and not args.save_figure,
                save_figure=args.save_figure,
            )
        except (ImageLoadError, MatrixCheckError, ValueError) as error:
            print(f"GALAT pada {path.name}: {error}", file=sys.stderr)
            failures += 1

    print("\n" + "=" * 78)
    if failures:
        print(f"SELESAI dengan {failures} dari {len(paths)} berkas gagal.")
        return 1
    print(f"SELESAI: {len(paths)} berkas berhasil diproses.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())