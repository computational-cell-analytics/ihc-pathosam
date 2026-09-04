#!/usr/bin/env python3
"""Open one selected level from label-pyramid HDF5 files in napari."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import dask.array as da
import h5py

# Some read-only Python installations prevent numba from locating its package
# cache. Use a writable cache before napari imports numba.
os.environ.setdefault(
    "NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "napari-numba-cache")
)
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "wsi-napari-cache")
)
import napari
from napari.utils.colormaps import DirectLabelColormap


DEFAULT_COLORS = {
    0: "transparent",
    1: "#2efc41",
    2: "#c80000",
    255: "#ffd92f",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show one level from each HDF5 label pyramid in napari."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="HDF5 files or directories (default: hdf5)",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=-1,
        help=(
            "Pyramid level to show. Negative values count from the coarsest "
            "level; -1 selects the coarsest level (default: -1)."
        ),
    )
    parser.add_argument(
        "--gap-um",
        type=float,
        default=500.0,
        help="Horizontal gap between samples in micrometres (default: 500)",
    )
    parser.add_argument(
        "--opacity",
        type=float,
        default=0.8,
        help="Initial label opacity from 0 to 1 (default: 0.8)",
    )
    return parser.parse_args()


def resolve_paths(inputs: list[Path]) -> list[Path]:
    if not inputs:
        inputs = [Path("hdf5")]
    paths = []
    for input_path in inputs:
        if input_path.is_dir():
            paths.extend(sorted(input_path.glob("*.h5")))
            paths.extend(sorted(input_path.glob("*.hdf5")))
        elif input_path.is_file():
            paths.append(input_path)
        else:
            raise FileNotFoundError(input_path)
    unique_paths = list(dict.fromkeys(path.resolve() for path in paths))
    if not unique_paths:
        raise FileNotFoundError("No .h5 or .hdf5 files found")
    return unique_paths


def resolve_level(labels: h5py.Group, requested_level: int) -> int:
    levels = sorted(int(name) for name in labels)
    if requested_level >= 0:
        if requested_level not in levels:
            raise ValueError(
                f"Level {requested_level} is unavailable; available levels: {levels}"
            )
        return requested_level
    try:
        return levels[requested_level]
    except IndexError as error:
        raise ValueError(
            f"Level {requested_level} is unavailable; available levels: {levels}"
        ) from error


def main() -> None:
    args = parse_args()
    paths = resolve_paths(args.paths)
    if not 0.0 <= args.opacity <= 1.0:
        raise ValueError("--opacity must be between 0 and 1")

    open_files: list[h5py.File] = []
    try:
        viewer = napari.Viewer(title="WSI annotation pyramids")
        offset_x_um = 0.0
        for path in paths:
            data = h5py.File(path, "r")
            open_files.append(data)
            if "labels" not in data:
                raise ValueError(
                    f"{path} has no label pyramid. Re-run export_hdf5.py."
                )
            labels = data["labels"]
            level = resolve_level(labels, args.level)
            dataset = labels[str(level)]
            sample_id = str(data.attrs.get("sample_id", path.stem))
            label_names = json.loads(str(data.attrs.get("label_names", "{}")))
            scale = (
                float(dataset.attrs["pixel_size_y_um"]),
                float(dataset.attrs["pixel_size_x_um"]),
            )
            downsample = float(dataset.attrs["downsample"])
            lazy_labels = da.from_array(
                dataset,
                chunks=dataset.chunks,
                lock=True,
                asarray=False,
            )

            colors = {
                int(label): DEFAULT_COLORS.get(int(label), "white")
                for label in label_names
            }
            colors[None] = "transparent"
            viewer.add_labels(
                lazy_labels,
                name=f"{sample_id} | level {level} ({downsample:g}x)",
                opacity=args.opacity,
                colormap=DirectLabelColormap(color_dict=colors),
                metadata={
                    "source": str(path),
                    "labels": label_names,
                    "level": level,
                    "downsample": downsample,
                    "pixel_size_um": {"y": scale[0], "x": scale[1]},
                },
                scale=scale,
                translate=(0.0, offset_x_um),
                units=("µm", "µm"),
            )
            offset_x_um += dataset.shape[1] * scale[1] + args.gap_um

        viewer.reset_view()
        napari.run()
    finally:
        for data in open_files:
            data.close()


if __name__ == "__main__":
    main()
