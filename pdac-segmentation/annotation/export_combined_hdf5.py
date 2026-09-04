#!/usr/bin/env python3
"""Combine one Zarr WSI level with its matching HDF5 annotation level."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import zarr


DATA_ROOT = Path(
    "/mnt/vast-nhr/projects/nim00007/data/histopatho/pdac-kfo/data_20260310"
)
DEFAULT_ZARR_DIR = DATA_ROOT / "converted_zarr_he"
DEFAULT_ANNOTATION_DIR = DATA_ROOT / "annotations"
DEFAULT_OUTPUT_DIR = DATA_ROOT / "combined_hdf5_level4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write one portable HDF5 file per matching Zarr WSI and annotation "
            "HDF5, with image and label data stored under separate keys."
        )
    )
    parser.add_argument(
        "--zarr-dir",
        type=Path,
        default=DEFAULT_ZARR_DIR,
        help=f"Directory containing SAMPLE.zarr WSIs (default: {DEFAULT_ZARR_DIR})",
    )
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        default=DEFAULT_ANNOTATION_DIR,
        help=(
            "Directory containing SAMPLE.h5 annotation pyramids "
            f"(default: {DEFAULT_ANNOTATION_DIR})"
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=4,
        help="Pyramid level to combine (default: 4)",
    )
    parser.add_argument(
        "--image-key",
        default="raw",
        help="HDF5 output key for the YXC image dataset (default: raw)",
    )
    parser.add_argument(
        "--label-key",
        default="labels",
        help="HDF5 output key for the YX label dataset (default: labels)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Spatial HDF5 chunk size (default: 512)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=2048,
        help="Spatial copy block size (default: 2048)",
    )
    parser.add_argument(
        "--compression",
        choices=("gzip", "lzf", "none"),
        default="gzip",
        help="HDF5 compression filter (default: gzip)",
    )
    parser.add_argument(
        "--sample",
        action="append",
        help="Process only this sample ID; repeat to select multiple samples",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing combined files",
    )
    return parser.parse_args()


def find_matching_pairs(
    zarr_dir: Path,
    annotation_dir: Path,
    requested_samples: list[str] | None,
) -> tuple[list[tuple[str, Path, Path]], list[str], list[str]]:
    if not zarr_dir.is_dir():
        raise FileNotFoundError(f"Zarr directory does not exist: {zarr_dir}")
    if not annotation_dir.is_dir():
        raise FileNotFoundError(
            f"Annotation directory does not exist: {annotation_dir}"
        )

    zarr_paths = {path.stem: path for path in sorted(zarr_dir.glob("*.zarr"))}
    annotation_paths = {
        path.stem: path for path in sorted(annotation_dir.glob("*.h5"))
    }
    if requested_samples:
        requested = set(requested_samples)
        known = set(zarr_paths) | set(annotation_paths)
        if unknown := sorted(requested - known):
            raise ValueError(f"Unknown sample ID: {', '.join(unknown)}")
        zarr_paths = {
            sample: path for sample, path in zarr_paths.items() if sample in requested
        }
        annotation_paths = {
            sample: path
            for sample, path in annotation_paths.items()
            if sample in requested
        }

    matched_samples = sorted(set(zarr_paths) & set(annotation_paths))
    pairs = [
        (sample, zarr_paths[sample], annotation_paths[sample])
        for sample in matched_samples
    ]
    missing_annotations = sorted(set(zarr_paths) - set(annotation_paths))
    missing_images = sorted(set(annotation_paths) - set(zarr_paths))
    if not pairs:
        raise RuntimeError("No matching Zarr WSI and annotation HDF5 pairs found")
    return pairs, missing_annotations, missing_images


def iter_blocks(
    height: int,
    width: int,
    block_size: int,
) -> Iterator[tuple[slice, slice]]:
    for y in range(0, height, block_size):
        y_slice = slice(y, min(y + block_size, height))
        for x in range(0, width, block_size):
            x_slice = slice(x, min(x + block_size, width))
            yield y_slice, x_slice


def compression_options(compression: str) -> dict[str, Any]:
    if compression == "none":
        return {}
    options: dict[str, Any] = {
        "compression": compression,
        "shuffle": True,
        "fletcher32": True,
    }
    if compression == "gzip":
        options["compression_opts"] = 4
    return options


def copy_attributes(source: h5py.AttributeManager, target: h5py.AttributeManager) -> None:
    for key, value in source.items():
        target[key] = value


def combine_pair(
    sample: str,
    zarr_path: Path,
    annotation_path: Path,
    output_path: Path,
    level: int,
    image_key: str,
    label_key: str,
    chunk_size: int,
    block_size: int,
    compression: str,
) -> None:
    zarr_key = f"s{level}/image"
    annotation_key = f"labels/{level}"
    image_group = zarr.open_group(zarr_path, mode="r")
    if zarr_key not in image_group:
        raise KeyError(f"{zarr_key!r} is missing from {zarr_path}")
    image_source = image_group[zarr_key]

    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.unlink(missing_ok=True)
    try:
        with h5py.File(annotation_path, "r") as annotation_file:
            if annotation_key not in annotation_file:
                raise KeyError(f"{annotation_key!r} is missing from {annotation_path}")
            label_source = annotation_file[annotation_key]

            if len(image_source.shape) != 3 or image_source.shape[2] != 3:
                raise ValueError(
                    f"Expected YXC RGB image data at {zarr_path}/{zarr_key}; "
                    f"found shape {image_source.shape}"
                )
            if len(label_source.shape) != 2:
                raise ValueError(
                    f"Expected YX labels at {annotation_path}:{annotation_key}; "
                    f"found shape {label_source.shape}"
                )
            if tuple(image_source.shape[:2]) != tuple(label_source.shape):
                raise ValueError(
                    f"Image shape {image_source.shape[:2]} and label shape "
                    f"{label_source.shape} differ for {sample} at level {level}"
                )

            height, width, channels = image_source.shape
            image_chunks = (
                min(chunk_size, height),
                min(chunk_size, width),
                channels,
            )
            label_chunks = (
                min(chunk_size, height),
                min(chunk_size, width),
            )
            dataset_options = compression_options(compression)

            with h5py.File(temporary_path, "w") as output_file:
                copy_attributes(annotation_file.attrs, output_file.attrs)
                output_file.attrs["combined_export_version"] = 1
                output_file.attrs["sample_id"] = sample
                output_file.attrs["scale_level"] = level
                output_file.attrs["downsample"] = float(2**level)
                output_file.attrs["source_image_zarr"] = zarr_path.name
                output_file.attrs["source_image_key"] = zarr_key
                output_file.attrs["source_annotation_hdf5"] = annotation_path.name
                output_file.attrs["source_annotation_key"] = annotation_key

                image_output = output_file.create_dataset(
                    image_key,
                    shape=image_source.shape,
                    dtype=image_source.dtype,
                    chunks=image_chunks,
                    **dataset_options,
                )
                image_output.attrs["axes"] = "YXC"
                image_output.attrs["level"] = level
                image_output.attrs["downsample"] = float(2**level)
                if "pixel_size_x_um" in label_source.attrs:
                    image_output.attrs["pixel_size_x_um"] = label_source.attrs[
                        "pixel_size_x_um"
                    ]
                if "pixel_size_y_um" in label_source.attrs:
                    image_output.attrs["pixel_size_y_um"] = label_source.attrs[
                        "pixel_size_y_um"
                    ]

                label_output = output_file.create_dataset(
                    label_key,
                    shape=label_source.shape,
                    dtype=label_source.dtype,
                    chunks=label_chunks,
                    fillvalue=0,
                    **dataset_options,
                )
                copy_attributes(label_source.attrs, label_output.attrs)

                blocks = list(iter_blocks(height, width, block_size))
                for block_index, (y_slice, x_slice) in enumerate(blocks, start=1):
                    image_output[y_slice, x_slice, :] = image_source[
                        y_slice, x_slice, :
                    ]
                    label_output[y_slice, x_slice] = label_source[y_slice, x_slice]
                    print(
                        f"\r  copying block {block_index}/{len(blocks)}",
                        end="",
                        flush=True,
                    )
                print()

        os.replace(temporary_path, output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def main() -> int:
    args = parse_args()
    try:
        if args.level < 0:
            raise ValueError("--level must be non-negative")
        if args.chunk_size < 1:
            raise ValueError("--chunk-size must be positive")
        if args.block_size < args.chunk_size:
            raise ValueError("--block-size must be at least --chunk-size")
        if args.image_key == args.label_key:
            raise ValueError("--image-key and --label-key must be different")

        pairs, missing_annotations, missing_images = find_matching_pairs(
            args.zarr_dir,
            args.annotation_dir,
            args.sample,
        )
        if missing_annotations:
            print(
                "warning: no annotations for: " + ", ".join(missing_annotations),
                file=sys.stderr,
            )
        if missing_images:
            print(
                "warning: no Zarr image for: " + ", ".join(missing_images),
                file=sys.stderr,
            )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        for sample, zarr_path, annotation_path in pairs:
            output_path = args.output_dir / f"{sample}_level{args.level}.h5"
            if output_path.exists() and not args.overwrite:
                print(f"{sample}: output exists, skipping {output_path}")
                continue
            print(f"{sample}: combining level {args.level}")
            combine_pair(
                sample,
                zarr_path,
                annotation_path,
                output_path,
                args.level,
                args.image_key,
                args.label_key,
                args.chunk_size,
                args.block_size,
                args.compression,
            )
            print(f"  wrote {output_path}")
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
