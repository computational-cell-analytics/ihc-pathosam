#!/usr/bin/env python3
"""Convert QuPath annotations to label-only multiscale HDF5 files."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import bioimage_py
import h5py
import numpy as np
from PIL import Image, ImageDraw


DEFAULT_CLASS_MAP = {"Normal pancreas": 1, "Tumor": 2}
UNCLASSIFIED_NAME = "Unclassified"
Point = tuple[int, int]
Ring = tuple[Point, ...]
Polygon = tuple[Ring, ...]


@dataclass(frozen=True)
class PyramidLevel:
    index: int
    downsample: float
    width: int
    height: int


@dataclass(frozen=True)
class PreparedFeature:
    label: int
    class_name: str
    polygons: tuple[Polygon, ...]
    bounds: tuple[float, float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rasterize QuPath annotations at level 0 and create a label pyramid "
            "that matches the WSI pyramid metadata."
        )
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        type=Path,
        default=Path("data"),
        help="Directory containing .vsi and .qpdata files (default: data)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("hdf5"),
        help="Output directory (default: hdf5)",
    )
    parser.add_argument(
        "--qupath",
        default="qupath",
        help="QuPath executable used to read .qpdata files (default: qupath)",
    )
    parser.add_argument(
        "--class-map",
        action="append",
        metavar="NAME=LABEL",
        help=(
            "Override the mask classes. Repeat this option for each class. "
            "Defaults to 'Normal pancreas=1' and 'Tumor=2'."
        ),
    )
    parser.add_argument(
        "--unclassified-label",
        type=int,
        default=255,
        help="Label for annotations without a QuPath class (default: 255)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="HDF5 and processing block size in pixels (default: 512)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of bioimage_py block workers (default: 1)",
    )
    parser.add_argument(
        "--sample",
        action="append",
        help="Export only this sample ID. Repeat to select multiple samples.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing HDF5 files",
    )
    return parser.parse_args()


def parse_class_map(values: Sequence[str] | None) -> dict[str, int]:
    if not values:
        return DEFAULT_CLASS_MAP.copy()

    class_map: dict[str, int] = {}
    for value in values:
        name, separator, label_text = value.rpartition("=")
        if not separator or not name.strip():
            raise ValueError(f"Invalid class mapping {value!r}; expected NAME=LABEL")
        try:
            label = int(label_text)
        except ValueError as error:
            raise ValueError(f"Invalid label in class mapping {value!r}") from error
        if not 1 <= label <= 254:
            raise ValueError(f"Class label must be between 1 and 254: {value!r}")
        class_map[name.strip()] = label

    if len(set(class_map.values())) != len(class_map):
        raise ValueError("Each class must have a different label")
    return class_map


def find_pairs(input_dir: Path) -> list[tuple[Path, Path]]:
    vsi_files = sorted(input_dir.glob("*.vsi"))
    if not vsi_files:
        raise FileNotFoundError(f"No .vsi files found in {input_dir}")

    pairs = []
    for vsi_path in vsi_files:
        matches = sorted(input_dir.glob(f"{vsi_path.name}*.qpdata"))
        if len(matches) != 1:
            names = ", ".join(path.name for path in matches) or "none"
            raise RuntimeError(
                f"Expected one QuPath file for {vsi_path.name}, found: {names}"
            )
        pairs.append((vsi_path, matches[0]))
    return pairs


def read_qpdata_metadata(qpdata_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = qpdata_path.read_bytes()
    marker = b'{"dataVersion"'
    start = raw.find(marker)
    if start < 0:
        raise RuntimeError(f"QuPath metadata JSON was not found in {qpdata_path}")

    serialized_text = raw[start:].decode("utf-8", errors="replace")
    header, _ = json.JSONDecoder().raw_decode(serialized_text)
    try:
        metadata = header["server"]["metadata"]
    except (KeyError, TypeError) as error:
        raise RuntimeError(
            f"Image server metadata is missing in {qpdata_path}"
        ) from error
    return header, metadata


def read_pyramid_levels(metadata: dict[str, Any]) -> list[PyramidLevel]:
    raw_levels = metadata.get("levels")
    if not raw_levels:
        raise RuntimeError("The serialized VSI server metadata has no pyramid levels")

    levels = [
        PyramidLevel(
            index=index,
            downsample=float(level["downsample"]),
            width=int(level["width"]),
            height=int(level["height"]),
        )
        for index, level in enumerate(raw_levels)
    ]
    if levels[0].width != int(metadata["width"]):
        raise RuntimeError("Level-0 width does not match the WSI metadata")
    if levels[0].height != int(metadata["height"]):
        raise RuntimeError("Level-0 height does not match the WSI metadata")
    if not math.isclose(levels[0].downsample, 1.0):
        raise RuntimeError("The first WSI pyramid level is not level 0")

    for previous, current in zip(levels, levels[1:], strict=False):
        if not math.isclose(current.downsample / previous.downsample, 2.0):
            raise RuntimeError("Only successive 2x WSI pyramid levels are supported")
        expected_width = (previous.width + 1) // 2
        expected_height = (previous.height + 1) // 2
        if (current.width, current.height) != (expected_width, expected_height):
            raise RuntimeError(
                f"WSI level {current.index} has unexpected dimensions "
                f"{current.width}x{current.height}; expected "
                f"{expected_width}x{expected_height}"
            )
    return levels


def export_geojson(
    pairs: Sequence[tuple[Path, Path]],
    destinations: Sequence[Path],
    qupath: str,
) -> None:
    executable = shutil.which(qupath) if Path(qupath).name == qupath else qupath
    if not executable or not Path(executable).exists():
        raise FileNotFoundError(
            f"QuPath executable {qupath!r} was not found. Install QuPath or pass "
            "--qupath /path/to/QuPath."
        )

    helper = Path(__file__).with_name("export_qupath_annotations.groovy")
    command = [str(executable), "script"]
    for (_, qpdata_path), destination in zip(pairs, destinations, strict=True):
        command.extend(["--args", str(qpdata_path.resolve())])
        command.extend(["--args", str(destination.resolve())])
    command.append(str(helper.resolve()))

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        details = "\n".join((result.stdout + result.stderr).splitlines()[-30:])
        raise RuntimeError(f"QuPath annotation export failed:\n{details}")
    missing = [str(path) for path in destinations if not path.exists()]
    if missing:
        raise RuntimeError(f"QuPath did not create: {', '.join(missing)}")


def load_features(geojson_path: Path) -> list[dict[str, Any]]:
    with geojson_path.open(encoding="utf-8") as file:
        geojson = json.load(file)
    if isinstance(geojson, list):
        return geojson
    if geojson.get("type") == "FeatureCollection":
        return geojson["features"]
    if geojson.get("type") == "Feature":
        return [geojson]
    raise ValueError(f"Unsupported GeoJSON root in {geojson_path}")


def geometry_to_polygons(
    geometry: dict[str, Any],
) -> tuple[tuple[Polygon, ...], tuple[float, float, float, float]]:
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if geometry_type == "Polygon":
        raw_polygons = [coordinates]
    elif geometry_type == "MultiPolygon":
        raw_polygons = coordinates
    else:
        raise ValueError(f"Unsupported annotation geometry: {geometry_type}")

    polygons = []
    all_points = []
    for raw_polygon in raw_polygons:
        rings = []
        for raw_ring in raw_polygon:
            points = np.asarray(raw_ring, dtype=np.float64)
            if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] < 2:
                raise ValueError("Invalid polygon coordinates in QuPath GeoJSON")
            points = points[:, :2]
            all_points.append(points)
            rounded = np.rint(points).astype(np.int64)
            rings.append(tuple((int(x), int(y)) for x, y in rounded))
        if rings:
            polygons.append(tuple(rings))
    if not polygons:
        raise ValueError("Empty polygon in QuPath GeoJSON")

    stacked = np.concatenate(all_points, axis=0)
    bounds = (
        float(stacked[:, 0].min()),
        float(stacked[:, 1].min()),
        float(stacked[:, 0].max()),
        float(stacked[:, 1].max()),
    )
    return tuple(polygons), bounds


def prepare_features(
    features: Sequence[dict[str, Any]],
    class_map: dict[str, int],
    unclassified_label: int,
) -> tuple[list[PreparedFeature], dict[str, int]]:
    prepared = []
    counts: dict[str, int] = {}
    for feature in features:
        properties = feature.get("properties") or {}
        classification = properties.get("classification")
        if classification:
            class_name = classification.get("name")
            if class_name not in class_map:
                raise ValueError(
                    f"QuPath class {class_name!r} is not mapped. Add "
                    f"--class-map {class_name!r}=LABEL."
                )
            label = class_map[class_name]
        else:
            class_name = UNCLASSIFIED_NAME
            label = unclassified_label

        polygons, bounds = geometry_to_polygons(feature["geometry"])
        prepared.append(PreparedFeature(label, class_name, polygons, bounds))
        counts[class_name] = counts.get(class_name, 0) + 1
    prepared.sort(key=lambda feature: feature.label)
    return prepared, counts


def dataset_chunks(shape: tuple[int, int], chunk_size: int) -> tuple[int, int]:
    return min(chunk_size, shape[0]), min(chunk_size, shape[1])


def create_label_dataset(
    group: h5py.Group,
    level: PyramidLevel,
    chunk_size: int,
) -> h5py.Dataset:
    shape = (level.height, level.width)
    dataset = group.create_dataset(
        str(level.index),
        shape=shape,
        dtype=np.uint8,
        chunks=dataset_chunks(shape, chunk_size),
        compression="lzf",
        shuffle=True,
        fillvalue=0,
    )
    dataset.attrs["axes"] = "YX"
    dataset.attrs["level"] = level.index
    dataset.attrs["downsample"] = level.downsample
    return dataset


def blocks_overlapping_features(
    features: Sequence[PreparedFeature],
    shape: tuple[int, int],
    block_shape: tuple[int, int],
    coordinate_scale: float,
) -> list[int]:
    from bioimage_py.util import get_blocking

    height, width = shape
    blocking = get_blocking(shape, block_shape)
    block_ids = set()
    for feature in features:
        min_x, min_y, max_x, max_y = feature.bounds
        min_x /= coordinate_scale
        min_y /= coordinate_scale
        max_x /= coordinate_scale
        max_y /= coordinate_scale
        if max_x < 0 or max_y < 0 or min_x >= width or min_y >= height:
            continue
        begin = [max(0, math.floor(min_y) - 1), max(0, math.floor(min_x) - 1)]
        end = [
            min(height, math.ceil(max_y) + 2),
            min(width, math.ceil(max_x) + 2),
        ]
        block_ids.update(blocking.get_block_ids_overlapping_bounding_box(begin, end))
    return sorted(block_ids)


def shifted_ring(points: Ring, x: int, y: int) -> list[Point]:
    return [(point_x - x, point_y - y) for point_x, point_y in points]


def rasterize_level_zero_blockwise(
    dataset: h5py.Dataset,
    features: Sequence[PreparedFeature],
    num_workers: int,
) -> None:
    def rasterize(block, inputs, outputs, mask) -> None:
        y0, x0 = (int(value) for value in block.begin)
        y1, x1 = (int(value) for value in block.end)
        block_shape = (y1 - y0, x1 - x0)
        block_labels = np.zeros(block_shape, dtype=np.uint8)

        for feature in features:
            min_x, min_y, max_x, max_y = feature.bounds
            if max_x < x0 or max_y < y0 or min_x >= x1 or min_y >= y1:
                continue
            feature_image = Image.new("1", (block_shape[1], block_shape[0]), 0)
            draw = ImageDraw.Draw(feature_image)
            for polygon in feature.polygons:
                exterior = shifted_ring(polygon[0], x0, y0)
                draw.polygon(exterior, fill=1)
                for hole in polygon[1:]:
                    interior = shifted_ring(hole, x0, y0)
                    draw.polygon(interior, fill=0)
            feature_mask = np.asarray(feature_image, dtype=bool)
            block_labels[feature_mask] = feature.label

        if np.any(block_labels):
            roi = bioimage_py.to_roi(block)
            outputs[0][roi] = block_labels

    block_ids = blocks_overlapping_features(
        features, dataset.shape, dataset.chunks, coordinate_scale=1.0
    )
    if not block_ids:
        return
    runner = bioimage_py.get_runner("local")
    runner.run(
        rasterize,
        [],
        outputs=[dataset],
        block_shape=dataset.chunks,
        block_ids=block_ids,
        num_workers=num_workers,
        name="rasterize",
    )


def downsample_with_bioimage_py(
    source: h5py.Dataset,
    target: h5py.Dataset,
    features: Sequence[PreparedFeature],
    downsample: float,
    num_workers: int,
) -> None:
    block_ids = blocks_overlapping_features(
        features,
        target.shape,
        target.chunks,
        coordinate_scale=downsample,
    )
    bioimage_py.downsample(
        source,
        2,
        output=target,
        order=0,
        anti_aliasing=False,
        block_shape=target.chunks,
        block_ids=block_ids,
        num_workers=num_workers,
    )


def calibration_value(metadata: dict[str, Any], key: str) -> float:
    try:
        calibration = metadata["pixelCalibration"][key]
        if calibration["unit"] not in ("µm", "um"):
            raise ValueError(f"Unexpected pixel-size unit {calibration['unit']!r}")
        return float(calibration["value"])
    except (KeyError, TypeError) as error:
        raise RuntimeError(f"Missing {key} in QuPath pixel calibration") from error


def write_pyramid(
    output_path: Path,
    vsi_path: Path,
    qpdata_path: Path,
    header: dict[str, Any],
    metadata: dict[str, Any],
    levels: Sequence[PyramidLevel],
    features: Sequence[PreparedFeature],
    class_map: dict[str, int],
    unclassified_label: int,
    annotation_counts: dict[str, int],
    chunk_size: int,
    num_workers: int,
) -> None:
    base_pixel_size_x = calibration_value(metadata, "pixelWidth")
    base_pixel_size_y = calibration_value(metadata, "pixelHeight")
    magnification = float(metadata.get("magnification", math.nan))
    label_names = {0: "Background"}
    label_names.update({label: name for name, label in class_map.items()})
    if UNCLASSIFIED_NAME in annotation_counts:
        label_names[unclassified_label] = UNCLASSIFIED_NAME

    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if temporary_path.exists():
        temporary_path.unlink()
    try:
        with h5py.File(temporary_path, "w") as output:
            output.attrs["sample_id"] = vsi_path.stem
            output.attrs["source_vsi"] = vsi_path.name
            output.attrs["source_qpdata"] = qpdata_path.name
            output.attrs["pyramid_metadata_source"] = (
                "QuPath serialized VSI image-server metadata"
            )
            output.attrs["pyramid_downsampler"] = "bioimage_py.downsample"
            output.attrs["pyramid_sampling"] = (
                "nearest-neighbor from the immediately preceding level"
            )
            output.attrs["objective_magnification"] = magnification
            output.attrs["base_pixel_size_x_um"] = base_pixel_size_x
            output.attrs["base_pixel_size_y_um"] = base_pixel_size_y
            output.attrs["label_names"] = json.dumps(label_names, sort_keys=True)
            output.attrs["annotation_counts"] = json.dumps(
                annotation_counts, sort_keys=True
            )
            output.attrs["qupath_server_metadata"] = json.dumps(
                header["server"], ensure_ascii=False, separators=(",", ":")
            )
            output.attrs["pyramid_levels"] = json.dumps(
                [
                    {
                        "level": level.index,
                        "downsample": level.downsample,
                        "width": level.width,
                        "height": level.height,
                    }
                    for level in levels
                ],
                separators=(",", ":"),
            )

            labels = output.create_group("labels")
            labels.attrs["axes"] = "YX"
            labels.attrs["level_count"] = len(levels)

            level_zero = create_label_dataset(labels, levels[0], chunk_size)
            level_zero.attrs["pixel_size_x_um"] = base_pixel_size_x
            level_zero.attrs["pixel_size_y_um"] = base_pixel_size_y
            print(
                f"  level 0: rasterizing {level_zero.shape[1]}x"
                f"{level_zero.shape[0]}",
                flush=True,
            )
            rasterize_level_zero_blockwise(level_zero, features, num_workers)

            previous = level_zero
            for level in levels[1:]:
                current = create_label_dataset(labels, level, chunk_size)
                current.attrs["pixel_size_x_um"] = (
                    base_pixel_size_x * level.downsample
                )
                current.attrs["pixel_size_y_um"] = (
                    base_pixel_size_y * level.downsample
                )
                downsample_with_bioimage_py(
                    previous,
                    current,
                    features,
                    level.downsample,
                    num_workers,
                )
                print(
                    f"  level {level.index}: {current.shape[1]}x"
                    f"{current.shape[0]} from level {level.index - 1}",
                    flush=True,
                )
                previous = current

        os.replace(temporary_path, output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def main() -> int:
    args = parse_args()
    try:
        class_map = parse_class_map(args.class_map)
        if not 1 <= args.unclassified_label <= 255:
            raise ValueError("--unclassified-label must be between 1 and 255")
        if args.unclassified_label in class_map.values():
            raise ValueError("--unclassified-label conflicts with a mapped class label")
        if args.chunk_size < 64:
            raise ValueError("--chunk-size must be at least 64")
        if args.workers < 1:
            raise ValueError("--workers must be at least 1")

        pairs = find_pairs(args.input_dir)
        if args.sample:
            requested = set(args.sample)
            pairs = [pair for pair in pairs if pair[0].stem in requested]
            found = {pair[0].stem for pair in pairs}
            if missing := sorted(requested - found):
                raise ValueError(f"Unknown sample ID: {', '.join(missing)}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_paths = [args.output_dir / f"{vsi.stem}.h5" for vsi, _ in pairs]
        existing = [path for path in output_paths if path.exists()]
        if existing and not args.overwrite:
            names = ", ".join(path.name for path in existing)
            raise FileExistsError(
                f"Output exists: {names}. Pass --overwrite to replace it."
            )

        with tempfile.TemporaryDirectory(prefix="qupath-annotations-") as temp_dir:
            geojson_paths = [
                Path(temp_dir) / f"annotations-{index}.geojson"
                for index in range(len(pairs))
            ]
            export_geojson(pairs, geojson_paths, args.qupath)

            for (vsi_path, qpdata_path), geojson_path, output_path in zip(
                pairs, geojson_paths, output_paths, strict=True
            ):
                header, metadata = read_qpdata_metadata(qpdata_path)
                levels = read_pyramid_levels(metadata)
                raw_features = load_features(geojson_path)
                features, counts = prepare_features(
                    raw_features, class_map, args.unclassified_label
                )
                print(
                    f"{vsi_path.stem}: {len(features)} annotations, "
                    f"{len(levels)} pyramid levels",
                    flush=True,
                )
                write_pyramid(
                    output_path,
                    vsi_path,
                    qpdata_path,
                    header,
                    metadata,
                    levels,
                    features,
                    class_map,
                    args.unclassified_label,
                    counts,
                    args.chunk_size,
                    args.workers,
                )
                print(f"  wrote {output_path}", flush=True)
    except (FileNotFoundError, RuntimeError, ValueError, IndexError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
