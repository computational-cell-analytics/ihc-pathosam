import argparse
import json
import os
import multiprocessing as mp
from concurrent import futures
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from skimage.measure import regionprops_table, find_contours, approximate_polygon
from tqdm import tqdm


def _ring_area(coords):
    # coords: (N,2) closed
    x = np.asarray([p[0] for p in coords])
    y = np.asarray([p[1] for p in coords])
    return 0.5 * abs(np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:]))


def _drop_consecutive_duplicates(contour):
    keep = np.ones(len(contour), dtype=bool)
    keep[1:] = np.any(contour[1:] != contour[:-1], axis=1)
    return contour[keep]


def extract_polygons(instances, props, global_offset):
    halo = 2
    min_area = 8

    def extract_mask(row_id):
        prop = props.iloc[row_id]
        bbox = (slice(prop.bbox_0, prop.bbox_2), slice(prop.bbox_1, prop.bbox_3))
        bbox = tuple(slice(
            max(0, bb.start - halo), min(sh, bb.stop + halo)
        ) for bb, sh in zip(bbox, instances.shape))
        mask = (instances[bbox] == prop.label).astype("uint8")
        contour = find_contours(mask, level=0.5)
        contour = max(contour, key=len)
        contour[:, 0] += (bbox[0].start + global_offset[0])
        contour[:, 1] += (bbox[1].start + global_offset[1])

        contour = _drop_consecutive_duplicates(contour)

        # Disable the global offset for debugging on local crops.
        # contour[:, 0] += bbox[0].start
        # contour[:, 1] += bbox[1].start

        contour_approx = approximate_polygon(contour, tolerance=1.0)
        contour_approx = _drop_consecutive_duplicates(contour_approx)

        # Ensure contours have enough points and are closed,
        # otherwise QuPath does not like them.
        if len(contour_approx) > 4:
            contour = contour_approx

        if not np.array_equal(contour[0], contour[-1]):
            contour = np.vstack([contour, contour[0]])

        # Drop small contours that don't work for QuPath.
        if len(contour) < 4 or _ring_area(contour.tolist()) < min_area:
            return None

        return contour

    n_rows = len(props)
    n_threads = max(16, mp.cpu_count())
    with futures.ThreadPoolExecutor(n_threads) as tp:
        masks = list(tqdm(tp.map(extract_mask, range(n_rows)), total=n_rows, desc="Extract Polygons"))
    masks = [mask for mask in masks if mask is not None]

    return masks


def _to_qupath_geojson(masks):
    features = []
    for obj_id, mask in enumerate(masks, 1):
        # Switch from yx to xy.
        assert mask.shape[1] == 2
        mask = mask[:, ::-1].astype(float).tolist()
        features.append({
            "type": "Feature",
            "properties": {
                "object_id": int(obj_id),
                "name": f"mask_{int(obj_id)}",
                "objectType": "annotation",
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [mask],
            },
        })

    return {"type": "FeatureCollection", "features": features}


def process_wsi(input_path, output_path, instance_key, semantic_key, format_, roi, global_offset):
    f = zarr.open(input_path, mode="r")
    instances = f[instance_key][roi]

    def majority_label(regionmask, intensity_image):
        values = intensity_image[regionmask].astype("uint16")
        return np.bincount(values).argmax()

    print("Compute regionprops ...")
    if semantic_key is None:
        props = pd.DataFrame(regionprops_table(instances, properties=("label", "bbox")))
    else:
        semantic = f[semantic_key][:]
        props = pd.DataFrame(
            regionprops_table(
                instances, semantic, properties=("label", "bbox"), extra_properties=(majority_label,)
            )
        )
        props = props[props.majority_label == 2]
    props = props.rename(columns={f"bbox-{i}": f"bbox_{i}" for i in range(4)})

    masks = extract_polygons(instances, props, global_offset)
    print("Extracted", len(masks), "masks")

    if format_ == "custom":
        output = {"cells": [mask.tolist() for mask in masks]}
    elif format_ == "qupath":
        output = _to_qupath_geojson(masks)
    else:
        raise ValueError(f"Invalid format: {format_}, choose one of 'custom', 'qupath'.")

    out_folder = os.path.split(output_path)[0]
    if out_folder != "":
        os.makedirs(out_folder, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", required=True)
    parser.add_argument("-k", "--instance_key", required=True)
    parser.add_argument("-o", "--output_path", required=True)
    parser.add_argument("-s", "--semantic_key")
    parser.add_argument("-f", "--format", default="custom")
    parser.add_argument("--pattern")
    parser.add_argument("--roi", nargs=4, type=int)
    args = parser.parse_args()

    if args.roi:
        roi = np.s_[args.roi[0]:args.roi[1], args.roi[2]:args.roi[3]]
        global_offset = [args.roi[0], args.roi[2]]
    else:
        roi = np.s_[:]
        global_offset = [0, 0]

    if args.pattern is None:
        process_wsi(
            args.input_path, args.output_path, args.instance_key, args.semantic_key, args.format, roi, global_offset
        )
    else:
        inputs = sorted(glob(os.path.join(args.input_path, "**", args.pattern), recursive=True))
        output_root = args.output_path
        for input_path in tqdm(inputs):
            fname = Path(input_path).stem
            folder = os.path.split(os.path.relpath(input_path, args.input_path))[0]
            output_path = os.path.join(output_root, folder, f"{fname}.json")
            process_wsi(
                input_path, output_path, args.instance_key, args.semantic_key, args.format, roi, global_offset
            )


if __name__ == "__main__":
    main()
