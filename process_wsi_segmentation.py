import argparse
import json
import os
import multiprocessing as mp
from concurrent import futures

import numpy as np
import pandas as pd
import zarr

from skimage.measure import regionprops_table, find_contours, approximate_polygon
from tqdm import tqdm


def extract_polygons(instances, props, global_offset):
    halo = 2

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
        contour = approximate_polygon(contour, tolerance=2.0).tolist()
        return contour

    n_rows = len(props)
    n_threads = max(16, mp.cpu_count())
    with futures.ThreadPoolExecutor(n_threads) as tp:
        masks = list(tqdm(tp.map(extract_mask, range(n_rows)), total=n_rows, desc="Extract Polygons"))

    return masks


def _to_qupath_geojson(masks):
    features = []
    for obj_id, mask in enumerate(masks, 1):
        # Switch from yx to xy.
        mask = np.array(mask)
        assert mask.shape[1] == 2
        mask = np.array([mask[:, 1], mask[:, 0]])
        features.append({
            "type": "Feature",
            "properties": {
                "object_id": int(obj_id),
                "name": f"mask_{int(obj_id)}",
                "isLocked": False,
                "classification": None,
                "level": 0,
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [mask.tolist()],
            },
        })

    return {"type": "FeatureCollection", "features": features}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", required=True)
    parser.add_argument("-k", "--instance_key", required=True)
    parser.add_argument("-o", "--output_path", required=True)
    parser.add_argument("-s", "--semantic_key")
    parser.add_argument("-f", "--format", default="custom")
    parser.add_argument("--roi", nargs=4, type=int)
    args = parser.parse_args()

    if args.roi:
        roi = np.s_[args.roi[0]:args.roi[1], args.roi[2]:args.roi[3]]
        global_offset = [args.roi[0], args.roi[2]]
    else:
        roi = np.s_[:]
        global_offset = [0, 0]

    f = zarr.open(args.input_path, mode="r")
    instances = f[args.instance_key][roi]

    def majority_label(regionmask, intensity_image):
        values = intensity_image[regionmask].astype("uint16")
        return np.bincount(values).argmax()

    print("Compute regionprops ...")
    if args.semantic_key is None:
        props = pd.DataFrame(regionprops_table(instances, properties=("label", "bbox")))
    else:
        semantic = f[args.semantic_key][:]
        props = pd.DataFrame(
            regionprops_table(
                instances, semantic, properties=("label", "bbox"), extra_properties=(majority_label,)
            )
        )
        props = props[props.majority_label == 2]
    props = props.rename(columns={f"bbox-{i}": f"bbox_{i}" for i in range(4)})

    masks = extract_polygons(instances, props, global_offset)

    if args.format == "custom":
        output = {"cells": masks}
    elif args.format == "qupath":
        output = _to_qupath_geojson(masks)

    os.makedirs(os.path.split(args.output_path)[0], exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(output, f)


if __name__ == "__main__":
    main()
