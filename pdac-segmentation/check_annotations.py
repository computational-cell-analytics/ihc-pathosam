"""Extracting the annotations from QuPath's annotation format and the corresponding tif crops from Leonie.

NOTE: The annotations bring both semantic and instance segmentation for 6 crops right now.
"""


import json
import os
from pathlib import Path

import h5py
import numpy as np
import imageio.v3 as imageio
from skimage.draw import polygon as draw_polygon


ROOT = "/mnt/vast-nhr/projects/cidas/cca/data/pdac_umg_histopatho"
OUTPUT_ROOT = os.path.join(ROOT, "extracted_data")


def geojson_to_label_array(geojson_path, image_shape):
    """Parse QuPath geojson and rasterize nucleus detections into an instance label array.

    Returns:
        labels: uint32 array of shape (H, W), 0=background, instance ids start at 1
        semantic: uint8 array of shape (H, W), 0=background, 1=negative, 2=positive
    """
    with open(geojson_path, "r") as f:
        data = json.load(f)

    features = data["features"]
    detections = [f for f in features if f.get("properties", {}).get("objectType") == "detection"]

    H, W = image_shape[:2]
    labels = np.zeros((H, W), dtype=np.uint32)

    positive_ids = []
    negative_ids = []

    for inst_id, feat in enumerate(detections, start=1):
        geom = feat["geometry"]
        if geom["type"] != "Polygon":
            continue

        coords = np.array(geom["coordinates"][0], dtype=np.float32)  # (N, 2) as [x, y]
        xs = coords[:, 0]
        ys = coords[:, 1]

        # rasterize: skimage.draw.polygon takes (row, col) = (y, x)
        rr, cc = draw_polygon(ys, xs, shape=(H, W))
        labels[rr, cc] = inst_id

        cls_name = feat.get("properties", {}).get("classification", {}).get("name", "")
        if cls_name == "Positive":
            positive_ids.append(inst_id)
        elif cls_name == "Negative":
            negative_ids.append(inst_id)

    positive_ids = np.array(positive_ids, dtype=np.uint32)
    negative_ids = np.array(negative_ids, dtype=np.uint32)

    semantic = np.zeros((H, W), dtype=np.uint8)
    semantic[np.isin(labels, negative_ids)] = 1
    semantic[np.isin(labels, positive_ids)] = 2

    return labels, semantic


def process_folder(folder_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    tif_files = sorted(Path(folder_path).glob("*.tif"))
    print(f"Found {len(tif_files)} tif files in {folder_path}")

    for tif_path in tif_files:
        geojson_path = tif_path.with_suffix(".geojson")
        if not geojson_path.exists():
            print(f"Skipping {tif_path.name}: no matching geojson")
            continue

        out_path = Path(output_dir) / (tif_path.stem.replace(" ", "_") + ".h5")
        if out_path.exists():
            print(f"Skipping {tif_path.name}: output already exists")
            continue

        print(f"Processing {tif_path.name} ...")
        image = imageio.imread(str(tif_path))  # (H, W, 3) uint8

        labels, semantic = geojson_to_label_array(str(geojson_path), image.shape)

        n_nuclei = len(np.unique(labels)) - 1  # exclude background
        print(f"Image shape: {image.shape}, nuclei: {n_nuclei}")

        with h5py.File(str(out_path), "w") as f:
            f.create_dataset("raw", data=image, compression="gzip")
            f.create_dataset("labels/nucleus/instances", data=labels, compression="gzip")
            f.create_dataset("labels/nucleus/semantic", data=semantic, compression="gzip")

        print(f" Saved to {out_path}")


def main():
    for entry in sorted(os.listdir(ROOT)):
        folder = os.path.join(ROOT, entry)
        if not os.path.isdir(folder) or entry.endswith(".zip"):
            continue
        output_dir = os.path.join(OUTPUT_ROOT, entry)
        process_folder(folder, output_dir)


if __name__ == "__main__":
    main()
