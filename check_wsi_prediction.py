import argparse
import json

import napari
import numpy as np
import zarr
from util import load_tif_as_zarr
from skimage.draw import polygon2mask


def _filter_polygons_by_roi(polys, roi):
    xs, ys = roi
    y0, y1 = ys.start, ys.stop
    x0, x1 = xs.start, xs.stop
    out = []
    for pol in polys:
        x = pol[:, 0]
        y = pol[:, 1]
        if (x >= x0).any() and (x < x1).any() and (y >= y0).any() and (y < y1).any():
            out.append(pol)
    return out


def _load_segmentation(prediction_path, bbox):
    with open(prediction_path, "r") as f:
        masks = json.load(f)["cells"]
    masks = [np.array(mask) for mask in masks]

    # Filter the masks in the BB.
    n_total = len(masks)
    masks = _filter_polygons_by_roi(masks, bbox)
    print(len(masks), "/", n_total, "masks are in the bounding box", bbox)

    # Write the masks to the segmentation.
    offset = np.array([bb.start for bb in bbox])
    out_shape = tuple(bb.stop - bb.start for bb in bbox)
    segmentation = np.zeros(out_shape, dtype="uint64")

    for seg_id, polygon in enumerate(masks, 1):
        polygon -= offset
        mask = polygon2mask(out_shape, polygon)
        segmentation[mask] = seg_id

    return segmentation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", required=True)
    parser.add_argument("-p", "--prediction_path", required=True)
    parser.add_argument("-c", "--center", nargs=2, type=int)
    parser.add_argument("--halo", nargs=2, type=int, default=(2048, 2048))
    parser.add_argument("--segmentation_path")
    parser.add_argument("--segmentation_key")
    args = parser.parse_args()

    wsi = load_tif_as_zarr(args.input_path)
    if args.center is None:
        center = [sh // 2 for sh in wsi.shape]
    else:
        center = args.center
    bbox = tuple(slice(ce - ha, ce + ha) for ce, ha in zip(center, args.halo))

    segmentation = _load_segmentation(args.prediction_path, bbox)
    image = wsi[bbox]

    if args.segmentation_path is None:
        original_segmentation = None
    else:
        f = zarr.open(args.segmentation_path, mode="r")
        original_segmentation = f[args.segmentation_key][bbox]

    v = napari.Viewer()
    v.add_image(image)
    v.add_labels(segmentation)
    if original_segmentation is not None:
        v.add_labels(original_segmentation)
    napari.run()


if __name__ == "__main__":
    main()
