import json
from pathlib import Path

import h5py
import numpy as np
import tifffile
import zarr

from skimage.draw import polygon as draw_polygon
from scipy.ndimage import uniform_filter
from elf.wrapper.resized_volume import ResizedVolume


def load_tif_as_zarr(path, scale_level=0):
    tif = tifffile.TiffFile(path)
    store = tif.aszarr()
    groups = zarr.open(store, mode="r")
    try:
        data = groups[str(scale_level)]
    except IndexError:
        data = groups
    return data


def extract_region_instance_id_arrays(json_path, pad=0):
    """
    Returns:
      dict[region_index] = (bbox_slice, seg)

    bbox_slice: (slice(y0, y1), slice(x0, x1))  # python slicing, y first
    seg: uint32 array of shape (y1-y0, x1-x0) with 0=background and
         instance ids = (cell_list_index + 1)
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    regions = data.get("annotated_regions", [])
    cells = data.get("cells", [])

    # --- region bboxes from region point lists ---
    region_bboxes = []
    for pts in regions:
        if not (isinstance(pts, list) and len(pts) >= 2 and isinstance(pts[0], list) and len(pts[0]) == 2):
            continue
        arr = np.asarray(pts, dtype=np.float32)
        x0 = int(np.floor(arr[:, 0].min())) - pad
        y0 = int(np.floor(arr[:, 1].min())) - pad
        x1 = int(np.ceil(arr[:, 0].max())) + pad
        y1 = int(np.ceil(arr[:, 1].max())) + pad
        if x1 > x0 and y1 > y0:
            region_bboxes.append((x0, y0, x1, y1))

    if not region_bboxes:
        return {}

    # allocate outputs
    out = {}
    for r_i, (x0, y0, x1, y1) in enumerate(region_bboxes):
        bbox_slice = (slice(y0, y1), slice(x0, x1))
        seg = np.zeros((y1 - y0, x1 - x0), dtype=np.uint32)
        out[r_i] = (bbox_slice, seg)

    # --- assign each cell to first region whose bbox contains its centroid ---
    for cell_idx, cell in enumerate(cells):
        # polygon nodes are assumed to be the last element; also allow polygon-only cells
        poly = cell
        if not (isinstance(poly, list) and len(poly) >= 3 and isinstance(poly[0], list) and len(poly[0]) == 2):
            poly = cell[-1] if isinstance(cell, list) and len(cell) else None
        if not (isinstance(poly, list) and len(poly) >= 3 and isinstance(poly[0], list) and len(poly[0]) == 2):
            continue

        pts = np.asarray(poly, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 3:
            continue

        cx = float(pts[:, 0].mean())
        cy = float(pts[:, 1].mean())

        inst_id = int(cell_idx + 1)

        for r_i, (x0, y0, x1, y1) in enumerate(region_bboxes):
            if (cx >= x0) and (cx < x1) and (cy >= y0) and (cy < y1):
                bbox_slice, seg = out[r_i]

                # shift polygon into region-local coords
                x = pts[:, 0] - x0
                y = pts[:, 1] - y0

                # quick reject
                h, w = seg.shape
                if x.max() < 0 or y.max() < 0 or x.min() >= w or y.min() >= h:
                    break

                rr, cc = draw_polygon(y, x, shape=seg.shape)
                seg[rr, cc] = inst_id
                break  # regions are non-overlapping; assigned

    return out


def get_mask(input_path):
    scale = 3
    image = load_tif_as_zarr(input_path, scale_level=scale)[:]

    thresh1 = np.array([230, 230, 230])[None, None]
    thresh2 = np.array([250, 0, 0])[None, None]
    thresh3 = np.array([10, 10, 10])[None, None]
    mask = (image > thresh1).all(axis=-1) | (image > thresh2).all(axis=-1) | (image < thresh3).all(axis=-1)
    window = (32, 32)
    mask = uniform_filter(mask.astype("float32"), size=window, mode="reflect")
    min_mask_fraction = 0.75
    mask = ~(mask > min_mask_fraction)

    # import napari
    # v = napari.Viewer()
    # v.add_image(image)
    # v.add_labels(mask)
    # napari.run()

    full_shape = load_tif_as_zarr(input_path, scale_level=0).shape[:2]
    mask = ResizedVolume(mask, shape=full_shape)
    return mask


def load_image(input_path):
    ext = Path(input_path).suffix
    if ext == ".h5":
        with h5py.File(input_path, "r") as f:
            image = f["image"][:]
    elif ext in (".tif", ".tiff"):
        image = load_tif_as_zarr(input_path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}.")
    return image


if __name__ == "__main__":
    input_path = "data/philips/cd8/RE-000139-1_1_4-CD8_CRC_CancerScout-2024-07-12T08-43-25.tiff"
    mask = get_mask(input_path)
    print(mask.shape)
