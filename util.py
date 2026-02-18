import json

import numpy as np
import tifffile
import zarr
from skimage.draw import polygon as draw_polygon


def load_tif_as_zarr(path, scale_level=0):
    tif = tifffile.TiffFile(path)
    store = tif.aszarr()
    groups = zarr.open(store, mode="r")
    data = groups[str(scale_level)]
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
