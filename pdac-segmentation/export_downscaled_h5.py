"""Export downscaled (s3 = 1/8 of s0) raw + segmentation labels to h5 for visualization."""

from pathlib import Path

import h5py
import numpy as np
import zarr
from skimage.transform import resize

HE_ZARR_DIR = Path("/mnt/vast-nhr/projects/nim00007/data/histopatho/pdac-kfo/data_20260310/converted_zarr_he")
SEG_DIR = Path(
    "/mnt/vast-nhr/projects/nim00007/data/histopatho/pdac-kfo/data_20260310/converted_zarr_he_tissue_segmentation"
)
OUT_DIR = Path("/mnt/vast-nhr/home/archit/u12090/ihc-pathosam/pdac-segmentation")

STAGES = ["tb", "epithelium", "multi-tissue"]
SLIDES = ["TM11_HE", "TM50_HE", "TM90_HE", "TM105_HE", "TM120_HE"]


def export_slide(name):
    out_path = OUT_DIR / f"{name}_s3.h5"
    if out_path.exists():
        print(f"{name}: already exists, skipping")
        return

    src = zarr.open(str(HE_ZARR_DIR / f"{name}.zarr"), mode="r")
    seg = zarr.open(str(SEG_DIR / f"{name}.zarr"), mode="r")

    print(f"{name}: loading s3 raw ...")
    raw = src["s3/image"][:]
    h, w = raw.shape[:2]
    print(f"  s3 shape: {raw.shape}")

    with h5py.File(out_path, "w") as f:
        f.create_dataset("raw", data=raw, chunks=(512, 512, 3), compression="gzip")
        print("  raw saved")

        for stage in STAGES:
            key = f"{stage}_labels"
            print(f"  {stage}: loading s0 labels and downscaling ...")
            labels_s0 = seg[key][:]
            labels_s3 = resize(labels_s0, (h, w), order=0, anti_aliasing=False, preserve_range=True).astype(np.uint8)
            f.create_dataset(key, data=labels_s3, chunks=(512, 512), compression="gzip")
            print(f"  {stage}: saved  shape={labels_s3.shape}")

    print(f"{name}: done -> {out_path}")


for slide in SLIDES:
    export_slide(slide)
