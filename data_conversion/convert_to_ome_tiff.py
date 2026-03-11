import os
import subprocess
from pathlib import Path
from glob import glob

import tifffile

os.environ["BF_MAX_MEM"] = "32g"


def _find_largest_series(path):
    largest_idx, largest_size, largest_shape = None, 0, None

    with tifffile.TiffFile(path) as f:
        series = f.series
        for idx, ser in enumerate(series):
            if ser.size > largest_size:
                largest_size = ser.size
                largest_shape = ser.shape
                largest_idx = idx
    return largest_idx, largest_shape


def convert_file(path, output_root):
    if "HE" in path:
        output_folder = os.path.join(output_root, "HE")
    else:
        output_folder = os.path.join(output_root, "DAB")
    name = Path(path).stem.replace(" ", "_")
    # The final output path.
    output_path = os.path.join(output_folder, f"{name}.ome.tif")
    if os.path.exists(output_path):
        return

    print("Converting", path, "...")

    # First conversion: from vsi format to tiff
    output1 = os.path.join(output_folder, f"{name}-1.ome.tif")
    cmd = ["bfconvert", path, output1]
    subprocess.run(cmd)

    # Second conversion: largest series to ome.tiff
    series_id, shape = _find_largest_series(output1)
    print("Largest series is", series_id, "with shape:", shape)
    cmd = [
        "bfconvert", "-series", str(series_id),
        "-tilex", "512", "-tiley", "512",
        "-pyramid-scale", "2", "-pyramid-resolutions", "6",
        "-compression", "JPEG",
        output1, output_path
    ]
    subprocess.run(cmd)

    # Remove intermediate file.
    os.remove(output1)


def main():
    root = "/mnt/lustre-grete/usr/u12086/data/pdac-histo-kfo/data_20260310"
    input_folder = os.path.join(root, "shared_data")
    output_root = os.path.join(root, "converted_data")

    input_files = glob(os.path.join(input_folder, "*.vsi"))
    for ff in input_files:
        convert_file(ff, output_root)


if __name__ == "__main__":
    main()
