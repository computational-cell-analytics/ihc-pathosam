import argparse
import os
from pathlib import Path

import h5py
import napari
from util import load_tif_as_zarr, extract_region_instance_id_arrays


def main():
    parser = argparse.ArgumentParser()
    # "data/philips/cd8/RE-000072-1_1_4-CD8_CRC_CancerScout-2024-07-12T11-04-16.tiff"
    parser.add_argument("-i", "--image_path", required=True)
    # "data/philips/cd8/RE-000072-1_1_4-CD8_CRC_CancerScout-2024-07-12T11-04-16_annotations.json"
    parser.add_argument("-l", "--label_path", required=True)
    parser.add_argument("-o", "--output_folder", required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)

    scale_level = 0
    data = load_tif_as_zarr(args.image_path, scale_level)
    fname = Path(args.image_path).stem

    regions = extract_region_instance_id_arrays(args.label_path)
    for i, (bb, seg) in enumerate(regions.values()):
        image = data[bb]
        assert image.shape[:2] == seg.shape
        out_path = os.path.join(output_folder, f"{fname}_tile{i:02}.h5")

        if args.check:
            v = napari.Viewer()
            v.add_image(image)
            v.add_labels(seg)
            napari.run()

        with h5py.File(out_path, "w") as f:
            f.create_dataset("image", data=image, compression="gzip")
            f.create_dataset("labels/ihc", data=seg, compression="gzip")


if __name__ == "__main__":
    main()
