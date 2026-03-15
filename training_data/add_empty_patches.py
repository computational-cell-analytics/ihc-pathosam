import os
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.append("..")
from util import load_tif_as_zarr  # noqa


def process_image(image_path, output_path, center, halo, check=True):
    scale_level = 0
    print(image_path)
    data = load_tif_as_zarr(image_path, scale_level)
    # Transpose the center?
    bb = tuple(slice(ce - ha, ce + ha) for ce, ha in zip(center, halo))
    image = data[bb]
    labels = np.zeros(image.shape[:2], dtype="uint8")
    print(labels.shape)

    if check:
        import napari
        v = napari.Viewer()
        v.add_image(image)
        napari.run()
        return

    with h5py.File(output_path, "w") as f:
        f.create_dataset("image", data=image, compression="gzip")
        f.create_dataset("labels/silver/semantic", data=labels, compression="gzip")
        f.create_dataset("labels/silver/nuclei", data=labels, compression="gzip")


def main():
    root = "/mnt/vast-nhr/projects/nim00007/data/histopatho/cancerscout"
    output_folder = "/mnt/vast-nhr/projects/nim00007/data/histopatho/cancerscout/training_data/v2/empty"
    halo = (2048, 2048)

    inputs = {
        "cd3/RE-000148-1_1_3-CD3_CRC_CancerScout-2024-07-02T18-33-41.tiff": [(48400, 22700), (59600, 3500)],
        "cd8/RE-000022-1_1_4-CD8_CRC_CancerScout-2024-07-16T12-31-15.tiff": [(55640, 40439), (83475, 5000)],
    }

    os.makedirs(output_folder, exist_ok=True)
    for name, centers in inputs.items():
        fname = Path(name).stem
        for i, center in enumerate(centers):
            image_path = os.path.join(root, name)
            output_path = os.path.join(output_folder, f"{fname}_tile{i}.h5")
            process_image(image_path, output_path, center, halo, check=False)


if __name__ == "__main__":
    main()
