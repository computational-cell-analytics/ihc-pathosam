import os
from glob import glob

import h5py
import napari


def main():
    root = "/mnt/vast-nhr/projects/nim00007/data/histopatho/cancerscout/training_data/v2"
    files = sorted(glob(os.path.join(root, "**", "*.h5"), recursive=True))

    for ff in files:
        with h5py.File(ff, mode="r") as f:
            image = f["image"][:]
            instances = f["labels/silver/nuclei"][:]
            semantic = f["labels/silver/semantic"][:]

        folder, name = os.path.split(ff)
        name = os.path.join(os.path.split(folder)[1], name)
        print(name)

        v = napari.Viewer()
        v.add_image(image)
        v.add_labels(instances)
        v.add_labels(semantic)
        v.title = name
        napari.run()


if __name__ == "__main__":
    main()
