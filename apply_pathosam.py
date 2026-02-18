import argparse
import os
from pathlib import Path

import h5py
from micro_sam.automatic_segmentation import automatic_instance_segmentation, get_predictor_and_segmenter


def _load_image(input_path):
    ext = Path(input_path).suffix
    if ext == ".h5":
        with h5py.File(input_path, "r") as f:
            image = f["image"][:]
    else:
        raise ValueError(f"Unsupported file extension {ext}")
    return image


def apply_pathosam(input_path, output_path, check):
    image = _load_image(input_path)

    if os.path.exists(output_path):
        with h5py.File(output_path, "r") as f:
            segmentation = f["segmentation"][:]
    else:
        predictor, segmenter = get_predictor_and_segmenter(model_type="vit_b_histopathology", is_tiled=True)
        tile_shape, halo = (512, 512), (64, 64)
        segmentation = automatic_instance_segmentation(
            predictor, segmenter, image, tile_shape=tile_shape, halo=halo, verbose=True, ndim=2
        )

    if check:
        import napari
        v = napari.Viewer()
        v.add_image(image)
        v.add_image(segmentation)
        napari.run()
        return

    if not os.path.exists(output_path):
        output_folder = os.path.split(output_path)[0]
        os.makedirs(output_folder, exist_ok=True)
        # Write the segmentation.
        with h5py.File(output_path, "w") as f:
            f.create_dataset("segmentation", data=segmentation, compression="gzip")


# TODO: support for masking, WSI inference, semantic segmentation and/or object classification
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_path", required=True)
    # TODO: make the model settable
    parser.add_argument("-o", "--output_path")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    apply_pathosam(args.image_path, args.output_path, args.check)


if __name__ == "__main__":
    main()
