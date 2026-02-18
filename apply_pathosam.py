import argparse
import os
from pathlib import Path

import h5py
from micro_sam.automatic_segmentation import automatic_instance_segmentation, get_predictor_and_segmenter
from util import load_tif_as_zarr, get_mask


def _load_image(input_path):
    ext = Path(input_path).suffix
    if ext == ".h5":
        with h5py.File(input_path, "r") as f:
            image = f["image"][:]
    elif ext in (".tif", ".tiff"):
        image = load_tif_as_zarr(input_path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}.")
    return image


def apply_pathosam(image_path, output_path, model_path, use_mask, check):
    image = _load_image(image_path)

    if os.path.exists(output_path):
        with h5py.File(output_path, "r") as f:
            segmentation = f["segmentation"][:]
    else:
        predictor, segmenter = get_predictor_and_segmenter(
            model_type="vit_b_histopathology", is_tiled=True, checkpoint=model_path,
        )
        tile_shape, halo = (376, 376), (64, 64)
        mask = get_mask(image_path) if use_mask else None
        segmentation = automatic_instance_segmentation(
            predictor, segmenter, image, tile_shape=tile_shape,
            halo=halo, verbose=True, ndim=2, batch_size=16,
            mask_path=mask,
        )

    if check:
        import napari
        v = napari.Viewer()
        v.add_image(image)
        v.add_labels(segmentation)
        napari.run()

    if not os.path.exists(output_path):
        output_folder = os.path.split(output_path)[0]
        os.makedirs(output_folder, exist_ok=True)
        # Write the segmentation.
        with h5py.File(output_path, "w") as f:
            f.create_dataset("segmentation", data=segmentation, compression="gzip")


# TODO: semantic segmentation and/or object classification
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_path", required=True)
    parser.add_argument("-o", "--output_path")
    parser.add_argument("-m", "--model_path")
    parser.add_argument("--use_mask", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    apply_pathosam(args.image_path, args.output_path, args.model_path, args.use_mask, args.check)


if __name__ == "__main__":
    main()
