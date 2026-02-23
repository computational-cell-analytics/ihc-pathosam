import argparse
import os

import h5py
from micro_sam.automatic_segmentation import automatic_instance_segmentation, get_predictor_and_segmenter
from util import load_image


def apply_pathosam(image_path, output_path, model_path, use_mask, check, batch_size, output_key):
    output_folder = os.path.split(output_path)[0]
    os.makedirs(output_folder, exist_ok=True)

    image = load_image(image_path)

    if os.path.exists(output_path) and output_key in h5py.File(output_path, "r"):
        with h5py.File(output_path, "r") as f:
            segmentation = f[output_key][:]
    else:
        predictor, segmenter = get_predictor_and_segmenter(
            model_type="vit_b_histopathology", is_tiled=True, checkpoint=model_path,
        )
        tile_shape, halo = (376, 376), (64, 64)
        segmentation = automatic_instance_segmentation(
            predictor, segmenter, image, tile_shape=tile_shape,
            halo=halo, verbose=True, ndim=2, batch_size=batch_size,
        )

    if check:
        import napari
        v = napari.Viewer()
        v.add_image(image)
        v.add_labels(segmentation)
        napari.run()

    # Write the segmentation.
    with h5py.File(output_path, "a") as f:
        if output_key in f:
            return
        f.create_dataset(output_key, data=segmentation, compression="gzip")


# TODO: semantic segmentation and/or object classification
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_path", required=True)
    parser.add_argument("-o", "--output_path")
    parser.add_argument("-m", "--model_path")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("-k", "--output_key", default="segmentation")
    args = parser.parse_args()

    apply_pathosam(
        args.image_path, args.output_path, args.model_path,
        args.check, args.batch_size, args.output_key,
    )


if __name__ == "__main__":
    main()
