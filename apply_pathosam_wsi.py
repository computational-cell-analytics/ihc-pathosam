import argparse
import os
from tempfile import TemporaryDirectory

import zarr
from micro_sam.automatic_segmentation import automatic_instance_segmentation, get_predictor_and_segmenter
from util import load_image, get_mask


def apply_pathosam_wsi(image_path, output_path, model_path, batch_size, output_key):
    output_folder = os.path.split(output_path)[0]
    os.makedirs(output_folder, exist_ok=True)
    if os.path.exists(output_path) and output_key in zarr.open(output_path, "r"):
        return

    image = load_image(image_path)

    predictor, segmenter = get_predictor_and_segmenter(
        model_type="vit_b_histopathology", is_tiled=True, checkpoint=model_path,
    )
    tile_shape, halo = (376, 376), (64, 64)
    mask = get_mask(image_path)

    shards = tuple(4 * ts for ts in tile_shape)
    segmentation = zarr.open(output_path, mode="a").create_array(
        name="seg", shape=image.shape[:2], dtype="uint64", chunks=tile_shape, shards=shards,
    )

    with TemporaryDirectory() as tmp:
        tmp_embed_path = os.path.join(tmp, "tmp_embeds.zarr")
        automatic_instance_segmentation(
            predictor, segmenter, image, tile_shape=tile_shape,
            halo=halo, verbose=True, ndim=2, batch_size=batch_size,
            mask_path=mask, embedding_path=tmp_embed_path,
            optimize_memory=True, segmentation=segmentation,
        )


# TODO: semantic segmentation and/or object classification
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_path", required=True)
    parser.add_argument("-o", "--output_path")
    parser.add_argument("-m", "--model_path")
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("-k", "--output_key", default="segmentation")
    args = parser.parse_args()

    apply_pathosam_wsi(
        args.image_path, args.output_path, args.model_path, args.batch_size, args.output_key,
    )


if __name__ == "__main__":
    main()
