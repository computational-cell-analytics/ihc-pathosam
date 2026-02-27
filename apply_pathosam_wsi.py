import argparse
import os
from tempfile import TemporaryDirectory

import zarr
from elf.wrapper import RoiWrapper
from micro_sam.automatic_segmentation import automatic_instance_segmentation, get_predictor_and_segmenter
from micro_sam.util import precompute_image_embeddings
from patho_sam.semantic_segmentation import get_semantic_predictor_and_segmenter
from util import load_image, get_mask


def _get_predictor_and_segmenter(model_type, model_path, semantic):
    if semantic:
        predictor, segmenter = get_semantic_predictor_and_segmenter(
            model_type="vit_b_histopathology", is_tiled=True, checkpoint=model_path, num_classes=3
        )

    else:
        predictor, segmenter = get_predictor_and_segmenter(
            model_type="vit_b_histopathology", is_tiled=True, checkpoint=model_path,
        )

    return predictor, segmenter


def _run_semantic_segmentation(
    predictor, segmenter, image_data, segmentation, mask, embedding_path, tile_shape, halo, batch_size
):
    image_embeddings = precompute_image_embeddings(
        predictor=predictor,
        input_=image_data,
        save_path=embedding_path,
        ndim=2,
        tile_shape=tile_shape,
        halo=halo,
        verbose=True,
        batch_size=batch_size,
        mask=mask,
    )
    segmenter.initialize(
        image=image_data,
        image_embeddings=image_embeddings,
        num_classes=3,
        tile_shape=tile_shape,
        halo=halo,
        verbose=True,
        batch_size=batch_size,
        semantic_segmentation=segmentation,
        mask=mask,
    )


def apply_pathosam_wsi(image_path, output_path, model_path, batch_size, output_key, semantic, mask, roi):
    output_folder = os.path.split(output_path)[0]
    os.makedirs(output_folder, exist_ok=True)
    if os.path.exists(output_path) and output_key in zarr.open(output_path, mode="r"):
        return

    image = load_image(image_path)
    if roi is not None:
        roi = (slice(roi[0], roi[1]), slice(roi[2], roi[3]), slice(0, 3))
        image = RoiWrapper(image, roi)
    predictor, segmenter = _get_predictor_and_segmenter(
        model_type="vit_b_histopathology", model_path=model_path, semantic=semantic,
    )

    tile_shape, halo = (376, 376), (64, 64)
    mask = None if mask in ("None", "none") else get_mask(image_path, masking_method=mask)

    dtype = "uint8" if semantic else "uint64"
    shards = tuple(4 * ts for ts in tile_shape)
    segmentation = zarr.open(output_path, mode="a").create_array(
        name=output_key, shape=image.shape[:2], dtype=dtype, chunks=tile_shape, shards=shards,
    )

    with TemporaryDirectory() as tmp:
        tmp_embed_path = os.path.join(tmp, "tmp_embeds.zarr")
        if semantic:
            _run_semantic_segmentation(
                predictor, segmenter, image, segmentation, mask, tmp_embed_path, tile_shape, halo, batch_size
            )
        else:
            automatic_instance_segmentation(
                predictor, segmenter, image, tile_shape=tile_shape,
                halo=halo, verbose=True, ndim=2, batch_size=batch_size,
                mask_path=mask, embedding_path=tmp_embed_path,
                optimize_memory=True, segmentation=segmentation,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_path", required=True)
    parser.add_argument("-o", "--output_path")
    parser.add_argument("-m", "--model_path")
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("-k", "--output_key", default="segmentation")
    parser.add_argument("--semantic", action="store_true")
    parser.add_argument("--mask", default="cd8")
    parser.add_argument("--roi", nargs=4, type=int)
    args = parser.parse_args()

    apply_pathosam_wsi(
        image_path=args.image_path, output_path=args.output_path, model_path=args.model_path,
        batch_size=args.batch_size, output_key=args.output_key,
        semantic=args.semantic, mask=args.mask, roi=args.roi,
    )


if __name__ == "__main__":
    main()
