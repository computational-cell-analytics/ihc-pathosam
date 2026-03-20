import argparse
import os
from glob import glob
from pathlib import Path
from tempfile import TemporaryDirectory

import zarr
from elf.wrapper import RoiWrapper
from micro_sam.automatic_segmentation import automatic_instance_segmentation
from micro_sam.util import precompute_image_embeddings
from patho_sam.semantic_segmentation import get_semantic_predictor_and_segmenter
from util import load_image, get_mask, get_instance_segmentation_model, get_obap_model


def _get_predictor_and_segmenter(model_type, model_path, semantic):
    if semantic:
        predictor, segmenter = get_semantic_predictor_and_segmenter(
            model_type="vit_b_histopathology", is_tiled=True, checkpoint=model_path, num_classes=3
        )

    else:
        predictor, segmenter = get_instance_segmentation_model(model_path)

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


def apply_classification_model(
    classification_model, image, segmentation, embed_path, mask, batch_size, tile_shape, halo, predictor, output,
):
    from deap_objects.inference import run_inference, project_to_semantic_segmentation

    model_cfg = get_obap_model(classification_model)
    class_pred = run_inference(
        model_cfg, classification_model, image, segmentation,
        mask=mask, batch_size=batch_size,
        tile_shape=tile_shape, halo=halo,
        embedding_path=embed_path,
        predictor=predictor,
    )
    project_to_semantic_segmentation(segmentation, class_pred, output)


def check_pred(image, segmentation, semantic_segmentation):
    import napari
    v = napari.Viewer()
    v.add_image(image[:], name="image")
    v.add_labels(segmentation[:], name="segmentation")
    if semantic_segmentation is not None:
        v.add_labels(semantic_segmentation[:], name="semantic-segmentation")
    napari.run()


def apply_pathosam_wsi(
    image_path, output_path, model_path,
    batch_size, output_key, semantic, mask, roi,
    classification_model, check=False,
):
    output_folder = os.path.split(output_path)[0]
    os.makedirs(output_folder, exist_ok=True)

    image = load_image(image_path)
    if roi is not None:
        roi = (slice(roi[0], roi[1]), slice(roi[2], roi[3]), slice(0, 3))
        image = RoiWrapper(image, roi)

    have_pred = os.path.exists(output_path) and output_key in zarr.open(output_path, mode="r")
    sem_seg_key = "semantic_from_class"
    if check and have_pred:
        f = zarr.open(output_path, mode="r")
        segmentation = f[output_key]
        semantic_segmentation = f[sem_seg_key] if sem_seg_key in f else None
        check_pred(image, segmentation, semantic_segmentation)
    elif have_pred:
        return

    predictor, segmenter = _get_predictor_and_segmenter(
        model_type="vit_b_histopathology", model_path=model_path, semantic=semantic,
    )

    tile_shape, halo = (384, 384), (64, 64)
    mask = None if mask in ("None", "none") else get_mask(image_path, masking_method=mask)

    dtype = "uint8" if semantic else "uint64"
    shards = tuple(4 * ts for ts in tile_shape)
    segmentation = zarr.open(output_path, mode="a").create_array(
        name=output_key, shape=image.shape[:2], dtype=dtype, chunks=tile_shape, shards=shards,
    )

    with TemporaryDirectory() as tmp:
        tmp_embed_path = os.path.join(tmp, "tmp_embeds.zarr")
        if semantic:
            assert classification_model is None, "Classification model for semantic segmentation is not supported"
            _run_semantic_segmentation(
                predictor, segmenter, image, segmentation, mask, tmp_embed_path, tile_shape, halo, batch_size
            )
        else:
            automatic_instance_segmentation(
                predictor, segmenter, image, tile_shape=tile_shape,
                halo=halo, verbose=True, ndim=2, batch_size=batch_size,
                mask_path=mask, embedding_path=tmp_embed_path,
                optimize_memory=True, segmentation=segmentation
            )
            if classification_model is None:
                semantic_segmentation = None
            else:
                semantic_segmentation = zarr.open(output_path, mode="a").create_array(
                    name=sem_seg_key, shape=image.shape[:2], dtype=dtype, chunks=tile_shape, shards=shards,
                )
                apply_classification_model(
                    classification_model, image, segmentation, tmp_embed_path, mask,
                    batch_size=batch_size, tile_shape=tile_shape, halo=halo,
                    predictor=predictor, output=semantic_segmentation,
                )

        if check:
            check_pred(image, segmentation, semantic_segmentation)


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
    parser.add_argument("--pattern")
    parser.add_argument("--classification_model")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.pattern is None:
        apply_pathosam_wsi(
            image_path=args.image_path, output_path=args.output_path, model_path=args.model_path,
            batch_size=args.batch_size, output_key=args.output_key,
            semantic=args.semantic, mask=args.mask, roi=args.roi,
            classification_model=args.classification_model,
            check=args.check,
        )
    else:
        input_folder, output_folder = args.image_path, args.output_path
        assert os.path.isdir(input_folder)
        os.makedirs(output_folder, exist_ok=True)
        assert os.path.exists(output_folder)

        inputs = glob(os.path.join(input_folder, args.pattern))
        for img_path in inputs:
            out_path = os.path.join(output_folder, f"{Path(img_path).stem}.zarr")
            apply_pathosam_wsi(
                image_path=img_path, output_path=out_path, model_path=args.model_path,
                batch_size=args.batch_size, output_key=args.output_key,
                semantic=args.semantic, mask=args.mask, roi=args.roi,
                classification_model=args.classification_model, check=args.check,
            )


if __name__ == "__main__":
    main()
