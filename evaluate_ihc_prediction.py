import argparse
import os
from pathlib import Path

import h5py
from micro_sam.automatic_segmentation import automatic_instance_segmentation, get_predictor_and_segmenter
from elf.evaluation.matching import mean_segmentation_accuracy

DEFAULT_INPUT = "data/annotated_data/cd8/RE-000072-1_1_4-CD8_CRC_CancerScout-2024-07-12T11-04-16_tile02.h5"


def run_prediction(input_path, instance_model_path, semantic_model_path, cache):
    instance_model_name = Path(instance_model_path).parent.stem
    cache_folder = f"./data/cache/{instance_model_name}"
    if semantic_model_path is not None:
        semantic_model_name = Path(semantic_model_path).parent.stem
        cache_folder += f"_{semantic_model_name}"
    cache_path = os.path.join(cache_folder, os.path.basename(input_path))
    if cache and os.path.exists(cache_path):
        with h5py.File(cache_path, "r") as f:
            pred = f["segmentation"][:]
        return pred

    with h5py.File(input_path, "r") as f:
        image = f["image"][:]

    # Run instance segmentation.
    tile_shape, halo = (376, 376), (64, 64)
    predictor, segmenter = get_predictor_and_segmenter(
        model_type="vit_b_histopathology", is_tiled=True, checkpoint=instance_model_path,
    )
    pred = automatic_instance_segmentation(
        predictor, segmenter, image, tile_shape=tile_shape,
        halo=halo, verbose=True, ndim=2, batch_size=8, min_size=40,
    )

    # TODO: run semantic segmentation and filter the instance segmentation.
    if semantic_model_path is not None:
        raise NotImplementedError

    if cache:
        os.makedirs(cache_folder, exist_ok=True)
        with h5py.File(cache_path, mode="a") as f:
            f.create_dataset("segmentation", data=pred, compression="gzip")

    return pred


def run_eval(input_path, pred, check):
    with h5py.File(input_path, "r") as f:
        labels = f["labels/ihc"][:]

    # TODO: visualize other predictions.
    if check:
        with h5py.File(input_path, "r") as f:
            image = f["image"][:]

        import napari
        v = napari.Viewer()
        v.add_image(image)
        v.add_labels(labels)
        v.add_labels(pred)
        napari.run()

    msa, sas = mean_segmentation_accuracy(pred, labels, return_accuracies=True)
    print("mSA :", msa)
    print("SA50:", sas[0])
    print("SA75:", sas[5])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", default=DEFAULT_INPUT)
    parser.add_argument("--instance_model", required=True)
    parser.add_argument("--semantic_model")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    input_path = args.input
    pred = run_prediction(input_path, args.instance_model, args.semantic_model, args.cache)

    run_eval(input_path, pred, args.check)


if __name__ == "__main__":
    main()
