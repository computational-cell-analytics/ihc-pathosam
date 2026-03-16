import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np

from elf.evaluation.matching import mean_segmentation_accuracy
from micro_sam.util import precompute_image_embeddings
from micro_sam.automatic_segmentation import automatic_instance_segmentation
from patho_sam.semantic_segmentation import get_semantic_predictor_and_segmenter

sys.path.append("..")
from util import get_instance_segmentation_model  # noqa

DEFAULT_INPUT = "splits/combined.json"


def run_semantic_segmentation(image, semantic_model_path, tile_shape, halo):
    tile_shape, halo = (384, 384), (64, 64)
    predictor, segmenter = get_semantic_predictor_and_segmenter(
        model_type="vit_b_histopathology", is_tiled=True, checkpoint=semantic_model_path, num_classes=3
    )
    image_embeddings = precompute_image_embeddings(
        predictor=predictor,
        input_=image,
        ndim=2,
        tile_shape=tile_shape,
        halo=halo,
        verbose=True,
        batch_size=8,
    )
    segmenter.initialize(
        image=image,
        image_embeddings=image_embeddings,
        num_classes=3,
        tile_shape=tile_shape,
        halo=halo,
        verbose=True,
        batch_size=8,
    )
    semantic_segmentation = segmenter._semantic_segmentation
    return semantic_segmentation


def filter_segmentation(instances, semantic):
    from skimage.measure import regionprops

    def majority_label(regionmask, intensity_image):
        values = intensity_image[regionmask]
        return np.bincount(values).argmax()

    props = regionprops(instances, semantic, extra_properties=(majority_label,))
    semantic_id_map = {prop.label: prop.majority_label for prop in props}
    keep_ids = [seg_id for seg_id, sem_id in semantic_id_map.items() if sem_id == 2]
    filtered = instances.copy()
    filtered[~np.isin(filtered, keep_ids)] = 0
    return filtered


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
        return pred, cache_path

    with h5py.File(input_path, "r") as f:
        image = f["image"][:]

    # Run instance segmentation.
    tile_shape, halo = (384, 384), (64, 64)
    predictor, segmenter = get_instance_segmentation_model(instance_model_path)
    pred = automatic_instance_segmentation(
        predictor, segmenter, image, tile_shape=tile_shape,
        halo=halo, verbose=True, ndim=2, batch_size=8, min_size=40,
    )

    # Filter with semantic model if specified.
    if semantic_model_path is not None:
        instances = pred.copy()
        semantic = run_semantic_segmentation(image, semantic_model_path, tile_shape, halo)
        pred = filter_segmentation(instances, semantic)
        if cache:
            os.makedirs(cache_folder, exist_ok=True)
            with h5py.File(cache_path, mode="a") as f:
                f.create_dataset("instance_segmentation", data=instances, compression="gzip")
                f.create_dataset("semantic_segmentation", data=semantic, compression="gzip")

    if cache:
        os.makedirs(cache_folder, exist_ok=True)
        with h5py.File(cache_path, mode="a") as f:
            f.create_dataset("segmentation", data=pred, compression="gzip")

    return pred, cache_path


def eval_image(input_path, pred, check, cache_path):
    with h5py.File(input_path, "r") as f:
        labels = f["labels/ihc"][:]

    if check:
        with h5py.File(input_path, "r") as f:
            image = f["image"][:]
        if os.path.exists(cache_path):
            with h5py.File(cache_path, "r") as f:
                instance_seg = f["instance_segmentation"][:] if "instance_segmentation" in f else None
                semantic_seg = f["semantic_segmentation"][:] if "semantic_segmentation" in f else None
        else:
            instance_seg, semantic_seg = None, None

        import napari
        v = napari.Viewer()
        v.add_image(image)
        v.add_labels(labels)
        v.add_labels(pred)
        if instance_seg is not None:
            v.add_labels(instance_seg)
        if semantic_seg is not None:
            v.add_labels(semantic_seg)
        napari.run()

    else:
        instance_seg, semantic_seg = None, None

    msa, sas = mean_segmentation_accuracy(pred, labels, return_accuracies=True)
    return msa, sas[0], sas[5]


def _aggregate(vals, term):
    aggregated = [v for k, v in vals.items() if term in k]
    return np.mean(aggregated)


def run_eval(input_path, instance_model, semantic_model, check, cache):
    with open(input_path, "r") as f:
        test_paths = json.load(f)["test"]

    msas, sa50s, sa75s = {}, {}, {}
    for path in test_paths:
        pred, cache_path = run_prediction(path, instance_model, semantic_model, cache)
        msa, sa50, sa75 = eval_image(path, pred, check, cache_path)
        msas[path] = msa
        sa50s[path] = sa50
        sa75s[path] = sa75

    msa_cd3, msa_cd8 = _aggregate(msas, "cd3"), _aggregate(msas, "cd8")
    sa50_cd3, sa50_cd8 = _aggregate(sa50s, "cd3"), _aggregate(sa50s, "cd8")
    sa75_cd3, sa75_cd8 = _aggregate(sa75s, "cd3"), _aggregate(sa75s, "cd8")

    print("mSA :", msa_cd3, "(cd3)", msa_cd8, "(cd8)")
    print("SA50 :", sa50_cd3, "(cd3)", sa50_cd8, "(cd8)")
    print("SA75 :", sa75_cd3, "(cd3)", sa75_cd8, "(cd8)")


# TODO visualize once
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", default=DEFAULT_INPUT)
    parser.add_argument("--instance_model", required=True)
    parser.add_argument("--semantic_model")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--check", action="store_true")

    args = parser.parse_args()
    run_eval(args.input, args.instance_model, args.semantic_model, args.check, args.cache)


if __name__ == "__main__":
    main()
