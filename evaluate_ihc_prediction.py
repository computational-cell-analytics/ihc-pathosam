import argparse
import os
from collections import OrderedDict
from pathlib import Path

import h5py
import torch

from elf.evaluation.matching import mean_segmentation_accuracy
from micro_sam.automatic_segmentation import automatic_instance_segmentation, get_predictor_and_segmenter
from micro_sam.instance_segmentation import get_unetr
from micro_sam.util import get_sam_model
from patho_sam.training.util import histopathology_identity
from torch_em.util.prediction import predict_with_halo

DEFAULT_INPUT = "data/annotated_data/cd8/RE-000072-1_1_4-CD8_CRC_CancerScout-2024-07-12T11-04-16_tile02.h5"


def run_semantic_segmentation(image, semantic_model_path, tile_shape, halo):
    num_classes = 3

    # Load the model.
    device = torch.device("cuda")
    predictor, state = get_sam_model(model_type="vit_b_histopathology", device=device, return_state=True)
    decoder_state = OrderedDict(
        [(k, v) for k, v in state["decoder_state"].items() if not k.startswith("out_conv.")]
    )
    unetr = get_unetr(
        image_encoder=predictor.model.image_encoder,
        decoder_state=decoder_state,
        out_channels=num_classes,
        flexible_load_checkpoint=True,
    )
    model_state = torch.load(semantic_model_path, map_location="cpu", weights_only=False)["model_state"]

    # model_state = remap_keys(model_state)
    unetr.load_state_dict(model_state)
    unetr.to(device)
    unetr.eval()

    # Run semantic segmentation.
    input_ = image.transpose((2, 0, 1))
    semantic = predict_with_halo(
        input_, unetr, gpu_ids=[0], block_shape=tile_shape, halo=halo, preprocess=lambda x: x.astype("float32"),
        with_channels=True,
    )

    return semantic


# TODO
def filter_segmentation(instances, semantic):
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
        return pred

    with h5py.File(input_path, "r") as f:
        image = f["image"][:]

    # TODO reactivate
    # Run instance segmentation.
    tile_shape, halo = (376, 376), (64, 64)
    # predictor, segmenter = get_predictor_and_segmenter(
    #     model_type="vit_b_histopathology", is_tiled=True, checkpoint=instance_model_path,
    # )
    # pred = automatic_instance_segmentation(
    #     predictor, segmenter, image, tile_shape=tile_shape,
    #     halo=halo, verbose=True, ndim=2, batch_size=8, min_size=40,
    # )

    import numpy as np
    pred = np.zeros_like(image)

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
