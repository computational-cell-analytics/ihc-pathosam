import argparse
import os
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
import torch

from elf.evaluation.dice import dice_score

from micro_sam.util import get_sam_model
from micro_sam.instance_segmentation import get_unetr

from torch_em.util.prediction import predict_with_halo

from _util import get_split

SPLIT_JSON = Path(__file__).parent / "splits" / "split.json"
DEFAULT_MODEL = "/mnt/vast-nhr/projects/cidas/cca/data/pdac_umg_histopatho/models/v1/checkpoints/pathosam-nuclei-semantic/best.pt"  # noqa
NUM_CLASSES = 3


def load_semantic_model(model_path):
    device = torch.device("cuda")
    predictor, state = get_sam_model(model_type="vit_b_histopathology", device=device, return_state=True)
    decoder_state = OrderedDict(
        [(k, v) for k, v in state["decoder_state"].items() if not k.startswith("out_conv.")]
    )
    unetr = get_unetr(
        image_encoder=predictor.model.image_encoder,
        decoder_state=decoder_state,
        out_channels=NUM_CLASSES,
        flexible_load_checkpoint=True,
        final_activation=None,
    )
    model_state = torch.load(model_path, map_location="cpu", weights_only=False)["model_state"]
    unetr.load_state_dict(model_state)
    unetr.to(device)
    unetr.eval()
    return unetr


def run_prediction(input_path, unetr, cache_folder, cache):
    cache_path = os.path.join(cache_folder, os.path.basename(input_path))
    if cache and os.path.exists(cache_path):
        with h5py.File(cache_path, "r") as f:
            return f["segmentation"][:]

    with h5py.File(input_path, "r") as f:
        image = f["raw"][:]  # (H, W, C)

    tile_shape, halo = (384, 384), (64, 64)
    input_ = image.transpose((2, 0, 1))  # (H, W, C) -> (C, H, W)
    semantic = predict_with_halo(
        input_, unetr, gpu_ids=[0], block_shape=tile_shape, halo=halo,
        preprocess=lambda x: x.astype("float32"), with_channels=True,
    )
    pred = semantic.argmax(axis=0)

    if cache:
        os.makedirs(cache_folder, exist_ok=True)
        with h5py.File(cache_path, "w") as f:
            f.create_dataset("segmentation", data=pred, compression="gzip")

    return pred


def eval_image(input_path, pred):
    with h5py.File(input_path, "r") as f:
        labels = f["labels/nucleus/semantic"][:]

    scores = []
    for cls in range(NUM_CLASSES):
        pred_mask = (pred == cls).astype("uint8")
        label_mask = (labels == cls).astype("uint8")
        scores.append(dice_score(pred_mask, label_mask, threshold_seg=None, threshold_gt=None))

    return scores


def run_eval(model_path, split_json, cache):
    _, val_paths = get_split(split_json=split_json)

    model_name = Path(model_path).parent.stem
    cache_folder = f"./data/cache/{model_name}"

    unetr = load_semantic_model(model_path)

    all_scores = {cls: [] for cls in range(NUM_CLASSES)}
    for path in val_paths:
        print(f"Evaluating {path} ...")
        pred = run_prediction(path, unetr, cache_folder, cache)
        scores = eval_image(path, pred)
        for cls, score in enumerate(scores):
            all_scores[cls].append(score)

    class_names = ["background", "negative cells", "positive cells"]
    print("\nPer-class Dice:")
    mean_scores = []
    for cls in range(NUM_CLASSES):
        mean_score = np.mean(all_scores[cls])
        mean_scores.append(mean_score)
        print(f"  {class_names[cls]}: {mean_score:.4f}")
    print(f"Mean Dice: {np.mean(mean_scores):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_json", default=str(SPLIT_JSON))
    parser.add_argument("--cache", action="store_true")
    args = parser.parse_args()
    run_eval(DEFAULT_MODEL, args.split_json, args.cache)

    # Results from finetuned model for semantic segmentation:
    # Per-class dice score:
    # background: 0.9717
    # negative cells: 0.6637
    # positive cells: 0.8122
    # Mean Dice: 0.8159


if __name__ == "__main__":
    main()
