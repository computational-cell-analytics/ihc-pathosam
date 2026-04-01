import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np

from elf.evaluation.matching import mean_segmentation_accuracy
from micro_sam.automatic_segmentation import automatic_instance_segmentation, get_predictor_and_segmenter

from _util import get_split

sys.path.insert(0, str(Path(__file__).parent.parent))
from util import get_instance_segmentation_model  # noqa (ihc-pathosam/util.py)


SPLIT_JSON = Path(__file__).parent / "splits" / "split.json"
DEFAULT_MODEL = "/mnt/vast-nhr/projects/cidas/cca/data/pdac_umg_histopatho/models/v1/checkpoints/pathosam-nuclei-instances/best.pt"  # noqa
MODEL_TYPE = "vit_b_histopathology"


def run_prediction(input_path, model_path, cache_folder, cache):
    cache_path = os.path.join(cache_folder, os.path.basename(input_path))
    if cache and os.path.exists(cache_path):
        with h5py.File(cache_path, "r") as f:
            return f["segmentation"][:]

    with h5py.File(input_path, "r") as f:
        image = f["raw"][:]  # (H, W, C)

    tile_shape, halo = (384, 384), (64, 64)
    if model_path is None:
        predictor, segmenter = get_predictor_and_segmenter(
            model_type=MODEL_TYPE, is_tiled=True, segmentation_mode="ais",
        )
    else:
        predictor, segmenter = get_instance_segmentation_model(model_path)
    pred = automatic_instance_segmentation(
        predictor, segmenter, image, tile_shape=tile_shape,
        halo=halo, verbose=True, ndim=2, batch_size=8, min_size=40,
    )

    if cache:
        os.makedirs(cache_folder, exist_ok=True)
        with h5py.File(cache_path, "w") as f:
            f.create_dataset("segmentation", data=pred, compression="gzip")

    return pred


def eval_image(input_path, pred):
    with h5py.File(input_path, "r") as f:
        labels = f["labels/nucleus/instances"][:]
    msa, sas = mean_segmentation_accuracy(pred, labels, return_accuracies=True)
    return msa, sas[0], sas[5]


def run_eval(model_path, split_json, cache):
    _, val_paths = get_split(split_json=split_json)

    model_name = Path(model_path).parent.stem if model_path else "default_ais"
    cache_folder = f"./data/cache/{model_name}"

    msas, sa50s, sa75s = {}, {}, {}
    for path in val_paths:
        print(f"Evaluating {path} ...")
        pred = run_prediction(path, model_path, cache_folder, cache)
        msa, sa50, sa75 = eval_image(path, pred)
        msas[path] = msa
        sa50s[path] = sa50
        sa75s[path] = sa75

    print(f"\nmSA:  {np.mean(list(msas.values())):.4f}")
    print(f"SA50: {np.mean(list(sa50s.values())):.4f}")
    print(f"SA75: {np.mean(list(sa75s.values())):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetuned", action="store_true", help="Use the finetuned model instead of the default.")
    parser.add_argument("--split_json", default=str(SPLIT_JSON))
    parser.add_argument("--cache", action="store_true")
    args = parser.parse_args()
    run_eval(DEFAULT_MODEL if args.finetuned else None, args.split_json, args.cache)

    # Default vit_b_histopathology (AIS):
    # python evaluate_instances.py
    # mSA: 0.1378, SA50: 0.2879, SA75: 0.1182

    # Finetuned model (AIS):
    # python evaluate_instances.py --finetuned
    # mSA: 0.3135, SA50: 0.5235, SA75: 0.3325


if __name__ == "__main__":
    main()
