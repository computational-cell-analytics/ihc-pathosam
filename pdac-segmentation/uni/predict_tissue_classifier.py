"""Predict background, normal pancreas, tumor and other tissue on an H&E WSI with a trained UNI2-h tissue classifier.

The prediction is written to an HDF5 file at the classifier's pixel size. If labels are given, the Dice, precision,
recall and F1 per class are printed as well. Only glass and annotated tissue are scored: the annotations are not
exhaustive and other tissue is not annotated.
"""
import os
import argparse

import h5py
import numpy as np

import torch

from elf.evaluation import dice_score
from elf.evaluation.matching import f1, precision, recall

from tissue_util import (
    PATCH_SIZE, OTHER_TISSUE, IGNORE_LABEL, LinearProbe, add_context, load_labels, predict_tokens, get_slide_features,
)


def compute_scores(prediction, labels, tissue_mask, class_names):
    """Dice, precision, recall and F1 per class on glass and annotated tissue.

    A class that is not annotated on the slide is not scored.
    """
    rows = (np.arange(labels.shape[0]) * tissue_mask.shape[0] // labels.shape[0])
    cols = (np.arange(labels.shape[1]) * tissue_mask.shape[1] // labels.shape[1])
    tissue = tissue_mask[rows[:, None], cols[None, :]]
    valid = (labels != IGNORE_LABEL) & ~((labels == 0) & tissue)
    prediction, labels = prediction[valid], labels[valid]
    scores = {}
    for class_id, name in class_names.items():
        pred_mask, label_mask = prediction == class_id, labels == class_id
        if not label_mask.any():
            continue
        tp = int((pred_mask & label_mask).sum())
        fp, fn = int(pred_mask.sum()) - tp, int(label_mask.sum()) - tp
        scores[name] = {
            "dice": dice_score(pred_mask, label_mask), "precision": precision(tp, fp, fn),
            "recall": recall(tp, fp, fn), "f1": f1(tp, fp, fn),
        }
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--image", required=True, help="WSI as zarr (s{level}/image) or TIFF pyramid.")
    parser.add_argument("-m", "--model", required=True, help="Classifier saved by train_tissue_classifier.py.")
    parser.add_argument("-o", "--output", required=True, help="Where to write the prediction (.h5).")
    parser.add_argument("--pixel_size", type=float, required=True, help="Microns per pixel at level 0.")
    parser.add_argument("--cache_dir", help="Directory to cache the UNI features of the slide.")
    parser.add_argument("--labels", help="Optional HDF5 label pyramid to evaluate the prediction against.")
    parser.add_argument("--label_key", default="labels/{level}", help="Key of the label pyramid level in the HDF5.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    checkpoint = torch.load(args.model, weights_only=False)
    probe = LinearProbe.from_state_dict(checkpoint["probe"])
    image, level, features, tissue_mask = get_slide_features(args.image, args.pixel_size, args.cache_dir)
    features = add_context(features, checkpoint["context"])
    token_prediction, probabilities = predict_tokens(
        probe, features, checkpoint["smooth"], checkpoint["other_threshold"]
    )

    prediction = np.repeat(np.repeat(token_prediction, PATCH_SIZE, axis=0), PATCH_SIZE, axis=1)
    prediction = prediction[:image.shape[0], :image.shape[1]]
    with h5py.File(args.output, "w") as f:
        f.create_dataset("prediction", data=prediction, compression="gzip")
        f.create_dataset("probabilities", data=probabilities.astype(np.float16), compression="gzip")
        f.attrs["pixel_size"] = checkpoint["pixel_size"]
        f.attrs["class_names"] = str(checkpoint["class_names"])
        f.attrs["source"] = str(args.image)
        f.attrs["source_level"] = level
    print("Saved the prediction to", args.output)

    if args.labels is not None:
        labels = load_labels(args.labels, args.label_key, level, image.shape[:2])
        scores = compute_scores(prediction, labels, tissue_mask, checkpoint["class_names"])
        for name, class_scores in scores.items():
            print(f"{name}: " + ", ".join(f"{metric} {score:.4f}" for metric, score in class_scores.items()))
        for metric in ["dice", "precision", "recall", "f1"]:
            print(f"Mean {metric}: {np.mean([class_scores[metric] for class_scores in scores.values()]):.4f}")
        other_share = 100 * (prediction[prediction != 0] == OTHER_TISSUE).mean()
        print(f"Other tissue: {other_share:.1f}% of the predicted tissue (not scored)")


if __name__ == "__main__":
    main()
