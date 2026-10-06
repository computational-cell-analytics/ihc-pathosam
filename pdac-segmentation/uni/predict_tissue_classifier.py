"""Predict background, normal pancreas and tumor on an H&E WSI with a trained UNI2-h tissue classifier.

The prediction is written to an HDF5 file at the classifier's pixel size. If labels are given, the Dice per class
is printed as well; unannotated tissue is not scored, because the annotations are not exhaustive.
"""
import os
import argparse

import h5py
import numpy as np

import torch

from tissue_util import (
    PATCH_SIZE, IGNORE_LABEL, LinearProbe, add_context, load_labels, predict_tokens, get_slide_features,
)


def compute_dice(prediction, labels, tissue_mask, class_names):
    """Dice per class on glass and annotated tissue. A class that is not annotated on the slide is not scored."""
    rows = (np.arange(labels.shape[0]) * tissue_mask.shape[0] // labels.shape[0])
    cols = (np.arange(labels.shape[1]) * tissue_mask.shape[1] // labels.shape[1])
    tissue = tissue_mask[rows[:, None], cols[None, :]]
    valid = (labels != IGNORE_LABEL) & ~((labels == 0) & tissue)
    prediction, labels = prediction[valid], labels[valid]
    scores = {}
    for class_id, name in class_names.items():
        pred_mask, label_mask = prediction == class_id, labels == class_id
        if label_mask.any():
            scores[name] = 2 * (pred_mask & label_mask).sum() / (pred_mask.sum() + label_mask.sum())
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
    token_prediction, probabilities = predict_tokens(probe, features, checkpoint["smooth"])

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
        scores = compute_dice(prediction, labels, tissue_mask, checkpoint["class_names"])
        print("Dice per class:", {name: round(float(score), 4) for name, score in scores.items()})
        print("Mean Dice:", round(float(np.mean(list(scores.values()))), 4))


if __name__ == "__main__":
    main()
