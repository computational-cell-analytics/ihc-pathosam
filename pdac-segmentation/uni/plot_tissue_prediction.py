"""Plot a tissue classifier prediction next to the slide and its annotation.

Panels: raw slide, ground truth, whole slide prediction, and per annotated class the true positives, false positives
and false negatives on glass and annotated tissue (the pixels that are scored by predict_tissue_classifier.py).
"""
import argparse

import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from skimage.transform import downscale_local_mean

from tissue_util import IGNORE_LABEL, TISSUE_DOWNSCALE, load_labels, load_image, compute_tissue_mask

matplotlib.use("Agg")

DOWNSCALE = 4
WHITE, GREY = (255, 255, 255), (217, 216, 212)
CLASS_COLORS = {1: (42, 120, 214), 2: (74, 58, 167), 3: (138, 109, 75)}
TP, FP, FN = (27, 175, 122), (208, 59, 59), (250, 178, 25)
# Same panel order as the held-out figures: tumor first.
CLASSES = {2: "tumor", 1: "normal pancreas"}


def to_rgb(label_map, tissue, colors):
    rgb = np.full(label_map.shape + (3,), WHITE, dtype=np.uint8)
    rgb[tissue] = GREY
    for value, color in colors.items():
        rgb[label_map == value] = color
    return rgb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--image", required=True, help="The predicted WSI.")
    parser.add_argument("-p", "--prediction", required=True, help="Prediction written by predict_tissue_classifier.py.")
    parser.add_argument("-l", "--labels", required=True, help="HDF5 label pyramid of the slide.")
    parser.add_argument("-o", "--output", required=True, help="Where to save the figure (.png).")
    parser.add_argument("--pixel_size", type=float, required=True, help="Microns per pixel at level 0.")
    parser.add_argument("--title", default="", help="Text appended to the title of the raw slide panel.")
    parser.add_argument("--label_key", default="labels/{level}", help="Key of the label pyramid level in the HDF5.")
    args = parser.parse_args()

    with h5py.File(args.prediction, "r") as f:
        prediction, level = f["prediction"][:], int(f.attrs["source_level"])
    image, _ = load_image(args.image, args.pixel_size)
    tissue_mask = compute_tissue_mask(image)
    labels = load_labels(args.labels, args.label_key, level, prediction.shape)

    rows = (np.arange(prediction.shape[0]) // TISSUE_DOWNSCALE).clip(0, tissue_mask.shape[0] - 1)
    cols = (np.arange(prediction.shape[1]) // TISSUE_DOWNSCALE).clip(0, tissue_mask.shape[1] - 1)
    tissue = tissue_mask[np.ix_(rows, cols)]
    valid = (labels != IGNORE_LABEL) & ~((labels == 0) & tissue)
    dice = {}
    for class_id in CLASSES:
        pred_mask, label_mask = (prediction == class_id) & valid, (labels == class_id) & valid
        if label_mask.any():
            dice[class_id] = 2 * (pred_mask & label_mask).sum() / (pred_mask.sum() + label_mask.sum())

    # The panels are shown at the resolution of the prediction, downscaled by nearest neighbor.
    index = np.ix_(np.arange(0, prediction.shape[0], DOWNSCALE), np.arange(0, prediction.shape[1], DOWNSCALE))
    prediction, labels, tissue, valid = prediction[index], labels[index], tissue[index], valid[index]
    # The raw slide is averaged instead, to avoid aliasing.
    raw = downscale_local_mean(image, (DOWNSCALE, DOWNSCALE, 1)).astype(np.uint8)
    raw = raw[:prediction.shape[0], :prediction.shape[1]]

    panels = [
        (raw, args.title),
        (to_rgb(np.where(valid, labels, 0), tissue, CLASS_COLORS), "ground truth"),
        (to_rgb(prediction, prediction != 0, CLASS_COLORS), "prediction (whole slide)"),
    ]
    for class_id, name in CLASSES.items():
        pred_mask, label_mask = (prediction == class_id) & valid, (labels == class_id) & valid
        rgb = to_rgb(np.zeros_like(labels), tissue, {})
        if class_id in dice:
            rgb[pred_mask & label_mask], rgb[pred_mask & ~label_mask], rgb[~pred_mask & label_mask] = TP, FP, FN
            title = f"{name}: errors (Dice {dice[class_id]:.3f})"
        else:
            title = f"{name}: errors (not annotated on this slide)"
        panels.append((rgb, title))

    fig, axes = plt.subplots(1, len(panels), figsize=(18, 4.8))
    for ax, (panel, title) in zip(axes, panels):
        ax.imshow(panel, interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]), ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#cccccc")

    def patch(color, label):
        return Patch(facecolor=np.array(color) / 255, edgecolor="#999999", label=label)

    class_legend = [
        patch(CLASS_COLORS[1], "normal pancreas"), patch(CLASS_COLORS[2], "tumor"),
        patch(CLASS_COLORS[3], "other tissue (prediction)"), patch(WHITE, "background (glass)"),
        patch(GREY, "unannotated tissue (not scored)"),
    ]
    error_legend = [patch(TP, "true positive"), patch(FP, "false positive"), patch(FN, "false negative")]
    fig.legend(handles=class_legend, loc="lower center", bbox_to_anchor=(0.36, 0.02), ncol=5, frameon=False,
               title="classes (ground truth and prediction)", fontsize=9, title_fontsize=9)
    fig.legend(handles=error_legend, loc="lower center", bbox_to_anchor=(0.8, 0.02), ncol=3, frameon=False,
               title="errors per class", fontsize=9, title_fontsize=9)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.18, wspace=0.04)
    fig.savefig(args.output, dpi=100)
    print("Saved the figure to", args.output)


if __name__ == "__main__":
    main()
