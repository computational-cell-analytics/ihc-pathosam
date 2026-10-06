"""Train the UNI2-h tissue classifier (background, normal pancreas, tumor) on annotated H&E WSIs.

Each slide is embedded with UNI2-h at TARGET_PIXEL_SIZE, every token gets the mean of its 9 x 9 neighborhood
appended, and a linear probe is trained on a class-balanced sample of the annotated tokens.
"""
import os
import argparse

import numpy as np

import torch

from apply_uni import get_uni_model_and_transform
from tissue_util import (
    CONTEXT, SMOOTH, CLASS_NAMES, IGNORE_LABEL, TARGET_PIXEL_SIZE, LinearProbe,
    add_context, get_device, load_labels, sample_balanced, tissue_at_tokens, get_slide_features, compute_token_labels,
)


def collect_token_labels(args, model):
    """Compute the token labels of every slide; the UNI features are computed (and cached) on the way."""
    pixel_sizes = args.pixel_size * len(args.images) if len(args.pixel_size) == 1 else args.pixel_size
    slide_labels = []
    for image_path, label_path, pixel_size in zip(args.images, args.labels, pixel_sizes):
        image, level, features, tissue_mask = get_slide_features(image_path, pixel_size, args.cache_dir, model)
        labels = load_labels(label_path, args.label_key, level, image.shape[:2])
        token_labels = compute_token_labels(labels, tissue_at_tokens(tissue_mask, features.shape[:2]))
        slide_labels.append(token_labels.ravel())
        print(image_path, "labeled tokens per class:", dict(zip(*np.unique(token_labels, return_counts=True))))
    return slide_labels, pixel_sizes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", nargs="+", required=True, help="WSIs as zarr (s{level}/image) or TIFF pyramids.")
    parser.add_argument("--labels", nargs="+", required=True, help="HDF5 label pyramids, one per image.")
    parser.add_argument("--pixel_size", nargs="+", type=float, required=True, help="Microns per pixel at level 0.")
    parser.add_argument("--label_key", default="labels/{level}", help="Key of the label pyramid level in the HDF5.")
    parser.add_argument("-o", "--output", required=True, help="Where to save the classifier (.pt).")
    parser.add_argument("--cache_dir", help="Where to cache the UNI features. Defaults to <output>_features.")
    parser.add_argument("--n_tokens", type=int, default=10000, help="Number of class-balanced training tokens.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    assert len(args.images) == len(args.labels), "Every image needs a label file."
    assert len(args.pixel_size) in (1, len(args.images)), "Pass one pixel size, or one per image."
    if args.cache_dir is None:
        args.cache_dir = f"{os.path.splitext(args.output)[0]}_features"

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    model, _ = get_uni_model_and_transform(get_device())
    slide_labels, pixel_sizes = collect_token_labels(args, model)

    slide_ids = np.concatenate([np.full(len(labels), i) for i, labels in enumerate(slide_labels)])
    token_ids = np.concatenate([np.arange(len(labels)) for labels in slide_labels])
    all_labels = np.concatenate(slide_labels)
    labeled = np.flatnonzero(all_labels != IGNORE_LABEL)
    sampled = labeled[sample_balanced(all_labels[labeled], args.n_tokens, np.random.default_rng(args.seed))]

    # Load the context features of the sampled tokens only, one slide at a time.
    x = None
    for i, (image_path, pixel_size) in enumerate(zip(args.images, pixel_sizes)):
        selected = slide_ids[sampled] == i
        if not selected.any():
            continue
        features = get_slide_features(image_path, pixel_size, args.cache_dir, model)[2]
        features = add_context(features).reshape(-1, 2 * features.shape[-1])
        if x is None:
            x = np.empty((len(sampled), features.shape[-1]), dtype=np.float32)
        x[selected] = features[token_ids[sampled][selected]]
    y = all_labels[sampled]
    print("Training tokens per class:", dict(zip(*np.unique(y, return_counts=True))))

    probe = LinearProbe(seed=args.seed).fit(x, y)
    torch.save({
        "probe": probe.state_dict(), "context": CONTEXT, "smooth": SMOOTH, "pixel_size": TARGET_PIXEL_SIZE,
        "class_names": CLASS_NAMES, "training_images": [str(path) for path in args.images],
    }, args.output)
    print("Saved the classifier to", args.output)


if __name__ == "__main__":
    main()
