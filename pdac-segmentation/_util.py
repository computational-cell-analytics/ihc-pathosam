import json
import random
from pathlib import Path

import h5py
import numpy as np

import torch_em
from micro_sam.training import default_sam_loader

from patho_sam.training.util import histopathology_identity


ROOT = "/mnt/vast-nhr/projects/cidas/cca/data/pdac_umg_histopatho/extracted_data"
SPLIT_JSON = Path(__file__).parent / "splits" / "split.json"


def get_split(root=ROOT, split_json=None, val_fraction=0.4, seed=42):
    """Load train/val split from JSON if available, otherwise create and save it.

    Args:
        root: Root directory to search for h5 files recursively.
        split_json: Path to the JSON file storing the split. If None, splits are not persisted.
        val_fraction: Fraction of files to use for validation.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_paths, val_paths) as lists of strings.
    """
    if split_json is not None and Path(split_json).exists():
        with open(split_json) as f:
            split = json.load(f)
        print(f"Loaded split from {split_json}: {len(split['train'])} train, {len(split['val'])} val")
        return split["train"], split["val"]

    all_paths = sorted(str(p) for p in Path(root).rglob("*.h5"))
    if not all_paths:
        raise RuntimeError(f"No h5 files found under {root}")

    rng = random.Random(seed)
    shuffled = all_paths[:]
    rng.shuffle(shuffled)

    n_val = max(1, int(len(shuffled) * val_fraction))
    val_paths = shuffled[:n_val]
    train_paths = shuffled[n_val:]

    split = {"train": train_paths, "val": val_paths}

    if split_json is not None:
        Path(split_json).parent.mkdir(parents=True, exist_ok=True)
        with open(split_json, "w") as f:
            json.dump(split, f, indent=2)
        print(f"Saved split to {split_json}: {len(train_paths)} train, {len(val_paths)} val")
    else:
        print(f"Created split (not persisted): {len(train_paths)} train, {len(val_paths)} val")

    return train_paths, val_paths


def _load_data(paths, label_key):
    """Load raw images and labels from h5 files into memory.

    Args:
        paths: List of h5 file paths.
        label_key: HDF5 dataset key for the label array.

    Returns:
        Tuple of (raw_arrays, label_arrays) as lists of numpy arrays.
    """
    raw_arrays, label_arrays = [], []
    for path in paths:
        with h5py.File(path, "r") as f:
            raw_arrays.append(f["raw"][:].transpose(2, 0, 1))  # (H, W, C) -> (C, H, W)
            label_arrays.append(f[label_key][:])
    return raw_arrays, label_arrays


def get_loaders(label_key, split_json=SPLIT_JSON, batch_size=4, n_train=800, n_val=80):
    """Create train and validation data loaders.

    Args:
        split_json: Path to JSON file with train/val split. Created if missing.
        label_key: HDF5 dataset key for the label array (e.g. 'labels/nucleus/semantic').
        batch_size: Batch size for both loaders.
        n_train: Number of training samples per epoch.
        n_val: Number of validation samples per epoch.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    train_paths, val_paths = get_split(split_json=split_json)

    train_data, train_labels = _load_data(train_paths, label_key)
    val_data, val_labels = _load_data(val_paths, label_key)

    train_loader = torch_em.default_segmentation_loader(
        raw_paths=train_data, label_paths=train_labels,
        raw_key=None, label_key=None,
        patch_shape=(512, 512), with_channels=True,
        batch_size=batch_size, n_samples=n_train,
        raw_transform=histopathology_identity,
    )
    val_loader = torch_em.default_segmentation_loader(
        raw_paths=val_data, label_paths=val_labels,
        raw_key=None, label_key=None,
        patch_shape=(512, 512), with_channels=True,
        batch_size=batch_size, n_samples=n_val,
        raw_transform=histopathology_identity,
    )

    return train_loader, val_loader


def get_class_weights(label_key, split_json=SPLIT_JSON, num_classes=None):
    """Compute balanced class weights from pixel frequencies in the training split."""
    train_paths, _ = get_split(split_json=split_json)

    counts = None if num_classes is None else np.zeros(num_classes, dtype=np.int64)
    for path in train_paths:
        with h5py.File(path, "r") as f:
            labels = f[label_key][:]

        bincount = np.bincount(labels.ravel(), minlength=0 if counts is None else len(counts))
        if counts is None:
            counts = bincount.astype(np.int64, copy=False)
        else:
            counts += bincount[:len(counts)]

    if counts is None or np.any(counts == 0):
        raise ValueError(f"Cannot compute class weights from counts: {counts}")

    total = counts.sum()
    return (total / (len(counts) * counts)).astype(np.float32).tolist()


def get_instance_loaders(label_key, split_json=SPLIT_JSON, batch_size=4, n_train=800, n_val=80):
    """Create train and validation data loaders for instance segmentation.

    Args:
        label_key: HDF5 dataset key for instance labels (e.g. 'labels/nucleus/instances').
        split_json: Path to JSON file with train/val split. Created if missing.
        batch_size: Batch size for both loaders.
        n_train: Number of training samples per epoch.
        n_val: Number of validation samples per epoch.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    train_paths, val_paths = get_split(split_json=split_json)

    train_data, train_labels = _load_data(train_paths, label_key)
    val_data, val_labels = _load_data(val_paths, label_key)

    train_loader = default_sam_loader(
        raw_paths=train_data, label_paths=train_labels,
        raw_key=None, label_key=None,
        patch_shape=(512, 512), with_channels=True,
        with_segmentation_decoder=True,
        train_instance_segmentation_only=True,
        batch_size=batch_size, n_samples=n_train,
    )
    val_loader = default_sam_loader(
        raw_paths=val_data, label_paths=val_labels,
        raw_key=None, label_key=None,
        patch_shape=(512, 512), with_channels=True,
        with_segmentation_decoder=True,
        train_instance_segmentation_only=True,
        batch_size=batch_size, n_samples=n_val,
    )

    return train_loader, val_loader
