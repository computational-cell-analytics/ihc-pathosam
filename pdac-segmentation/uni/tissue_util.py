"""Shared code for the UNI2-h tissue classifier: image loading, features, tissue mask, context and the linear probe.

The settings were chosen in the leave-one-slide-out benchmark described in BENCHMARK_RESULTS.md.
"""
from pathlib import Path

import h5py
import zarr
import numpy as np
import tifffile
from tqdm import tqdm
from skimage.color import rgb2hsv
from skimage.filters import gaussian
from skimage.transform import resize, downscale_local_mean
from skimage.morphology import remove_small_holes, remove_small_objects

import torch
import torch.nn as nn
import torch.nn.functional as F

from apply_uni import get_uni_model_and_transform


# Microns per pixel the features are computed at: pyramid level 3 of the 20x PDAC scans.
TARGET_PIXEL_SIZE = 2.19
TILE_SIZE = 224
PATCH_SIZE = 14
TOKENS_PER_TILE = TILE_SIZE // PATCH_SIZE
BATCH_SIZE = 128
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLASS_NAMES = {0: "background", 1: "normal_pancreas", 2: "tumor"}
IGNORE_LABEL = 255
TISSUE_DOWNSCALE = 4
TISSUE_SATURATION_THRESHOLD = 0.04
CONTEXT = 9
SMOOTH = 3


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_pyramid(path):
    """Return the pyramid levels of a WSI as (array, downsample) pairs, from the finest to the coarsest level.

    Supports the converted zarr layout of this project (s{level}/image) and TIFF pyramids such as SVS or OME-TIFF.
    """
    path = Path(path)
    if path.suffix == ".zarr":
        group = zarr.open_group(path, mode="r")
        arrays = [group[f"s{level}/image"] for level in range(len(group)) if f"s{level}" in group]
    else:
        arrays = [zarr.open(level.aszarr(), mode="r") for level in tifffile.TiffFile(path).series[0].levels]
    return [(array, arrays[0].shape[1] / array.shape[1]) for array in arrays]


def load_image(path, pixel_size):
    """Load a WSI at TARGET_PIXEL_SIZE from the closest finer pyramid level.

    Args:
        path: The zarr or TIFF pyramid.
        pixel_size: Microns per pixel at full resolution.

    Returns:
        The (H, W, 3) uint8 image and the pyramid level it was read from.
    """
    pyramid = get_pyramid(path)
    level = max(i for i, (_, downsample) in enumerate(pyramid) if pixel_size * downsample <= TARGET_PIXEL_SIZE * 1.05)
    array, downsample = pyramid[level]
    image = np.asarray(array[:])[..., :3]
    scale = pixel_size * downsample / TARGET_PIXEL_SIZE
    if abs(scale - 1) > 0.05:
        shape = (round(image.shape[0] * scale), round(image.shape[1] * scale))
        image = resize(image, shape, preserve_range=True, anti_aliasing=True).round().astype(np.uint8)
    return image, level


def load_labels(label_path, label_key, level, shape):
    """Load the annotation of a pyramid level and match it to the image shape (nearest neighbor)."""
    with h5py.File(label_path, "r") as f:
        labels = f[label_key.format(level=level)][:]
    if labels.shape != shape:
        labels = resize(labels, shape, order=0, preserve_range=True, anti_aliasing=False).astype(labels.dtype)
    return labels


def compute_tissue_mask(image):
    """Separate tissue from glass by thresholding the smoothed saturation of a downscaled image.

    The threshold sits just above the saturation of glass, so that pale stroma still counts as tissue.
    """
    small = downscale_local_mean(image, (TISSUE_DOWNSCALE, TISSUE_DOWNSCALE, 1)).astype(np.uint8)
    saturation = gaussian(rgb2hsv(small)[..., 1], sigma=2)
    mask = remove_small_objects(saturation > TISSUE_SATURATION_THRESHOLD, max_size=64)
    return remove_small_holes(mask, max_size=256)


def tissue_at_tokens(tissue_mask, grid_shape):
    """Look up the tissue mask at the token centers."""
    rows = ((np.arange(grid_shape[0]) + 0.5) * PATCH_SIZE / TISSUE_DOWNSCALE).astype(int)
    cols = ((np.arange(grid_shape[1]) + 0.5) * PATCH_SIZE / TISSUE_DOWNSCALE).astype(int)
    mask = tissue_mask[rows.clip(0, tissue_mask.shape[0] - 1)[:, None], cols.clip(0, tissue_mask.shape[1] - 1)[None]]
    # Tokens in the padding beyond the image are glass.
    mask[rows >= tissue_mask.shape[0]] = False
    mask[:, cols >= tissue_mask.shape[1]] = False
    return mask


def grid_shape_of(image_shape):
    n_rows, n_cols = -(-image_shape[0] // TILE_SIZE), -(-image_shape[1] // TILE_SIZE)
    return n_rows * TOKENS_PER_TILE, n_cols * TOKENS_PER_TILE


def compute_token_labels(labels, tissue_tokens):
    """Assign each token the majority label of its pixels.

    The annotations are not exhaustive, so only glass is used as background: tokens that are labeled background
    but lie on tissue are ignored.
    """
    grid_shape = grid_shape_of(labels.shape)
    padded = np.full((grid_shape[0] * PATCH_SIZE, grid_shape[1] * PATCH_SIZE), IGNORE_LABEL, dtype=np.uint8)
    padded[:labels.shape[0], :labels.shape[1]] = labels
    blocks = padded.reshape(grid_shape[0], PATCH_SIZE, grid_shape[1], PATCH_SIZE)
    values = np.array(list(CLASS_NAMES) + [IGNORE_LABEL], dtype=np.uint8)
    counts = np.stack([(blocks == value).sum(axis=(1, 3)) for value in values])
    token_labels = values[counts.argmax(axis=0)]
    token_labels[(token_labels == 0) & tissue_tokens] = IGNORE_LABEL
    return token_labels


def embed_image(image, model=None):
    """Compute the UNI2-h token grid of an image tile by tile, padding the border tiles with white (glass)."""
    device = get_device()
    if model is None:
        model, _ = get_uni_model_and_transform(device)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    grid_shape = grid_shape_of(image.shape)
    n_cols = grid_shape[1] // TOKENS_PER_TILE
    features = np.empty(grid_shape + (model.embed_dim,), dtype=np.float16)

    for row in tqdm(range(grid_shape[0] // TOKENS_PER_TILE), desc="Computing UNI features"):
        strip = np.full((TILE_SIZE, n_cols * TILE_SIZE, 3), 255, dtype=np.uint8)
        part = image[row * TILE_SIZE:(row + 1) * TILE_SIZE]
        strip[:part.shape[0], :part.shape[1]] = part
        tiles = np.ascontiguousarray(strip.reshape(TILE_SIZE, n_cols, TILE_SIZE, 3).transpose(1, 0, 2, 3))
        outputs = []
        with torch.inference_mode(), torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
            for start in range(0, n_cols, BATCH_SIZE):
                batch = torch.from_numpy(tiles[start:start + BATCH_SIZE]).to(device)
                batch = (batch.permute(0, 3, 1, 2).float() / 255 - mean) / std
                tokens = model.forward_features(batch)[:, model.num_prefix_tokens:]
                outputs.append(tokens.reshape(len(batch), TOKENS_PER_TILE, TOKENS_PER_TILE, -1).half().cpu().numpy())
        tokens = np.concatenate(outputs).transpose(1, 0, 2, 3).reshape(TOKENS_PER_TILE, grid_shape[1], -1)
        features[row * TOKENS_PER_TILE:(row + 1) * TOKENS_PER_TILE] = tokens
    return features


def get_slide_features(image_path, pixel_size, cache_dir=None, model=None):
    """Load the image, compute its UNI features and tissue mask, and cache both if a cache directory is given.

    Returns:
        The image, its pyramid level, the (H, W, C) float16 token features and the tissue mask.
    """
    image, level = load_image(image_path, pixel_size)
    cache_path = None if cache_dir is None else Path(cache_dir) / f"{Path(image_path).stem}.h5"
    if cache_path is not None and cache_path.exists():
        with h5py.File(cache_path, "r") as f:
            if tuple(f.attrs["image_shape"]) == image.shape[:2] and f.attrs["pixel_size"] == TARGET_PIXEL_SIZE:
                return image, level, f["features"][:], f["tissue_mask"][:]

    features, tissue_mask = embed_image(image, model), compute_tissue_mask(image)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(cache_path, "w") as f:
            f.create_dataset("features", data=features)
            f.create_dataset("tissue_mask", data=tissue_mask, compression="gzip")
            f.attrs["image_shape"] = image.shape[:2]
            f.attrs["pixel_size"] = TARGET_PIXEL_SIZE
            f.attrs["source"] = str(image_path)
    return image, level, features, tissue_mask


def add_context(features, context=CONTEXT):
    """Append to every token the mean over the context x context tokens around it."""
    device = get_device()
    features = torch.from_numpy(features).to(device)
    n_channels = features.shape[-1]
    output = torch.empty(features.shape[:2] + (2 * n_channels,), dtype=features.dtype, device=device)
    output[..., :n_channels] = features
    # Channel chunks bound the memory and avoid overflowing the int32 indexing of avg_pool2d on large grids.
    for start in range(0, n_channels, 256):
        grid = features[..., start:start + 256].permute(2, 0, 1)[None].float()
        grid = F.avg_pool2d(grid, context, stride=1, padding=context // 2, count_include_pad=False)
        output[..., n_channels + start:n_channels + start + 256] = grid[0].permute(1, 2, 0).half()
    output = output.cpu().numpy()
    torch.cuda.empty_cache()
    return output


def sample_balanced(labels, n_tokens, rng):
    """Sample up to `n_tokens`, splitting evenly over the classes and refilling from the larger ones."""
    classes = np.unique(labels)
    pools = {c: rng.permutation(np.flatnonzero(labels == c)) for c in classes}
    quotas = {c: 0 for c in classes}
    remaining = min(n_tokens, len(labels))
    open_classes = list(classes)
    while remaining > 0:
        share = max(remaining // len(open_classes), 1)
        for c in list(open_classes):
            take = min(share, len(pools[c]) - quotas[c], remaining)
            quotas[c] += take
            remaining -= take
            if quotas[c] == len(pools[c]):
                open_classes.remove(c)
            if remaining == 0:
                break
    return np.concatenate([pools[c][:quotas[c]] for c in classes])


class LinearProbe:
    """Multinomial logistic regression on standardized features, trained on the GPU."""

    def __init__(self, n_steps=2000, batch_size=4096, lr=1e-3, weight_decay=1e-4, seed=0):
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed
        self.device = get_device()

    def fit(self, x, y):
        torch.manual_seed(self.seed)
        self.classes_ = np.unique(y)
        x = torch.from_numpy(x).to(self.device)
        self.mean, self.std = x.mean(dim=0), x.std(dim=0) + 1e-6
        x = (x - self.mean) / self.std
        targets = torch.from_numpy(np.searchsorted(self.classes_, y)).to(self.device)
        self.model = nn.Linear(x.shape[1], len(self.classes_)).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for _ in range(self.n_steps):
            batch = torch.randint(len(x), (min(self.batch_size, len(x)),), device=self.device)
            loss = F.cross_entropy(self.model(x[batch]), targets[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return self

    def predict_proba(self, x):
        with torch.no_grad():
            x = (torch.from_numpy(x).to(self.device) - self.mean) / self.std
            return torch.softmax(self.model(x), dim=1).cpu().numpy()

    def state_dict(self):
        return {
            "weight": self.model.weight.detach().cpu(), "bias": self.model.bias.detach().cpu(),
            "mean": self.mean.cpu(), "std": self.std.cpu(), "classes": self.classes_,
        }

    @classmethod
    def from_state_dict(cls, state):
        probe = cls()
        probe.classes_ = np.asarray(state["classes"])
        probe.mean, probe.std = state["mean"].to(probe.device), state["std"].to(probe.device)
        probe.model = nn.Linear(state["weight"].shape[1], len(probe.classes_)).to(probe.device)
        probe.model.load_state_dict({"weight": state["weight"], "bias": state["bias"]})
        return probe


def predict_tokens(probe, features, smooth=SMOOTH, chunk_size=200_000):
    """Predict the class probabilities of every token and average them over a smooth x smooth window."""
    flat = features.reshape(-1, features.shape[-1])
    probabilities = np.concatenate([
        probe.predict_proba(flat[start:start + chunk_size].astype(np.float32))
        for start in range(0, len(flat), chunk_size)
    ]).reshape(features.shape[:2] + (-1,))
    if smooth > 1:
        grid = torch.from_numpy(probabilities).float().permute(2, 0, 1)[None]
        grid = F.avg_pool2d(grid, smooth, stride=1, padding=smooth // 2, count_include_pad=False)
        probabilities = grid[0].permute(1, 2, 0).numpy()
    return probe.classes_[probabilities.argmax(axis=-1)].astype(np.uint8), probabilities
