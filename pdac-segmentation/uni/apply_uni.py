import os
import json
import argparse
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

import torch
from torchvision import transforms

import timm
from nifty.tools import blocking


ROOT = Path(__file__).resolve().parent
DEFAULT_SPLIT_JSON = ROOT.parent / "splits" / "split.json"
DEFAULT_OUTPUT_PATH = ROOT / "pdac_val_uni"
PCA_COMPONENTS = 3
TILE_SIZE = 224
STRIDE = 224
BATCH_SIZE = 16


def get_model_path(model_folder="/mnt/vast-nhr/projects/cidas/cca/models/univ2"):
    filename = "pytorch_model.bin"
    model_path = os.path.join(model_folder, filename)
    if os.path.exists(model_path):
        return model_path

    from huggingface_hub import login, hf_hub_download
    login()  # login with your User Access Token, found at https://huggingface.co/settings/tokens
    os.makedirs(model_folder, exist_ok=True)  # create directory if it does not exist
    hf_hub_download("MahmoodLab/UNI2-h", filename=filename, local_dir=model_folder, force_download=True)
    return model_path


def get_uni_model_and_transform(device):
    model_path = get_model_path()
    model = timm.create_model(
        pretrained=False,
        model_name="vit_giant_patch14_224",
        img_size=224,
        patch_size=14,
        depth=24,
        num_heads=24,
        init_values=1e-5,
        embed_dim=1536,
        mlp_ratio=2.66667 * 2,
        num_classes=0,
        no_embed_class=True,
        mlp_layer=timm.layers.SwiGLUPacked,
        act_layer=torch.nn.SiLU,
        reg_tokens=8,
        dynamic_img_size=True,
    )
    model.to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=True)
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    model.eval()
    return model, transform


def _load_split(split):
    with open(DEFAULT_SPLIT_JSON) as f:
        split_data = json.load(f)
    if split not in split_data:
        raise KeyError(f"Split '{split}' not found in {DEFAULT_SPLIT_JSON}. Available keys: {sorted(split_data)}")
    return split_data[split]


def _iter_tiles(image):
    height, width = image.shape[:2]
    tiling = blocking([0, 0], [height, width], [STRIDE, STRIDE])

    for block_id in range(tiling.numberOfBlocks):
        block = tiling.getBlock(block_id)
        y0, x0 = block.begin
        y1, x1 = block.end
        tile = image[y0:y1, x0:x1]
        pad_h = TILE_SIZE - tile.shape[0]
        pad_w = TILE_SIZE - tile.shape[1]
        if pad_h > 0 or pad_w > 0:
            tile = np.pad(tile, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
        yield tile, (y0, x0)


def _extract_tile_tokens(model, transform, image, device):
    patch_h, patch_w = model.patch_embed.patch_size
    grid_h = model.patch_embed.img_size[0] // patch_h
    grid_w = model.patch_embed.img_size[1] // patch_w
    num_prefix_tokens = getattr(model, "num_prefix_tokens", 0)

    tile_tensors = []
    coords = []
    token_batches = []

    with torch.inference_mode():
        for tile, coord in _iter_tiles(image):
            tile_tensors.append(transform(Image.fromarray(tile)))
            coords.append(coord)
            if len(tile_tensors) == BATCH_SIZE:
                batch = torch.stack(tile_tensors, dim=0).to(device)
                patch_tokens = model.forward_features(batch)[:, num_prefix_tokens:, :]
                token_batches.append(patch_tokens.reshape(batch.shape[0], grid_h, grid_w, -1).cpu().numpy())
                tile_tensors.clear()
        if tile_tensors:
            batch = torch.stack(tile_tensors, dim=0).to(device)
            patch_tokens = model.forward_features(batch)[:, num_prefix_tokens:, :]
            token_batches.append(patch_tokens.reshape(batch.shape[0], grid_h, grid_w, -1).cpu().numpy())

    if not token_batches:
        raise RuntimeError("No token features were extracted from the input image.")

    return np.concatenate(token_batches, axis=0).astype(np.float32), np.asarray(coords, dtype=np.int32)


def _infer_grid_shape(coords):
    ys = np.unique(coords[:, 0])
    xs = np.unique(coords[:, 1])
    if len(ys) * len(xs) != len(coords):
        raise ValueError(
            f"Coordinates do not form a dense rectangular grid: {len(ys)} x {len(xs)} != {len(coords)}"
        )
    return ys, xs


def _stitch_tile_features(tile_features, coords):
    ys, xs = _infer_grid_shape(coords)
    tile_grid_h, tile_grid_w = len(ys), len(xs)
    patch_grid_h, patch_grid_w, channels = tile_features.shape[1:]
    y_to_idx = {int(y): idx for idx, y in enumerate(ys.tolist())}
    x_to_idx = {int(x): idx for idx, x in enumerate(xs.tolist())}

    stitched = np.zeros((tile_grid_h * patch_grid_h, tile_grid_w * patch_grid_w, channels), dtype=np.float32)
    for feature_tile, (y, x) in zip(tile_features, coords):
        y0 = y_to_idx[int(y)] * patch_grid_h
        x0 = x_to_idx[int(x)] * patch_grid_w
        stitched[y0:y0 + patch_grid_h, x0:x0 + patch_grid_w] = feature_tile
    return stitched


def _compute_tilewise_pca(tile_tokens):
    _, patch_grid_h, patch_grid_w, feature_dim = tile_tokens.shape
    tile_pca = np.empty(tile_tokens.shape[:3] + (PCA_COMPONENTS,), dtype=np.float32)

    for tile_idx, tokens in enumerate(tile_tokens):
        flat_tokens = tokens.reshape(-1, feature_dim).astype(np.float32, copy=False)
        centered = flat_tokens - flat_tokens.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        projected = centered @ vt[:PCA_COMPONENTS].T
        tile_pca[tile_idx] = projected.reshape(patch_grid_h, patch_grid_w, PCA_COMPONENTS)

    return tile_pca


def _normalize_channels(array):
    mins = array.min(axis=(0, 1), keepdims=True)
    maxs = array.max(axis=(0, 1), keepdims=True)
    denom = np.where((maxs - mins) < 1e-8, 1.0, maxs - mins)
    return ((array - mins) / denom).astype(np.float32)


def _upsample_rgb_grid(grid, image_shape):
    image_h, image_w = int(image_shape[0]), int(image_shape[1])
    rgb = (255.0 * _normalize_channels(grid)).clip(0, 255).astype(np.uint8)
    return np.asarray(Image.fromarray(rgb, mode="RGB").resize((image_w, image_h), resample=Image.BILINEAR))


def _resolve_output_path(input_path):
    DEFAULT_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    return DEFAULT_OUTPUT_PATH / f"{Path(input_path).stem}_uni_features.h5"


def _save_embeddings(
    output_path, *, input_path, raw_image, instance_segmentation, semantic_segmentation, upsampled_pca
):
    with h5py.File(output_path, "w") as f:
        f.create_dataset("raw", data=raw_image, compression="gzip")
        f.create_dataset("labels/nucleus/instances", data=instance_segmentation, compression="gzip")
        f.create_dataset("labels/nucleus/semantic", data=semantic_segmentation, compression="gzip")
        f.create_dataset("pca", data=upsampled_pca, compression="gzip")
        f.attrs["input_path"] = str(input_path)


def apply_uni(model, transform, input_path, device):
    output_path = _resolve_output_path(input_path)
    if output_path.exists():
        print(f"Skipping {input_path}: {output_path} already exists")
        return output_path

    with h5py.File(input_path, "r") as f:
        image = f["raw"][:]
        instance_segmentation = f["labels/nucleus/instances"][:]
        semantic_segmentation = f["labels/nucleus/semantic"][:]

    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected RGB image in dataset 'raw', got shape {image.shape} from {input_path}")

    tile_tokens, coords = _extract_tile_tokens(model, transform, image, device)
    stitched_pca = _stitch_tile_features(_compute_tilewise_pca(tile_tokens), coords)
    upsampled_pca = _upsample_rgb_grid(stitched_pca, image.shape)

    _save_embeddings(
        output_path,
        input_path=input_path,
        raw_image=image,
        instance_segmentation=instance_segmentation,
        semantic_segmentation=semantic_segmentation,
        upsampled_pca=upsampled_pca,
    )

    print(f"Saved embeddings for {input_path} to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, transform = get_uni_model_and_transform(device=device)
    for input_path in _load_split(args.split):
        apply_uni(model=model, transform=transform, input_path=input_path, device=device)


if __name__ == "__main__":
    main()
