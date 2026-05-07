"""Run all pathology segmentation stages on H&E WSIs stored as zarr.

For each .zarr in HE_ZARR_DIR, reads s{scale}/image lazily and runs the TB,
epithelium, and multi-tissue models. Per-stage probabilities are streamed to a
temporary on-disk zarr in SEG_OUTPUT_DIR (deleted after argmax), so RAM stays
bounded to one block at a time. Final labels are written to SEG_OUTPUT_DIR.

Usage:
    python run_tsr_segmentation.py
    python run_tsr_segmentation.py --name TM11 --scale 0 --overwrite

Connected to https://doi.org/10.1371/journal.pone.0301969.
"""

import os
import argparse
from pathlib import Path

import yaml
import zarr
import numpy as np

import torch
import torch.nn as nn

from torch_em.util.prediction import predict_with_halo

import segmentation_models_pytorch as smp

from huggingface_hub import snapshot_download


HF_REPO_ID = "PierpaoloV93/pathology-segmentation-models"
MODELS_DIR = os.path.expanduser("~/.cache/ihc_pathosam/pathology-segmentation-models")
HE_ZARR_DIR = Path("/mnt/vast-nhr/projects/nim00007/data/histopatho/pdac-kfo/data_20260310/converted_zarr_he")
SEG_OUTPUT_DIR = Path(
    "/mnt/vast-nhr/projects/nim00007/data/histopatho/pdac-kfo/data_20260310/converted_zarr_he_tissue_segmentation"
)

STAGES = {
    "tb": {
        "allow_patterns": ["tb/**"],
        "model_dir": "tb",
        "halo": (96, 96),
    },
    "epithelium": {
        "allow_patterns": ["epithelium/**"],
        "model_dir": "epithelium/best_models",
        "halo": (100, 100),
    },
    "multi-tissue": {
        "allow_patterns": ["multi-tissue/**"],
        "model_dir": "multi-tissue/best_models",
        "halo": (100, 100),
    },
}

ARCHITECTURES = {
    "unet": smp.Unet,
    "unet-plus": smp.UnetPlusPlus,
    "manet": smp.MAnet,
    "linknet": smp.Linknet,
    "fpn": smp.FPN,
    "pspnet": smp.PSPNet,
    "deeplabv3": smp.DeepLabV3,
    "deeplabv3+": smp.DeepLabV3Plus,
    "pan": smp.PAN,
}


class SegmentationModel(nn.Module):
    """Single smp model with softmax output and out_channels for predict_with_halo."""

    def __init__(self, base, n_classes):
        super().__init__()
        self.base = base
        self.out_channels = n_classes

    def forward(self, x):
        with torch.amp.autocast("cuda"):
            return torch.softmax(self.base(x), dim=1)


class EnsembleModel(nn.Module):
    """Averages softmax predictions across multiple SegmentationModels."""

    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.out_channels = models[0].out_channels

    def forward(self, x):
        return torch.stack([m(x) for m in self.models]).mean(0)


def _load_model(pt_path, device):
    cfg_path = str(pt_path).split("_epoch")[0].split("_best")[0] + ".yaml"
    with open(cfg_path) as f:
        params = yaml.safe_load(f)
    model_params = params["model"]
    n_classes = len(set(params["sampler"]["training"]["label_map"].values()))
    arch = model_params.get("modelname", "unet")
    base = ARCHITECTURES[arch](encoder_name=model_params["backbone"], classes=n_classes, encoder_weights=None)
    state = torch.load(str(pt_path), map_location=device)
    base.load_state_dict(state)
    base.eval().to(device)
    print(f"  loaded {pt_path.name}  arch={arch}  classes={n_classes}")
    return SegmentationModel(base, n_classes)


def _download_stage(stage, cfg):
    model_dir = Path(MODELS_DIR) / cfg["model_dir"]
    if model_dir.exists() and any(model_dir.glob("*.pt")):
        return
    print(f"[{stage}] Downloading from {HF_REPO_ID} ...")
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="model",
        local_dir=MODELS_DIR,
        allow_patterns=cfg["allow_patterns"],
        ignore_patterns=["*.git*", "*.gitattributes"],
    )


def load_all_models(device):
    models = {}
    for stage, cfg in STAGES.items():
        _download_stage(stage, cfg)
        pt_files = sorted((Path(MODELS_DIR) / cfg["model_dir"]).glob("*.pt"))
        if not pt_files:
            raise FileNotFoundError(f"No .pt files in {Path(MODELS_DIR) / cfg['model_dir']}")
        loaded = [_load_model(p, device) for p in pt_files]
        models[stage] = loaded[0] if len(loaded) == 1 else EnsembleModel(loaded)
        if len(loaded) > 1:
            print(f"  [{stage}] ensemble of {len(loaded)} models")
    return models


class _ArgmaxAccumulator:
    """Receives per-block softmax predictions and writes argmax labels directly.

    predict_with_halo writes non-overlapping blocks, so each pixel is written
    exactly once - no running-max needed. Only a single (H, W) uint8 array is kept.
    """

    def __init__(self, n_classes, h, w):
        self.labels = np.zeros((h, w), dtype=np.uint8)
        self.shape = (n_classes, h, w)
        self.ndim = 3
        self.dtype = np.dtype("float32")

    def __setitem__(self, idx, val):
        _, h_sl, w_sl = idx
        self.labels[h_sl, w_sl] = val.argmax(axis=0).astype(np.uint8)


def run_stage(image, model, stage, gpu_ids):
    halo = STAGES[stage]["halo"]
    block_shape = (512 - 2 * halo[0], 512 - 2 * halo[1])
    n_classes = model.out_channels
    _, h, w = image.shape

    acc = _ArgmaxAccumulator(n_classes, h, w)
    predict_with_halo(
        image, model, gpu_ids=gpu_ids, block_shape=block_shape, halo=halo,
        with_channels=True, preprocess=lambda x: x, output=acc,
    )
    return acc.labels


class _Image:
    """Loads (H, W, C) uint8 zarr into RAM once, serves (C, H, W) float32 blocks on demand.

    Loading as uint8 (~24 GB at s0) avoids the ~96 GB float32 footprint while keeping
    all block reads in RAM so inference is not bottlenecked by network filesystem I/O.
    """

    def __init__(self, arr):
        print("  Loading image to RAM (uint8) ...")
        self._raw = arr[:]
        h, w, c = self._raw.shape
        self.shape = (c, h, w)
        self.ndim = 3
        self.dtype = np.dtype("float32")

    def __getitem__(self, idx):
        c_idx, h_idx, w_idx = idx
        crop = self._raw[h_idx, w_idx, :]
        return crop.transpose(2, 0, 1).astype(np.float32)[c_idx] / 255.0


def process_zarr(zarr_path, models, gpu_ids, scale, overwrite):
    name = Path(zarr_path).name
    print(f"\n{name}")
    src = zarr.open(str(zarr_path), mode="r")
    image = _Image(src[f"s{scale}/image"])
    print(f"  image shape (C, H, W): {image.shape}")

    out_path = SEG_OUTPUT_DIR / name
    dst = zarr.open(str(out_path), mode="a")

    for stage, model in models.items():
        seg_key = f"{stage}_labels"
        if seg_key in dst:
            if not overwrite:
                print(f"  [{stage}] skip (already exists)")
                continue
            del dst[seg_key]

        labels = run_stage(image, model, stage, gpu_ids)
        dst.create_array(seg_key, data=labels, chunks=(512, 512))
        print(f"  [{stage}] saved  labels={labels.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--scale", type=int, default=0)
    parser.add_argument("--name", default=None, help="process only the zarr whose filename contains this string")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    gpu_ids = [args.gpu] if torch.cuda.is_available() else ["cpu"]

    zarr_paths = sorted(HE_ZARR_DIR.glob("*.zarr"))
    if args.name:
        zarr_paths = [p for p in zarr_paths if args.name in p.name]

    models = load_all_models(device)

    print(f"\nProcessing {len(zarr_paths)} file(s) ...")
    for i, zarr_path in enumerate(zarr_paths, 1):
        print(f"[{i}/{len(zarr_paths)}]", end=" ")
        process_zarr(zarr_path, models, gpu_ids, args.scale, args.overwrite)


if __name__ == "__main__":
    main()
