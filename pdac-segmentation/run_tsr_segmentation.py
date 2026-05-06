"""Run all pathology segmentation stages on H&E WSIs stored as zarr.

For each .zarr in HE_ZARR_DIR, reads s{scale}/image as (H, W, C) uint8 and runs the
TB, epithelium, and multi-tissue models. Predictions are written back under
'predictions/pathology/<stage>' as (n_classes, H, W) float32.

Usage:
    python run_tsr_segmentation.py
    python run_tsr_segmentation.py --scale 1 --overwrite

Connected to https://doi.org/10.1371/journal.pone.0301969.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
import zarr
import segmentation_models_pytorch as smp
from huggingface_hub import snapshot_download

from torch_em.util.prediction import predict_with_halo


HF_REPO_ID = "PierpaoloV93/pathology-segmentation-models"
MODELS_DIR = os.path.expanduser("~/.cache/ihc_pathosam/pathology-segmentation-models")
HE_ZARR_DIR = Path("/mnt/vast-nhr/projects/nim00007/data/histopatho/pdac-kfo/data_20260310/converted_zarr_he")

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


def run_stage(image, model, stage, gpu_ids):
    halo = STAGES[stage]["halo"]
    block_shape = (512 - 2 * halo[0], 512 - 2 * halo[1])
    return predict_with_halo(
        image, model, gpu_ids=gpu_ids, block_shape=block_shape, halo=halo, with_channels=True, preprocess=lambda x: x,
    )


class _LazyImage:
    """Wraps a (H, W, C) zarr array as a lazy (C, H, W) float32 view for predict_with_halo."""

    def __init__(self, arr):
        h, w, c = arr.shape
        self._arr = arr
        self.shape = (c, h, w)
        self.ndim = 3
        self.dtype = np.dtype("float32")

    def __getitem__(self, idx):
        c_idx, h_idx, w_idx = idx
        crop = self._arr[h_idx, w_idx, :]
        return crop.transpose(2, 0, 1).astype(np.float32)[c_idx] / 255.0


def process_zarr(zarr_path, models, gpu_ids, scale, overwrite):
    print(f"\n{Path(zarr_path).name}")
    z = zarr.open(str(zarr_path), mode="a")
    image = _LazyImage(z[f"s{scale}/image"])
    print(f"  image shape (C, H, W): {image.shape}")

    for stage, model in models.items():
        seg_key = f"predictions/pathology/{stage}_labels"
        if seg_key in z:
            if not overwrite:
                print(f"  [{stage}] skip (already exists)")
                continue
            del z[seg_key]

        output = run_stage(image, model, stage, gpu_ids)
        labels = np.argmax(output, axis=0).astype(np.uint8)

        z.create_array(seg_key, data=labels, chunks=(512, 512))
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
