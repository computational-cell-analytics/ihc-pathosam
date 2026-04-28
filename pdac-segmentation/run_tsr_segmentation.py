"""Run all pathology segmentation stages on h5 split files.

For each h5 file in the split, reads raw (H, W, C) uint8 and runs the
TB, epithelium, and multi-tissue models. Predictions are written back under
'predictions/pathology/<stage>' as (n_classes, H, W) float32.

Usage:
    python run_pathology_segmentation.py
    python run_pathology_segmentation.py --split train --overwrite

Connected to https://doi.org/10.1371/journal.pone.0301969.
"""

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import yaml
import segmentation_models_pytorch as smp
from huggingface_hub import snapshot_download

from torch_em.util.prediction import predict_with_halo


HF_REPO_ID = "PierpaoloV93/pathology-segmentation-models"
SPLIT_JSON = Path(__file__).parent / "splits" / "split.json"
MODELS_DIR = os.path.expanduser("~/.cache/ihc_pathosam/pathology-segmentation-models")

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


def process_file(h5_path, models, gpu_ids, overwrite):
    print(f"\n{Path(h5_path).name}")
    with h5py.File(h5_path, "r") as f:
        raw = f["raw"][:]
    image = raw[..., :3].transpose(2, 0, 1).astype(np.float32) / 255.0

    for stage, model in models.items():
        out_key = f"predictions/pathology/{stage}"
        with h5py.File(h5_path, "a") as f:
            if out_key in f:
                if not overwrite:
                    print(f"  [{stage}] skip (already exists)")
                    continue
                del f[out_key]

        output = run_stage(image, model, stage, gpu_ids)

        with h5py.File(h5_path, "a") as f:
            f.create_dataset(out_key, data=output, compression="gzip")

        print(f"  [{stage}] saved  shape={output.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val", "all"], default="all")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    gpu_ids = [args.gpu] if torch.cuda.is_available() else ["cpu"]

    with open(SPLIT_JSON) as f:
        split_data = json.load(f)
    if args.split == "train":
        paths = split_data["train"]
    elif args.split == "val":
        paths = split_data["val"]
    else:
        paths = split_data["train"] + split_data["val"]

    models = load_all_models(device)

    print(f"\nProcessing {len(paths)} file(s) ...")
    for i, h5_path in enumerate(paths, 1):
        print(f"[{i}/{len(paths)}]", end=" ")
        process_file(h5_path, models, gpu_ids, args.overwrite)


if __name__ == "__main__":
    main()
