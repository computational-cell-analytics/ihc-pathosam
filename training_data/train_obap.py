import json
import os
from pathlib import Path

import imageio.v3 as imageio
import h5py
import torch

from deap.xdict import xdict
from deap_objects.train import train
from deap_objects.utils.util import LabelRemapper, get_object_conf

DATA_ROOT = "/mnt/vast-nhr/projects/nim00007/data/histopatho/cancerscout/training_data_obap"
CKPT_ROOT = "/mnt/vast-nhr/projects/nim00007/data/histopatho/cancerscout/checkpoints/v2/checkpoints"


def get_class_ids():
    return [1, 2]


def _require_tifs(path, tile_id, im, labels, masks):
    folder, fname = os.path.split(path)
    top_name = os.path.basename(folder)
    fname = Path(fname).stem

    out_folder = os.path.join(DATA_ROOT, top_name)
    tif_im = os.path.join(out_folder, f"{fname}-tile{tile_id}-im.tif")
    tif_labels = os.path.join(out_folder, f"{fname}-tile{tile_id}-labels.tif")
    tif_masks = os.path.join(out_folder, f"{fname}-tile{tile_id}-masks.tif")

    if not os.path.exists(tif_im):
        os.makedirs(out_folder, exist_ok=True)
        imageio.imwrite(tif_im, im, compression="zlib")
        imageio.imwrite(tif_labels, labels, compression="zlib")
        imageio.imwrite(tif_masks, masks, compression="zlib")

    return tif_im, tif_labels, tif_masks


def get_data_paths(splits, split):
    import nifty.tools as nt

    hdf5_paths = splits[split]
    im_paths, label_paths, instance_paths = [], [], []

    tile_shape = (512, 512)
    for path in hdf5_paths:
        with h5py.File(path, "r") as f:
            im = f["image"][:]
            labels = f["labels/silver/semantic"][:]
            masks = f["labels/silver/nuclei"][:]

        tiles = nt.blocking((0, 0), im.shape[:2], tile_shape)
        for tile_id in range(tiles.numberOfBlocks):
            tile = tiles.getBlock(tile_id)
            bb = tuple(slice(beg, end) for beg, end in zip(tile.begin, tile.end))
            this_masks = masks[bb]
            shape = this_masks.shape
            if shape != tile_shape:  # Only keep patches with the full shape
                continue
            max_id = this_masks.max()
            if max_id == 0:  # Skip empty patches
                continue

            tif_im, tif_labels, tif_masks = _require_tifs(path, tile_id, im[bb], labels[bb], this_masks)
            im_paths.append(tif_im)
            label_paths.append(tif_labels)
            instance_paths.append(tif_masks)

    return im_paths, label_paths, instance_paths


def build_run_cfg(split_file, backbone, ckpt_dir):
    with open(split_file) as f:
        splits = json.load(f)

    train_image_paths, train_label_paths, train_instance_paths = get_data_paths(splits, split="train")
    val_image_paths, val_label_paths, val_instance_paths = get_data_paths(splits, split="val")

    # TODO use all objects. Don't conflate object counts in training and per image.
    total_objects = 1000
    max_objects, _ = get_object_conf(total_objects, len(train_image_paths))

    n_classes = len(get_class_ids())
    label_map = {cid: i + 1 for i, cid in enumerate(get_class_ids())}
    label_transform = LabelRemapper(label_map)

    inv_label_map = {v: k for k, v in label_map.items()}

    semseg_dataset = xdict(
        __class__='deap_objects.datasets.datasets.ObjectDataset',
        label_transform=label_transform
    )
    # configurations for training
    default = xdict(
        lr=0.001,
        n_iterations=10000,
        n_workers=4,
        no_wandb=True,
        amp=True,
        bs=6,
        wd=0.001,
        val_interval=10000,  # no validation
        patience=10,
    )

    # model configuration
    # Max Objects is set to 1000 if not given.
    model_cfg = xdict(
        __class__='deap_objects.models.object_attentive_probing.SelfAttReadouts_o',
        max_objects=max_objects,
        dim=32,
        inp_img_size=1024,
        up=(2, 2, 2),
    )

    semseg = xdict(
        task=xdict(
            __class__='deap_objects.tasks.objectclass.ObjectClassificationTask',
            n_classes=n_classes,
            key_name='class',
            inverse_label_map=inv_label_map,
        ),
        dataset=xdict(
            image_paths=train_image_paths,
            semantic_paths=train_label_paths,
            instance_paths=train_instance_paths,
            max_objects=max_objects,
        ) + semseg_dataset,
        dataset_val=xdict(
            image_paths=val_image_paths,
            semantic_paths=val_label_paths,
            instance_paths=val_instance_paths,
        ) + semseg_dataset,
        model=xdict(
            outputs=(('class', n_classes),),
        ) + model_cfg,
    ) + default

    run_name = f"{backbone}_{total_objects}objects"
    semseg_run = xdict(
        name=run_name,
        model=xdict(backbone_name=backbone),
        checkpoint_dir=str(ckpt_dir),
    ) + semseg

    return semseg_run


def run_training(split_file):
    name = Path(split_file).stem
    ckpt_dir = Path(os.path.join(CKPT_ROOT, name))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    backbone = f"pathoSAM-IHC-{name}"
    semseg_run = build_run_cfg(split_file, backbone=backbone, ckpt_dir=ckpt_dir)

    device = torch.device("cuda")
    train(semseg_run, device=device)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("split")
    # parser.add_argument("--slurm", action="store_true")
    args = parser.parse_args()

    run_training(args.split)


if __name__ == "__main__":
    main()
