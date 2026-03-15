import os
import shutil
from glob import glob
from pathlib import Path
from time import time

import torch
from deap.xdict import xdict
from deap_objects.train import train, evaluate
from deap_objects.utils.util import LabelRemapper, get_object_conf


def get_data_paths(out_dir, split):
    image_paths = sorted(glob(os.path.join(out_dir, split, "images", "*.tif")))
    label_paths = sorted(glob(os.path.join(out_dir, split, "labels", "*.tif")))
    instance_paths = sorted(glob(os.path.join(out_dir, split, "masks", "*.tif")))
    assert len(image_paths) > 0
    return image_paths, label_paths, instance_paths


# This is a list of semantic classes that are actually used.
# Here we use both foreground classes which are 1 and 2.
def get_class_ids():
    return [1, 2]


def build_run_cfg(backbone, ckpt_dir, out_dir):

    train_image_paths, train_label_paths, train_instance_paths = get_data_paths(out_dir, split="train")
    val_image_paths, val_label_paths, val_instance_paths = get_data_paths(out_dir, split="val")
    test_image_paths, test_label_paths, test_instance_paths = get_data_paths(out_dir, split="test")

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
        dataset_test=xdict(
            image_paths=test_image_paths,
            semantic_paths=test_label_paths,
            instance_paths=test_instance_paths,
            for_inference=True,
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
        save_path=str(out_dir / "object_classification.csv"),
    ) + semseg

    return semseg_run


def training_and_evaluation(out_dir):
    checkpoint_dir = "./checkpoints"
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    backbone = "pathoSAM-IHC"
    semseg_run = build_run_cfg(backbone=backbone, ckpt_dir=ckpt_dir, out_dir=out_dir)

    weights = semseg_run["name"] + "-weights.pth"
    weights_path = ckpt_dir / weights

    device = torch.device("cuda")
    if not weights_path.exists():
        print(f"Checkpoint not found: {weights_path}")
        print("Starting training...")

        start = time()
        train(semseg_run, device=device)
        print(f"Training finished in {time() - start:.2f}s")

        shutil.move(checkpoint_dir / "last_weights.pth", weights_path)

        if not weights_path.exists():
            raise RuntimeError(
                f"Training finished but checkpoint was not found at {weights_path}"
            )
    else:
        print(f"Using existing checkpoint: {weights_path}")

    out = evaluate(
        semseg_run,
        weights=str(weights_path),
        device=device,
        predict_semseg=True,
    )

    time_file = out_dir / "time.csv"
    with open(time_file, "w") as f:
        f.write("inference_time\n")
        f.write(f"{out['inference_time']}\n")

    print("Done.")
    print(f"Checkpoint: {weights_path}")
    print(f"Results: {out_dir}")


def data_preparation(out_dir):
    import h5py
    import imageio
    import numpy as np
    from nifty.tools import blocking

    tile_shape = (512, 512)
    halo = [0, 0]
    full_shape = tuple(ts + ha for ts, ha in zip(tile_shape, halo))

    def _save_split(split, images, labels, instances):
        assert len(images) == len(labels)
        assert len(images) == len(instances)
        root = os.path.join(out_dir, split)
        image_folder, label_folder, mask_folder = os.path.join(root, "images"), os.path.join(root, "labels"), os.path.join(root, "masks")  # noqa
        os.makedirs(image_folder, exist_ok=True)
        os.makedirs(label_folder, exist_ok=True)
        os.makedirs(mask_folder, exist_ok=True)
        for i, (im, lab, masks) in enumerate(zip(images, labels, instances)):
            fname = f"tile-{i}.tif"
            imageio.imwrite(os.path.join(image_folder, fname), im, compression="zlib")
            imageio.imwrite(os.path.join(label_folder, fname), lab, compression="zlib")
            imageio.imwrite(os.path.join(mask_folder, fname), masks, compression="zlib")

    def _create_split(split, images, labels, instances):
        assert len(images) == len(labels)
        assert len(images) == len(instances)
        tiled_images, tiled_labels, tiled_instances = [], [], []
        for im, lab, masks in zip(images, labels, instances):
            tiles = blocking((0, 0), im.shape[:2], tile_shape)
            for tile_id in range(tiles.numberOfBlocks):
                tile = tiles.getBlockWithHalo(tile_id, halo).outerBlock
                bb = tuple(slice(beg, end) for beg, end in zip(tile.begin, tile.end))
                this_masks = masks[bb]
                shape = this_masks.shape
                if shape != full_shape:  # Only keep patches with the full shape
                    continue
                max_id = this_masks.max()
                if max_id == 0:  # Skip empty patches
                    continue
                tiled_images.append(im[bb])
                tiled_labels.append(lab[bb])
                tiled_instances.append(this_masks)
        print("Create", split, "split with", len(tiled_images), "images")
        _save_split(split, tiled_images, tiled_labels, tiled_instances)

    image_key = "image"
    label_key = "labels/silver/semantic"
    instance_key = "labels/silver/nuclei"

    train_paths = [
        "data/annotated_data/cd8/RE-000072-1_1_4-CD8_CRC_CancerScout-2024-07-12T11-04-16_tile00.h5",
        "data/annotated_data/cd8/RE-000072-1_1_4-CD8_CRC_CancerScout-2024-07-12T11-04-16_tile01.h5",
    ]
    images, labels, instances = [], [], []
    for path in train_paths:
        with h5py.File(path, "r") as f:
            images.append(f[image_key][:])
            labels.append(f[label_key][:])
            instances.append(f[instance_key][:])
    _create_split("train", images, labels, instances)

    path = "data/annotated_data/cd8/RE-000072-1_1_4-CD8_CRC_CancerScout-2024-07-12T11-04-16_tile02.h5"
    val_bb, test_bb = np.s_[:4000], np.s_[4000:]
    for split, bb in zip(["val", "test"], [val_bb, test_bb]):
        with h5py.File(path, "r") as f:
            image = f[image_key][bb]
            lab = f[label_key][bb]
            masks = f[instance_key][bb]
        _create_split(split, [image], [lab], [masks])


def main():
    out_dir = "./data/obap"
    # data_preparation(out_dir)
    training_and_evaluation(out_dir)


if __name__ == "__main__":
    main()
