import json
from pathlib import Path

import h5py
from micro_sam.training import train_instance_segmentation, default_sam_loader


def _load_data(paths, label_key):
    data, labels = [], []
    for path in paths:
        with h5py.File(path, "r") as f:
            dat = f["image"][:].transpose((2, 0, 1))
            lab = f[label_key][:]
        assert dat.shape[1:] == lab.shape, f"{dat.shape}, {lab.shape}, {path}"
        data.append(dat)
        labels.append(lab)
    return data, labels


def get_loaders(split_path):
    with open(split_path, "r") as f:
        split = json.load(f)
    train_paths, val_paths = split["train"], split["val"]
    label_key = "labels/silver/nuclei"

    train_data, train_labels = _load_data(train_paths, label_key)
    val_data, val_labels = _load_data(val_paths, label_key)

    batch_size = 4
    train_loader = default_sam_loader(
        raw_paths=train_data, label_paths=train_labels,
        raw_key=None, label_key=None,
        patch_shape=(512, 512), with_channels=True,
        with_segmentation_decoder=True,
        train_instance_segmentation_only=True,
        batch_size=batch_size, n_samples=800,
    )
    val_loader = default_sam_loader(
        raw_paths=val_data, label_paths=val_labels,
        raw_key=None, label_key=None,
        patch_shape=(512, 512), with_channels=True,
        with_segmentation_decoder=True,
        train_instance_segmentation_only=True,
        batch_size=batch_size, n_samples=80,
    )

    return train_loader, val_loader


def run_training(split_path):
    train_loader, val_loader = get_loaders(split_path)
    name = f"pathosam-nuclei-{Path(split_path).stem}"
    save_root = "/mnt/vast-nhr/projects/nim00007/data/histopatho/cancerscout/checkpoints/v2"
    train_instance_segmentation(
        name=name, model_type="vit_b_histopathology",
        train_loader=train_loader, val_loader=val_loader,
        early_stopping=25, n_epochs=250, lr=1e-4,
        save_root=save_root,
    )


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("split")
    parser.add_argument("--slurm", action="store_true")
    args = parser.parse_args()

    if args.slurm:
        import subprocess
        cmd = ["sbatch", "train_instances.sh", args.split]
        subprocess.run(cmd)
    else:
        run_training(args.split)


if __name__ == "__main__":
    main()
