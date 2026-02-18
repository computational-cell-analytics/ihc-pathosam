import h5py
import numpy as np
from micro_sam.training import train_sam, default_sam_loader


def get_loaders():
    # We use the first two tiles for training.
    train_paths = [
        "data/annotated_data/cd8/RE-000072-1_1_4-CD8_CRC_CancerScout-2024-07-12T11-04-16_tile00.h5",
        "data/annotated_data/cd8/RE-000072-1_1_4-CD8_CRC_CancerScout-2024-07-12T11-04-16_tile01.h5",
    ]
    # And half of the last one for validation (the other half is used for testing).
    val_path = "data/annotated_data/cd8/RE-000072-1_1_4-CD8_CRC_CancerScout-2024-07-12T11-04-16_tile02.h5"

    train_data, train_labels = [], []
    for path in train_paths:
        with h5py.File(path, "r") as f:
            data = f["image"][:].transpose((2, 0, 1))
            labels = f["labels/ihc"][:]
        train_data.append(data)
        train_labels.append(labels)

    val_bb = np.s_[:4000]
    with h5py.File(val_path, "r") as f:
        val_data = [f["image"][val_bb].transpose((2, 0, 1))]
        val_labels = [f["labels/ihc"][val_bb]]

    train_loader = default_sam_loader(
        raw_paths=train_data, label_paths=train_labels,
        raw_key=None, label_key=None,
        patch_shape=(512, 512), with_channels=True,
        with_segmentation_decoder=True,
        batch_size=8, n_samples=800,
    )
    val_loader = default_sam_loader(
        raw_paths=val_data, label_paths=val_labels,
        raw_key=None, label_key=None,
        patch_shape=(512, 512), with_channels=True,
        with_segmentation_decoder=True,
        batch_size=8, n_samples=400,
    )

    return train_loader, val_loader


def main():
    train_loader, val_loader = get_loaders()
    train_sam(
        name="pathosam-ihc", model_type="vit_b_histopathology",
        train_loader=train_loader, val_loader=val_loader,
    )


if __name__ == "__main__":
    main()
