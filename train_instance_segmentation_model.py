import h5py
import numpy as np
from micro_sam.training import train_sam, default_sam_loader


def get_loaders(version):
    # We use the first two tiles for training.
    train_paths = [
        "data/annotated_data/cd8/RE-000072-1_1_4-CD8_CRC_CancerScout-2024-07-12T11-04-16_tile00.h5",
        "data/annotated_data/cd8/RE-000072-1_1_4-CD8_CRC_CancerScout-2024-07-12T11-04-16_tile01.h5",
    ]
    # And half of the last one for validation (the other half is used for testing).
    val_path = "data/annotated_data/cd8/RE-000072-1_1_4-CD8_CRC_CancerScout-2024-07-12T11-04-16_tile02.h5"

    if version == 1:
        label_key = "labels/ihc"
    else:
        label_key = "labels/silver/nuclei"

    train_data, train_labels = [], []
    for path in train_paths:
        with h5py.File(path, "r") as f:
            data = f["image"][:].transpose((2, 0, 1))
            labels = f[label_key][:]
        train_data.append(data)
        train_labels.append(labels)

    val_bb = np.s_[:4000]
    with h5py.File(val_path, "r") as f:
        val_data = [f["image"][val_bb].transpose((2, 0, 1))]
        val_labels = [f[label_key][val_bb]]

    batch_size = 6
    train_loader = default_sam_loader(
        raw_paths=train_data, label_paths=train_labels,
        raw_key=None, label_key=None,
        patch_shape=(512, 512), with_channels=True,
        with_segmentation_decoder=True,
        batch_size=batch_size, n_samples=800,
    )
    val_loader = default_sam_loader(
        raw_paths=val_data, label_paths=val_labels,
        raw_key=None, label_key=None,
        patch_shape=(512, 512), with_channels=True,
        with_segmentation_decoder=True,
        batch_size=batch_size, n_samples=100,
    )

    return train_loader, val_loader


def train_v1():
    train_loader, val_loader = get_loaders(version=1)
    train_sam(
        name="pathosam-ihc", model_type="vit_b_histopathology",
        train_loader=train_loader, val_loader=val_loader,
    )


def train_v2():
    train_loader, val_loader = get_loaders(version=2)
    train_sam(
        name="pathosam-all-nuc", model_type="vit_b_histopathology",
        train_loader=train_loader, val_loader=val_loader, early_stopping=25,
        n_epochs=200, n_objects_per_batch=10, lr=1e-4,
    )


# TODO implement segmentationdecoder only training to make this more efficient.
# TODO need to train on fully empty patches to avoid spurious segmentations.
def main():
    # train_v1()
    train_v2()


if __name__ == "__main__":
    main()
