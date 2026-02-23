from collections import OrderedDict

import h5py
import numpy as np
import torch
import torch_em

import micro_sam.training as sam_training
from micro_sam.instance_segmentation import get_unetr

from patho_sam.training import SemanticInstanceTrainer
from patho_sam.training.util import calculate_class_weights_for_loss_weighting, histopathology_identity


def get_loaders():
    # We use the first two tiles for training.
    train_paths = [
        "data/annotated_data/cd8/RE-000072-1_1_4-CD8_CRC_CancerScout-2024-07-12T11-04-16_tile00.h5",
        "data/annotated_data/cd8/RE-000072-1_1_4-CD8_CRC_CancerScout-2024-07-12T11-04-16_tile01.h5",
    ]
    # And half of the last one for validation (the other half is used for testing).
    val_path = "data/annotated_data/cd8/RE-000072-1_1_4-CD8_CRC_CancerScout-2024-07-12T11-04-16_tile02.h5"

    label_key = "labels/silver/semantic"

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

    batch_size = 8
    train_loader = torch_em.default_segmentation_loader(
        raw_paths=train_data, label_paths=train_labels,
        raw_key=None, label_key=None,
        patch_shape=(512, 512), with_channels=True,
        with_segmentation_decoder=True,
        batch_size=batch_size, n_samples=800,
        raw_transform=histopathology_identity,
    )
    val_loader = torch_em.default_segmentation_loader(
        raw_paths=val_data, label_paths=val_labels,
        raw_key=None, label_key=None,
        patch_shape=(512, 512), with_channels=True,
        with_segmentation_decoder=True,
        batch_size=batch_size, n_samples=200,
        raw_transform=histopathology_identity,
    )

    return train_loader, val_loader


def train_semantic_segmentation():
    # Hyperparameters for training
    num_classes = 3
    checkpoint_name = "pathosam-semantic"

    train_loader, val_loader = get_loaders()

    # Get the trainable Segment Anything Model.
    model, state = sam_training.get_trainable_sam_model(
        model_type="vit_b_histopathology", return_state=True,
    )

    # Remove the output layer weights as we have new target class for the new task.
    decoder_state = OrderedDict(
        [(k, v) for k, v in state["decoder_state"].items() if not k.startswith("out_conv.")]
    )

    # Get the UNETR model for semantic segmentation pipeline
    unetr = get_unetr(
        image_encoder=model.sam.image_encoder,
        decoder_state=decoder_state,
        out_channels=num_classes,
        flexible_load_checkpoint=True,
    )

    # All other stuff we need for training
    optimizer = torch.optim.AdamW(unetr.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=5)

    # This class creates all the training data for each batch (inputs and semantic labels)
    convert_inputs = sam_training.util.ConvertToSemanticSamInputs()

    # The trainer which performs the semantic segmentation training and validation (implemented using 'torch_em')
    trainer = SemanticInstanceTrainer(
        name=checkpoint_name,
        train_loader=train_loader,
        val_loader=val_loader,
        model=unetr,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        log_image_interval=100,
        mixed_precision=True,
        compile_model=False,
        convert_inputs=convert_inputs,
        num_classes=num_classes,
        dice_weight=0,
        class_weights=calculate_class_weights_for_loss_weighting(),
    )
    trainer.fit(iterations=int(1e5), overwrite_training=False)


def main():
    train_semantic_segmentation()


if __name__ == "__main__":
    main()
