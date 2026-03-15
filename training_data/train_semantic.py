import json
from collections import OrderedDict
from pathlib import Path

import torch
import torch_em

import micro_sam.training as sam_training
from micro_sam.instance_segmentation import get_unetr

from patho_sam.training import SemanticInstanceTrainer
from patho_sam.training.util import histopathology_identity

from train_instances import _load_data


def get_loaders(split_path):
    with open(split_path, "r") as f:
        split = json.load(f)
    train_paths, val_paths = split["train"], split["val"]
    label_key = "labels/silver/semantic"

    train_data, train_labels = _load_data(train_paths, label_key)
    val_data, val_labels = _load_data(val_paths, label_key)

    batch_size = 4
    train_loader = torch_em.default_segmentation_loader(
        raw_paths=train_data, label_paths=train_labels,
        raw_key=None, label_key=None,
        patch_shape=(512, 512), with_channels=True,
        batch_size=batch_size, n_samples=800,
        raw_transform=histopathology_identity,
    )
    val_loader = torch_em.default_segmentation_loader(
        raw_paths=val_data, label_paths=val_labels,
        raw_key=None, label_key=None,
        patch_shape=(512, 512), with_channels=True,
        batch_size=batch_size, n_samples=80,
        raw_transform=histopathology_identity,
    )

    return train_loader, val_loader


def run_training(split_path):
    name = f"pathosam-semantic-{Path(split_path).stem}"
    save_root = "/mnt/vast-nhr/projects/nim00007/data/histopatho/cancerscout/checkpoints/v2"
    train_loader, val_loader = get_loaders(split_path)

    # Hyperparameters for training
    num_classes = 3
    class_weights = [1, 2, 4]

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
        final_activation=None,
    )

    # All other stuff we need for training
    optimizer = torch.optim.AdamW(unetr.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=5)

    # This class creates all the training data for each batch (inputs and semantic labels)
    convert_inputs = sam_training.util.ConvertToSemanticSamInputs()

    # The trainer which performs the semantic segmentation training and validation (implemented using 'torch_em')
    trainer = SemanticInstanceTrainer(
        name=name,
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
        class_weights=class_weights,
        device=torch.device("cuda"),
        save_root=save_root,
        early_stopping=25,
    )
    trainer.fit(epochs=250, overwrite_training=False)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("split")
    parser.add_argument("--slurm", action="store_true")
    args = parser.parse_args()

    if args.slurm:
        import subprocess
        cmd = ["sbatch", "train_semantic.sh", args.split]
        subprocess.run(cmd)
    else:
        run_training(args.split)


if __name__ == "__main__":
    main()
