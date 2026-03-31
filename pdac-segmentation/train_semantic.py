from collections import OrderedDict

import torch

import micro_sam.training as sam_training
from micro_sam.instance_segmentation import get_unetr

from patho_sam.training import SemanticInstanceTrainer

from util import get_loaders


def run_training():
    name = "pathosam-nuclei-semantic"
    save_root = "/mnt/vast-nhr/projects/cidas/cca/data/pdac_umg_histopatho/models/v1"

    label_key = "labels/nucleus/semantic"
    train_loader, val_loader = get_loaders(label_key)

    # Hyperparameters for training
    num_classes = 3

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
    optimizer = torch.optim.AdamW(unetr.parameters(), lr=5e-5)
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
        device=torch.device("cuda"),
        save_root=save_root,
        early_stopping=None,
    )
    trainer.fit(epochs=250, overwrite_training=False)


def main():
    run_training()


if __name__ == "__main__":
    main()
