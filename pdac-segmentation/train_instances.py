from micro_sam.training import train_instance_segmentation

from util import get_instance_loaders


def run_training():
    name = "pathosam-nuclei-instances"
    save_root = "/mnt/vast-nhr/projects/cidas/cca/data/pdac_umg_histopatho/models/v1"

    label_key = "labels/nucleus/instances"
    train_loader, val_loader = get_instance_loaders(label_key, n_val=200)

    train_instance_segmentation(
        name=name,
        model_type="vit_b_histopathology",
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=250,
        lr=5e-5,
        save_root=save_root,
        early_stopping=None,
    )


def main():
    run_training()


if __name__ == "__main__":
    main()
