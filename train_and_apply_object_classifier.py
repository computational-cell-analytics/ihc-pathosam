from pathlib import Path
from time import time

from deap.xdict import xdict
from deap.train import train, evaluate
from deap_objects.utils.util import LabelRemapper, get_object_conf


# TODO: what formats do we expect here?
def get_data_paths():
    return image_paths, label_paths, instance_paths


# TODO: What is this?
def get_class_ids():
    pass


def build_run_cfg(
    dataset,
    backbone,
    total_objects,
    ckpt_dir,
    out_dir,
):

    train_image_paths, train_label_paths, train_instance_paths = get_data_paths()
    val_image_paths, val_label_paths, val_instance_paths = get_data_paths()
    test_image_paths, test_label_paths, test_instance_paths = get_data_paths()

    max_objects, _ = get_object_conf(total_objects, len(train_image_paths))

    n_classes = len(get_class_ids(dataset))
    label_map = {cid: i + 1 for i, cid in enumerate(get_class_ids(dataset))}
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

    run_name = f"{dataset}_{backbone}_{total_objects}objects"

    semseg_run = xdict(
        name=run_name,
        model=xdict(backbone_name=backbone),
        checkpoint_dir=str(ckpt_dir),
        save_path=str(out_dir / "object_classification.csv"),
    ) + semseg

    return semseg_run


def main(args):
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    weights_path = ckpt_dir / args.weights

    semseg_run = build_run_cfg(
        dataset=args.dataset,
        backbone=args.backbone,
        total_objects=args.total_objects,
        ckpt_dir=ckpt_dir,
        out_dir=out_dir,
    )

    if not weights_path.exists():
        print(f"Checkpoint not found: {weights_path}")
        print("Starting training...")

        start = time()
        train(
            semseg_run,
            device=args.device,
            seed=args.seed,
        )
        print(f"Training finished in {time() - start:.2f}s")

        if not weights_path.exists():
            raise RuntimeError(
                f"Training finished but checkpoint was not found at {weights_path}"
            )
    else:
        print(f"Using existing checkpoint: {weights_path}")

    out = evaluate(
        semseg_run,
        weights=str(weights_path),
        device=args.device,
        predict_semseg=True,
    )

    time_file = out_dir / "time.csv"
    with open(time_file, "w") as f:
        f.write("inference_time\n")
        f.write(f"{out['inference_time']}\n")

    print("Done.")
    print(f"Checkpoint: {weights_path}")
    print(f"Results: {out_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # TODO: How do we use a customly initialized backbone?
    parser.add_argument('--backbone', type=str, default='SAM')
    # TODO: How do we train on all available objects?
    parser.add_argument('--total_objects', type=int, default=100)

    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        required=True,
        help="Directory where checkpoints are stored / should be written"
    )
    parser.add_argument(
        '--weights',
        type=str,
        default='last_weights.pth',
        help="Checkpoint filename"
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        default='inference_results'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42
    )

    args = parser.parse_args()
    main(args)
