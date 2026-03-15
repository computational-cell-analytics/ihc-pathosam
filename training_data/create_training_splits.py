import os
import json
from glob import glob

from sklearn.model_selection import train_test_split


def _save_split(out, train_paths, val_paths, test_paths):
    with open(out, "w") as f:
        json.dump({
            "train": train_paths,
            "val": val_paths,
            "test": test_paths,
        }, f, indent=2)


def _aggregate(paths):
    agg = {}
    for path in paths:
        prefix = path[:-len("_tile04.h5")]
        if prefix in agg:
            agg[prefix].append(path)
        else:
            agg[prefix] = [path]
    return agg


def _make_splits(agg):
    train, test = [], []
    for i, paths in enumerate(agg.values()):
        if i == (len(agg) - 1):
            test.extend(paths)
        else:
            train.extend(paths)
    train, val = train_test_split(train, test_size=0.15)
    return train, val, test


def create_splits_v2():
    root = "/mnt/vast-nhr/projects/nim00007/data/histopatho/cancerscout/training_data/v2"

    split_folder = "./splits"
    os.makedirs(split_folder, exist_ok=True)

    # These always go in the train split.
    empty = sorted(glob(os.path.join(root, "empty", "*.h5")))

    # CD3+ / CD8+ stainings.
    cd3 = sorted(glob(os.path.join(root, "cd3", "*.h5")))
    cd8 = sorted(glob(os.path.join(root, "cd8", "*.h5")))

    # Aggregate the data by WSI.
    cd3_agg, cd8_agg = _aggregate(cd3), _aggregate(cd8)

    # Train, Val, and Test data for CD3 and CD8.
    cd3_train, cd3_val, cd3_test = _make_splits(cd3_agg)
    cd8_train, cd8_val, cd8_test = _make_splits(cd8_agg)

    # Save the splits
    _save_split(f"{split_folder}/cd3.json", cd3_train + empty, cd3_val, cd3_test + cd8_test)
    _save_split(f"{split_folder}/cd8.json", cd8_train + empty, cd8_val, cd3_test + cd8_test)
    _save_split(
        f"{split_folder}/combined.json", cd3_train + cd8_train + empty, cd3_val + cd8_val, cd3_test + cd8_test
    )
    _save_split(
        f"{split_folder}/final.json", cd3_train + cd8_train + cd3_test + cd8_test + empty, cd3_val + cd8_val, []
    )


def main():
    create_splits_v2()


if __name__ == "__main__":
    main()
