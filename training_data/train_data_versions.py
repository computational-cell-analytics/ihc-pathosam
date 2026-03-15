import os
from glob import glob
from extract_training_data import process_image


def training_data_v2():
    data_root = "/mnt/vast-nhr/projects/nim00007/data/histopatho/cancerscout"
    pred_root = "/mnt/vast-nhr/home/pape41/u12086/Work/my_projects/ihc-pathosam/data/predictions-cancerscout"  # noqa

    label_paths = sorted(glob(os.path.join(data_root, "**", "*.json"), recursive=True))
    image_paths = []
    pred_paths = []
    for path in label_paths:
        im_path = path.replace("_annotations.json", ".tiff")
        assert os.path.exists(im_path), im_path
        image_paths.append(im_path)

        root, fname = os.path.split(im_path)
        folder = os.path.split(root)[1]
        pred_path = os.path.join(pred_root, folder, fname.replace(".tiff", ".zarr"))
        assert os.path.exists(pred_path)
        pred_paths.append(pred_path)

    assert len(pred_paths) == len(image_paths), f"{len(pred_paths)}, {len(image_paths)}"

    output_root = "/mnt/vast-nhr/projects/nim00007/data/histopatho/cancerscout/training_data/v2"
    pred_key = "segmentation"
    for im_path, label_path, pred_path in zip(image_paths, label_paths, pred_paths):
        print("Processing", im_path)
        output_folder = os.path.join(output_root, "cd3" if "cd3" in pred_path else "cd8")
        process_image(im_path, label_path, output_folder, pred_path, pred_key)


def main():
    training_data_v2()


if __name__ == "__main__":
    main()
