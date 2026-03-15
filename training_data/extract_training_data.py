import argparse
import os
import sys
from pathlib import Path

import napari
import numpy as np
from elf.io import open_file

sys.path.append("..")
from util import load_tif_as_zarr, extract_region_instance_id_arrays  # noqa


def merge_labels(seg, pred):
    import vigra
    import nifty.ground_truth as ngt

    overlap_threshold = 0.25
    ids = np.unique(seg)[1:]
    ovlp = ngt.overlap(seg, pred)

    remove_pred_ids = []
    for seg_id in ids:
        this_ovlps = ovlp.overlapArraysNormalized(seg_id)
        for pred_id, pred_ovlp in zip(*this_ovlps):
            if pred_id != 0 and pred_ovlp > overlap_threshold:
                remove_pred_ids.append(pred_id)

    ret = pred.copy()
    ret[np.isin(pred, remove_pred_ids)] = 0
    ret, _, _ = vigra.analysis.relabelConsecutive(ret, start_label=1, keep_zeros=True)

    offset = int(ret.max())
    insert_mask = seg != 0
    ret[insert_mask] = (seg[insert_mask] + offset)

    semantic = np.zeros(ret.shape, dtype="uint8")
    semantic[ret > offset] = 2
    semantic[(ret <= offset) & (ret != 0)] = 1

    pos_ids = np.unique(ret[semantic == 2])
    neg_ids = np.unique(ret[semantic == 1])

    return ret, semantic, pos_ids, neg_ids


def process_image(image_path, label_path, output_folder, pred_path, pred_key, check=False):
    scale_level = 0
    data = load_tif_as_zarr(image_path, scale_level)
    fname = Path(image_path).stem

    regions = extract_region_instance_id_arrays(label_path)
    for i, (bb, seg) in enumerate(regions.values()):
        image = data[bb]
        assert image.shape[:2] == seg.shape

        if pred_path is None:
            pred = None
        else:
            with open_file(pred_path, mode="r") as f:
                pred = f[pred_key][bb]
            merged, semantic, pos_ids, neg_ids = merge_labels(seg, pred)

        if check:
            v = napari.Viewer()
            v.add_image(image)
            v.add_labels(seg)
            if pred is not None:
                v.add_labels(pred, visible=False)
                v.add_labels(merged)
                v.add_labels(semantic)
            napari.run()

        if output_folder is None:
            continue
        os.makedirs(output_folder, exist_ok=True)

        out_path = os.path.join(output_folder, f"{fname}_tile{i:02}.h5")
        with open_file(out_path, "w") as f:
            f.create_dataset("image", data=image, compression="gzip")
            f.create_dataset("labels/ihc", data=seg, compression="gzip")
            if pred is not None:
                f.create_dataset("labels/silver/semantic", data=semantic, compression="gzip")
                f.create_dataset("labels/silver/nuclei", data=merged, compression="gzip")
                f.create_dataset("labels/silver/positive_ids", data=pos_ids)
                f.create_dataset("labels/silver/negative_ids", data=neg_ids)


def main():
    parser = argparse.ArgumentParser()
    # "data/philips/cd8/RE-000072-1_1_4-CD8_CRC_CancerScout-2024-07-12T11-04-16.tiff"
    parser.add_argument("-i", "--image_path", required=True)
    # "data/philips/cd8/RE-000072-1_1_4-CD8_CRC_CancerScout-2024-07-12T11-04-16_annotations.json"
    parser.add_argument("-l", "--label_path", required=True)
    parser.add_argument("-o", "--output_folder")
    parser.add_argument("--pred_path")
    parser.add_argument("--pred_key")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    process_image(
        args.image_path, args.label_path, args.output_folder, args.pred_path, args.pred_key, args.check
    )


if __name__ == "__main__":
    main()
