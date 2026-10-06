This creates initial code for embedding computation with UNI.

## Tissue classification with UNI2-h

Classifies H&E WSIs into background (glass), normal pancreas and tumor with a linear probe on UNI2-h features.
See `BENCHMARK_RESULTS.md` for how the method was chosen and how well it works.

Training on annotated slides (zarr pyramids with `s{level}/image`, or TIFF pyramids such as SVS / OME-TIFF, and HDF5
label pyramids with `labels/{level}` as written by `../annotation/export_hdf5.py`):

```bash
python train_tissue_classifier.py \
    --images TM105_HE.zarr TM11_HE.zarr ... --labels TM105_HE.h5 TM11_HE.h5 ... \
    --pixel_size 0.2738 -o uni_tissue_classifier.pt
```

Prediction on a new slide:

```bash
python predict_tissue_classifier.py -i slide.zarr -m uni_tissue_classifier.pt -o slide_prediction.h5 --pixel_size 0.2738
```

- `--pixel_size` is the size of a pixel at full resolution in microns. The slide is read from the closest finer
  pyramid level and resampled to 2.19 microns per pixel.
- The output contains `prediction` (0 background, 1 normal pancreas, 2 tumor at 2.19 microns per pixel) and
  `probabilities` (per 31 micron token).
- Pass `--labels` to also print the Dice per class against an annotation.
- The model has no class for other tissue (stroma, fat, muscle, ...), which is predicted as tumor or normal pancreas.
- It is trained on H&E only and is not meant for IHC slides.

The classifier trained on the five PDAC slides:
`/mnt/vast-nhr/projects/cidas/cca/experiments/pdac_uni_semantic/models/uni_tissue_classifier_v1.pt`.
