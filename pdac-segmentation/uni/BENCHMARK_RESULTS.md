# UNI2-h tissue classification benchmark (PDAC H&E)

This documents how the settings of `train_tissue_classifier.py` / `predict_tissue_classifier.py` were chosen.
The benchmark ran on 2026-10-05/06. The benchmark script itself was removed afterwards; everything it did is
described here, and the training / prediction scripts implement the selected configuration.

## Key takeaways

- Selected configuration: linear probe on UNI2-h tokens at 2.19 um/px (level 3 of the 20x scans), with the
  9 x 9 token context mean appended, 3 x 3 probability smoothing and 10k class-balanced training tokens.
  Held-out mean Dice 0.980 with the final scripts; nested estimate of the whole selection procedure 0.977.
- Only score glass and annotated regions: the annotations are not exhaustive, and counting unannotated tissue as
  background penalizes correct predictions (normal pancreas Dice 0.37 -> 0.97 once this is fixed).
- Context (the mean of the surrounding tokens) is the one big gain; everything else is within noise.
- 2.19 um/px is as good as or better than 1.1 um/px at a quarter of the cost.
- UNI2-h is robust to simulated stain and blur shifts (at most about -0.003); stain augmentation is not needed and
  per-slide feature standardization hurts.
- 1k training tokens are about as good as 10k; with about 100 tokens, a Random Forest with context 5 is better.
- The model has no class for other tissue (stroma, fat, muscle, ...); it predicts such tissue as tumor or normal
  pancreas. This needs annotations of other tissue before whole-slide maps can be trusted.

## Task and data

- Classes: 0 background, 1 normal pancreas, 2 tumor (255 = unclassified, ignored).
- Five H&E slides (no IHC), 20x, 0.2738 um/px at level 0: TM105, TM11, TM120, TM50, TM90.
  Images: `/mnt/vast-nhr/projects/nim00007/data/histopatho/pdac-kfo/data_20260310/converted_zarr_he/`,
  labels: `.../annotations/` (QuPath polygons rasterized by `../annotation/export_hdf5.py`).
- The annotations are coarse region outlines and are not exhaustive: a lot of tissue (including normal pancreas
  and tumor) is left unannotated. TM50 and TM11 have no normal pancreas annotation; TM11 only has a few thin tumor
  strips.
- Benchmark outputs (features, per-run CSVs, summaries):
  `/mnt/vast-nhr/projects/cidas/cca/experiments/pdac_uni_semantic/`.

## Method

1. Read the slide at a pyramid level, cut it into 224 x 224 tiles (white padding at the border) and embed each
   tile with UNI2-h (ImageNet normalization, fp16). Each tile gives a 16 x 16 grid of 1536-d tokens (14 px each).
2. Each token gets the majority label of its 14 x 14 pixels.
3. Optionally append context: the mean token feature over a k x k token window (average pooling, stride 1,
   only existing tokens are averaged at the slide border).
4. Sample a class-balanced set of training tokens and fit a classifier.
5. Predict every token of the held-out slide, optionally average the class probabilities over an s x s window,
   take the argmax and score the prediction with Dice per class.

### Context in detail

- A token is one UNI2-h patch of 14 x 14 pixels, about 31 um at 2.19 um/px (not the 224 px tile fed to UNI2-h).
- Context k appends to every token the plain (unweighted) mean of the 1536-d features of the k x k tokens centered
  on it, including the token itself and any glass in the window. The token keeps its own 1536 features, so the
  classifier sees 3072 features per token. With k = 9 the window covers about 280 um.
- The window runs over the stitched token grid of the whole slide, so it crosses the borders of the 224 px tiles.
  At the slide border only existing tokens are averaged (no zero padding).
- It is average pooling with stride 1 on the GPU, in chunks of 256 channels. Nothing is learned. Training and
  prediction use the same function (`add_context` in `tissue_util.py`), and the window size is stored in the model.
- Smoothing s is the same averaging applied to the predicted class probabilities before the argmax.

## Evaluation protocol

- Leave-one-slide-out with 4 test slides (TM105, TM120, TM50, TM90). TM11 is used for training only, because its
  annotation is too sparse to evaluate on.
- Configurations were selected by nested leave-one-slide-out: inside every outer fold, each of the 3 remaining
  test slides is held out once as validation slide (training on the other 3 slides + TM11). The score used for
  selection is the mean validation Dice; the outer test slide is never used for choosing.
- Mean Dice is the mean over the classes annotated on the slide (a class that is not annotated is not scored).
- Robustness: every outer test slide was also predicted after global stain / scanner shifts it was never trained
  on (scaling of the hematoxylin and eosin optical densities: pale 0.7/0.7, dark 1.35/1.35, very pale 0.5/0.5,
  blue 1.5/0.6, pink 0.6/1.5, and a Gaussian blur with sigma 1.5 px).
- Seed variability: rerunning the sampling with other seeds changes the mean test Dice by about +-0.002, so
  differences below about 0.003 are noise.

### Data split

- Outer folds: 4 classifiers, each trained on 4 slides (always including TM11) and tested on the fifth.
- Inner folds (for choosing the configuration): within each outer fold, 3 runs that train on 3 slides and validate
  on the remaining test-capable slide.
- Final model: trained on all 5 slides.
- Labeled tokens per slide at 2.19 um/px (glass background / normal pancreas / tumor; unannotated tissue excluded):

| Slide | Background (glass) | Normal pancreas | Tumor |
| --- | --- | --- | --- |
| TM105 | 80k | 6.1k | 147k |
| TM11 (train only) | 456k | 0 | 3.3k |
| TM120 | 351k | 3.3k | 71k |
| TM50 | 150k | 0 | 77k |
| TM90 | 193k | 3.4k | 86k |

Normal pancreas is the scarce class (about 13k tokens over all slides). A 10k training sample has about 3.3k
tokens per class.

## What was tried, in order

### 1. First run: dense annotation labels, level 3 (2.19 um/px)

Unannotated tissue counted as background. One fold (test TM105), Random Forest (150 trees, depth 20):

| Tokens | Background | Normal | Tumor | Mean |
| --- | --- | --- | --- | --- |
| 100 | 0.774 | 0.354 | 0.842 | 0.657 |
| 100k | 0.819 | 0.372 | 0.887 | 0.692 |
| all (2.3M) | 0.833 | 0.380 | 0.890 | 0.701 |

Normal pancreas was "wrong" mostly because the model found normal pancreas that was not annotated.

### 2. Ignore unannotated tissue ("tissue_ignore")

Background is restricted to glass; annotated-background tokens on tissue are ignored in training and scoring.

- Tissue mask v1 (Otsu threshold on HSV saturation): mean Dice over 5 folds 0.55 -> 0.69 at 100k tokens. TM11 failed
  (tumor 0.18): pale stroma was classified as glass by the mask, and TM11 is barely annotated.
- The existing `tb_labels` from `converted_zarr_he_tissue_segmentation` are not a glass mask (they mark dense
  regions inside the tissue).
- Tissue mask v2 (used from here on): HSV saturation at level 5, Gaussian sigma 2, threshold 0.04 (glass has median
  saturation about 0.01, tissue 0.16-0.38), small objects (64 px) and holes (256 px) removed. It covers 97-99% of
  the annotated tissue. TM11 became train-only.

RF, level 3, 4 test folds: mean Dice 0.911 (100 tokens), 0.948 (1k), 0.960 (10k), 0.961 (100k).

### 3. Classifier and context (level 3 and level 2)

Mean over the 4 test folds, 100k tokens:

| Level | Configuration | Normal | Tumor | Mean |
| --- | --- | --- | --- | --- |
| 3 | RF, no context | 0.903 | 0.970 | 0.961 |
| 3 | linear probe, no context | 0.957 | 0.968 | 0.973 |
| 3 | RF, context 5 | 0.964 | 0.976 | 0.979 |
| 3 | linear probe, context 5 | 0.971 | 0.973 | 0.979 |
| 2 | RF, no context | 0.834 | 0.967 | 0.943 |
| 2 | linear probe, context 5 | 0.957 | 0.972 | 0.974 |
| 2 | RF, context 5 | 0.949 | 0.978 | 0.976 |

Level 2 (1.1 um/px) is not better than level 3 and costs 4x more, so all later work used level 3.
The linear probe is multinomial logistic regression on standardized features (AdamW, lr 1e-3, weight decay
1e-4, 2000 steps, batch 4096) and trains in about 1 s on the GPU.

### 4. Nested grid (overnight), level 3, budgets 100 / 1k / 10k tokens

Grid: classifier {linear, RF} x context {1, 5, 9, 15} x per-slide feature standardization {off, on} x stain
augmentation of the training slides (H/E scaled 1.3/0.75 and 0.75/1.3) {off, on}; then smoothing {3, 5},
an ensemble of RF + linear (averaged probabilities) and contexts 21 / 31; then budgets 100k / all and extra seeds
for linear context 9 / 15. 44 configurations, about 900 runs.

Selected rows at 10k tokens (validation = nested inner score; test = outer test slides, original stain;
worst shift = lowest test score over the six shifts):

| Configuration | Validation | Test | Worst shift | Normal | Tumor |
| --- | --- | --- | --- | --- | --- |
| ensemble, context 9, smooth 3 | 0.981 | 0.980 | 0.980 | 0.973 | 0.976 |
| ensemble, context 9 | 0.980 | 0.980 | 0.979 | 0.974 | 0.975 |
| linear, context 9, smooth 3 (selected) | 0.980 | 0.979 | 0.979 | 0.972 | 0.975 |
| linear, context 15, smooth 3 | 0.980 | 0.978 | 0.977 | 0.961 | 0.977 |
| linear, context 9 | 0.979 | 0.980 | 0.974 | 0.975 | 0.974 |
| RF, context 9 | 0.978 | 0.979 | 0.978 | 0.970 | 0.973 |
| linear, context 5 | 0.977 | 0.977 | 0.975 | 0.972 | 0.971 |
| linear, context 21 | 0.978 | 0.975 | 0.974 | 0.954 | 0.976 |
| linear, context 31 | 0.975 | 0.973 | 0.971 | 0.939 | 0.979 |
| RF, context 15 | 0.972 | 0.972 | 0.972 | 0.948 | 0.970 |
| linear, context 9, per-slide standardization | 0.969 | 0.970 | 0.959 | 0.961 | 0.963 |
| linear, no context | 0.959 | 0.962 | 0.953 | 0.922 | 0.965 |
| RF, no context | 0.960 | 0.960 | 0.958 | 0.902 | 0.967 |

Nested estimates (the honest number for the whole selection procedure, 10k tokens): 0.976 when always taking the
best validation configuration, 0.977 with the one-standard-error rule (simplest configuration within one SE of
the best). Under all six shifts the nested estimate stays at 0.975-0.979.

Budget (nested estimate, one-SE rule): 100 tokens 0.970, 1k 0.977, 10k 0.977. With very few tokens RF with
context 5 is better than the linear probe. Using all ~1M tokens (linear, context 9) gives 0.983 on the original
slides but drops to 0.970 under blur, so it generalizes worse than 10k.

### Findings

- Context is the one large gain (normal pancreas 0.92 -> 0.97). Windows of 9-15 tokens (about 280-460 um) work
  best; 21 and 31 lose normal pancreas.
- Per-slide feature standardization hurts in every configuration, most under blur.
- Stain augmentation is neutral: UNI2-h features are already robust to stain shifts (at most about -0.003).
- Probability smoothing (3 x 3) adds robustness to blur for the linear probe (worst shift 0.974 -> 0.979).
- The top configurations are within seed noise of each other.

### Resolution

Only two pyramid levels were compared (100k tokens, mean over the 4 test folds):

| Configuration | Level 3 (2.19 um/px) | Level 2 (1.1 um/px) |
| --- | --- | --- |
| linear probe, context 5 | 0.979 | 0.974 |
| RF, context 5 | 0.979 | 0.976 |
| RF, no context | 0.961 | 0.943 |

- Level 3 is as good or better everywhere and costs a quarter of the compute and storage, so it is used.
- Level 1 (0.55 um/px, the scale UNI2-h was pretrained at) was not tested: it costs about 16x more than level 3,
  and level 2 was already not better.
- Level 4 (4.4 um/px) was not tested. Since coarser has been better so far it may work as well at 4x less cost, but
  it may lose detail at the tumor / normal boundaries. It is a cheap follow-up experiment.
- The pipeline is defined in microns per pixel, not by level: every slide is resampled to 2.19 um/px, so slides
  from other scanners or magnifications end up at the same scale.

## Selected configuration

Linear probe on UNI2-h tokens at 2.19 um/px, with the 9 x 9 token context mean appended (3072 features per token),
3 x 3 probability smoothing, trained on 10k class-balanced tokens, no per-slide standardization, no stain
augmentation. It is the simplest configuration within noise of the best and the most robust linear variant.

Leave-one-slide-out check of the final scripts (scored at 2.19 um/px, which makes background slightly lower than
in the benchmark that scored at level 4):

| Test slide | Background | Normal | Tumor | Mean |
| --- | --- | --- | --- | --- |
| TM105 | 0.971 | 0.986 | 0.984 | 0.980 |
| TM120 | 0.990 | 0.962 | 0.951 | 0.968 |
| TM50 | 0.995 | - | 0.991 | 0.993 |
| TM90 | 0.988 | 0.975 | 0.973 | 0.978 |

The classifier trained on all five slides is
`/mnt/vast-nhr/projects/cidas/cca/experiments/pdac_uni_semantic/models/uni_tissue_classifier_v1.pt`.

The held-out predictions were rendered as PNGs (`<slide>_heldout_prediction.png`): raw slide, ground truth, whole
slide prediction, and per class the true positives (green), false positives (red) and false negatives (orange),
following the colors of `elf.visualisation.metric_visualization`; unannotated tissue is grey (not scored). The
leave-one-slide-out classifiers and predictions are in
`/mnt/vast-nhr/projects/cidas/cca/experiments/pdac_uni_semantic/pipeline_test/`.

## Inference for users

```bash
python predict_tissue_classifier.py -i slide.zarr -m uni_tissue_classifier_v1.pt -o slide_prediction.h5 --pixel_size 0.2738
```

- Input: a WSI pyramid (zarr with `s{level}/image`, or a TIFF pyramid such as SVS / OME-TIFF) and its pixel size
  at full resolution. The closest finer level is read and resampled to 2.19 um/px.
- Output: HDF5 with `prediction` (0 background, 1 normal pancreas, 2 tumor at 2.19 um/px), `probabilities` (per
  token) and the source and pixel size as attributes. `--labels` also prints the Dice against an annotation.
- Needs a GPU and access to the UNI2-h weights (gated on Hugging Face; local copy in
  `/mnt/vast-nhr/projects/cidas/cca/models/univ2`). About 25 s per slide on an A100.
- Open points: reading the pixel size from the slide metadata, a batch mode for folders, export to QuPath (GeoJSON)
  and overview PNGs. Reading SVS / TIFF pyramids is implemented but not yet tested on real slides.

## Limitations

- The scores only cover glass and annotated regions. The model has no class for other tissue (stroma, fat, muscle,
  duodenal mucosa, lymph nodes): it predicts such tissue as tumor or normal pancreas. Annotating a few regions of
  other tissue per slide and adding a class for it is the next step before others rely on whole-slide maps.
- Five slides from one center and one scanner. The stain and blur shifts are simulations, not real other-site data.
- H&E only. IHC (hematoxylin + DAB, no eosin) looks very different to UNI2-h and to the saturation-based glass
  mask; it would need annotated IHC slides and a retrained classifier.
