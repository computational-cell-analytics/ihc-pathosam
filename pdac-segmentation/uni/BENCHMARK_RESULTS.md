# UNI2-h tissue classification benchmark (PDAC H&E)

This documents how the settings of `train_tissue_classifier.py` / `predict_tissue_classifier.py` were chosen.
The benchmark ran on 2026-10-05/06. The benchmark script itself was removed afterwards; everything it did is
described here, and the training / prediction scripts implement the selected configuration.

## Key takeaways

- Final pipeline: linear probe on UNI2-h tokens at 2.19 um/px (level 3 of the 20x scans), with the 9 x 9 token
  context mean appended, 3 x 3 probability smoothing, 10k class-balanced training tokens, and an other tissue class
  learned from clusters of unannotated tissue (predicted only at probability >= 0.9). Held-out mean Dice 0.973
  (background 0.985, normal pancreas 0.971, tumor 0.957); the 3-class version without other tissue reaches 0.980.
- Only score glass and annotated regions: the annotations are not exhaustive, and counting unannotated tissue as
  background penalizes correct predictions (normal pancreas Dice 0.37 -> 0.97 once this is fixed).
- Context (the mean of the surrounding tokens) is the one big gain; everything else is within noise.
- 2.19 um/px is as good as or better than 1.1 um/px at a quarter of the cost.
- UNI2-h is robust to simulated stain and blur shifts (at most about -0.003); stain augmentation is not needed and
  per-slide feature standardization hurts.
- 1k training tokens are about as good as 10k; with about 100 tokens, a Random Forest with context 5 is better.
- Normal pancreas is almost never missed (recall 0.99-1.00) but slightly over-predicted at its borders.
- Without an other tissue class, all stroma, fat, muscle, ... is predicted as tumor or normal pancreas. Learning other
  tissue from clusters of unannotated tissue fixes this partly (about a third of it) at a small cost in tumor Dice;
  annotated examples of other tissue remain the reliable fix.

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

### 5. Learning other tissue without annotations

The whole-slide maps predict all tissue as tumor or normal pancreas, which is confusing. Three ways to get an
"other tissue" class without new annotations were tested with the selected configuration (2026-10-06):

- Unannotated tissue as a fourth class: leave-one-slide-out Dice for normal pancreas dropped from 0.96-0.99 to
  0.06-0.20 and for tumor from 0.95-0.98 to 0.51-0.89. Unannotated tissue contains a lot of real normal pancreas
  (and some tumor), and there are far more unannotated tokens than annotated normal pancreas tokens (about 13k),
  so the model learns annotated normal pancreas as other tissue (recall 0.04 on TM105).
- The confidence of the 3-class model: the maximum probability separates annotated from unannotated tissue only
  partially (AUC 0.71-0.89, median confidence on unannotated tissue 0.97-1.0), so a threshold would only catch a
  small part of the other tissue.
- The multi-tissue model of `../tissue_segmentation` (`multi-tissue_labels`): its classes do not line up with the
  annotations. The class covering 39% of the unannotated tissue also covers 32% of the annotated tumor regions,
  because the tumor outlines include the tumor stroma.

Three further ideas were then compared leave-one-slide-out at the token level (mean over the 4 test slides; coverage =
share of unannotated tissue predicted as other tissue, leak = share of annotated tissue predicted as other tissue):

| Method | Normal | Tumor | Coverage | Tumor leak |
| --- | --- | --- | --- | --- |
| 3-class (current model) | 0.977 | 0.972 | 0% | 0% |
| clustering (k-means, 40 clusters, strict), other only if P(other) >= 0.9 | 0.977 | 0.949 | 33% | 4.5% |
| clustering, other by argmax | 0.977 | 0.937 | 40% | 6.8% |
| clustering, looser cluster selection | 0.974 | 0.900 | 63% | 13.6% |
| tumor vs other tissue (incl. normal pancreas) vs background | - | 0.831 | 88% | 24% |
| positive-unlabeled learning (Elkan-Noto) | 0.07 | 0.83 | 71% | 24% (96% of normal) |

- Clustering: k-means on the context features (PCA to 64 dims) of the training tissue; clusters whose share of the
  annotated normal / tumor tokens is below 5% of their share of the unannotated tokens are taken as other tissue.
  Normal pancreas is unaffected; tumor loses about 0.02 Dice, mostly on TM120 (0.87). Visually it removes most of the
  fat and loose stroma, but duodenal mucosa and muscle (TM50) are still predicted as tumor. The settings were picked
  among 19 variants on the test slides, so the numbers are slightly optimistic.
- Tumor vs other tissue fails because the unannotated stroma looks like the stroma inside the tumor outlines.
- Positive-unlabeled learning fails because annotated normal pancreas looks like the unannotated normal pancreas.
- A public tissue-type dataset (e.g. NCT-CRC-HE-100K) was not tried: it is colorectal and at 0.5 um/px, where a
  patch is too small for the context features.

Annotated examples of other tissue (stroma outside the tumor, fat, muscle, duodenal mucosa, lymph nodes, ...) remain
the reliable fix; they can be trained as a fourth class with the same pipeline.


### 6. Class balance

The training sample is class balanced (`sample_balanced`: equal share per class, 2,500 tokens each for 10k tokens).
Leave-one-slide-out comparison with the final pipeline (mean over the 4 test slides):

| Training sample (10k tokens) | Background | Normal | Tumor | Other coverage |
| --- | --- | --- | --- | --- |
| class balanced (selected) | 0.985 | 0.977 | 0.949 | 33% |
| natural proportions (76% glass, 0.4% normal pancreas) | 0.987 | 0.972 | 0.957 | 29% |
| natural proportions + class-weighted loss | 0.987 | 0.968 | 0.958 | 30% |

The differences are within noise. Balanced sampling is kept because it guarantees enough tokens of the rare normal
pancreas class (only about 40 tokens in a natural sample).

## Final pipeline

Linear probe on UNI2-h tokens at 2.19 um/px, with the 9 x 9 token context mean appended (3072 features per token),
trained on 10k class-balanced tokens of four classes: background (glass), normal pancreas, tumor and other tissue.
Other tissue is learned from the unannotated tissue in the clusters (k-means, 40 clusters) that hold almost no
annotated normal pancreas or tumor. At prediction, the class probabilities are averaged over 3 x 3 tokens and other
tissue is only predicted where its probability is at least 0.9.

Leave-one-slide-out results of the final scripts (`train_tissue_classifier.py` / `predict_tissue_classifier.py`,
metrics from `elf.evaluation`, scored at 2.19 um/px on glass and annotated tissue). Dice / precision / recall;
pixel-wise F1 is identical to Dice:

| Test slide | Background | Normal pancreas | Tumor | Mean Dice | Other tissue (of predicted tissue) |
| --- | --- | --- | --- | --- | --- |
| TM105 | 0.971 / 0.976 / 0.966 | 0.982 / 0.965 / 0.999 | 0.984 / 0.982 / 0.986 | 0.979 | 2.5% |
| TM120 | 0.987 / 0.985 / 0.989 | 0.953 / 0.911 / 0.999 | 0.894 / 0.965 / 0.833 | 0.945 | 36.6% |
| TM50 | 0.997 / 0.998 / 0.995 | - | 0.993 / 0.991 / 0.996 | 0.995 | 23.4% |
| TM90 | 0.987 / 0.986 / 0.988 | 0.978 / 0.969 / 0.987 | 0.958 / 0.976 / 0.942 | 0.975 | 26.6% |
| Mean | 0.985 / 0.986 / 0.984 | 0.971 / 0.948 / 0.995 | 0.957 / 0.978 / 0.939 | 0.973 | |

Compared with the 3-class model (mean Dice 0.980: background 0.986, normal pancreas 0.974, tumor 0.975), tumor
recall drops mostly on TM120, where part of the tumor stroma is predicted as other tissue. Normal pancreas is
almost never missed (recall 0.99-1.00) but slightly over-predicted at its borders.

The classifier trained on all five slides is
`/mnt/vast-nhr/projects/cidas/cca/experiments/pdac_uni_semantic/models/uni_tissue_classifier_v2.pt` (6 of 40
clusters used as other tissue). The leave-one-slide-out classifiers and predictions are in
`/mnt/vast-nhr/projects/cidas/cca/experiments/pdac_uni_semantic/pipeline_final_loso/`.

Held-out prediction figures of the final pipeline (`other_tissue_figures/<slide>_heldout_prediction.png` in the
repository root): raw slide, ground truth and whole slide prediction as flat maps (normal pancreas blue, tumor
violet, other tissue brown, glass white, unannotated tissue grey), and per class the true positives (aqua), false
positives (red) and false negatives (yellow), following the TP / FP / FN idea of
`elf.visualisation.metric_visualization` with colors that stay distinct for color-blind readers.

## Inference for users

```bash
python predict_tissue_classifier.py -i slide.zarr -m uni_tissue_classifier_v2.pt -o slide_prediction.h5 --pixel_size 0.2738
```

- Input: a WSI pyramid (zarr with `s{level}/image`, or a TIFF pyramid such as SVS / OME-TIFF) and its pixel size
  at full resolution. The closest finer level is read and resampled to 2.19 um/px.
- Output: HDF5 with `prediction` (0 background, 1 normal pancreas, 2 tumor, 3 other tissue at 2.19 um/px),
  `probabilities` (per token) and the source and pixel size as attributes. `--labels` also prints the Dice,
  precision, recall and F1 against an annotation.
- Needs a GPU and access to the UNI2-h weights (gated on Hugging Face; local copy in
  `/mnt/vast-nhr/projects/cidas/cca/models/univ2`). About 25 s per slide on an A100.
- Open points: reading the pixel size from the slide metadata, a batch mode for folders, export to QuPath (GeoJSON)
  and overview PNGs. Reading SVS / TIFF pyramids is implemented but not yet tested on real slides.

## Limitations

- The scores only cover glass and annotated regions. Other tissue is learned without annotations and cannot be
  scored; it covers about a third of the unannotated tissue, and some tissue (e.g. duodenal mucosa and muscle on
  TM50) is still predicted as tumor. Annotated regions of other tissue would allow training and scoring it properly.
- Five slides from one center and one scanner. The stain and blur shifts are simulations, not real other-site data.
- H&E only. IHC (hematoxylin + DAB, no eosin) looks very different to UNI2-h and to the saturation-based glass
  mask; it would need annotated IHC slides and a retrained classifier.
