# PDAC Segmentation Results

Last Updated: April 1st, 2026

Model: Semantic segmentation and instance segmentation models trained on 6 crops, annotated by Leonie using QuPath.

Results:

1. Instance Segmentation (PathoSAM) (_segmentation accuracy reported_):
    - mSA: 0.1378, SA50: 0.2879, SA75: 0.1182
2. Instance Segmentation (Finetuned starting from PathoSAM) (_segmentation accuracy reported_):
    - mSA: 0.3135, SA50: 0.5235, SA75: 0.3325
3. Semantic Segmentation (Finetuned starting from PathoSAM) (_dice score reported_):
    - background: 0.9717, negative cells: 0.6637, positive cells: 0.8122, mean: 0.8159 (unweighted)
    - background: 0.9506, negative cells: 0.6420, positive cells: 0.7773, mean: 0.7900 (weighted)

Next Steps:
1. Add class weighting for positive / negative cells.
2. Check results on WSIs.
3. Send it off to Leonie.
