# IHC Segmentation with PathoSAM

This repository contains scripts for IHC segmentation in histopathology with [PathoSAM](https://github.com/computational-cell-analytics/patho-sam).

The most important scripts are:

`apply_pathosam_wsi.py`: To apply a PathoSAM model on a WSI image. The path to the WSI and the PathoSAM model can be specified via CLI.

`extract_training_data.py`: To extract the training masks and ROIs from the JSON and WSIs and to combine them with PathoSAM predictions for a silver ground-truth of all nuclei, and for semantic labels distinguishing IHC positive and negative labels.

`train_instance_segmentation_model.py`: Finetune a model for nucleus instance segmentation.

`train_semantic_segmentation_model.py`: Finetune a model for semantic segmentation (IHC positive vs. negative nuclei).

`process_wsi_segmentation.py`: Extract polygon list from WSI segmentation result and (optionally) filter instance segmentation with semantic segmentation.


## Set-up

The best way to set-up an environment with all dependencies is with [micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html) (conda also works, but it's much slower).

You need to: 
- Create an environment with micro_sam:
    - `micromamba create -c conda-forge micro_sam -n patho-sam`
    - Make sure that version 1.7.4 of micro_sam (or later) is installed, which is required for efficient WSI segmentation.
- Install PathoSAM in it:
    - `micromamba activate patho-sam`
    - `git clone https://github.com/computational-cell-analytics/patho-sam`
    - `cd patho-sam`
    - `pip install -e .`


## Models & Evaluation

I have trained two different models on the data shared by Felipe. The shared data contained 3 annotated regions, 2 were used for training, half of the other one for validation, the other half as a test spilt.
- Model v1: Instance segmentation model finetuned to only segment the IHC-positive cells. This did not work well (see eval below).
- Model(s) v2: Instance segmentation model finetuned to segment all nuclei (based on silver GT produced via initial PathoSAM model) and semantic segmentation model to segment IHC positive vs. negative cells. The results from both models can be combined to filter only the IHC positive instances. 
    - This worked much better, the results look quite good qualitatively; the eval scores are not so great because the GT annotations are not super precise; the model also has some FPs and FNs.


Evaluation on half of region 3 from the first batch of data shared by Felipe:

- Model v1 (Instance Segmentation IHCs):
    - mSA : 0.08988952006600967
    - SA50: 0.1596650998824912
    - SA75: 0.09167588495575221
- Model v2 (Full Instance Segmentation + Semantic Segmentation):
    - mSA : 0.2603745601924928
    - SA50: 0.5
    - SA75: 0.24342105263157895
