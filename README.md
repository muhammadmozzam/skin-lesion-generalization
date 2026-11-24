# How well do lightweight skin lesion classifiers generalize?

A cross-dataset study of accuracy, calibration and simple fixes for skin cancer detection.
Models are trained on one source (ISIC 2019 dermoscopy) and tested, without retraining,
on images from other hospitals and devices (HAM10000, PAD-UFES-20 smartphone photos).

> **Research prototype, not for diagnosis.**

**Live demo:** _coming soon (Hugging Face Spaces)_

## The question
Skin cancer models often score high on the dataset they were trained on, then drop on
images from another hospital or camera. How big is that drop, and which simple fixes
(colour augmentation, colour normalisation, test-time augmentation) reduce it?

## Setup
| | |
|---|---|
| Task | Binary: malignant (melanoma, BCC, SCC) vs benign (nevus, benign keratosis, dermatofibroma, vascular). Actinic keratosis excluded (pre-malignant and labelled inconsistently across datasets). |
| Train / validation / in-domain test | ISIC 2019 training set **with all HAM10000 images removed** (ISIC 2019 contains HAM10000), split 70/15/15 by lesion |
| External test 1 | HAM10000 (Vienna, dermoscopy) |
| External test 2 | PAD-UFES-20 (Brazil, smartphone clinical photos) |
| Models | ResNet-50, EfficientNet-B0, DeiT-Small (ImageNet pretrained, `timm`) |
| Fixes | none · strong colour augmentation · Shades-of-Gray colour normalisation · test-time augmentation |
| Metrics | AUC, balanced accuracy, F1, sensitivity, specificity, expected calibration error (ECE), Brier |
| Calibration | Temperature scaling fitted on ISIC validation only, applied unchanged to every outside dataset |
| Leakage ablation | Same model trained *with* the HAM10000 images left inside ISIC 2019, to measure how much this common overlap inflates HAM10000 results |
| Protocol | Checkpoint chosen on ISIC validation only; 3 seeds per setting, mean ± std reported |

## Results
_Filled in from real runs: see `results/summary.csv` and `results/auc_drop.csv`._

## How to run
Data is **not** included. Download each dataset from its official source and accept its licence:
- ISIC 2019: https://challenge.isic-archive.com/data/ (CC BY-NC)
- HAM10000: https://doi.org/10.7910/DVN/DBW86T (see licence on the page)
- PAD-UFES-20: https://data.mendeley.com/datasets/zr7vgbcyr2/1 (see licence on the page)

```bash
pip install -r requirements.txt
cd src
python data.py --isic /path/isic2019 --ham /path/ham10000 --pad /path/pad-ufes-20 --out ../data/splits.csv
python preprocess.py --splits ../data/splits.csv --out ../data/cache
python train.py --splits ../data/splits.csv --model efficientnet_b0 --fix none --seed 0 --out ../results
python summarize.py --results ../results
python gradcam.py --run ../results/efficientnet_b0_none_s0 --splits ../data/splits.csv
python export_onnx.py --run ../results/efficientnet_b0_none_s0 --splits ../data/splits.csv
```
### On Kaggle (free GPU)
Create a notebook, attach the three datasets (`andrewmvd/isic-2019`,
`kmader/skin-cancer-mnist-ham10000`, `mahdavi1202/skin-cancer`), turn on GPU and Internet, then:
```bash
!git clone https://github.com/muhammadmozzam/skin-lesion-generalization /kaggle/working/skin-lesion-generalization
!pip install -q timm grad-cam onnx onnxruntime
!python /kaggle/working/skin-lesion-generalization/run_kaggle.py --stage prep
!python /kaggle/working/skin-lesion-generalization/run_kaggle.py --stage check
!python /kaggle/working/skin-lesion-generalization/run_kaggle.py --stage train --seeds 0
!python /kaggle/working/skin-lesion-generalization/run_kaggle.py --stage summary
```

## Repository layout
```
src/           data prep, training, metrics, calibration, Grad-CAM, ONNX export
app/           Gradio demo for Hugging Face Spaces
run_kaggle.py  one-command runner for Kaggle
results/       metrics tables and figures (no images, no weights)
```

## Citation
_Added once the preprint is online._

## Acknowledgements
Please cite the dataset papers: ISIC 2019 (Tschandl et al. 2018; Codella et al. 2017;
Combalia et al. 2019), HAM10000 (Tschandl et al. 2018), PAD-UFES-20 (Pacheco et al. 2020).

## Licence
Code: MIT. Datasets keep their own licences.
