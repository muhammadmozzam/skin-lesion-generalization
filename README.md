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
| Protocol | Checkpoint chosen on ISIC validation only; 10 epochs; 2 seeds per setting (0 and 1), mean reported |

## Results (Kaggle run, version 2, 27 Nov 2026)
Data after removing the HAM10000 overlap: ISIC 2019 train 10,181 / val 2,156 / test 2,242 images;
HAM10000 9,688; PAD-UFES-20 1,568. 9,885 ISIC 2019 images were removed because they are HAM10000 images.

### 1. Every model drops on new hospitals and cameras
AUC, mean of 2 seeds, no test-time augmentation. "Drop" = ISIC test AUC minus outside AUC.

| Model | Fix | ISIC 2019 test | HAM10000 | PAD-UFES-20 | Drop to HAM | Drop to PAD |
|---|---|---|---|---|---|---|
| EfficientNet-B0 | none | 0.883 | 0.826 | 0.726 | 0.057 | **0.157** |
| EfficientNet-B0 | colour augmentation | 0.893 | 0.842 | 0.771 | 0.051 | 0.122 |
| EfficientNet-B0 | Shades-of-Gray | 0.881 | 0.797 | 0.760 | 0.085 | 0.121 |
| ResNet-50 | none | 0.900 | 0.829 | 0.819 | 0.071 | 0.082 |
| ResNet-50 | colour augmentation | 0.888 | 0.834 | 0.812 | 0.054 | 0.076 |
| ResNet-50 | Shades-of-Gray | 0.893 | 0.800 | 0.816 | 0.093 | 0.077 |
| DeiT-Small | none | 0.892 | 0.835 | 0.854 | 0.057 | **0.038** |
| DeiT-Small | colour augmentation | 0.893 | 0.859 | 0.809 | 0.035 | 0.085 |
| DeiT-Small | Shades-of-Gray | 0.905 | 0.834 | 0.818 | 0.070 | 0.087 |

What this shows:
- **In-domain scores hide the problem.** All nine settings score 0.88–0.91 on the ISIC test set, but 0.73–0.85 on smartphone photos from Brazil (PAD-UFES-20).
- **The smallest model drops the most.** EfficientNet-B0 loses 0.12–0.16 AUC on PAD-UFES-20; ResNet-50 and DeiT-Small lose 0.04–0.09.
- **Colour augmentation helps EfficientNet-B0 but is not a general fix.** It cut EfficientNet's PAD-UFES-20 drop from 0.157 to 0.122, changed little for ResNet-50, and for DeiT-Small helped on HAM10000 but hurt on PAD-UFES-20.
- **Shades-of-Gray colour normalisation hurt on HAM10000 for all three models** (drop 0.070–0.093 vs 0.057–0.071 without it).
- **Seeds disagree.** On PAD-UFES-20 the two seeds of the same setting differ by up to 0.08 AUC (DeiT-Small, no fix: 0.813 vs 0.895). Two seeds are not enough to rank the fixes confidently; differences under about 0.04 should be read as noise.
- **Test-time augmentation** (flips) raised outside AUC slightly in almost every setting (e.g. EfficientNet-B0 on PAD-UFES-20: 0.726 → 0.757).

### 2. Leakage test: what happens if the HAM10000 overlap is left in
ISIC 2019 contains the HAM10000 images. Many papers train on ISIC 2019 and then report HAM10000 as an "external" test.
To see the effect, EfficientNet-B0 (seed 0) was retrained with the overlap kept.

| Training data | ISIC 2019 test | HAM10000 | PAD-UFES-20 |
|---|---|---|---|
| Overlap removed (correct) | 0.875 | 0.819 | 0.739 |
| Overlap kept (leaky) | 0.908 | **0.952** | 0.865 |

With the overlap kept, the model scores **higher on HAM10000 than on its own test set** (a negative drop, −0.044),
which is the signature of test images having been seen in training. But the leaky training set is also about 70% larger
(17,185 vs 10,181 images), and PAD-UFES-20 (which has no overlap) improved by a similar amount. So part of the HAM10000 gain is simply
more data. A same-size control (leaky vs clean with equal training size) is planned to separate the two.

### 3. Calibration: confidence is least reliable where it matters most
Expected calibration error (ECE, lower is better), before and after temperature scaling fitted on ISIC validation only (18 runs: 3 models × 3 fixes × 2 seeds).

| | ISIC 2019 test | HAM10000 | PAD-UFES-20 |
|---|---|---|---|
| ECE before, range | 0.02–0.10 | 0.02–0.09 | **0.03–0.28** |
| Runs where temperature scaling lowered ECE | 10 of 18 | 7 of 18 | **16 of 18** |

Models are most over-confident on the smartphone photos. Temperature scaling usually reduced that error
(e.g. DeiT-Small + Shades-of-Gray, seed 1: 0.231 → 0.089) but PAD-UFES-20 ECE often stayed above 0.10,
and on HAM10000 it made calibration worse more often than better. A temperature fitted on the training hospital does not transfer reliably.

### Raw numbers
Per-run metrics (AUC, balanced accuracy, F1, ECE, with and without test-time augmentation) are printed in the
Kaggle run log and saved by `src/summarize.py` and `src/calibrate.py` to `results/`.

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
!python /kaggle/working/skin-lesion-generalization/run_kaggle.py --stage train --seeds 0,1
!python /kaggle/working/skin-lesion-generalization/run_kaggle.py --stage summary
```

## Repository layout
```
src/           data prep, training, metrics, calibration, Grad-CAM, ONNX export
app/           Gradio demo for Hugging Face Spaces
run_kaggle.py  one-command runner for Kaggle
results/       metrics tables and figures (no images, no weights)
```

## Updates
| Date | What changed |
|---|---|
| 26 Nov 2026 | Version 1 run: EfficientNet-B0 and ResNet-50, seed 0. Fixed a half-precision overflow that produced NaN predictions on a few images; added resume support so finished runs are not retrained. |
| 27 Nov 2026 | Version 2 run: added DeiT-Small, seed 1 for every setting, the leakage test and temperature-scaling calibration. Results above. |
| Next | Same-size leakage control, Grad-CAM figures, ONNX export and the Hugging Face demo. Paper preprint in preparation. |

## Citation
_Added once the preprint is online._

## Acknowledgements
Please cite the dataset papers: ISIC 2019 (Tschandl et al. 2018; Codella et al. 2017;
Combalia et al. 2019), HAM10000 (Tschandl et al. 2018), PAD-UFES-20 (Pacheco et al. 2020).

## Licence
Code: MIT. Datasets keep their own licences.
