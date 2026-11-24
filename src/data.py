"""Build one clean table (path, label, source, group) for each dataset.

Task: binary skin cancer detection
    malignant (1) = melanoma, basal cell carcinoma, squamous cell carcinoma
    benign    (0) = nevus, benign keratosis / seborrheic keratosis, dermatofibroma, vascular
Excluded everywhere: actinic keratosis (pre-malignant, labelled differently in each
dataset: HAM10000 "akiec" also mixes in Bowen's disease) and ISIC "UNK".

IMPORTANT: ISIC 2019 training data already contains every HAM10000 image.
We remove them from ISIC 2019, so HAM10000 is a genuinely unseen test set.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

IMG_EXT = {".jpg", ".jpeg", ".png"}

ISIC_MAP = {"MEL": 1, "BCC": 1, "SCC": 1, "NV": 0, "BKL": 0, "DF": 0, "VASC": 0}
HAM_MAP = {"mel": 1, "bcc": 1, "nv": 0, "bkl": 0, "df": 0, "vasc": 0}
PAD_MAP = {"MEL": 1, "BCC": 1, "SCC": 1, "NEV": 0, "SEK": 0}


def index_images(root):
    """Map file stem -> full path for every image under root."""
    out = {}
    for dirpath, _, files in os.walk(root):
        for f in files:
            p = Path(dirpath) / f
            if p.suffix.lower() in IMG_EXT:
                out.setdefault(p.stem, str(p))
    return out


def find_file(root, name):
    for dirpath, _, files in os.walk(root):
        if name in files:
            return str(Path(dirpath) / name)
    raise FileNotFoundError(f"{name} not found under {root}")


def load_ham(root):
    meta = pd.read_csv(find_file(root, "HAM10000_metadata.csv"))
    imgs = index_images(root)
    meta = meta[meta["dx"].isin(HAM_MAP)].copy()
    meta["label"] = meta["dx"].map(HAM_MAP)
    meta["path"] = meta["image_id"].map(imgs)
    meta["group"] = meta["lesion_id"]
    meta["source"] = "HAM10000"
    meta["image"] = meta["image_id"]
    return meta.dropna(subset=["path"])[["image", "path", "label", "group", "source"]]


def load_isic2019(root, ham_ids):
    gt = pd.read_csv(find_file(root, "ISIC_2019_Training_GroundTruth.csv"))
    try:
        meta = pd.read_csv(find_file(root, "ISIC_2019_Training_Metadata.csv"))
        gt = gt.merge(meta[["image", "lesion_id"]], on="image", how="left")
    except FileNotFoundError:
        gt["lesion_id"] = np.nan
    classes = [c for c in ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"] if c in gt]
    gt["dx"] = gt[classes].idxmax(axis=1)
    gt = gt[gt["dx"].isin(ISIC_MAP)].copy()
    n0 = len(gt)
    gt = gt[~gt["image"].isin(ham_ids)]  # remove the HAM10000 overlap
    print(f"ISIC 2019: removed {n0 - len(gt)} images that are also in HAM10000")
    imgs = index_images(root)
    gt["label"] = gt["dx"].map(ISIC_MAP)
    gt["path"] = gt["image"].map(imgs)
    gt["group"] = gt["lesion_id"].fillna(gt["image"])
    gt["source"] = "ISIC2019"
    return gt.dropna(subset=["path"])[["image", "path", "label", "group", "source"]]


def load_pad(root):
    meta = pd.read_csv(find_file(root, "metadata.csv"))
    imgs = index_images(root)
    meta = meta[meta["diagnostic"].isin(PAD_MAP)].copy()
    meta["label"] = meta["diagnostic"].map(PAD_MAP)
    meta["image"] = meta["img_id"].str.replace(r"\.png$", "", regex=True)
    meta["path"] = meta["image"].map(imgs)
    meta["group"] = meta["patient_id"]
    meta["source"] = "PAD-UFES-20"
    return meta.dropna(subset=["path"])[["image", "path", "label", "group", "source"]]


def group_split(df, val_frac=0.15, test_frac=0.15, seed=42):
    """Split by lesion so the same lesion never sits in two splits."""
    rng = np.random.default_rng(seed)
    groups = np.array(df["group"].astype(str).unique(), dtype=object)
    rng.shuffle(groups)
    n = len(groups)
    n_test, n_val = int(n * test_frac), int(n * val_frac)
    test_g, val_g = set(groups[:n_test]), set(groups[n_test:n_test + n_val])
    g = df["group"].astype(str)
    df = df.copy()
    df["split"] = np.where(g.isin(test_g), "test",
                           np.where(g.isin(val_g), "val", "train"))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--isic", required=True, help="folder with ISIC 2019 images + csvs")
    ap.add_argument("--ham", required=True, help="folder with HAM10000 images + metadata")
    ap.add_argument("--pad", required=True, help="folder with PAD-UFES-20 images + metadata.csv")
    ap.add_argument("--out", default="data/splits.csv")
    ap.add_argument("--keep_ham_overlap", action="store_true",
                    help="ablation only: keep HAM10000 images in ISIC training (leaky setup many papers use)")
    args = ap.parse_args()

    ham = load_ham(args.ham)
    ham_ids = set(pd.read_csv(find_file(args.ham, "HAM10000_metadata.csv"))["image_id"])
    if args.keep_ham_overlap:
        print("WARNING: leaky ablation - HAM10000 images stay in the ISIC 2019 training data")
        ham_ids = set()
    isic = group_split(load_isic2019(args.isic, ham_ids))
    ham["split"] = "external"
    pad = load_pad(args.pad)
    pad["split"] = "external"

    df = pd.concat([isic, ham, pad], ignore_index=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.groupby(["source", "split", "label"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
