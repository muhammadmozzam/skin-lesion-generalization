"""Combine every run into one table: mean ± std over seeds, plus the drop
from the in-domain ISIC test to each outside dataset."""
import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    files = list(Path(a.results).glob("*/metrics.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df.to_csv(Path(a.results) / "all_runs.csv", index=False)

    keys = ["model", "fix", "tta", "dataset"]
    cols = ["auc", "bal_acc", "f1", "sensitivity", "specificity", "ece"]
    g = df.groupby(keys)[cols]
    mean, std, n = g.mean(), g.std().fillna(0), g.size()
    table = mean.round(3).astype(str) + " ± " + std.round(3).astype(str)
    table["seeds"] = n
    table.reset_index().to_csv(Path(a.results) / "summary.csv", index=False)

    piv = mean["auc"].unstack("dataset")
    for ext in ["HAM10000", "PAD-UFES-20"]:
        if ext in piv and "ISIC2019-test" in piv:
            piv[f"AUC drop -> {ext}"] = piv["ISIC2019-test"] - piv[ext]
    piv.round(3).to_csv(Path(a.results) / "auc_drop.csv")
    print(piv.round(3).to_string())


if __name__ == "__main__":
    main()
