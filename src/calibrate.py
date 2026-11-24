"""Post-hoc temperature scaling (Guo et al., 2017), fitted on the ISIC validation
split only, then applied unchanged to every test set. Reports ECE before/after.

Binary softmax probabilities are turned back into logits with log(p / (1 - p)),
so no retraining is needed. Runs without validation predictions (older runs) get
them recomputed from best.pt."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from metrics import all_metrics


def to_logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def nll(t, z, y):
    p = 1 / (1 + np.exp(-z / t))
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def val_predictions(run, splits):
    """Recompute ISIC validation probabilities for runs saved before val preds existed."""
    import torch
    import timm
    from torch.utils.data import DataLoader
    from train import MODELS, ImgDS, build_tf, predict
    args = json.load(open(run / "args.json"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(MODELS[args["model"]], pretrained=False, num_classes=2)
    model.load_state_dict(torch.load(run / "best.pt", map_location=dev))
    model.to(dev)
    df = pd.read_csv(splits)
    va = df[df.split == "val"]
    col = "cache_sog" if args["fix"] == "sog" else "cache_raw"
    col = col if col in va else "path"
    dl = DataLoader(ImgDS(va, build_tf(args["fix"], False), col), batch_size=64, num_workers=2)
    return va.label.values, predict(model, dl, dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--splits", default="data/splits.csv")
    a = ap.parse_args()
    rows = []
    for run in sorted(p.parent for p in Path(a.results).glob("*/predictions.csv")):
        pr = pd.read_csv(run / "predictions.csv")
        pr = pr[~pr.tta]
        v = pr[pr.dataset == "ISIC2019-val"]
        if len(v):
            yv, pv = v.label.values, v.prob.values
        elif (run / "best.pt").exists():
            yv, pv = val_predictions(run, a.splits)
        else:
            print("skip (no validation data):", run.name)
            continue
        t = minimize_scalar(nll, bounds=(0.05, 20), method="bounded", args=(to_logit(pv), yv)).x
        for ds, d in pr[pr.dataset != "ISIC2019-val"].groupby("dataset"):
            z = to_logit(d.prob.values)
            before = all_metrics(d.label.values, d.prob.values)
            after = all_metrics(d.label.values, 1 / (1 + np.exp(-z / t)))
            rows.append({"run": run.name, "dataset": ds, "temperature": t,
                         "ece_before": before["ece"], "ece_after": after["ece"],
                         "brier_before": before["brier"], "brier_after": after["brier"],
                         "auc": before["auc"]})
    out = pd.DataFrame(rows)
    out.to_csv(Path(a.results) / "calibration.csv", index=False)
    print(out.round(4).to_string())


if __name__ == "__main__":
    main()
