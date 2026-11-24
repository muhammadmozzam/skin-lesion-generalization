"""Train on ISIC 2019 (without HAM10000), pick the checkpoint on the ISIC
validation split only, then test on ISIC test + HAM10000 + PAD-UFES-20.

Fixes compared (--fix):
  none      standard flips / rotations / crops
  coloraug  + strong colour jitter, gamma, random grayscale
  sog       Shades-of-Gray colour normalisation on train AND test images
Test-time augmentation (8 flips/rotations) is always reported as an extra row.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from metrics import all_metrics

MODELS = {
    "resnet50": "resnet50.a1_in1k",
    "efficientnet_b0": "efficientnet_b0.ra_in1k",
    "deit_small": "deit_small_patch16_224.fb_in1k",
}
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def build_tf(fix, train, size=224):
    if not train:
        return T.Compose([T.Resize(size), T.CenterCrop(size), T.ToTensor(), T.Normalize(MEAN, STD)])
    t = [T.RandomResizedCrop(size, scale=(0.6, 1.0)), T.RandomHorizontalFlip(),
         T.RandomVerticalFlip(), T.RandomRotation(90)]
    if fix == "coloraug":
        t += [T.ColorJitter(0.4, 0.4, 0.4, 0.1), T.RandomGrayscale(0.1),
              T.RandomApply([T.GaussianBlur(5)], p=0.2)]
    else:
        t += [T.ColorJitter(0.1, 0.1, 0.1, 0.0)]
    return T.Compose(t + [T.ToTensor(), T.Normalize(MEAN, STD)])


class ImgDS(Dataset):
    def __init__(self, df, tf, col):
        self.p, self.y, self.tf = df[col].tolist(), df["label"].tolist(), tf

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        return self.tf(Image.open(self.p[i]).convert("RGB")), self.y[i]


@torch.no_grad()
def predict(model, loader, dev, tta=False):
    model.eval()
    out = []
    for x, _ in loader:
        x = x.to(dev, non_blocking=True)
        views = [x]
        if tta:
            views = [torch.rot90(v, k, (2, 3)) for v in (x, x.flip(3)) for k in range(4)]
        with torch.autocast("cuda", enabled=dev == "cuda"):
            pr = torch.stack([model(v).float().softmax(1)[:, 1] for v in views]).mean(0)
        out.append(pr.cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="data/splits.csv")
    ap.add_argument("--model", default="efficientnet_b0", choices=MODELS)
    ap.add_argument("--fix", default="none", choices=["none", "coloraug", "sog"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="debug: use only N train images")
    ap.add_argument("--no_pretrained", action="store_true", help="offline testing only")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    set_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    run = f"{a.model}_{a.fix}_s{a.seed}"
    out = Path(a.out) / run
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(a.splits)
    col = "cache_sog" if a.fix == "sog" else "cache_raw"
    if col not in df:
        col = "path"
    tr = df[df.split == "train"]
    if a.limit:
        tr = tr.sample(min(a.limit, len(tr)), random_state=a.seed)
    va = df[df.split == "val"]

    mk = lambda d, train: DataLoader(ImgDS(d, build_tf(a.fix, train), col), batch_size=a.bs,
                                     shuffle=train, num_workers=a.workers, pin_memory=True,
                                     drop_last=train and len(d) > a.bs)
    tl, vl = mk(tr, True), mk(va, False)

    model = timm.create_model(MODELS[a.model], pretrained=not a.no_pretrained, num_classes=2).to(dev)
    pos = tr.label.mean()
    w = torch.tensor([0.5 / (1 - pos), 0.5 / pos], dtype=torch.float32, device=dev)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * len(tl), pct_start=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=dev == "cuda")

    best, log = -1.0, []
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); tot = 0.0
        for x, y in tl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast("cuda", enabled=dev == "cuda"):
                loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item()
        m = all_metrics(va.label, predict(model, vl, dev))
        log.append({"epoch": ep, "loss": tot / len(tl), **{f"val_{k}": v for k, v in m.items()}})
        print(f"[{run}] ep{ep} loss {tot/len(tl):.4f} val_auc {m['auc']:.4f} "
              f"val_bacc {m['bal_acc']:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if m["auc"] > best:  # selection uses ISIC validation only
            best = m["auc"]
            torch.save(model.state_dict(), out / "best.pt")
    pd.DataFrame(log).to_csv(out / "train_log.csv", index=False)

    model.load_state_dict(torch.load(out / "best.pt", map_location=dev))
    rows, preds = [], []
    sets = {"ISIC2019-test": df[df.split == "test"],
            "HAM10000": df[df.source == "HAM10000"],
            "PAD-UFES-20": df[df.source == "PAD-UFES-20"]}
    for name, d in sets.items():
        loader = mk(d, False)
        for tta in (False, True):
            p = predict(model, loader, dev, tta)
            rows.append({"run": run, "model": a.model, "fix": a.fix, "seed": a.seed,
                         "tta": tta, "dataset": name, **all_metrics(d.label, p)})
            preds.append(pd.DataFrame({"image": d.image.values, "dataset": name, "tta": tta,
                                       "label": d.label.values, "prob": p}))
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    pd.concat(preds).to_csv(out / "predictions.csv", index=False)
    json.dump(vars(a), open(out / "args.json", "w"), indent=2)
    print(pd.DataFrame(rows)[["dataset", "tta", "auc", "bal_acc", "f1", "ece"]].round(4))


if __name__ == "__main__":
    main()
