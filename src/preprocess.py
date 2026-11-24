"""Resize every image once (short side 256) so training is not slowed by
decoding 1024px JPEGs. Also stores a Shades-of-Gray colour-normalised copy."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from colorconst import shades_of_gray


def _resize(img, short=256):
    w, h = img.size
    s = short / min(w, h)
    return img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)


def work(args):
    src, dst_raw, dst_sog = args
    if Path(dst_raw).exists() and Path(dst_sog).exists():
        return
    img = _resize(Image.open(src).convert("RGB"))
    img.save(dst_raw, quality=95)
    Image.fromarray(shades_of_gray(np.asarray(img))).save(dst_sog, quality=95)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="data/splits.csv")
    ap.add_argument("--out", default="data/cache")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    df = pd.read_csv(a.splits)
    raw, sog = Path(a.out) / "raw", Path(a.out) / "sog"
    raw.mkdir(parents=True, exist_ok=True)
    sog.mkdir(parents=True, exist_ok=True)
    names = df["source"] + "__" + df["image"].astype(str) + ".jpg"
    jobs = [(p, str(raw / n), str(sog / n)) for p, n in zip(df["path"], names)]
    with ProcessPoolExecutor(a.workers) as ex:
        for i, _ in enumerate(ex.map(work, jobs, chunksize=64)):
            if i % 2000 == 0:
                print(f"{i}/{len(jobs)}", flush=True)
    df["cache_raw"] = [j[1] for j in jobs]
    df["cache_sog"] = [j[2] for j in jobs]
    df.to_csv(a.splits, index=False)
    print("done", len(df))


if __name__ == "__main__":
    main()
