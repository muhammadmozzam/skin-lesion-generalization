"""One-command runner for Kaggle (also works on Colab: set SLG_WORK and SLG_INPUT).

  python run_kaggle.py --stage prep     # find datasets, build splits, resize images
  python run_kaggle.py --stage check    # 1-epoch tiny run to test the setup
  python run_kaggle.py --stage train --models efficientnet_b0,resnet50,deit_small \
                       --fixes none,coloraug,sog --seeds 0 --epochs 10
  python run_kaggle.py --stage summary
  python run_kaggle.py --stage extras --best efficientnet_b0_coloraug_s0
  python run_kaggle.py --stage calibrate  # temperature scaling fitted on ISIC validation
  python run_kaggle.py --stage leaky --seeds 0,1,2  # ablation: keep HAM10000 inside ISIC training
  python run_kaggle.py --stage pack     # results.zip without weights/images
"""
import argparse
import itertools
import os
import subprocess
import sys

W = os.environ.get("SLG_WORK", "/kaggle/working")
INPUT = os.environ.get("SLG_INPUT", "/kaggle/input")
SRC = os.path.join(W, "skin-lesion-generalization", "src")
SPLITS = os.path.join(W, "data", "splits.csv")
RES = os.path.join(W, "results")


def sh(*args):
    print(">>", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], cwd=SRC, check=True)


def dataset_root(test):
    """Folder of the attached dataset that holds a file matching `test`:
    INPUT/<slug> or, in Kaggle's newer layout, INPUT/datasets/<owner>/<slug>."""
    for d, _, files in os.walk(INPUT):
        for f in files:
            if test(d, f):
                rel = os.path.relpath(d, INPUT).split(os.sep)
                keep = 3 if rel[0] == "datasets" and len(rel) >= 3 else 1
                return os.path.join(INPUT, *rel[:keep])
    return None


def is_pad(d, f):
    if f != "metadata.csv":
        return False
    with open(os.path.join(d, f)) as fh:
        return "diagnostic" in fh.readline()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["prep", "check", "train", "summary", "extras", "calibrate", "leaky", "pack"])
    ap.add_argument("--models", default="efficientnet_b0,resnet50,deit_small")
    ap.add_argument("--fixes", default="none,coloraug,sog")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--epochs", default="10")
    ap.add_argument("--best", default="")
    a = ap.parse_args()

    if a.stage == "prep":
        isic = dataset_root(lambda d, f: f == "ISIC_2019_Training_GroundTruth.csv")
        ham = dataset_root(lambda d, f: f == "HAM10000_metadata.csv")
        pad = dataset_root(is_pad)
        print("ISIC:", isic, "\nHAM:", ham, "\nPAD:", pad)
        assert isic and ham and pad, "Attach all three datasets with Add Input"
        sh("data.py", "--isic", isic, "--ham", ham, "--pad", pad, "--out", SPLITS)
        sh("preprocess.py", "--splits", SPLITS, "--out", os.path.join(W, "data", "cache"), "--workers", "4")
    elif a.stage == "check":
        sh("train.py", "--splits", SPLITS, "--model", "efficientnet_b0", "--fix", "none",
           "--seed", "99", "--epochs", "1", "--limit", "1500", "--out", os.path.join(W, "debug"))
    elif a.stage == "train":
        for m, f, s in itertools.product(a.models.split(","), a.fixes.split(","), a.seeds.split(",")):
            if os.path.exists(os.path.join(RES, f"{m}_{f}_s{s}", "metrics.csv")):
                print("skip (done):", m, f, s)
                continue
            sh("train.py", "--splits", SPLITS, "--model", m, "--fix", f, "--seed", s,
               "--epochs", a.epochs, "--out", RES)
    elif a.stage == "summary":
        sh("summarize.py", "--results", RES)
    elif a.stage == "extras":
        run = os.path.join(RES, a.best)
        sh("gradcam.py", "--run", run, "--splits", SPLITS)
        sh("export_onnx.py", "--run", run, "--splits", SPLITS)
    elif a.stage == "calibrate":
        sh("calibrate.py", "--results", RES, "--splits", SPLITS)
    elif a.stage == "leaky":
        isic = dataset_root(lambda d, f: f == "ISIC_2019_Training_GroundTruth.csv")
        ham = dataset_root(lambda d, f: f == "HAM10000_metadata.csv")
        pad = dataset_root(is_pad)
        leaky = os.path.join(W, "data", "splits_leaky.csv")
        sh("data.py", "--isic", isic, "--ham", ham, "--pad", pad, "--out", leaky, "--keep_ham_overlap")
        sh("preprocess.py", "--splits", leaky, "--out", os.path.join(W, "data", "cache"), "--workers", "4")
        out = os.path.join(W, "results_leaky")
        for s in a.seeds.split(","):
            if not os.path.exists(os.path.join(out, f"efficientnet_b0_none_s{s}", "metrics.csv")):
                sh("train.py", "--splits", leaky, "--model", "efficientnet_b0", "--fix", "none",
                   "--seed", s, "--epochs", a.epochs, "--out", out)
        sh("summarize.py", "--results", out)
    elif a.stage == "pack":
        subprocess.run(f'cd {W} && zip -qr results.zip $(ls -d results results_leaky 2>/dev/null) -x "*.pt" "*.onnx" && ls -lh results.zip',
                       shell=True, check=True)


if __name__ == "__main__":
    main()
