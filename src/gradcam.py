"""Save Grad-CAM heatmaps for a few correct and wrong predictions per dataset."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from train import MODELS, build_tf


def target_layer(model, name):
    if name == "resnet50":
        return [model.layer4[-1]]
    if name == "efficientnet_b0":
        return [model.conv_head]
    return [model.blocks[-1].norm1]  # DeiT


def vit_reshape(t, h=14, w=14):
    t = t[:, -h * w:, :].reshape(t.size(0), h, w, t.size(2))
    return t.permute(0, 3, 1, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="results/<run> folder")
    ap.add_argument("--splits", default="data/splits.csv")
    ap.add_argument("--n", type=int, default=4)
    a = ap.parse_args()
    run = Path(a.run)
    import json
    args = json.load(open(run / "args.json"))
    model = timm.create_model(MODELS[args["model"]], pretrained=False, num_classes=2)
    model.load_state_dict(torch.load(run / "best.pt", map_location="cpu"))
    model.eval()
    cam = GradCAM(model=model, target_layers=target_layer(model, args["model"]),
                  reshape_transform=vit_reshape if args["model"] == "deit_small" else None)

    df = pd.read_csv(a.splits).set_index("image")
    col = "cache_sog" if args["fix"] == "sog" else "cache_raw"
    preds = pd.read_csv(run / "predictions.csv")
    preds = preds[~preds.tta]
    preds["correct"] = (preds.prob >= 0.5).astype(int) == preds.label
    tf = build_tf(args["fix"], False)
    out = run / "gradcam"
    out.mkdir(exist_ok=True)
    for ds, d in preds.groupby("dataset"):
        for ok in (True, False):
            for _, r in d[d.correct == ok].sample(frac=1, random_state=0).head(a.n).iterrows():
                path = df.loc[r.image, col] if col in df else df.loc[r.image, "path"]
                if isinstance(path, pd.Series):
                    path = path.iloc[0]
                img = Image.open(path).convert("RGB")
                x = tf(img).unsqueeze(0)
                g = cam(input_tensor=x, targets=[ClassifierOutputTarget(1)])[0]
                rgb = np.asarray(img.resize((224, 224))).astype(np.float32) / 255
                vis = show_cam_on_image(rgb, g, use_rgb=True)
                tag = "correct" if ok else "wrong"
                Image.fromarray(np.hstack([(rgb * 255).astype(np.uint8), vis])).save(
                    out / f"{ds}_{tag}_{r.image}_y{r.label}_p{r.prob:.2f}.jpg")
    print("saved to", out)


if __name__ == "__main__":
    main()
