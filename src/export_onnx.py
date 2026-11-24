"""Export the best model to ONNX, quantise to 8-bit, and compare size,
CPU speed and AUC on PAD-UFES-20 before and after."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import timm
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from PIL import Image

from metrics import all_metrics
from train import MODELS, build_tf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--splits", default="data/splits.csv")
    ap.add_argument("--eval_n", type=int, default=500)
    a = ap.parse_args()
    run = Path(a.run)
    args = json.load(open(run / "args.json"))
    model = timm.create_model(MODELS[args["model"]], pretrained=False, num_classes=2)
    model.load_state_dict(torch.load(run / "best.pt", map_location="cpu"))
    model.eval()

    fp32, int8 = run / "model.onnx", run / "model_int8.onnx"
    torch.onnx.export(model, torch.randn(1, 3, 224, 224), fp32, input_names=["image"],
                      output_names=["logits"], dynamic_axes={"image": {0: "b"}, "logits": {0: "b"}},
                      opset_version=17, dynamo=False)
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)

    df = pd.read_csv(a.splits)
    d = df[df.source == "PAD-UFES-20"].sample(min(a.eval_n, (df.source == "PAD-UFES-20").sum()), random_state=0)
    col = "cache_sog" if args["fix"] == "sog" else "cache_raw"
    col = col if col in d else "path"
    tf = build_tf(args["fix"], False)
    xs = np.stack([tf(Image.open(p).convert("RGB")).numpy() for p in d[col]])

    rows = []
    for name, f in [("fp32", fp32), ("int8", int8)]:
        s = ort.InferenceSession(str(f), providers=["CPUExecutionProvider"])
        t0 = time.time()
        logits = np.concatenate([s.run(None, {"image": xs[i:i + 1]})[0] for i in range(len(xs))])
        ms = (time.time() - t0) / len(xs) * 1000
        e = np.exp(logits - logits.max(1, keepdims=True))
        p = (e / e.sum(1, keepdims=True))[:, 1]
        rows.append({"format": name, "size_mb": f.stat().st_size / 1e6, "cpu_ms_per_image": ms,
                     **all_metrics(d.label.values, p)})
    r = pd.DataFrame(rows)
    r.to_csv(run / "onnx_compare.csv", index=False)
    print(r.round(4))


if __name__ == "__main__":
    main()
