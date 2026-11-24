"""Gradio demo for Hugging Face Spaces. Put model_int8.onnx next to this file."""
import gradio as gr
import numpy as np
import onnxruntime as ort
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
sess = ort.InferenceSession("model_int8.onnx", providers=["CPUExecutionProvider"])


def prep(img):
    w, h = img.size
    s = 224 / min(w, h)
    img = img.resize((round(w * s), round(h * s)), Image.BICUBIC)
    w, h = img.size
    l, t = (w - 224) // 2, (h - 224) // 2
    x = np.asarray(img.crop((l, t, l + 224, t + 224)), dtype=np.float32) / 255
    return ((x - MEAN) / STD).transpose(2, 0, 1)[None]


def classify(img):
    logits = sess.run(None, {"image": prep(img.convert("RGB"))})[0][0]
    e = np.exp(logits - logits.max())
    p = e / e.sum()
    return {"Benign": float(p[0]), "Malignant (MEL / BCC / SCC)": float(p[1])}


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil", label="Skin lesion photo"),
    outputs=gr.Label(num_top_classes=2),
    title="Skin lesion classifier: cross-dataset study",
    description=("**Research prototype, not for diagnosis.** Trained on ISIC 2019 dermoscopy "
                 "images and tested on HAM10000 and PAD-UFES-20 smartphone photos. "
                 "Always see a dermatologist about a skin lesion."),
)

if __name__ == "__main__":
    demo.launch()
