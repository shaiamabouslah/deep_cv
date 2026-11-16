"""Evaluates the trained segmentation model against a small local labeled
sample (data/samples/images + data/samples/masks), if present. The full
training/validation set lives on Kaggle (see notebooks/); this stage is
meant for a quick local sanity check and for the DVC/CI metrics report."""
import json
import os

from PIL import Image

from src.data import build_transforms, load_params
from src.metrics import iou_score
from src.model import load_model


def evaluate(params_path="params.yaml"):
    params = load_params(params_path)
    model_path = params["model"]["model_path"]
    samples_dir = "data/samples"

    os.makedirs(os.path.dirname(params["paths"]["metrics_out"]), exist_ok=True)

    if not os.path.exists(model_path) or not os.path.isdir(os.path.join(samples_dir, "images")):
        metrics = {"status": "skipped", "reason": "missing model.pt or data/samples/"}
        with open(params["paths"]["metrics_out"], "w") as f:
            json.dump(metrics, f, indent=2)
        print(metrics)
        return metrics

    model, image_size = load_model(model_path)
    transform = build_transforms(image_size)

    image_dir = os.path.join(samples_dir, "images")
    mask_dir = os.path.join(samples_dir, "masks")

    ious = []
    for name in sorted(os.listdir(image_dir)):
        image = Image.open(os.path.join(image_dir, name)).convert("RGB")
        mask = Image.open(os.path.join(mask_dir, name)).convert("L").resize((image_size, image_size))

        import torch
        tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            pred = (torch.sigmoid(model(tensor)) > 0.5).float()
        true = (torch.tensor(list(mask.getdata())).reshape(1, 1, image_size, image_size) > 127).float()
        ious.append(iou_score(pred, true))

    metrics = {"status": "ok", "mean_iou": sum(ious) / len(ious), "n_samples": len(ious)}
    with open(params["paths"]["metrics_out"], "w") as f:
        json.dump(metrics, f, indent=2)
    print(metrics)
    return metrics


if __name__ == "__main__":
    evaluate()
