"""Evaluates the trained segmentation model against a small local sample
(data/samples/images + data/samples/masks).

Those 5 fixtures are SYNTHETIC (a rectangle on a flat background, generated
deterministically — see the generation snippet in the project history),
not real satellite imagery. They exist to validate the pipeline mechanics
end-to-end (model loads, inference runs, IoU computes correctly, DVC/CI
report a real metrics.json instead of silently skipping) — the mean IoU
they produce is NOT a measure of real-world segmentation accuracy. The
actual accuracy claim comes from the Kaggle training/validation run (see
notebooks/kaggle_train_segmentation.ipynb and its MLflow run)."""
import json
import os

import numpy as np
import torch
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

        tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            pred = (torch.sigmoid(model(tensor)) > 0.5).float()
        true = (torch.from_numpy(np.array(mask)).reshape(1, 1, image_size, image_size) > 127).float()
        ious.append(iou_score(pred, true))

    metrics = {
        "status": "ok",
        "mean_iou": sum(ious) / len(ious),
        "n_samples": len(ious),
        "note": "synthetic smoke-test fixtures — validates pipeline mechanics, not real-world accuracy",
    }
    with open(params["paths"]["metrics_out"], "w") as f:
        json.dump(metrics, f, indent=2)
    print(metrics)
    return metrics


if __name__ == "__main__":
    evaluate()
