"""Evaluates the segmentation model on the real evaluation set in data/eval/
(Sentinel-2 tiles of the monitored dams at three seasons, NDWI-labelled —
see src/build_eval_set.py), with exactly the serving preprocessing: plain
resize, the checkpoint's threshold, no-data pixels excluded.

Scored only on pixels that are neither cloud nor no-data:
  - iou: dataset-level IoU (total intersection / total union), not a mean
    of per-image IoUs, which over-weights tiles with almost no water
  - precision / recall of the water class
  - water_pct_mae: mean absolute error, in percentage points, of each
    tile's water share — the number the app actually shows to users
Per-image details go to reports/eval_per_image.json.
"""
import json
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.data import build_normalize_transform, load_params
from src.model import load_model
from src.preprocess import nodata_mask

EVAL_DIR = "data/eval"
PER_IMAGE_OUT = "reports/eval_per_image.json"


def _write(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _predict(model, image, image_size, threshold):
    resized = transforms.Resize((image_size, image_size))(image)
    with torch.no_grad():
        prob = torch.sigmoid(model(build_normalize_transform()(resized).unsqueeze(0)))[0, 0].numpy()
    return prob > threshold, ~nodata_mask(np.array(resized))


def score_checkpoint(model_path, eval_dir=EVAL_DIR):
    """Scores one checkpoint on the eval set. Returns (metrics, per_image)."""
    model, image_size, threshold = load_model(model_path)
    with open(os.path.join(eval_dir, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)

    tp = fp = fn = 0
    per_image = []
    for entry in manifest:
        image = Image.open(os.path.join(eval_dir, "images", entry["file"])).convert("RGB")
        label = np.array(
            Image.open(os.path.join(eval_dir, "masks", entry["file"])).convert("L")
            .resize((image_size, image_size), Image.NEAREST)
        )
        pred, valid = _predict(model, image, image_size, threshold)
        valid &= (label < 64) | (label > 192)
        truth = (label > 192) & valid
        pred &= valid

        tile_tp = int((pred & truth).sum())
        tile_fp = int((pred & ~truth).sum())
        tile_fn = int((~pred & truth).sum())
        tp, fp, fn = tp + tile_tp, fp + tile_fp, fn + tile_fn
        n_valid = max(int(valid.sum()), 1)
        per_image.append({
            **entry,
            "true_water_pct": round(truth.sum() / n_valid * 100, 2),
            "pred_water_pct": round(pred.sum() / n_valid * 100, 2),
            "tp": tile_tp, "fp": tile_fp, "fn": tile_fn, "valid_fraction": round(float(valid.mean()), 3),
        })

    metrics = {
        "status": "ok",
        "n_images": len(per_image),
        "threshold": threshold,
        "iou": round(tp / max(tp + fp + fn, 1), 4),
        "precision": round(tp / max(tp + fp, 1), 4),
        "recall": round(tp / max(tp + fn, 1), 4),
        "water_pct_mae": round(
            float(np.mean([abs(p["pred_water_pct"] - p["true_water_pct"]) for p in per_image])), 3
        ),
    }
    return metrics, per_image


def evaluate(params_path="params.yaml", eval_dir=EVAL_DIR, per_image_out=PER_IMAGE_OUT, track=False):
    params = load_params(params_path)
    model_path = params["model"]["model_path"]
    metrics_out = params["paths"]["metrics_out"]
    os.makedirs(os.path.dirname(metrics_out), exist_ok=True)

    if not os.path.exists(model_path) or not os.path.exists(os.path.join(eval_dir, "manifest.json")):
        metrics = {"status": "skipped", "reason": "missing model.pt or data/eval/"}
        _write(metrics_out, metrics)
        print(metrics)
        return metrics

    metrics, per_image = score_checkpoint(model_path, eval_dir)
    _write(metrics_out, metrics)
    _write(per_image_out, per_image)
    print(metrics)
    if track:
        from src import tracking

        tracking.log_experiment("segmentation-water", "evaluate served model", params={"checkpoint": model_path, "eval_dir": eval_dir},
                                metrics=metrics, artifacts=[metrics_out], tags={"source": "dvc stage evaluate"})
    return metrics


if __name__ == "__main__":
    evaluate(track=True)
