"""Trains and evaluates the radar water model against its baselines.

Splits (see src/sar_model.split_sites): world sites split by site into
train / val / test_world, and the six Tunisian dams are held out entirely
(`test_tunisia`): the model never sees Tunisia, which is exactly where it is
meant to be used. Every method is scored on the same patches, pooled over
valid pixels only.

Compared: four classical baselines (VV threshold, median-filtered VV
threshold, VH threshold, gradient-boosted trees on hand-made features) and the
U-Net, initialised from ImageNet or from scratch (does transfer learning help
on radar, a very different image type?).

    python -m src.train_sar
"""
import json
import os

import torch

from src import sar_model as sm

MODEL_PATH = "models/sar_unet.pt"
REPORT_PATH = "reports/sar_eval.json"
EPOCHS = 25
SPLITS = ("val", "test_world", "test_tunisia")


def run(epochs=EPOCHS, sar_dir=sm.SAR_DIR, report_path=REPORT_PATH, model_path=MODEL_PATH, track=False):
    manifest = sm.load_manifest(sar_dir)
    splits = sm.split_sites(manifest)
    print({k: f"{len(v)} patches / {len({r['site_id'] for r in v})} sites" for k, v in splits.items()}, flush=True)

    results = {}
    baselines = {
        "threshold_vv": sm.ThresholdBaseline("vv"), "threshold_vv_median": sm.ThresholdBaseline("vv", median=True),
        "threshold_vh": sm.ThresholdBaseline("vh"), "gbm_features": sm.GbmBaseline(),
    }
    for name, baseline in baselines.items():
        baseline.fit(splits["train"], sar_dir)
        results[name] = {s: sm.evaluate_baseline(baseline, splits[s], sar_dir) for s in SPLITS}
        if hasattr(baseline, "threshold"):
            results[name]["threshold_db"] = baseline.threshold
        print(name, results[name], flush=True)

    best_model, best_val = None, -1.0
    for name, pretrained in (("unet_imagenet_init", True), ("unet_random_init", False)):
        model, history = sm.train(splits["train"], splits["val"], epochs=epochs, sar_dir=sar_dir, pretrained=pretrained)
        results[name] = {s: sm.evaluate_model(model, splits[s], sar_dir) for s in SPLITS}
        results[name]["best_val_iou"] = max(history)
        results[name]["val_curve"] = history
        print(name, {k: v for k, v in results[name].items() if k != "val_curve"}, flush=True)
        if max(history) > best_val:
            best_model, best_val, best_name = model, max(history), name

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save({"state_dict": best_model.state_dict(), "encoder": "resnet18", "variant": best_name,
                "metrics": {s: results[best_name][s] for s in SPLITS}}, model_path)
    results["data"] = {k: {"patches": len(v), "sites": len({r["site_id"] for r in v})} for k, v in splits.items()}
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print("saved", model_path, report_path)
    if track:
        log_run(results, best_name, epochs, model_path, report_path)
    return results


def log_run(results, best_name, epochs, model_path, report_path):
    from src import registry, tracking

    metrics = {k: {m: v for m, v in r.items() if m != "val_curve"} for k, r in results.items() if k != "data"}
    tracking.log_experiment(
        "segmentation-sar-radar", f"train {best_name}",
        params={"encoder": "resnet18", "epochs": epochs, "lr": 3e-4, "batch_size": 8, "loss": "masked dice+bce",
                "best_variant": best_name, **{f"n_{k}": v["patches"] for k, v in results["data"].items()}},
        metrics=metrics, curves={f"val_iou.{n}": results[n]["val_curve"] for n in ("unet_imagenet_init", "unet_random_init")},
        artifacts=[report_path], tags={"source": "train_sar"})
    best = results[best_name]
    version = registry.register_segmentation(
        model_path, f"Radar U-Net ({best_name}); held-out Tunisia IoU {best['test_tunisia']['iou']}.",
        {"test_tunisia_iou": best["test_tunisia"]["iou"], "test_world_iou": best["test_world"]["iou"]},
        name=registry.SAR, loader=registry.load_radar)
    promoted = registry.promote_if_better(registry.SAR, version, "test_tunisia_iou")
    print(f"registered {registry.SAR} v{version}; champion: {promoted}")


if __name__ == "__main__":
    run(track=True)
