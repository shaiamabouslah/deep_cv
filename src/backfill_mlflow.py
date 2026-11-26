"""Logs the experiments already run on this project into MLflow, and builds
the Model Registry's real history.

These experiments ran before MLflow tracking was wired into the scripts, so
each backfilled run is tagged `source=backfill` and `backfilled_from=<file>`:
the numbers are read from the result files the experiments wrote (nothing is
re-trained or invented), and the model evaluations are recomputed from the
saved checkpoints. Runs that already exist are skipped, so it is safe to
re-run. From now on the training scripts log their own runs.

    python -m src.backfill_mlflow
"""
import os

from src import registry, tracking
from src.evaluate import score_checkpoint

SEG_EXP, SAR_EXP, FORECAST_EXP, CALIB_EXP = (
    "segmentation-water", "segmentation-sar-radar", "forecasting-water-level", "fill-rate-calibration")
SEGMENTATION_MODELS = [
    # (name, checkpoint, registry alias, description)
    ("v1", "models/model_previous.pt", "previous",
     "v1: U-Net ResNet34 (ImageNet init), 10 epochs on the Kaggle water-bodies dataset, Dice loss, no augmentation. "
     "Served until 2026-11-26. Its training metrics were never recorded (the Kaggle run was not kept)."),
    ("v2", "models/model_v2.pt", "rejected",
     "v2: same architecture, Kaggle retrain with augmentation, Dice+BCE, 30 epochs, best-epoch selection. Better on "
     "the Kaggle test set (IoU 0.81) but REJECTED: collapsed on a bright dry-season Tunisian scene (Tunisia IoU 0.56) "
     "and no better than v1 worldwide."),
    ("v3", "models/model.pt", "champion",
     "v3: v1 fine-tuned on 306 worldwide Sentinel-2 tiles with colour and scale augmentation. PROMOTED by the gate: "
     "Tunisia IoU 0.76 (+0.035), world benchmark IoU 0.57 (+0.12), gain largest far from training sites."),
]


def already_logged(experiment, run_name, tracking_uri=None):
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(tracking_uri or tracking.TRACKING_URI)
    exp = mlflow.get_experiment_by_name(experiment)
    if exp is None:
        return False
    runs = MlflowClient().search_runs([exp.experiment_id], filter_string=f"tags.mlflow.runName = '{run_name}'")
    return bool(runs)


def log_once(experiment, run_name, **kwargs):
    if already_logged(experiment, run_name):
        print("  exists:", experiment, "/", run_name)
        return None
    run_id = tracking.log_experiment(experiment, run_name, **kwargs)
    print("  logged:", experiment, "/", run_name)
    return run_id


def segmentation_runs():
    import torch

    from src import world_benchmark as wb

    print("segmentation")
    scores = {}
    for name, path, _, _ in SEGMENTATION_MODELS:
        tunisia, _ = score_checkpoint(path, "data/eval")
        _, rows = score_checkpoint(path, "data/world")
        clean = [r for r in rows if wb.tile_quality(r) is None]
        scores[name] = {
            "tunisia": {k: tunisia[k] for k in ("iou", "precision", "recall", "water_pct_mae")},
            "world": wb.pooled(clean, 1000), "world_reservoirs": wb.pooled([r for r in clean if r["type"] == "reservoir"], 1000),
            "world_lakes": wb.pooled([r for r in clean if r["type"] == "lake"], 1000),
        }
        log_once(SEG_EXP, f"evaluate {name}", params={"checkpoint": path, "tunisia_tiles": 18, "world_tiles": len(clean)},
                 metrics=scores[name], tags={"source": "recomputed from checkpoint", "model": name})

    kaggle = torch.load("models/model_v2.pt", map_location="cpu")
    kaggle_metrics = kaggle.get("metrics", {})
    log_once(SEG_EXP, "train v2 (Kaggle)", params={
        "encoder": "resnet34", "init": "imagenet", "epochs": 30, "lr": 3e-4, "batch_size": 16, "loss": "dice+bce",
        "scheduler": "cosine", "augmentation": "nodata strip, flips, rot90", "seed": 42, "image_size": 256,
        "dataset": "franciscoescobar/satellite-images-of-water-bodies"},
        metrics={k: v for k, v in kaggle_metrics.items() if isinstance(v, (int, float))},
        tags={"source": "backfill", "backfilled_from": "models/model_v2.pt (metrics saved by the Kaggle notebook)",
              "model": "v2"})

    v3 = torch.load("models/model.pt", map_location="cpu")["metrics"]
    log_once(SEG_EXP, "finetune v3 (worldwide tiles)", params={
        "init": v3.get("init"), "epochs": 12, "lr": 1e-4, "batch_size": 8, "loss": "masked dice+bce",
        "color_augmentation": v3.get("color_augmentation"), "zoom_crop": "0.45-1.0", "n_train_tiles": v3.get("n_train_tiles"),
        "train_sites": "371 HydroLAKES lakes/reservoirs, 69 countries, disjoint from the benchmark"},
        metrics={**scores["v3"]}, curves={"val_iou": v3.get("val_iou_curve", [])},
        tags={"source": "backfill", "backfilled_from": "models/model.pt (checkpoint metrics) + recomputed scores", "model": "v3"})
    return scores


def register_models(scores):
    print("registry")
    existing = {v["description"][:2] for v in registry.lineage(registry.SEGMENTATION)} if _has_model(registry.SEGMENTATION) else set()
    for name, path, alias, description in SEGMENTATION_MODELS:
        if name in existing:
            print("  exists:", registry.SEGMENTATION, name)
            continue
        tags = {"tunisia_iou": scores[name]["tunisia"]["iou"], "world_iou": scores[name]["world"]["iou"],
                "world_reservoir_iou": scores[name]["world_reservoirs"]["iou"],
                "world_lake_iou": scores[name]["world_lakes"]["iou"], "model_name": name}
        version = registry.register_segmentation(path, description, tags, metrics=scores[name], alias=alias)
        print(f"  registered {registry.SEGMENTATION} v{version} ({name}) alias={alias}")

    if not _has_model(registry.SAR) and os.path.exists("models/sar_unet.pt"):
        sar = tracking.read_json("reports/sar_eval.json")
        version = registry.register_segmentation(
            "models/sar_unet.pt",
            "Radar (Sentinel-1 VV/VH) water U-Net ResNet18, ImageNet init, trained on 323 radar patches with optical "
            "labels. Held-out Tunisia IoU 0.87 vs 0.80 for the best classical baseline.",
            {"test_tunisia_iou": sar["unet_imagenet_init"]["test_tunisia"]["iou"],
             "test_world_iou": sar["unet_imagenet_init"]["test_world"]["iou"]},
            name=registry.SAR, alias="champion", loader=registry.load_radar)
        print(f"  registered {registry.SAR} v{version}")

    if not _has_model(registry.FORECASTER) and os.path.exists("models/forecaster.pt"):
        import torch

        ck = torch.load("models/forecaster.pt", map_location="cpu")
        version = registry.register_forecaster(
            "models/forecaster.pt",
            "GRU ensemble (5 seeds, weather history), +1/+2/+4 weeks. Test MAE (pts) vs no-change baseline: "
            f"{ck['test_mae']} vs {ck['persistence_mae']}.",
            {f"test_mae_{k}": v for k, v in ck["test_mae"].items()} | {f"baseline_mae_{k}": v for k, v in ck["persistence_mae"].items()},
            metrics={"test_mae": ck["test_mae"], "persistence_mae": ck["persistence_mae"]}, alias="champion")
        print(f"  registered {registry.FORECASTER} v{version}")


def _has_model(name):
    import mlflow
    from mlflow import MlflowClient
    mlflow.set_tracking_uri(tracking.TRACKING_URI)
    return any(m.name == name for m in MlflowClient().search_registered_models())


def sar_runs():
    print("radar")
    results = tracking.read_json("reports/sar_eval.json")
    data = results.get("data", {})
    for method, metrics in results.items():
        if method == "data":
            continue
        curve = metrics.pop("val_curve", None) if isinstance(metrics, dict) else None
        is_unet = method.startswith("unet")
        params = {"method": method, "train_patches": data.get("train", {}).get("patches"),
                  "train_sites": data.get("train", {}).get("sites")}
        if is_unet:
            params.update({"encoder": "resnet18", "init": "imagenet" if "imagenet" in method else "random",
                           "epochs": 25, "lr": 3e-4, "batch_size": 8, "loss": "masked dice+bce",
                           "augmentation": "flips, rot90"})
        log_once(SAR_EXP, method, params=params, metrics=metrics, curves={"val_iou": curve} if curve else None,
                 tags={"source": "backfill", "backfilled_from": "reports/sar_eval.json", "task": "radar water segmentation"})
    if os.path.exists("reports/sar_gapfill.json"):
        gap = tracking.read_json("reports/sar_gapfill.json")
        for name, report in gap.items():
            log_once(SAR_EXP, f"gapfill {name}", params={"predictor": name, "official_readings": "ONAGRI 2024-2025"},
                     metrics=report, tags={"source": "backfill", "backfilled_from": "reports/sar_gapfill.json",
                                           "task": "radar-only fill-rate estimate vs official"})


def forecaster_runs():
    print("forecaster")
    report = tracking.read_json("reports/forecast_eval.json")
    params = {"window_weeks": 12, "horizons": "1,2,4", "train_end": report["split"]["train_end"],
              "val_end": report["split"]["val_end"], "seeds": report["seeds"], "n_windows": report["n_windows"],
              "n_test_windows": report["n_test"], "model": "GRU hidden 32, ~8k parameters", "loss": "masked L1"}
    for protocol in ("temporal_holdout", "leave_one_dam_out"):
        for method, metrics in report[protocol].items():
            log_once(FORECAST_EXP, f"{protocol} / {method}", params={**params, "protocol": protocol, "method": method},
                     metrics=metrics, tags={"source": "backfill", "backfilled_from": "reports/forecast_eval.json",
                                            "protocol": protocol})


def calibration_runs():
    print("calibration")
    calibration = tracking.read_json("data/official/calibration.json")
    for dam, entry in calibration["dams"].items():
        log_once(CALIB_EXP, dam, params={"dam": dam, "method": "isotonic regression anchored at (0,0),(100,100)",
                                         "cv": "5 contiguous time blocks", "official_source": "ONAGRI bulletins 2024-01..2025-10"},
                 metrics={"n_points": entry["n_points"], "usable": entry["usable"], "typical_error_pts": entry["typical_error_pts"],
                          "cv": entry["cv"] or {}},
                 tags={"source": "backfill", "backfilled_from": "data/official/calibration.json"})


if __name__ == "__main__":
    scores = segmentation_runs()
    register_models(scores)
    sar_runs()
    forecaster_runs()
    calibration_runs()
    print("done; browse with: mlflow ui --backend-store-uri sqlite:///mlflow.db")
