"""Serving side of the deep-learning forecaster (training: src/train_forecaster.py).

For a site it fetches the last weeks of satellite water area and weather from
Earth Engine with the very code that built the training set, turns them into
the model input, and returns the ensemble's prediction for +1, +2 and +4
weeks. Results are cached per site: the underlying data changes weekly.

Honesty notes carried in the result: each horizon comes with the typical error
measured on held-out weeks (not the spread between ensemble members, which is
tiny and would understate the real uncertainty), and the error of simply
assuming "no change", the baseline the model has to beat.
"""
import datetime
import os
import time

import numpy as np
import pandas as pd
import torch

from src import build_forecast_dataset as dataset
from src import forecast_model as fm
from src.alerts import ALERT_TIERS, classify_alert_level

CHECKPOINT_PATH = "models/forecaster.pt"
CACHE_TTL_S = 6 * 3600
LOOKBACK_WEEKS = fm.WINDOW + fm.MAX_FFILL + 3

_bundle = {"path": None, "value": None}
_cache = {}


def load_bundle(path=CHECKPOINT_PATH):
    if not os.path.exists(path):
        return None
    checkpoint = torch.load(path, map_location="cpu")
    models = []
    for state in checkpoint["state_dicts"]:
        model = fm.GRUForecaster(checkpoint["n_features"], checkpoint["n_future"])
        model.load_state_dict(state)
        models.append(model.eval())
    return {"models": models, **{k: checkpoint[k] for k in (
        "stats", "n_future", "horizons", "test_mae", "persistence_mae", "train_end", "val_end", "n_test")}}


def get_bundle():
    if _bundle["path"] != CHECKPOINT_PATH:
        _bundle.update(path=CHECKPOINT_PATH, value=load_bundle(CHECKPOINT_PATH))
    return _bundle["value"]


def model_available():
    return get_bundle() is not None


def fetch_recent_weekly(dam, today=None, project=None):
    """Weekly table of the last LOOKBACK_WEEKS complete weeks (the current,
    unfinished week is left out so every row means the same as in training)."""
    import ee

    from src.gee_client import _ensure_initialized

    _ensure_initialized(project)
    today = today or datetime.date.today()
    end = today - datetime.timedelta(days=today.weekday())
    start = end - datetime.timedelta(weeks=LOOKBACK_WEEKS)
    region = ee.Geometry.Point([dam["lon"], dam["lat"]]).buffer(dam.get("buffer_m", 6000))
    scenes = dataset.fetch_scene_rows(region, start, end)
    rain, climate = dataset.fetch_daily_weather(region, start, end)
    return dataset.weekly_table(dam["name"], scenes, rain, climate, str(start), str(end))


def _alert_level(current, predicted, horizons):
    """Alert tier of the worst predicted drop, counting only drops larger than
    the model's typical error at that horizon. A relative drop alone would
    flag a nearly empty reservoir for a 0.7-point move that is pure noise."""
    if current <= 0:
        return None
    resolvable = [
        current - float(value)
        for value, h in zip(predicted, horizons)
        if current - float(value) > h["typical_error_pts"]
    ]
    return classify_alert_level(max(resolvable, default=0.0) / current * 100, ALERT_TIERS)


def predict(bundle, weekly):
    window = fm.latest_window(weekly, bundle["stats"], use_weather=True)
    if window is None:
        return {
            "available": False,
            "reason": "Pas assez d'images claires récentes (réservoir visible à 70 % ou plus) "
                      "pour alimenter le modèle.",
        }
    x, current, week = window
    inputs = torch.tensor(x)
    future = torch.zeros(1, bundle["n_future"])
    with torch.no_grad():
        deltas = np.stack([m(inputs, future).numpy()[0] * fm.DELTA_SCALE for m in bundle["models"]])
    predicted = np.clip(current + deltas.mean(axis=0), 0, 100)

    horizons = []
    for j, weeks_ahead in enumerate(bundle["horizons"]):
        key = f"h{weeks_ahead}"
        horizons.append({
            "weeks": weeks_ahead,
            "target_week": str((week + pd.Timedelta(weeks=weeks_ahead)).date()),
            "predicted_pct": round(float(predicted[j]), 1),
            "typical_error_pts": bundle["test_mae"][key],
            "no_change_error_pts": bundle["persistence_mae"][key],
        })

    return {
        "available": True,
        "origin_week": str(week.date()),
        "current_pct": round(current, 1),
        "horizons": horizons,
        "projected_alert_level": _alert_level(current, predicted, horizons),
        "n_models": len(bundle["models"]),
        "evaluated_on": f"{bundle['n_test']} semaines-barrages après {bundle['val_end']}",
    }


def cached_forecast(site):
    hit = _cache.get(site)
    if hit and time.time() - hit[0] < CACHE_TTL_S:
        return hit[1]
    return None


def deep_forecast(dam, project=None, fetcher=None):
    """None when no trained model is installed; otherwise the prediction, or
    {"available": False, "reason": ...} when the recent data is not usable."""
    bundle = get_bundle()
    if bundle is None:
        return None
    hit = cached_forecast(dam["name"])
    if hit is not None:
        return hit
    weekly = (fetcher or fetch_recent_weekly)(dam, project=project)
    result = predict(bundle, weekly)
    if result["available"]:
        _cache[dam["name"]] = (time.time(), result)
    return result
