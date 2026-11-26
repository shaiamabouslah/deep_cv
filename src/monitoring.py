"""Model monitoring: data drift on uploaded images, prediction confidence,
and forecast accuracy over time.

What is stored: for every /predict call, a handful of numbers describing the
input (brightness, contrast, sharpness, mean colour) and the output (water
share, mean confidence). The image itself is never stored.

Data drift: the distribution of each input feature over the most recent
predictions is compared with a reference sample taken from the images the
model was trained on (data/monitoring/reference.json), with two standard
tests:
  - PSI (population stability index): < 0.1 stable, 0.1-0.25 moderate shift,
    > 0.25 major shift
  - two-sample Kolmogorov-Smirnov test
A feature is flagged when PSI is above its threshold AND the KS test rejects
equality, so a single noisy statistic does not raise an alarm.

Concept / performance drift: every deep-learning forecast is logged with the
level it started from; once the real week is observed the error is computed
and compared with the "no change" baseline. If the model's rolling error
stops beating that baseline, the model has stopped earning its place and the
continuous-training pipeline retrains it.

    python -m src.monitoring build-reference     # reference sample from the training tiles
    python -m src.monitoring simulate-drift      # replay normal then darkened images (demo)
    python -m src.monitoring check               # print the drift / performance report
"""
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats

REFERENCE_PATH = "data/monitoring/reference.json"
DEFAULT_DB = "data/monitoring.db"
FEATURES = ["brightness", "contrast", "sharpness", "red_mean", "green_mean", "blue_mean", "confidence", "water_pct"]
PSI_WARN, PSI_ALERT = 0.1, 0.25
KS_P_THRESHOLD = 0.01
MIN_SAMPLES = 30
WINDOW = 200
FORECAST_WINDOW = 8
FORECAST_TOLERANCE = 1.10


def _connect(db_path):
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(f"""CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, site TEXT, source TEXT,
        {', '.join(f'{f} REAL' for f in FEATURES)})""")
    conn.execute("""CREATE TABLE IF NOT EXISTS forecasts (
        id INTEGER PRIMARY KEY AUTOINCREMENT, made_at TEXT NOT NULL, site TEXT NOT NULL, origin_week TEXT NOT NULL,
        target_week TEXT NOT NULL, horizon_weeks INTEGER NOT NULL, base_pct REAL NOT NULL, predicted_pct REAL NOT NULL,
        source TEXT DEFAULT 'live', UNIQUE (site, origin_week, horizon_weeks, source))""")
    return conn


def image_features(rgb, sharpness, probabilities=None, valid=None):
    """Numbers describing one input image and, if given, the model's output on it."""
    rgb = np.asarray(rgb, dtype=float)
    gray = rgb.mean(axis=2)
    out = {
        "brightness": float(gray.mean()), "contrast": float(gray.std()), "sharpness": float(sharpness),
        "red_mean": float(rgb[..., 0].mean()), "green_mean": float(rgb[..., 1].mean()), "blue_mean": float(rgb[..., 2].mean()),
        "confidence": None, "water_pct": None,
    }
    if probabilities is not None:
        p = np.asarray(probabilities, dtype=float)
        keep = np.ones_like(p, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
        if keep.any():
            out["confidence"] = float((np.abs(p[keep] - 0.5) * 2).mean())
            out["water_pct"] = float((p[keep] > 0.5).mean() * 100)
    return out


def log_prediction(db_path, site, features, source="upload"):
    conn = _connect(db_path)
    conn.execute(
        f"INSERT INTO predictions (timestamp, site, source, {', '.join(FEATURES)}) VALUES (?, ?, ?, {', '.join('?' * len(FEATURES))})",
        (datetime.now(timezone.utc).isoformat(), site, source, *[features.get(f) for f in FEATURES]),
    )
    conn.commit()
    conn.close()


def recent_predictions(db_path, n=WINDOW):
    conn = _connect(db_path)
    frame = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC LIMIT ?", conn, params=(n,))
    conn.close()
    return frame


def psi(reference, current, bins=10):
    """Population stability index of `current` against `reference`, with the
    reference's quantiles as bin edges (so every bin starts equally full)."""
    reference, current = np.asarray(reference, float), np.asarray(current, float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    expected = np.histogram(reference, edges)[0] / len(reference)
    actual = np.histogram(current, edges)[0] / len(current)
    expected, actual = np.clip(expected, 1e-4, None), np.clip(actual, 1e-4, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def feature_status(psi_value, ks_p):
    if psi_value >= PSI_ALERT and ks_p < KS_P_THRESHOLD:
        return "alert"
    if psi_value >= PSI_WARN and ks_p < KS_P_THRESHOLD:
        return "warn"
    return "ok"


def drift_report(reference, current, min_samples=MIN_SAMPLES):
    """`reference` = {feature: [values]}; `current` = DataFrame of recent predictions."""
    if reference is None:
        return {"status": "no_reference", "n_current": int(len(current)), "features": {}}
    if len(current) < min_samples:
        return {"status": "insufficient_data", "n_current": int(len(current)), "min_samples": min_samples, "features": {}}
    features = {}
    for name in FEATURES:
        values = current[name].dropna().values if name in current else []
        ref = np.asarray(reference.get(name, []), float)
        if len(values) < min_samples or len(ref) < min_samples:
            continue
        p = psi(ref, values)
        ks = stats.ks_2samp(ref, values)
        features[name] = {"psi": round(p, 4), "ks_p": float(ks.pvalue), "ks_stat": round(float(ks.statistic), 4),
                          "status": feature_status(p, ks.pvalue),
                          "reference_mean": round(float(ref.mean()), 3), "current_mean": round(float(np.mean(values)), 3)}
    order = {"alert": 2, "warn": 1, "ok": 0}
    worst = max((f["status"] for f in features.values()), key=order.get, default="ok")
    return {"status": worst, "n_current": int(len(current)), "features": features,
            "flagged": sorted(n for n, f in features.items() if f["status"] != "ok")}


def load_reference(path=None):
    path = path or REFERENCE_PATH
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)["features"]


# --- forecast accuracy over time -------------------------------------------------------------

def log_forecast(db_path, site, deep, source="live"):
    """Stores each horizon of a deep-learning forecast result (src/forecast_dl.predict)."""
    conn = _connect(db_path)
    for h in deep["horizons"]:
        conn.execute(
            "INSERT OR IGNORE INTO forecasts (made_at, site, origin_week, target_week, horizon_weeks, base_pct, predicted_pct, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), site, deep["origin_week"], h["target_week"], h["weeks"],
             deep["current_pct"], h["predicted_pct"], source),
        )
    conn.commit()
    conn.close()


def forecast_performance(db_path, observed, window=FORECAST_WINDOW, tolerance=FORECAST_TOLERANCE):
    """Matures logged forecasts against `observed` (DataFrame site, week, water_pct):
    model error vs the error of assuming no change, over the last `window`
    matured target weeks per horizon. Status `degraded` when the model's
    rolling error exceeds the baseline's by more than `tolerance`."""
    conn = _connect(db_path)
    logged = pd.read_sql_query("SELECT * FROM forecasts", conn)
    conn.close()
    if logged.empty:
        return {"status": "no_forecasts", "horizons": {}}
    logged["target_week"] = pd.to_datetime(logged["target_week"])
    obs = observed.rename(columns={"week": "target_week", "water_pct": "actual"})[["site", "target_week", "actual"]]
    obs = obs.assign(target_week=pd.to_datetime(obs["target_week"]))
    matured = logged.merge(obs, on=["site", "target_week"]).dropna(subset=["actual"])
    if matured.empty:
        return {"status": "waiting_for_actuals", "n_logged": int(len(logged)), "horizons": {}}
    matured["model_error"] = (matured["predicted_pct"] - matured["actual"]).abs()
    matured["baseline_error"] = (matured["base_pct"] - matured["actual"]).abs()
    horizons, degraded = {}, False
    for horizon, group in matured.groupby("horizon_weeks"):
        recent = group.sort_values("target_week").tail(window * max(1, group["site"].nunique()))
        model, base = float(recent["model_error"].mean()), float(recent["baseline_error"].mean())
        horizons[int(horizon)] = {"n": int(len(recent)), "model_mae": round(model, 3), "no_change_mae": round(base, 3),
                                  "skill": round(1 - model / base, 3) if base > 0 else None}
        degraded = degraded or (len(recent) >= 5 and model > base * tolerance)
    return {"status": "degraded" if degraded else "ok", "n_matured": int(len(matured)), "horizons": horizons}


def check_all(db_path=DEFAULT_DB, weekly_path="data/forecast/weekly.csv", report_path="reports/monitoring.json"):
    """Drift + forecast performance + the retraining decision the pipeline uses."""
    drift = drift_report(load_reference(), recent_predictions(db_path))
    observed = pd.read_csv(weekly_path, parse_dates=["week"]) if os.path.exists(weekly_path) else pd.DataFrame(
        columns=["site", "week", "water_pct"])
    performance = forecast_performance(db_path, observed)
    reasons = []
    if drift["status"] == "alert":
        reasons.append("input data drift: " + ", ".join(drift.get("flagged", [])))
    if performance["status"] == "degraded":
        reasons.append("forecaster no longer beats the no-change baseline")
    report = {"generated_at": datetime.now(timezone.utc).isoformat(), "data_drift": drift,
              "forecast_performance": performance, "retrain_needed": bool(reasons), "reasons": reasons}
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    return report


# --- reference sample and demo ---------------------------------------------------------------

def build_reference(tiles_dir="data/world_train", model_path="models/model.pt", n=300, path=REFERENCE_PATH):
    """Reference distributions from tiles the model was trained on, computed
    exactly like at serving time (same resize, same model, same threshold)."""
    import torch
    from PIL import Image
    from torchvision import transforms

    from src.data import build_normalize_transform
    from src.model import load_model
    from src.preprocess import nodata_mask, sharpness_score

    with open(os.path.join(tiles_dir, "manifest.json"), encoding="utf-8") as f:
        files = [m["file"] for m in json.load(f)][:n]
    model, size, _ = load_model(model_path)
    rows = {name: [] for name in FEATURES}
    for name in files:
        image = Image.open(os.path.join(tiles_dir, "images", name)).convert("RGB")
        sharp = sharpness_score(image)
        resized = transforms.Resize((size, size))(image)
        with torch.no_grad():
            prob = torch.sigmoid(model(build_normalize_transform()(resized).unsqueeze(0)))[0, 0].numpy()
        features = image_features(np.array(resized), sharp, prob, ~nodata_mask(np.array(resized)))
        for key, value in features.items():
            if value is not None:
                rows[key].append(round(value, 4))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"source": tiles_dir, "n_images": len(files), "model": model_path, "features": rows}, f)
    return rows


def simulate_drift(db_path=DEFAULT_DB, tiles_dir="data/world", model_path="models/model.pt", n=80, darkening=0.55):
    """Demo: replays benchmark tiles as normal traffic, then the same tiles
    darkened and hazed (e.g. a different sensor or heavy atmosphere). Uses
    the real model, so confidence and water share move as they would live."""
    import torch
    from PIL import Image, ImageEnhance
    from torchvision import transforms

    from src.data import build_normalize_transform
    from src.model import load_model
    from src.preprocess import nodata_mask, sharpness_score

    with open(os.path.join(tiles_dir, "manifest.json"), encoding="utf-8") as f:
        files = [m["file"] for m in json.load(f)][:n]
    model, size, _ = load_model(model_path)

    def run(transform, source):
        for name in files:
            image = transform(Image.open(os.path.join(tiles_dir, "images", name)).convert("RGB"))
            resized = transforms.Resize((size, size))(image)
            with torch.no_grad():
                prob = torch.sigmoid(model(build_normalize_transform()(resized).unsqueeze(0)))[0, 0].numpy()
            features = image_features(np.array(resized), sharpness_score(image), prob, ~nodata_mask(np.array(resized)))
            log_prediction(db_path, "simulation", features, source)

    run(lambda im: im, "simulated: normal traffic")
    run(lambda im: ImageEnhance.Contrast(ImageEnhance.Brightness(im).enhance(darkening)).enhance(0.6), "simulated: darkened and hazy")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "check"
    db = os.environ.get("MONITORING_DB", DEFAULT_DB)
    if command == "build-reference":
        build_reference()
        print("reference written to", REFERENCE_PATH)
    elif command == "simulate-drift":
        simulate_drift(db)
        print("simulated traffic logged to", db)
    else:
        print(json.dumps(check_all(db), indent=1))
