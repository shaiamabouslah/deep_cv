"""FastAPI backend: water-body segmentation + history + alerting.

Exposes:
  POST /predict           image + site name -> mask overlay, water area %, alert
                           (runs the trained deep-learning segmentation model)
  GET  /live/{site}       real Sentinel-2 + CHIRPS analysis for a known site
  POST /analyze_location  real Sentinel-2 + CHIRPS analysis for any point/polygon
  POST /sites             register a new site (drawn/clicked on the map)
  GET  /sites             all sites (seed dams + custom) with latest status
  GET  /history/{site}    time series of past readings for that site
  GET  /alerts            currently triggered alerts across all sites
  GET  /briefing          AI agent synthesis of every site's status
  GET  /health
Serves the static dashboard (frontend) at /.
"""
import base64
import io
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import torch
from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from src.agent import generate_briefing
from src.alerts import (
    check_and_record_alert,
    get_active_alerts,
    get_highest_alert_level,
    get_history,
    get_latest_reading,
    get_previous_ndvi,
    has_active_alert,
    record_reading,
)
from src.data import build_normalize_transform, load_params
from src.gee_client import (
    GEENotAuthenticated,
    analyze_site_live,
    assess_water_balance,
    backfill_monthly_history,
    explain_level_change,
)
from src.metrics import water_area_fraction
from src.model import build_segmentation_model, load_model
from src.preprocess import sharpness_score, standardize
from src.rate_limit import rate_limit
from src.site_store import add_custom_site, get_custom_sites
from src.sites import TUNISIA_DAMS
from src.water_quality import analyze as analyze_water_quality

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("water_monitoring")

_state = {"model": None, "image_size": 256, "params": None, "using_fallback": False}


def _load_model_state():
    params = load_params()
    _state["params"] = params
    model_path = params["model"]["model_path"]

    if os.path.exists(model_path):
        model, image_size = load_model(model_path)
        _state["model"] = model
        _state["image_size"] = image_size
        _state["using_fallback"] = False
        logger.info("Loaded trained model from %s (image_size=%s)", model_path, image_size)
    else:
        # No trained checkpoint yet (e.g. before the Kaggle export is dropped in):
        # fall back to an untrained model so the API/dashboard stay testable.
        _state["model"] = build_segmentation_model(
            params["model"]["encoder"], encoder_weights=None
        ).eval()
        _state["image_size"] = params["model"]["image_size"]
        _state["using_fallback"] = True
        logger.warning("No model.pt found at %s — serving an untrained fallback model", model_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model_state()
    yield


app = FastAPI(title="Water Monitoring API", lifespan=lifespan)

gee_rate_limit = rate_limit(max_calls=10, window_seconds=60)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _state["model"] is not None,
        "using_fallback_untrained_model": _state["using_fallback"],
    }


MIN_SHARPNESS = 15.0


@app.post("/predict")
async def predict(file: UploadFile = File(...), site: str = Form(...)):
    image_bytes = await file.read()
    raw_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Classical CV preprocessing before the image reaches the deep learning
    # model: blur check, then aspect-ratio-preserving resize (letterbox)
    # instead of a plain stretch — the same standardized image is reused
    # below for the water-quality color analysis, so mask and RGB stay
    # pixel-aligned.
    blur_score = float(sharpness_score(raw_image))
    image = standardize(raw_image, _state["image_size"])

    tensor = build_normalize_transform()(image).unsqueeze(0)

    with torch.no_grad():
        logits = _state["model"](tensor)
        mask = (torch.sigmoid(logits) > 0.5).float()

    area_pct = water_area_fraction(mask) * 100
    db_path = _state["params"]["paths"]["history_db"]

    # Alert check must run BEFORE the new reading is recorded, otherwise the
    # historical baseline it compares against includes the point being
    # tested, diluting drop detection.
    alert = check_and_record_alert(
        db_path, site, area_pct, tiers=_state["params"]["alerts"]["tiers"]
    )
    record_reading(db_path, site, area_pct)

    mask_array = mask.squeeze().numpy()
    resized_rgb = np.array(image)
    quality = analyze_water_quality(resized_rgb, mask_array)

    mask_image = Image.fromarray((mask_array * 255).astype("uint8"))
    buf = io.BytesIO()
    mask_image.save(buf, format="PNG")
    mask_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    logger.info(
        "predict site=%s water_area_pct=%.1f alert=%s blur_score=%.1f",
        site, area_pct, bool(alert), blur_score,
    )

    return {
        "site": site,
        "water_area_pct": round(area_pct, 2),
        "mask_png_base64": mask_b64,
        "alert": alert,
        "water_quality": quality,
        "image_quality": {
            "sharpness": round(blur_score, 1),
            "blurry": bool(blur_score < MIN_SHARPNESS),
        },
        "using_fallback_untrained_model": _state["using_fallback"],
    }


def _run_live_analysis(site_name, lat, lon, polygon_coords=None):
    db_path = _state["params"]["paths"]["history_db"]
    previous_history = get_history(db_path, site_name)
    previous_ndvi = get_previous_ndvi(db_path, site_name)

    try:
        result = analyze_site_live(
            lat, lon,
            project=_state["params"]["gee"]["project"],
            polygon_coords=polygon_coords,
        )
    except GEENotAuthenticated as exc:
        logger.error("GEE not authenticated: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        logger.warning("No GEE imagery for site=%s: %s", site_name, exc)
        raise HTTPException(status_code=404, detail=str(exc))

    water_area_proxy = max(0.0, min(100.0, (result["indices"]["ndwi"] + 1) * 50))
    current_ndvi = result["indices"]["ndvi"]

    level_change = None
    water_balance = None
    if previous_history:
        pct_change = water_area_proxy - previous_history[-1]["water_area_pct"]
        level_change = explain_level_change(pct_change, result["rainfall"])
        if previous_ndvi is not None:
            water_balance = assess_water_balance(pct_change, current_ndvi - previous_ndvi)

    # Same ordering fix as /predict: check the alert against history BEFORE
    # writing the new point into it.
    alert = check_and_record_alert(
        db_path, site_name, water_area_proxy, tiers=_state["params"]["alerts"]["tiers"]
    )
    record_reading(db_path, site_name, water_area_proxy, ndvi=current_ndvi, source="live")

    logger.info(
        "live_analysis site=%s water_area_proxy=%.1f alert_level=%s rainfall_mm=%s",
        site_name, water_area_proxy, alert["level"] if alert else None, result["rainfall"]["mm"],
    )

    return {
        "site": site_name, "alert": alert,
        "level_change_explanation": level_change,
        "water_balance": water_balance,
        **result,
    }


@app.get("/live/{site}", dependencies=[Depends(gee_rate_limit)])
def live_analysis(site: str):
    """Real satellite analysis via Google Earth Engine: fetches the latest
    Sentinel-2 bands for this site's coordinates and computes validated
    spectral indices (NDVI/NDCI/NDTI), instead of the RGB color heuristic
    used in /predict for an uploaded image."""
    all_sites = TUNISIA_DAMS + get_custom_sites(_state["params"]["paths"]["history_db"])
    dam = next((d for d in all_sites if d["name"] == site), None)
    if dam is None:
        raise HTTPException(status_code=404, detail=f"Site inconnu : {site}")

    return _run_live_analysis(site, dam["lat"], dam["lon"])


class AnalyzeLocationRequest(BaseModel):
    lat: float
    lon: float
    name: Optional[str] = None
    polygon: Optional[list] = None  # list of [lon, lat] pairs, as drawn on the map


@app.post("/analyze_location", dependencies=[Depends(gee_rate_limit)])
def analyze_location(req: AnalyzeLocationRequest):
    """Ad-hoc analysis for any point or hand-drawn area on the map — not
    limited to the predefined/registered sites."""
    site_name = req.name or f"Point ({req.lat:.4f}, {req.lon:.4f})"
    return _run_live_analysis(site_name, req.lat, req.lon, polygon_coords=req.polygon)


class NewSiteRequest(BaseModel):
    name: str
    lat: float
    lon: float


@app.post("/sites")
def create_site(req: NewSiteRequest):
    """Registers a site drawn/clicked by the user so it persists and shows
    up on the map like the seed dams — the platform isn't limited to a
    fixed list."""
    seed_match = next((d for d in TUNISIA_DAMS if d["name"] == req.name), None)
    if seed_match is not None:
        return {
            "id": req.name, "name": req.name,
            "lat": seed_match["lat"], "lon": seed_match["lon"],
            "name_taken": True,
        }

    db_path = _state["params"]["paths"]["history_db"]
    result = add_custom_site(db_path, req.name, req.lat, req.lon)
    if result["name_taken"]:
        logger.info("Site name '%s' already exists, rejected new coords", req.name)
    return result


@app.get("/sites")
def sites():
    db_path = _state["params"]["paths"]["history_db"]
    all_sites = TUNISIA_DAMS + get_custom_sites(db_path)
    result = []
    for dam in all_sites:
        latest = get_latest_reading(db_path, dam["name"])
        result.append({
            **dam,
            "latest_reading": latest,
            "alert_active": has_active_alert(db_path, dam["name"]),
            "alert_level": get_highest_alert_level(db_path, dam["name"]),
        })
    return result


def _run_backfill(site_name, lat, lon, months_back):
    try:
        points = backfill_monthly_history(
            lat, lon, months_back=months_back, project=_state["params"]["gee"]["project"]
        )
        db_path = _state["params"]["paths"]["history_db"]
        for point in points:
            record_reading(
                db_path, site_name, point["water_area_pct"],
                source="backfill", timestamp=point["date"],
            )
        logger.info("Backfill done for site=%s: %d points", site_name, len(points))
    except Exception:
        logger.exception("Backfill failed for site=%s", site_name)


@app.post("/sites/{site}/backfill", dependencies=[Depends(gee_rate_limit)])
def backfill_site(site: str, background_tasks: BackgroundTasks, months_back: int = 6):
    """Pulls real multi-month history from the Sentinel-2 archive for this
    site in the background (several sequential GEE calls — can take a few
    minutes). Poll /history/{site} to see points appear as they land."""
    all_sites = TUNISIA_DAMS + get_custom_sites(_state["params"]["paths"]["history_db"])
    dam = next((d for d in all_sites if d["name"] == site), None)
    if dam is None:
        raise HTTPException(status_code=404, detail=f"Site inconnu : {site}")

    background_tasks.add_task(_run_backfill, site, dam["lat"], dam["lon"], months_back)
    return {"status": "started", "site": site, "months_back": months_back}


@app.get("/briefing")
def briefing():
    """AI agent layer: synthesizes every site's status into a short,
    actionable, plain-language briefing."""
    sites_data = sites()
    active_alerts = get_active_alerts(_state["params"]["paths"]["history_db"])
    return generate_briefing(sites_data, active_alerts)


@app.get("/history/{site}")
def history(site: str):
    return get_history(_state["params"]["paths"]["history_db"], site)


@app.get("/alerts")
def alerts():
    return get_active_alerts(_state["params"]["paths"]["history_db"])


if os.path.isdir("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")
