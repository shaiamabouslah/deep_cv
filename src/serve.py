"""FastAPI backend: water-body segmentation + history + alerting.

Exposes:
  POST /predict        image + site name -> mask overlay, water area %, alert
  GET  /history/{site} -> time series of past readings for that site
  GET  /alerts          -> currently triggered alerts across all sites
  GET  /health
Serves the static dashboard (frontend) at /.
"""
import base64
import io
import os

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image

from src.alerts import (
    check_and_record_alert,
    get_active_alerts,
    get_history,
    get_latest_reading,
    has_active_alert,
    record_reading,
)
from src.data import build_transforms, load_params
from src.gee_client import GEENotAuthenticated, analyze_site_live
from src.metrics import water_area_fraction
from src.model import build_segmentation_model, load_model
from src.sites import TUNISIA_DAMS
from src.water_quality import analyze as analyze_water_quality

app = FastAPI(title="Water Monitoring API")

_state = {"model": None, "image_size": 256, "params": None, "using_fallback": False}


@app.on_event("startup")
def startup():
    params = load_params()
    _state["params"] = params
    model_path = params["model"]["model_path"]

    if os.path.exists(model_path):
        model, image_size = load_model(model_path)
        _state["model"] = model
        _state["image_size"] = image_size
        _state["using_fallback"] = False
    else:
        # No trained checkpoint yet (e.g. before the Kaggle export is dropped in):
        # fall back to an untrained model so the API/dashboard stay testable.
        _state["model"] = build_segmentation_model(
            params["model"]["encoder"], encoder_weights=None
        ).eval()
        _state["image_size"] = params["model"]["image_size"]
        _state["using_fallback"] = True


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _state["model"] is not None,
        "using_fallback_untrained_model": _state["using_fallback"],
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...), site: str = Form(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    transform = build_transforms(_state["image_size"])
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = _state["model"](tensor)
        mask = (torch.sigmoid(logits) > 0.5).float()

    area_pct = water_area_fraction(mask) * 100
    record_reading(_state["params"]["paths"]["history_db"], site, area_pct)
    alert = check_and_record_alert(
        _state["params"]["paths"]["history_db"],
        site,
        area_pct,
        _state["params"]["alerts"]["drop_threshold_pct"],
    )

    mask_array = mask.squeeze().numpy()
    resized_rgb = np.array(image.resize((_state["image_size"], _state["image_size"])))
    quality = analyze_water_quality(resized_rgb, mask_array)

    mask_image = Image.fromarray((mask_array * 255).astype("uint8"))
    buf = io.BytesIO()
    mask_image.save(buf, format="PNG")
    mask_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "site": site,
        "water_area_pct": round(area_pct, 2),
        "mask_png_base64": mask_b64,
        "alert": alert,
        "water_quality": quality,
        "using_fallback_untrained_model": _state["using_fallback"],
    }


@app.get("/live/{site}")
def live_analysis(site: str):
    """Real satellite analysis via Google Earth Engine: fetches the latest
    Sentinel-2 bands for this site's coordinates and computes validated
    spectral indices (NDVI/NDCI/NDTI), instead of the RGB color heuristic
    used in /predict for an uploaded image."""
    dam = next((d for d in TUNISIA_DAMS if d["name"] == site), None)
    if dam is None:
        raise HTTPException(status_code=404, detail=f"Site inconnu : {site}")

    try:
        result = analyze_site_live(dam["lat"], dam["lon"])
    except GEENotAuthenticated as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    db_path = _state["params"]["paths"]["history_db"]
    water_area_proxy = max(0.0, min(100.0, (result["indices"]["ndwi"] + 1) * 50))
    record_reading(db_path, site, water_area_proxy)

    return {"site": site, **result}


@app.get("/sites")
def sites():
    db_path = _state["params"]["paths"]["history_db"]
    result = []
    for dam in TUNISIA_DAMS:
        latest = get_latest_reading(db_path, dam["name"])
        result.append({
            **dam,
            "latest_reading": latest,
            "alert_active": has_active_alert(db_path, dam["name"]),
        })
    return result


@app.get("/history/{site}")
def history(site: str):
    return get_history(_state["params"]["paths"]["history_db"], site)


@app.get("/alerts")
def alerts():
    return get_active_alerts(_state["params"]["paths"]["history_db"])


if os.path.isdir("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")
