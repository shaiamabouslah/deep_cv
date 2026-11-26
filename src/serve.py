"""FastAPI backend: water-body segmentation + history + alerting.

Exposes:
  POST /predict           image + site name -> mask overlay, water area % of the photo
                           (runs the trained deep-learning segmentation model;
                           one-off check, not written to the site history)
  GET  /live/{site}       real Sentinel-2 + CHIRPS analysis for a known site
  POST /analyze_location  real Sentinel-2 + CHIRPS analysis for any point/polygon
  POST /sites             register a new site (drawn/clicked on the map)
  GET  /sites             all sites (seed dams + custom) with latest status
  GET  /history/{site}    time series of past readings for that site
  GET  /forecast/{site}   deep-learning (GRU) forecast at +1/+2/+4 weeks when a trained
                           model is installed, plus the linear trend + 7-day rain (Open-Meteo)
  GET  /world/benchmark   model results per site on the worldwide benchmark (world map)
  GET  /lake_at           lake/reservoir outline + facts at any point of the world (HydroLAKES)
  GET  /alerts            currently triggered alerts across all sites
  GET  /briefing          AI agent synthesis of every site's status
  GET  /health
  GET  /metrics           Prometheus metrics (traffic, latency, errors, CPU/memory, drift, forecast skill)
  GET  /monitoring        drift and performance report as JSON (page: /monitoring.html)
Serves the static dashboard (frontend) at /.
"""
import base64
import io
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import torch
from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms

load_dotenv()  # picks up a local .env (GROQ_API_KEY, etc.) if present — gitignored, never committed

from src.agent import generate_briefing, generate_fused_recommendation
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
from src import calibration, forecast_dl, lakes, monitoring, serve_metrics
from src.forecast import fetch_rain_forecast, project_risk
from src.gee_client import (
    GEENotAuthenticated,
    analyze_site_live,
    assess_water_balance,
    backfill_monthly_history,
    explain_level_change,
)
from src.metrics import water_area_fraction
from src.model import build_segmentation_model, load_model
from src.preprocess import nodata_mask, sharpness_score
from src.rate_limit import rate_limit
from src.site_store import add_custom_site, get_custom_sites
from src.sites import TUNISIA_DAMS
from src.water_quality import analyze as analyze_water_quality

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("water_monitoring")

_state = {"model": None, "image_size": 256, "threshold": 0.5, "params": None, "using_fallback": False}


def _load_model_state():
    params = load_params()
    _state["params"] = params
    _state["calibration"] = calibration.load()
    model_path = params["model"]["model_path"]

    if os.path.exists(model_path):
        model, image_size, threshold = load_model(model_path)
        _state["model"] = model
        _state["image_size"] = image_size
        _state["threshold"] = threshold
        _state["using_fallback"] = False
        logger.info("Loaded trained model from %s (image_size=%s)", model_path, image_size)
    else:
        # No trained checkpoint yet (e.g. before the Kaggle export is dropped in):
        # fall back to an untrained model so the API/dashboard stay testable.
        _state["model"] = build_segmentation_model(
            params["model"]["encoder"], encoder_weights=None
        ).eval()
        _state["image_size"] = params["model"]["image_size"]
        _state["threshold"] = 0.5
        _state["using_fallback"] = True
        logger.warning("No model.pt found at %s — serving an untrained fallback model", model_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model_state()
    yield


app = FastAPI(title="Water Monitoring API", lifespan=lifespan)
app.middleware("http")(serve_metrics.metrics_middleware)

gee_rate_limit = rate_limit(max_calls=10, window_seconds=60)
forecast_rate_limit = rate_limit(max_calls=20, window_seconds=60)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _state["model"] is not None,
        "using_fallback_untrained_model": _state["using_fallback"],
    }


MIN_SHARPNESS = 15.0
MONITORING_CACHE_SECONDS = 30
_monitoring_cache = {"at": 0.0, "report": None}


def _monitoring_db():
    return os.environ.get("MONITORING_DB", monitoring.DEFAULT_DB)


def _record_prediction(site, rgb, sharpness, probabilities, valid, inference_seconds):
    """Monitoring must never break serving: any failure is logged and dropped."""
    try:
        features = monitoring.image_features(rgb, sharpness, probabilities, valid)
        monitoring.log_prediction(_monitoring_db(), site, features)
        serve_metrics.record_prediction(inference_seconds, features)
    except Exception as exc:
        logger.warning("prediction monitoring skipped: %s", exc)


def _monitoring_report(max_age=MONITORING_CACHE_SECONDS):
    if _monitoring_cache["report"] is None or time.time() - _monitoring_cache["at"] > max_age:
        _monitoring_cache["report"] = monitoring.check_all(_monitoring_db())
        _monitoring_cache["at"] = time.time()
        serve_metrics.publish_monitoring(_monitoring_cache["report"])
    return _monitoring_cache["report"]


@app.get("/metrics", include_in_schema=False)
def prometheus_metrics():
    try:
        _monitoring_report()
    except Exception as exc:
        logger.warning("monitoring report unavailable for /metrics: %s", exc)
    return Response(serve_metrics.render(), media_type="text/plain; version=0.0.4")


@app.get("/monitoring")
def monitoring_status():
    """System resources + model monitoring: drift of the inputs against the
    training reference, forecast accuracy against the no-change baseline."""
    try:
        report = _monitoring_report()
    except Exception as exc:
        logger.warning("monitoring report failed: %s", exc)
        return JSONResponse(status_code=503, content={"detail": "monitoring indisponible"})
    recent = monitoring.recent_predictions(_monitoring_db(), 20)
    return {"system": serve_metrics.system_snapshot(), **report,
            "recent_predictions": recent[["timestamp", "site", "source", "brightness", "water_pct", "confidence"]]
            .round(2).to_dict("records")}


@app.post("/predict")
async def predict(
    request: Request,
    file: UploadFile = File(...),
    site: str = Form(...),
    fuse_satellite: bool = Form(False),
):
    db_path = _state["params"]["paths"]["history_db"]
    dam = next((d for d in TUNISIA_DAMS + get_custom_sites(db_path) if d["name"] == site), None)
    if dam is None:
        raise HTTPException(status_code=404, detail=f"Site inconnu : {site}")

    image_bytes = await file.read()
    raw_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    blur_score = float(sharpness_score(raw_image))
    # Plain stretch, exactly as in training. A letterbox (padding) looks
    # nicer but the model never saw padded inputs and reads the white bars
    # as water: on a 16:9 image it reported 54-87% water vs 1-20% real.
    # The same resized image feeds the color analysis so mask and RGB align.
    image = transforms.Resize((_state["image_size"], _state["image_size"]))(raw_image)

    tensor = build_normalize_transform()(image).unsqueeze(0)

    started = time.perf_counter()
    with torch.no_grad():
        logits = _state["model"](tensor)
        probabilities = torch.sigmoid(logits)
        mask = (probabilities > _state["threshold"]).float()
    inference_seconds = time.perf_counter() - started

    resized_rgb = np.array(image)
    valid = torch.from_numpy(~nodata_mask(resized_rgb))
    mask = mask * valid
    _record_prediction(site, resized_rgb, blur_score, probabilities[0, 0].numpy(), valid.numpy(), inference_seconds)

    # Not recorded and not checked for alerts: this is "% of this photo that
    # is water", while the site history holds "% of the dam's historical
    # max extent" from the satellite path. Mixing the two produced false
    # alerts (e.g. 75% by satellite vs ~12% for one uploaded tile).
    area_pct = water_area_fraction(mask, valid) * 100

    mask_array = mask.squeeze().numpy()
    quality = analyze_water_quality(resized_rgb, mask_array)

    mask_image = Image.fromarray((mask_array * 255).astype("uint8"))
    buf = io.BytesIO()
    mask_image.save(buf, format="PNG")
    mask_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    # Multi-input fusion: combine the segmentation result (this photo) with
    # real GEE spectral indices for the same site (if it's a registered
    # site with known coordinates) and let the agent produce ONE
    # recommendation instead of separate hard-coded templates. Opt-in
    # because the GEE round trip adds real latency.
    signals = {
        "site": site, "water_area_pct": area_pct, "source": "upload",
        "rgb_quality": quality, "gee_indices": None, "rainfall": None,
        "water_balance": None, "alert": None,
    }
    if fuse_satellite:
        gee_rate_limit(request)
        try:
            gee_result = analyze_site_live(
                dam["lat"], dam["lon"], project=_state["params"]["gee"]["project"],
                area_buffer_m=dam.get("buffer_m", 6000),
            )
            signals["gee_indices"] = gee_result["indices"]
            signals["rainfall"] = gee_result["rainfall"]
        except (GEENotAuthenticated, ValueError) as exc:
            logger.warning("GEE enrichment skipped for site=%s: %s", site, exc)

    recommendation = generate_fused_recommendation(signals)

    logger.info(
        "predict site=%s water_area_pct=%.1f blur_score=%.1f fuse_satellite=%s ai_generated=%s",
        site, area_pct, blur_score, fuse_satellite, recommendation["ai_generated"],
    )

    return {
        "site": site,
        "water_area_pct": round(area_pct, 2),
        "mask_png_base64": mask_b64,
        "water_quality": quality,
        "recommendation": recommendation,
        "image_quality": {
            "sharpness": round(blur_score, 1),
            "blurry": bool(blur_score < MIN_SHARPNESS),
        },
        "using_fallback_untrained_model": _state["using_fallback"],
    }


def _run_live_analysis(site_name, lat, lon, polygon_coords=None, area_buffer_m=6000):
    db_path = _state["params"]["paths"]["history_db"]
    previous_history = get_history(db_path, site_name)
    previous_ndvi = get_previous_ndvi(db_path, site_name)

    try:
        result = analyze_site_live(
            lat, lon,
            project=_state["params"]["gee"]["project"],
            polygon_coords=polygon_coords,
            area_buffer_m=area_buffer_m,
        )
    except GEENotAuthenticated as exc:
        logger.error("GEE not authenticated: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        logger.warning("No GEE imagery for site=%s: %s", site_name, exc)
        raise HTTPException(status_code=404, detail=str(exc))

    water_area_proxy = result["water_area"]["water_area_pct"] or 0.0
    current_ndvi = result["indices"]["ndvi"]

    level_change = None
    water_balance = None
    if previous_history:
        pct_change = water_area_proxy - previous_history[-1]["water_area_pct"]
        level_change = explain_level_change(pct_change, result["rainfall"])
        if previous_ndvi is not None:
            water_balance = assess_water_balance(pct_change, current_ndvi - previous_ndvi)

    # Check the alert against history BEFORE writing the new point into it,
    # otherwise the reading dilutes its own baseline.
    alert = check_and_record_alert(
        db_path, site_name, water_area_proxy, tiers=_state["params"]["alerts"]["tiers"]
    )
    record_reading(db_path, site_name, water_area_proxy, ndvi=current_ndvi, source="live")

    fill_estimate = calibration.estimate_fill(site_name, water_area_proxy, _state.get("calibration"))
    recommendation = generate_fused_recommendation({
        "site": site_name, "water_area_pct": water_area_proxy, "source": "satellite",
        "rgb_quality": None, "gee_indices": result["indices"], "fill_estimate": fill_estimate,
        "rainfall": result["rainfall"], "water_balance": water_balance, "alert": alert,
    })

    logger.info(
        "live_analysis site=%s water_area_proxy=%.1f alert_level=%s rainfall_mm=%s ai_generated=%s",
        site_name, water_area_proxy, alert["level"] if alert else None, result["rainfall"]["mm"],
        recommendation["ai_generated"],
    )

    return {
        "site": site_name, "alert": alert,
        "level_change_explanation": level_change,
        "water_balance": water_balance,
        "fill_estimate": fill_estimate,
        "recommendation": recommendation,
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

    return _run_live_analysis(site, dam["lat"], dam["lon"], area_buffer_m=dam.get("buffer_m", 6000))


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


def _with_fill_rates(site, deep):
    """The forecaster predicts the surface index; add the official-style fill
    rate through the dam's calibration curve (None if it has no usable curve)."""
    def fill(value):
        estimate = _fill_for(site, value)
        return estimate["fill_pct"] if estimate else None

    return {**deep, "current_fill_pct": fill(deep["current_pct"]),
            "horizons": [{**h, "fill_pct": fill(h["predicted_pct"])} for h in deep["horizons"]]}


def _fill_for(site, surface_pct):
    return calibration.estimate_fill(site, surface_pct, _state.get("calibration"))


@app.get("/sites")
def sites():
    db_path = _state["params"]["paths"]["history_db"]
    all_sites = TUNISIA_DAMS + get_custom_sites(db_path)
    result = []
    for dam in all_sites:
        latest = get_latest_reading(db_path, dam["name"])
        if latest:
            latest = {**latest, "fill_estimate": _fill_for(dam["name"], latest["water_area_pct"])}
        result.append({
            **dam,
            "latest_reading": latest,
            "alert_active": has_active_alert(db_path, dam["name"]),
            "alert_level": get_highest_alert_level(db_path, dam["name"]),
        })
    return result


def _run_backfill(site_name, lat, lon, months_back, buffer_m=6000):
    try:
        points = backfill_monthly_history(
            lat, lon, months_back=months_back, project=_state["params"]["gee"]["project"],
            buffer_m=buffer_m,
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
def backfill_site(site: str, background_tasks: BackgroundTasks, months_back: int = 12):
    """Pulls real multi-month history from the Sentinel-2 archive for this
    site in the background (several sequential GEE calls — can take a few
    minutes). Poll /history/{site} to see points appear as they land.
    Default is 12 months, a full seasonal cycle; capped at 36 to keep a
    single request from running away with the GEE quota."""
    months_back = max(1, min(months_back, 36))

    all_sites = TUNISIA_DAMS + get_custom_sites(_state["params"]["paths"]["history_db"])
    dam = next((d for d in all_sites if d["name"] == site), None)
    if dam is None:
        raise HTTPException(status_code=404, detail=f"Site inconnu : {site}")

    background_tasks.add_task(
        _run_backfill, site, dam["lat"], dam["lon"], months_back, dam.get("buffer_m", 6000)
    )
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
    points = get_history(_state["params"]["paths"]["history_db"], site)
    for point in points:
        estimate = _fill_for(site, point["water_area_pct"])
        point["fill_pct"] = estimate["fill_pct"] if estimate else None
    return points


@app.get("/forecast/{site}", dependencies=[Depends(forecast_rate_limit)])
def forecast(site: str, request: Request):
    """Projects where this site's water level is likely headed, by
    extrapolating its own recorded trend and cross-referencing a 7-day
    rain forecast (Open-Meteo) as a qualitative caveat — see
    src/forecast.py's docstring for why the rain forecast isn't folded
    numerically into the projection."""
    db_path = _state["params"]["paths"]["history_db"]
    all_sites = TUNISIA_DAMS + get_custom_sites(db_path)
    dam = next((d for d in all_sites if d["name"] == site), None)
    if dam is None:
        raise HTTPException(status_code=404, detail=f"Site inconnu : {site}")

    latest = get_latest_reading(db_path, site)
    if latest is None:
        raise HTTPException(
            status_code=404,
            detail=f"Pas encore de relevé pour {site} — lance une analyse live d'abord.",
        )

    history_points = get_history(db_path, site)
    rain = fetch_rain_forecast(dam["lat"], dam["lon"])
    risk = project_risk(latest["water_area_pct"], history_points, rain)

    deep = None
    if forecast_dl.model_available():
        if forecast_dl.cached_forecast(site) is None:
            gee_rate_limit(request)
        try:
            deep = forecast_dl.deep_forecast(dam, project=_state["params"]["gee"]["project"])
        except Exception as exc:
            logger.warning("deep forecast failed for site=%s: %s", site, exc)
            deep = {"available": False, "reason": "Prévision par modèle indisponible (Earth Engine injoignable)."}
    if deep and deep.get("available") and deep.get("horizons"):
        deep = _with_fill_rates(site, deep)
    if deep and deep.get("available", True) and deep.get("horizons"):
        try:
            monitoring.log_forecast(_monitoring_db(), site, deep)
        except Exception as exc:
            logger.warning("forecast monitoring skipped: %s", exc)
    return {"site": site, **risk, "deep_learning": deep}


WORLD_TILES_PATH = "reports/world_tiles.json"
WORLD_SUMMARY_PATH = "reports/world_benchmark.json"
TUNISIA_EVAL_PATH = "reports/metrics.json"


def _read_json(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@app.get("/world/benchmark")
def world_benchmark_map():
    """Per-site results of the served segmentation model on the worldwide
    benchmark (see src/world_benchmark.py), for the dashboard's world map."""
    tiles = _read_json(WORLD_TILES_PATH)
    if tiles is None:
        raise HTTPException(status_code=404, detail="Benchmark mondial non calculé (python -m src.world_benchmark).")
    summary = (_read_json(WORLD_SUMMARY_PATH) or {}).get(tiles["served_model"], {}).get("clean", {})
    tunisia = _read_json(TUNISIA_EVAL_PATH) or {}
    return {
        "served_model": tiles["served_model"], "tiles": tiles["tiles"],
        "summary": {"overall": summary.get("overall"), "by_type": summary.get("by_type"),
                    "tunisia_iou": tunisia.get("iou")},
    }


@app.get("/lake_at", dependencies=[Depends(gee_rate_limit)])
def lake_lookup(lat: float, lon: float):
    """The lake or reservoir at this point of the world (HydroLAKES), with its
    outline, so the analysis uses its real shape wherever on Earth."""
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise HTTPException(status_code=422, detail="Coordonnées hors limites.")
    try:
        lake = lakes.lake_at(lat, lon, project=_state["params"]["gee"]["project"])
    except GEENotAuthenticated as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.warning("lake lookup failed at (%s, %s): %s", lat, lon, exc)
        raise HTTPException(status_code=502, detail="Recherche du plan d'eau impossible pour le moment (Earth Engine).")
    if lake is None:
        raise HTTPException(status_code=404, detail="Aucun lac ou réservoir cartographié à cet endroit.")
    return lake


@app.get("/alerts")
def alerts():
    return get_active_alerts(_state["params"]["paths"]["history_db"])


if os.path.isdir("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")
