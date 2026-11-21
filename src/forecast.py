"""7-day rain forecast (Open-Meteo — free, no API key, no quota) crossed
with each site's own historical trend, to project where the water level
is headed instead of only reporting where it is.

This is NOT a calibrated hydrological model: there is no known
relationship in this project between forecast mm of rain and area %
gained for a specific reservoir (that would need a height-area-volume
curve per site, which isn't public — see reports/validation_vs_official.md).
The projection is a plain linear extrapolation of the recent trend; the
rain forecast is surfaced as an explicit qualitative caveat when it could
plausibly change the outcome, never folded numerically into the number."""
from datetime import datetime

import httpx

from src.alerts import ALERT_TIERS, classify_alert_level

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# (minimum cumulative mm over the forecast window, level).
RAIN_FORECAST_TIERS = [
    (40, "importante"),
    (15, "modérée"),
]

STABLE_SLOPE_PCT_PER_DAY = 0.05


def classify_rain_level(total_mm, tiers=RAIN_FORECAST_TIERS):
    for threshold, level in tiers:
        if total_mm >= threshold:
            return level
    return "faible"


def fetch_rain_forecast(lat, lon, days=7, timeout=10.0):
    params = {
        "latitude": lat, "longitude": lon,
        "daily": "precipitation_sum",
        "forecast_days": days,
        "timezone": "auto",
    }
    response = httpx.get(OPEN_METEO_URL, params=params, timeout=timeout)
    response.raise_for_status()
    daily = response.json()["daily"]
    total_mm = round(sum(daily["precipitation_sum"]), 1)
    return {
        "days": days,
        "dates": daily["time"],
        "daily_mm": daily["precipitation_sum"],
        "total_mm": total_mm,
        "level": classify_rain_level(total_mm),
    }


def compute_trend(history, max_points=30):
    """Linear regression slope of water_area_pct against time (pct/day),
    over up to the last `max_points` readings. None if there's too little
    history to fit a meaningful trend (fewer than 3 points, or they all
    fall on the same day)."""
    points = history[-max_points:]
    if len(points) < 3:
        return None

    # Readings mix timezone-aware timestamps ("live"/"alerts", UTC with an
    # offset) and plain dates ("backfill", no time component at all) —
    # normalized to naive UTC so they can be subtracted from each other.
    def parse(ts):
        dt = datetime.fromisoformat(ts)
        return dt.replace(tzinfo=None) if dt.tzinfo else dt

    t0 = parse(points[0]["timestamp"])
    xs = [(parse(p["timestamp"]) - t0).total_seconds() / 86400 for p in points]
    ys = [p["water_area_pct"] for p in points]

    if xs[-1] - xs[0] < 1:
        return None

    mean_x, mean_y = sum(xs) / len(xs), sum(ys) / len(ys)
    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom == 0:
        return None
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom


def project_risk(current_pct, history, rain_forecast, days_ahead=7, tiers=ALERT_TIERS):
    slope = compute_trend(history)
    if slope is None:
        return {
            "insufficient_history": True,
            "rain_forecast": rain_forecast,
            "note": "Historique insuffisant pour projeter une tendance "
                    "(3 points minimum, sur plusieurs jours).",
        }

    projected_pct = max(0.0, min(100.0, current_pct + slope * days_ahead))
    if slope > STABLE_SLOPE_PCT_PER_DAY:
        trend_label = "hausse"
    elif slope < -STABLE_SLOPE_PCT_PER_DAY:
        trend_label = "baisse"
    else:
        trend_label = "stable"

    drop_pct = max(0.0, (current_pct - projected_pct) / current_pct * 100) if current_pct > 0 else 0.0
    projected_alert_level = classify_alert_level(drop_pct, tiers)

    caveat = None
    if trend_label == "baisse" and rain_forecast["level"] in {"modérée", "importante"}:
        caveat = (
            f"Tendance à la baisse, mais une pluviométrie {rain_forecast['level']} est "
            f"prévue ({rain_forecast['total_mm']} mm sur {rain_forecast['days']} jours) — "
            "pourrait atténuer ou inverser cette baisse ; non intégré numériquement au "
            "calcul (pas de relation calibrée pluie→surface pour ce site)."
        )
    elif trend_label == "hausse" and rain_forecast["level"] == "faible":
        caveat = (
            "Tendance à la hausse basée sur l'historique récent, mais peu de pluie est "
            f"prévue ({rain_forecast['total_mm']} mm sur {rain_forecast['days']} jours) — "
            "cette hausse pourrait ne pas se maintenir."
        )

    return {
        "insufficient_history": False,
        "current_pct": round(current_pct, 1),
        "projected_pct": round(projected_pct, 1),
        "days_ahead": days_ahead,
        "slope_pct_per_day": round(slope, 2),
        "trend_label": trend_label,
        "projected_alert_level": projected_alert_level,
        "rain_forecast": rain_forecast,
        "caveat": caveat,
    }
