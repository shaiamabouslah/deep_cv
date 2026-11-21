import pytest

from src import forecast
from src.forecast import classify_rain_level, compute_trend, fetch_rain_forecast, project_risk


def test_classify_rain_level_tiers():
    assert classify_rain_level(5) == "faible"
    assert classify_rain_level(20) == "modérée"
    assert classify_rain_level(50) == "importante"


def _reading(day, pct):
    return {"timestamp": f"2026-01-{day:02d}T00:00:00+00:00", "water_area_pct": pct, "ndvi": None, "source": "live"}


def test_compute_trend_detects_declining_slope():
    history = [_reading(d, 80 - d) for d in range(1, 6)]
    slope = compute_trend(history)
    assert slope < 0


def test_compute_trend_none_with_too_little_history():
    assert compute_trend([_reading(1, 50)]) is None
    assert compute_trend([_reading(1, 50), _reading(1, 51)]) is None


def test_compute_trend_handles_mixed_naive_and_aware_timestamps():
    history = [
        {"timestamp": "2026-01-01", "water_area_pct": 80.0, "ndvi": None, "source": "backfill"},
        {"timestamp": "2026-01-02T00:00:00+00:00", "water_area_pct": 78.0, "ndvi": None, "source": "live"},
        {"timestamp": "2026-01-03T00:00:00+00:00", "water_area_pct": 76.0, "ndvi": None, "source": "live"},
    ]
    slope = compute_trend(history)
    assert slope is not None
    assert slope < 0


def test_project_risk_insufficient_history():
    rain = {"days": 7, "dates": [], "daily_mm": [], "total_mm": 0.0, "level": "faible"}
    result = project_risk(50.0, [_reading(1, 50)], rain)
    assert result["insufficient_history"] is True


def test_project_risk_declining_trend_with_significant_rain_gets_caveat():
    history = [_reading(d, 80 - d * 3) for d in range(1, 6)]
    rain = {"days": 7, "dates": [], "daily_mm": [], "total_mm": 60.0, "level": "importante"}
    result = project_risk(history[-1]["water_area_pct"], history, rain)
    assert result["trend_label"] == "baisse"
    assert result["caveat"] is not None
    assert "importante" in result["caveat"]


def test_project_risk_declining_trend_triggers_projected_alert_level():
    history = [_reading(d, 80 - d * 5) for d in range(1, 6)]
    rain = {"days": 7, "dates": [], "daily_mm": [], "total_mm": 0.0, "level": "faible"}
    result = project_risk(history[-1]["water_area_pct"], history, rain)
    assert result["projected_alert_level"] is not None


def test_project_risk_rising_trend_no_drop_alert():
    history = [_reading(d, 20 + d * 3) for d in range(1, 6)]
    rain = {"days": 7, "dates": [], "daily_mm": [], "total_mm": 0.0, "level": "faible"}
    result = project_risk(history[-1]["water_area_pct"], history, rain)
    assert result["trend_label"] == "hausse"
    assert result["projected_alert_level"] is None
    assert result["caveat"] is not None


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_fetch_rain_forecast_parses_open_meteo_payload(monkeypatch):
    payload = {"daily": {"time": ["2026-01-01", "2026-01-02"], "precipitation_sum": [3.0, 2.0]}}
    monkeypatch.setattr(forecast.httpx, "get", lambda *a, **k: _FakeResponse(payload))

    result = fetch_rain_forecast(36.0, 9.0, days=2)

    assert result["total_mm"] == 5.0
    assert result["level"] == "faible"
    assert result["daily_mm"] == [3.0, 2.0]
