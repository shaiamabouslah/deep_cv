import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.alerts import record_reading


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from src import serve
    # Never reach Earth Engine from a test: no deep forecast unless a test opts in.
    monkeypatch.setattr(serve.forecast_dl, "model_available", lambda: False)
    monkeypatch.setenv("MONITORING_DB", str(tmp_path / "monitoring.db"))
    monkeypatch.setitem(serve._monitoring_cache, "report", None)
    with TestClient(serve.app) as test_client:
        # Isolate each test run's history/alerts from the developer's real DB.
        serve._state["params"]["paths"]["history_db"] = str(tmp_path / "history.db")
        yield test_client


def _sample_image_bytes():
    img = Image.new("RGB", (64, 64), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_predict_returns_mask_and_area(client):
    response = client.post(
        "/predict",
        data={"site": "Sidi Salem"},
        files={"file": ("test.png", _sample_image_bytes(), "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    assert "water_area_pct" in body
    assert 0.0 <= body["water_area_pct"] <= 100.0
    assert "mask_png_base64" in body


def test_predict_unknown_site_returns_404(client):
    response = client.post(
        "/predict",
        data={"site": "Nowhere"},
        files={"file": ("test.png", _sample_image_bytes(), "image/png")},
    )
    assert response.status_code == 404


def test_predict_does_not_touch_site_history_or_alerts(client):
    from src import serve
    db_path = serve._state["params"]["paths"]["history_db"]
    record_reading(db_path, "Sidi Salem", 80.0, source="live")
    record_reading(db_path, "Sidi Salem", 82.0, source="live")

    response = client.post(
        "/predict",
        data={"site": "Sidi Salem"},
        files={"file": ("test.png", _sample_image_bytes(), "image/png")},
    )

    assert response.status_code == 200
    assert "alert" not in response.json()
    assert len(client.get("/history/Sidi Salem").json()) == 2
    assert client.get("/alerts").json() == []


def test_predict_resizes_like_training(client, monkeypatch):
    from src import serve
    seen = {}
    real_model = serve._state["model"]

    def spy(tensor):
        seen["shape"] = tuple(tensor.shape)
        seen["corner"] = tensor[0, :, 0, 0].clone()
        return real_model(tensor)

    monkeypatch.setitem(serve._state, "model", spy)
    img = Image.new("RGB", (160, 90), color=(20, 60, 120))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    client.post("/predict", data={"site": "Sidi Salem"}, files={"file": ("wide.png", buf, "image/png")})

    size = serve._state["image_size"]
    assert seen["shape"] == (1, 3, size, size)
    white_normalized = (1.0 - 0.485) / 0.229
    assert float(seen["corner"][0]) < white_normalized - 1.0


def test_alerts_endpoint_returns_list(client):
    response = client.get("/alerts")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_sites_endpoint_returns_known_dams(client):
    response = client.get("/sites")
    assert response.status_code == 200
    body = response.json()
    names = [s["name"] for s in body]
    assert "Sidi Salem" in names
    assert all("lat" in s and "lon" in s and "alert_level" in s for s in body)


def test_backfill_schedules_background_task_and_inserts_history(client, monkeypatch):
    monkeypatch.setattr(
        "src.serve.backfill_monthly_history",
        lambda *a, **k: [{"date": "2026-01-01", "water_area_pct": 50.0}],
    )
    response = client.post("/sites/Sidi Salem/backfill?months_back=1")
    assert response.status_code == 200
    assert response.json()["status"] == "started"

    history = client.get("/history/Sidi Salem").json()
    assert any(h["source"] == "backfill" for h in history)


def test_backfill_unknown_site_returns_404(client):
    response = client.post("/sites/Nowhere/backfill")
    assert response.status_code == 404


def test_create_custom_site_then_appears_in_sites(client):
    response = client.post("/sites", json={"name": "Test Lac", "lat": 36.2, "lon": 9.1})
    assert response.status_code == 200
    assert response.json()["name"] == "Test Lac"

    sites_response = client.get("/sites")
    names = [s["name"] for s in sites_response.json()]
    assert "Test Lac" in names


def test_briefing_endpoint_returns_text_without_api_key(client, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    response = client.get("/briefing")
    assert response.status_code == 200
    body = response.json()
    assert "text" in body
    assert body["ai_generated"] is False


def test_forecast_unknown_site_returns_404(client):
    response = client.get("/forecast/Nowhere")
    assert response.status_code == 404


def test_forecast_without_prior_reading_returns_404(client):
    response = client.get("/forecast/Sidi Salem")
    assert response.status_code == 404


def test_forecast_returns_projection(client, monkeypatch):
    monkeypatch.setattr(
        "src.serve.fetch_rain_forecast",
        lambda *a, **k: {"days": 7, "dates": [], "daily_mm": [], "total_mm": 0.0, "level": "faible"},
    )
    from src import serve
    record_reading(serve._state["params"]["paths"]["history_db"], "Sidi Salem", 75.0, source="live")

    response = client.get("/forecast/Sidi Salem")

    assert response.status_code == 200
    body = response.json()
    assert body["site"] == "Sidi Salem"
    assert body["insufficient_history"] is True
    assert body["deep_learning"] is None


def _forecast_with_deep(client, monkeypatch, deep):
    from src import serve
    monkeypatch.setattr(
        "src.serve.fetch_rain_forecast",
        lambda *a, **k: {"days": 7, "dates": [], "daily_mm": [], "total_mm": 0.0, "level": "faible"},
    )
    monkeypatch.setattr(serve.forecast_dl, "model_available", lambda: True)
    monkeypatch.setattr(serve.forecast_dl, "deep_forecast", deep)
    record_reading(serve._state["params"]["paths"]["history_db"], "Sidi Salem", 75.0, source="live")
    return client.get("/forecast/Sidi Salem")


def test_forecast_includes_the_deep_learning_prediction(client, monkeypatch):
    prediction = {"available": True, "current_pct": 70.0, "horizons": [{"weeks": 1, "predicted_pct": 69.0}]}

    response = _forecast_with_deep(client, monkeypatch, lambda dam, project=None: prediction)

    assert response.status_code == 200
    deep = response.json()["deep_learning"]
    assert deep["horizons"][0]["predicted_pct"] == 69.0 and deep["current_pct"] == 70.0
    assert "fill_pct" in deep["horizons"][0] and "current_fill_pct" in deep


def test_forecast_survives_a_deep_learning_failure(client, monkeypatch):
    def broken(dam, project=None):
        raise RuntimeError("earth engine down")

    response = _forecast_with_deep(client, monkeypatch, broken)

    assert response.status_code == 200
    body = response.json()
    assert body["deep_learning"]["available"] is False
    assert body["insufficient_history"] is True


def test_world_benchmark_endpoint_serves_the_map_data_and_summary(client, monkeypatch, tmp_path):
    from src import serve
    tiles = tmp_path / "tiles.json"
    tiles.write_text(json.dumps({"served_model": "v1 (served)", "tiles": [{"id": 1, "lat": 1.0, "lon": 2.0, "iou": {"v1 (served)": 0.7}}]}))
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"v1 (served)": {"clean": {"overall": {"iou": 0.45}, "by_type": {"reservoir": {"iou": 0.73}}}}}))
    tunisia = tmp_path / "metrics.json"
    tunisia.write_text(json.dumps({"iou": 0.72}))
    monkeypatch.setattr(serve, "WORLD_TILES_PATH", str(tiles))
    monkeypatch.setattr(serve, "WORLD_SUMMARY_PATH", str(summary))
    monkeypatch.setattr(serve, "TUNISIA_EVAL_PATH", str(tunisia))

    body = client.get("/world/benchmark").json()

    assert body["tiles"][0]["id"] == 1
    assert body["summary"]["by_type"]["reservoir"]["iou"] == 0.73 and body["summary"]["tunisia_iou"] == 0.72


def test_world_benchmark_without_results_is_a_clear_404(client, monkeypatch, tmp_path):
    from src import serve
    monkeypatch.setattr(serve, "WORLD_TILES_PATH", str(tmp_path / "missing.json"))
    assert client.get("/world/benchmark").status_code == 404


def test_lake_lookup_returns_the_outline_and_facts(client, monkeypatch):
    from src import serve
    lake = {"id": 5, "name": "Test", "type": "reservoir", "area_km2": 12.0, "polygon": [[0, 0], [1, 0], [1, 1], [0, 0]]}
    monkeypatch.setattr(serve.lakes, "lake_at", lambda lat, lon, project=None: lake)
    assert client.get("/lake_at", params={"lat": 36.6, "lon": 9.4}).json() == lake


def test_lake_lookup_404_when_no_water_body_and_422_for_bad_coordinates(client, monkeypatch):
    from src import serve
    monkeypatch.setattr(serve.lakes, "lake_at", lambda lat, lon, project=None: None)
    assert client.get("/lake_at", params={"lat": 20.0, "lon": 20.0}).status_code == 404
    assert client.get("/lake_at", params={"lat": 120.0, "lon": 20.0}).status_code == 422


def test_predict_is_logged_for_monitoring_and_counted_in_metrics(client):
    from src import serve

    def predictions_total():
        line = next(l for l in client.get("/metrics").text.splitlines() if l.startswith("model_predictions_total "))
        return float(line.split()[1])

    before = predictions_total()
    client.post("/predict", data={"site": "Sidi Salem"}, files={"file": ("t.png", _sample_image_bytes(), "image/png")})

    recent = serve.monitoring.recent_predictions(serve._monitoring_db())
    assert len(recent) == 1 and recent.loc[0, "site"] == "Sidi Salem"
    text = client.get("/metrics").text
    assert 'http_requests_total{method="POST",route="/predict",status="200"}' in text
    assert predictions_total() == before + 1
    assert "process_memory_bytes" in text and "host_cpu_percent" in text


def test_metrics_label_routes_by_template_not_by_url(client):
    client.get("/history/Sidi Salem")
    client.get("/history/Nowhere")
    text = client.get("/metrics").text
    assert 'route="/history/{site}"' in text and "Nowhere" not in text


def test_monitoring_endpoint_reports_system_and_model_state(client):
    body = client.get("/monitoring").json()
    assert {"system", "data_drift", "forecast_performance", "retrain_needed", "recent_predictions"} <= set(body)
    assert body["system"]["host_memory_percent"] > 0


def test_sites_and_history_carry_the_calibrated_fill_rate(client):
    from src import serve

    db_path = serve._state["params"]["paths"]["history_db"]
    record_reading(db_path, "Sidi Salem", 75.0, source="live")

    sites = {s["name"]: s for s in client.get("/sites").json()}
    fill = sites["Sidi Salem"]["latest_reading"]["fill_estimate"]
    assert fill is None or 0 <= fill["fill_pct"] <= 100
    history = client.get("/history/Sidi Salem").json()
    assert "fill_pct" in history[-1]
    assert (history[-1]["fill_pct"] is None) == (fill is None)


def test_geocode_endpoint_validates_and_forwards(client, monkeypatch):
    from src import serve

    monkeypatch.setattr(serve.place_report, "geocode", lambda q: [{"name": q, "lat": 1.0, "lon": 2.0}])
    assert client.get("/geocode", params={"q": "a"}).status_code == 422
    assert client.get("/geocode", params={"q": "Mead"}).json()[0]["name"] == "Mead"


def test_geocode_endpoint_hides_upstream_failures(client, monkeypatch):
    from src import serve

    def broken(q):
        raise RuntimeError("down")

    monkeypatch.setattr(serve.place_report, "geocode", broken)
    assert client.get("/geocode", params={"q": "Mead"}).status_code == 502


def test_place_report_returns_context_and_survives_a_failing_part(client, monkeypatch):
    from src import serve

    record = {"record": [{"year": 2000 + i, "area_km2": 10.0, "pct_of_max": 100.0} for i in range(20)], "max_km2": 10.0, "years": [2000, 2019]}
    monkeypatch.setattr(serve.place_report, "water_record", lambda *a, **k: record)

    def no_climate(*a, **k):
        raise RuntimeError("open-meteo down")

    monkeypatch.setattr(serve.place_report, "climate_summary", no_climate)
    body = client.post("/place_report", json={"lat": 36.1, "lon": -114.4}).json()

    assert body["climate"] is None and body["record"]["max_km2"] == 10.0
    assert any("Climat" in n for n in body["notes"]) and body["assessment"]["basis"] == "contexte"


def test_assess_endpoint_combines_a_measurement_with_context(client):
    body = client.post("/assess", json={"surface_pct": 35, "climate": {"anomaly_pct": 100}}).json()
    assert body["level"] == "alerte" and body["basis"] == "mesure"
