import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture()
def client(tmp_path):
    from src import serve
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


def test_history_accumulates_readings(client):
    client.post(
        "/predict",
        data={"site": "Sidi Salem"},
        files={"file": ("test.png", _sample_image_bytes(), "image/png")},
    )
    client.post(
        "/predict",
        data={"site": "Sidi Salem"},
        files={"file": ("test.png", _sample_image_bytes(), "image/png")},
    )
    history = client.get("/history/Sidi Salem").json()
    assert len(history) == 2


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
    client.post(
        "/predict",
        data={"site": "Sidi Salem"},
        files={"file": ("test.png", _sample_image_bytes(), "image/png")},
    )

    response = client.get("/forecast/Sidi Salem")

    assert response.status_code == 200
    body = response.json()
    assert body["site"] == "Sidi Salem"
    assert body["insufficient_history"] is True
