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
    assert all("lat" in s and "lon" in s for s in body)
