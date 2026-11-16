import numpy as np

from src.water_quality import analyze, compute_quality_metrics, recommend_usage


def test_compute_quality_metrics_none_when_no_water():
    assert compute_quality_metrics(np.empty((0, 3))) is None


def test_clear_water_recommends_irrigation_or_correct_quality():
    pixels = np.tile(np.array([80, 110, 160]), (100, 1))
    metrics = compute_quality_metrics(pixels)
    usage = recommend_usage(metrics)
    assert usage["label"] in {"qualité apparente correcte", "irrigation possible"}


def test_green_dominant_water_flags_treatment_required():
    pixels = np.tile(np.array([40, 160, 30]), (100, 1))
    metrics = compute_quality_metrics(pixels)
    usage = recommend_usage(metrics)
    assert usage["label"] == "traitement requis"


def test_dark_turbid_water_recommends_irrigation_only():
    pixels = np.tile(np.array([30, 35, 40]), (100, 1))
    metrics = compute_quality_metrics(pixels)
    usage = recommend_usage(metrics)
    assert usage["label"] == "irrigation possible"


def test_analyze_end_to_end_on_synthetic_image():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[:2, :, :] = [60, 90, 150]
    mask = np.zeros((4, 4), dtype=bool)
    mask[:2, :] = True

    result = analyze(image, mask)
    assert result["metrics"] is not None
    assert "label" in result["usage"]
