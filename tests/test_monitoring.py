import numpy as np
import pandas as pd
import pytest

from src import monitoring


def _reference(rng, n=300):
    return {name: list(rng.normal(50, 10, n)) for name in monitoring.FEATURES}


def _recent(rng, n=120, shift=0.0):
    return pd.DataFrame({name: rng.normal(50 + shift, 10, n) for name in monitoring.FEATURES})


def test_psi_is_near_zero_for_the_same_distribution_and_large_for_a_shifted_one():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 2000)
    assert monitoring.psi(base, rng.normal(0, 1, 2000)) < 0.05
    assert monitoring.psi(base, rng.normal(1.5, 1, 2000)) > 0.25


def test_feature_needs_both_psi_and_ks_to_alert():
    assert monitoring.feature_status(0.4, 0.0001) == "alert"
    assert monitoring.feature_status(0.15, 0.0001) == "warn"
    assert monitoring.feature_status(0.4, 0.5) == "ok"
    assert monitoring.feature_status(0.02, 0.0001) == "ok"


def test_no_alarm_on_normal_traffic_and_alarm_on_a_shift():
    rng = np.random.default_rng(1)
    reference = _reference(rng)
    assert monitoring.drift_report(reference, _recent(rng))["status"] == "ok"
    shifted = monitoring.drift_report(reference, _recent(rng, shift=25))
    assert shifted["status"] == "alert" and "brightness" in shifted["flagged"]


def test_drift_report_says_when_it_cannot_judge():
    rng = np.random.default_rng(2)
    assert monitoring.drift_report(None, _recent(rng))["status"] == "no_reference"
    assert monitoring.drift_report(_reference(rng), _recent(rng, n=5))["status"] == "insufficient_data"


def test_image_features_describe_the_image_and_the_prediction():
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[:, :4] = 200
    probabilities = np.zeros((8, 8))
    probabilities[:, :4] = 1.0
    features = monitoring.image_features(rgb, 123.0, probabilities, np.ones((8, 8), dtype=bool))
    assert features["brightness"] == pytest.approx(100) and features["sharpness"] == 123.0
    assert features["water_pct"] == pytest.approx(50) and features["confidence"] == pytest.approx(1.0)


def test_logged_predictions_round_trip(tmp_path):
    db = str(tmp_path / "m.db")
    for value in (10, 20, 30):
        monitoring.log_prediction(db, "Lakhmes", {**{f: value for f in monitoring.FEATURES}})
    frame = monitoring.recent_predictions(db, 2)
    assert len(frame) == 2 and list(frame["brightness"]) == [30, 20]


def _logged_forecast(db, base, predicted):
    deep = {"origin_week": "2025-01-06", "current_pct": base, "horizons": [
        {"weeks": 1, "target_week": "2025-01-13", "predicted_pct": predicted}]}
    monitoring.log_forecast(db, "Lakhmes", deep)


def _observed(actual):
    return pd.DataFrame({"site": ["Lakhmes"], "week": ["2025-01-13"], "water_pct": [actual]})


def test_forecast_performance_waits_for_the_real_week(tmp_path):
    db = str(tmp_path / "m.db")
    _logged_forecast(db, 50, 55)
    assert monitoring.forecast_performance(db, _observed(np.nan))["status"] == "waiting_for_actuals"


def test_forecast_performance_compares_with_the_no_change_baseline(tmp_path):
    good, bad = str(tmp_path / "good.db"), str(tmp_path / "bad.db")
    _logged_forecast(good, 50, 58)
    _logged_forecast(bad, 50, 70)
    ok = monitoring.forecast_performance(good, _observed(60))
    assert ok["horizons"][1]["model_mae"] == 2 and ok["horizons"][1]["no_change_mae"] == 10
    assert ok["horizons"][1]["skill"] == pytest.approx(0.8)
    worse = monitoring.forecast_performance(bad, _observed(52), window=1)
    assert worse["horizons"][1]["skill"] < 0


def test_degradation_needs_enough_matured_forecasts_to_trigger(tmp_path):
    db = str(tmp_path / "m.db")
    for site in "ABCDE":
        monitoring.log_forecast(db, site, {"origin_week": "2025-01-06", "current_pct": 50, "horizons": [
            {"weeks": 1, "target_week": "2025-01-13", "predicted_pct": 80}]})
    observed = pd.DataFrame({"site": list("ABCDE"), "week": ["2025-01-13"] * 5, "water_pct": [50] * 5})
    assert monitoring.forecast_performance(db, observed)["status"] == "degraded"
    assert monitoring.forecast_performance(db, observed.head(2))["status"] == "ok"


def test_check_all_turns_drift_into_a_retraining_decision(tmp_path, monkeypatch):
    rng = np.random.default_rng(3)
    ref = tmp_path / "ref.json"
    import json
    ref.write_text(json.dumps({"features": _reference(rng)}))
    monkeypatch.setattr(monitoring, "REFERENCE_PATH", str(ref))
    db = str(tmp_path / "m.db")
    for _, row in _recent(rng, 60, shift=25).iterrows():
        monitoring.log_prediction(db, "x", row.to_dict())
    report = monitoring.check_all(db, weekly_path=str(tmp_path / "none.csv"), report_path=str(tmp_path / "r.json"))
    assert report["retrain_needed"] is True and report["reasons"][0].startswith("input data drift")
    assert (tmp_path / "r.json").exists()
