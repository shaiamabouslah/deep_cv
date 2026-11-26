import numpy as np
import torch
from torch import nn

from src import forecast_dl as dl
from src import forecast_model as fm
from tests.test_forecast_model import _frame


def _bundle(models=None):
    return {
        "models": models or [fm.GRUForecaster(8, 0).eval() for _ in range(2)],
        "stats": fm.compute_stats(_frame()), "n_future": 0, "horizons": list(fm.HORIZONS),
        "test_mae": {"h1": 2.0, "h2": 2.5, "h4": 3.4}, "persistence_mae": {"h1": 2.1, "h2": 2.8, "h4": 4.0},
        "train_end": "2024-07-01", "val_end": "2025-07-01", "n_test": 425,
    }


class _Constant(nn.Module):
    def __init__(self, delta_pts):
        super().__init__()
        self.delta = delta_pts / fm.DELTA_SCALE

    def forward(self, x, future):
        return torch.full((len(x), len(fm.HORIZONS)), self.delta)


def test_predict_returns_every_horizon_with_dates_and_measured_errors():
    weekly = _frame(n_weeks=20)
    result = dl.predict(_bundle(), weekly)

    assert result["available"] is True
    assert [h["weeks"] for h in result["horizons"]] == [1, 2, 4]
    origin = weekly["week"].iloc[-1]
    assert result["origin_week"] == str(origin.date())
    assert result["horizons"][2]["target_week"] == str((origin + np.timedelta64(28, "D")).date())
    assert all(0 <= h["predicted_pct"] <= 100 for h in result["horizons"])
    assert result["horizons"][0]["typical_error_pts"] == 2.0
    assert result["horizons"][0]["no_change_error_pts"] == 2.1


def test_predict_refuses_when_recent_weeks_have_no_clear_scene():
    weekly = _frame(n_weeks=20)
    weekly.loc[6:, "water_pct"] = np.nan

    result = dl.predict(_bundle(), weekly)

    assert result["available"] is False
    assert "images claires" in result["reason"]


def test_predicted_collapse_raises_an_alert_level():
    weekly = _frame(n_weeks=20)
    weekly["water_pct"] = 60.0
    result = dl.predict(_bundle([_Constant(-40.0)]), weekly)

    assert result["horizons"][0]["predicted_pct"] == 20.0
    assert result["projected_alert_level"] == "critique"


def test_drop_smaller_than_the_models_own_error_raises_no_alert_even_on_a_tiny_base():
    weekly = _frame(n_weeks=20)
    weekly["water_pct"] = 3.4
    result = dl.predict(_bundle([_Constant(-0.7)]), weekly)

    assert result["horizons"][0]["predicted_pct"] == 2.7
    assert result["projected_alert_level"] is None


def test_stable_prediction_raises_no_alert():
    weekly = _frame(n_weeks=20)
    weekly["water_pct"] = 60.0
    assert dl.predict(_bundle([_Constant(0.0)]), weekly)["projected_alert_level"] is None


def test_missing_recent_weather_does_not_block_a_forecast():
    weekly = _frame(n_weeks=20)
    weekly.loc[17:, ["rain_mm", "evap_mm", "temp_c"]] = np.nan
    assert dl.predict(_bundle(), weekly)["available"] is True


def test_no_checkpoint_means_no_deep_forecast(monkeypatch):
    monkeypatch.setattr(dl, "get_bundle", lambda: None)
    assert dl.deep_forecast({"name": "X"}) is None


def test_forecast_is_cached_per_site(monkeypatch):
    dl._cache.clear()
    monkeypatch.setattr(dl, "get_bundle", _bundle)
    calls = []

    def fetcher(dam, project=None):
        calls.append(dam["name"])
        return _frame(n_weeks=20)

    first = dl.deep_forecast({"name": "A"}, fetcher=fetcher)
    second = dl.deep_forecast({"name": "A"}, fetcher=fetcher)
    dl.deep_forecast({"name": "B"}, fetcher=fetcher)

    assert first == second
    assert calls == ["A", "B"]


def test_unusable_data_is_not_cached_so_it_can_recover(monkeypatch):
    dl._cache.clear()
    monkeypatch.setattr(dl, "get_bundle", _bundle)
    stale = _frame(n_weeks=20)
    stale["water_pct"] = np.nan

    assert dl.deep_forecast({"name": "A"}, fetcher=lambda d, project=None: stale)["available"] is False
    assert dl.cached_forecast("A") is None


def test_checkpoint_round_trip_loads_with_torch_default_safe_mode(tmp_path):
    models = [fm.GRUForecaster(8, 0).eval() for _ in range(2)]
    bundle = _bundle(models)
    path = tmp_path / "forecaster.pt"
    torch.save({
        "state_dicts": [m.state_dict() for m in models], "stats": bundle["stats"], "n_features": 8, "n_future": 0,
        "horizons": bundle["horizons"], "test_mae": bundle["test_mae"], "persistence_mae": bundle["persistence_mae"],
        "train_end": "2024-07-01", "val_end": "2025-07-01", "n_test": 425,
    }, path)

    loaded = dl.load_bundle(str(path))
    weekly = _frame(n_weeks=20)

    assert len(loaded["models"]) == 2
    assert dl.predict(loaded, weekly)["horizons"] == dl.predict(bundle, weekly)["horizons"]
    assert dl.load_bundle(str(tmp_path / "missing.pt")) is None
