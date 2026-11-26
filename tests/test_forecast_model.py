import numpy as np
import pandas as pd
import pytest

from src import forecast_model as fm


def _frame(n_weeks=80, site="A", seed=0, wet_effect=6.0):
    """Synthetic dam: level rises right after rainy weeks and slowly falls
    otherwise — a pattern a model that reads the rain can learn."""
    rng = np.random.default_rng(seed)
    weeks = pd.date_range("2020-01-06", periods=n_weeks, freq="W-MON")
    rain = np.where(rng.random(n_weeks) < 0.25, rng.uniform(20, 60, n_weeks), 0.0)
    level = np.zeros(n_weeks)
    level[0] = 50
    for t in range(1, n_weeks):
        level[t] = np.clip(level[t - 1] - 0.5 + (wet_effect if rain[t] > 0 else 0), 0, 100)
    return pd.DataFrame({
        "site": site, "week": weeks, "water_pct": level, "visible_fraction": 1.0,
        "rain_mm": rain, "evap_mm": 10 + rng.normal(0, 1, n_weeks), "temp_c": 15 + rng.normal(0, 2, n_weeks),
    })


def test_windows_have_expected_shapes():
    df = _frame()
    windows = fm.make_windows(df, fm.compute_stats(df))
    assert windows.x.shape[1:] == (fm.WINDOW, 8)
    assert windows.future.shape[1] == fm.FUTURE_WEEKS
    assert windows.y.shape[1] == len(fm.HORIZONS)
    assert len(windows) == len(windows.base) == len(windows.trend)


def test_missing_weeks_are_carried_forward_not_interpolated():
    df = _frame()
    df.loc[40:42, "water_pct"] = np.nan
    windows = fm.make_windows(df, fm.compute_stats(df), use_weather=False)
    origin = windows.origin == df["week"].iloc[42].to_datetime64()
    assert origin.sum() == 1
    last_seen = df["water_pct"].iloc[39]
    assert windows.base[origin][0] == pytest.approx(last_seen)
    assert windows.x[origin][0, -1, 1] == 0.0
    assert windows.x[origin][0, -1, 2] == pytest.approx(3 / fm.MAX_FFILL)


def test_unobserved_target_weeks_are_masked_not_invented():
    df = _frame()
    df.loc[50, "water_pct"] = np.nan
    windows = fm.make_windows(df, fm.compute_stats(df))
    origin = windows.origin == df["week"].iloc[49].to_datetime64()
    assert np.isnan(windows.y[origin][0, 0])
    assert not np.isnan(windows.y[origin][0, 1])


def test_long_gaps_drop_the_window():
    df = _frame()
    df.loc[30:40, "water_pct"] = np.nan
    windows = fm.make_windows(df, fm.compute_stats(df))
    dropped = windows.origin == df["week"].iloc[38].to_datetime64()
    assert dropped.sum() == 0


def test_time_split_keeps_targets_inside_their_period():
    df = _frame(n_weeks=200)
    windows = fm.make_windows(df, fm.compute_stats(df))
    train_end, val_end = "2022-06-01", "2023-06-01"
    train, val, test = fm.split_by_time(windows, train_end, val_end)
    assert len(train) and len(val) and len(test)
    weeks_ahead = np.timedelta64(7 * max(fm.HORIZONS), "D")
    assert train.origin.max() + weeks_ahead <= np.datetime64(train_end)
    assert val.origin.min() > np.datetime64(train_end)
    assert val.origin.max() + weeks_ahead <= np.datetime64(val_end)
    assert test.origin.min() > np.datetime64(val_end)


def test_persistence_and_trend_baselines():
    df = _frame()
    df["water_pct"] = 40 + 2 * np.arange(len(df)) * 0.2
    windows = fm.make_windows(df, fm.compute_stats(df))
    assert np.allclose(fm.persistence_predict(windows), windows.base[:, None])
    perfect = fm.score(windows.trend, windows)
    assert perfect["h1"]["mae"] < 0.05
    assert fm.score(fm.persistence_predict(windows), windows)["h4"]["mae"] > 1.0


def test_score_reports_event_error_separately():
    df = _frame()
    windows = fm.make_windows(df, fm.compute_stats(df))
    result = fm.score(fm.persistence_predict(windows), windows)
    assert result["h1"]["n_event"] > 0
    assert result["h1"]["event_mae"] > result["h1"]["mae"]


def test_gru_learns_that_rain_raises_the_level():
    df = pd.concat([_frame(260, "A", 1), _frame(260, "B", 2), _frame(260, "C", 3)], ignore_index=True)
    stats = fm.compute_stats(df)
    windows = fm.make_windows(df, stats)
    train, val, test = fm.split_by_time(windows, "2023-12-01", "2024-12-01")

    model, best_epoch = fm.train_gru(train, val, seed=0, epochs=60)
    gru = fm.score(fm.predict_gru(model, test), test)
    persistence = fm.score(fm.persistence_predict(test), test)

    assert best_epoch >= 1
    assert gru["h1"]["mae"] < 0.5 * persistence["h1"]["mae"]


def test_gru_without_future_rain_cannot_see_the_next_week_rain():
    df = pd.concat([_frame(260, "A", 1), _frame(260, "B", 2)], ignore_index=True)
    windows = fm.make_windows(df, fm.compute_stats(df))
    train, val, test = fm.split_by_time(windows, "2023-12-01", "2024-12-01")

    with_rain, _ = fm.train_gru(train, val, seed=0, n_future=fm.FUTURE_WEEKS, epochs=60)
    without, _ = fm.train_gru(train, val, seed=0, n_future=0, epochs=60)

    assert fm.score(fm.predict_gru(with_rain, test), test)["h1"]["mae"] < fm.score(fm.predict_gru(without, test), test)["h1"]["mae"]


def test_ridge_and_seasonal_baselines_return_valid_predictions():
    df = _frame(n_weeks=200)
    windows = fm.make_windows(df, fm.compute_stats(df))
    train, val, test = fm.split_by_time(windows, "2022-06-01", "2023-06-01")
    for pred in (fm.ridge_predict(train, test), fm.seasonal_predict(train, test)):
        assert pred.shape == (len(test), len(fm.HORIZONS))
        assert (pred >= 0).all() and (pred <= 100).all()
