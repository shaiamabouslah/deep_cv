"""Is the forecaster's advantage real or noise? Paired block bootstrap of the
MAE difference between methods on the test windows, resampling whole
calendar weeks (the dams move together, so windows of one week are not
independent). Also reports seed spread and per-dam results.

    python -m src.forecast_significance
"""
import numpy as np
import pandas as pd

from src import forecast_model as fm
from src import train_forecaster as tf

COMPARISONS = [
    ("gru_no_future_rain", "persistence"), ("gru_no_future_rain", "linear_trend"),
    ("gru_no_future_rain", "gru_full"), ("gru_no_future_rain", "gru_no_weather"),
    ("gru_no_future_rain", "seasonal_delta"), ("persistence", "linear_trend"),
]


def block_bootstrap(errors_a, errors_b, known, weeks, n=4000, seed=0):
    """Mean and 95% interval of mean(errors_a) - mean(errors_b), resampling weeks."""
    rng = np.random.default_rng(seed)
    unique_weeks = np.unique(weeks)
    by_week = [np.where((weeks == w) & known)[0] for w in unique_weeks]
    diffs = np.empty(n)
    for i in range(n):
        idx = np.concatenate([by_week[p] for p in rng.integers(0, len(by_week), len(by_week))])
        diffs[i] = errors_a[idx].mean() - errors_b[idx].mean()
    return float(diffs.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def run(csv_path=tf.CSV_PATH, seeds=tf.SEEDS):
    df = pd.read_csv(csv_path, parse_dates=["week"])
    stats = fm.compute_stats(df[df["week"] < tf.TRAIN_END])
    full = fm.make_windows(df, stats, use_weather=True)
    no_weather = fm.make_windows(df, stats, use_weather=False)
    preds, test, _ = tf.experiment(full, no_weather, seeds)
    ensemble = {name: np.mean(runs, axis=0) for name, runs in preds.items()}

    print(f"{len(test)} test windows, {len(np.unique(test.origin))} distinct weeks")
    print("paired MAE difference (first minus second), 95% interval over weeks; negative = first better")
    for a, b in COMPARISONS:
        print(f"\n{a} vs {b}")
        for j, horizon in enumerate(fm.HORIZONS):
            known = ~np.isnan(test.y[:, j])
            err = lambda name: np.abs(ensemble[name][:, j] - test.y[:, j])
            mean, lo, hi = block_bootstrap(err(a), err(b), known, test.origin)
            verdict = "better" if hi < 0 else "worse" if lo > 0 else "not distinguishable"
            print(f"  +{horizon} wk: {mean:+.3f}  [{lo:+.3f}, {hi:+.3f}]  -> {verdict}")

    print("\nper-dam MAE at +4 weeks")
    j = fm.HORIZONS.index(4)
    for site in sorted(set(test.site)):
        mask = (test.site == site) & ~np.isnan(test.y[:, j])
        row = {name: float(np.abs(ensemble[name][mask, j] - test.y[mask, j]).mean())
               for name in ("gru_no_future_rain", "persistence", "linear_trend")}
        print(f"  {site:15s} " + "  ".join(f"{k}={v:.2f}" for k, v in row.items()))


if __name__ == "__main__":
    run()
