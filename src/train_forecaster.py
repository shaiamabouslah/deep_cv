"""Trains and evaluates the deep-learning forecaster against honest baselines.

Two protocols, both with time-disjoint train / validation / test periods:
  - temporal holdout: the model has seen every dam, but only up to the
    cutoff; it is scored on the weeks after (the real deployment situation)
  - leave-one-dam-out: each dam in turn is hidden from training and scored
    on the test period (how it would behave on a dam it has never seen,
    such as a site a user draws on the map)

Three GRU variants isolate what actually helps: full (history + weather +
the next two weeks of rain), no future rain, and no weather at all. Each is
trained with several seeds because the data is small; spread is reported.
Baselines run on exactly the same windows: persistence, the linear trend the
app used before, the seasonal average change, and ridge regression.

    python -m src.train_forecaster
"""
import json
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from src import forecast_model as fm

CSV_PATH = "data/forecast/weekly.csv"
MODEL_PATH = "models/forecaster.pt"
REPORT_PATH = "reports/forecast_eval.json"
TRAIN_END = "2024-07-01"
VAL_END = "2025-07-01"
SEEDS = 5
LODO_SEEDS = 3

VARIANTS = {
    "gru_full": {"weather": True, "n_future": fm.FUTURE_WEEKS},
    "gru_no_future_rain": {"weather": True, "n_future": 0},
    "gru_no_weather": {"weather": False, "n_future": 0},
}
# The one the app serves: the next-two-weeks rain forecast did not help (even
# with perfect rain in training), and leaving it out removes a train/serve
# mismatch (real rain at training, a noisy forecast at serving).
SERVED_VARIANT = "gru_no_future_rain"


def _split(windows, held_out=None):
    train, val, test = fm.split_by_time(windows, TRAIN_END, VAL_END)
    if held_out is not None:
        train, val = train.take(train.site != held_out), val.take(val.site != held_out)
        test = test.take(test.site == held_out)
    return train, val, test


def experiment(full, no_weather, seeds, held_out=None):
    """Returns ({method: [prediction array per run]}, test windows, the
    trained GRUs of the served variant)."""
    train, val, test = _split(full, held_out)
    train_nw, val_nw, test_nw = _split(no_weather, held_out)

    preds = {
        "persistence": [fm.persistence_predict(test)],
        "linear_trend": [test.trend],
        "seasonal_delta": [fm.seasonal_predict(train, test)],
        "ridge": [fm.ridge_predict(train, test)],
    }
    models = []
    for name, config in VARIANTS.items():
        tr, va, te = (train, val, test) if config["weather"] else (train_nw, val_nw, test_nw)
        runs = []
        for seed in range(seeds):
            model, _ = fm.train_gru(tr, va, seed=seed, n_future=config["n_future"])
            runs.append(fm.predict_gru(model, te))
            if name == SERVED_VARIANT:
                models.append(model)
        preds[name] = runs
    return preds, test, models


def summarise(preds, test):
    summary = {}
    ensembles = {f"{name}_ensemble": [np.mean(runs, axis=0)] for name, runs in preds.items() if len(runs) > 1}
    for method, runs in {**preds, **ensembles}.items():
        scores = [fm.score(run, test) for run in runs]
        summary[method] = {}
        for horizon in fm.HORIZONS:
            key = f"h{horizon}"
            maes = [s[key]["mae"] for s in scores]
            events = [s[key]["event_mae"] for s in scores if s[key]["event_mae"] is not None]
            summary[method][key] = {
                "mae": float(np.mean(maes)), "mae_std": float(np.std(maes)),
                "event_mae": float(np.mean(events)) if events else None,
                "n": scores[0][key]["n"], "n_event": scores[0][key]["n_event"],
            }
    return summary


def pooled(folds):
    """Concatenates held-out-dam folds into one test set per method/run."""
    methods = folds[0][0].keys()
    preds = {m: [np.concatenate([fold[0][m][r] for fold in folds]) for r in range(len(folds[0][0][m]))] for m in methods}
    test = SimpleNamespace(
        y=np.concatenate([fold[1].y for fold in folds]), base=np.concatenate([fold[1].base for fold in folds])
    )
    return preds, test


def format_table(summary, key="mae"):
    header = f"{'method':28s}" + "".join(f"{'+' + str(h) + ' wk':>12s}" for h in fm.HORIZONS)
    lines = [header]
    for method, horizons in summary.items():
        cells = []
        for h in fm.HORIZONS:
            value = horizons[f"h{h}"][key]
            cells.append(f"{value:12.2f}" if value is not None else f"{'-':>12s}")
        lines.append(f"{method:28s}" + "".join(cells))
    return "\n".join(lines)


def log_run(primary, lodo, n_windows, n_test):
    from src import registry, tracking

    params = {"window_weeks": fm.WINDOW, "horizons": ",".join(map(str, fm.HORIZONS)), "train_end": TRAIN_END,
              "val_end": VAL_END, "seeds": SEEDS, "n_windows": n_windows, "n_test_windows": n_test,
              "model": "GRU hidden 32", "loss": "masked L1", "served_variant": SERVED_VARIANT}
    for protocol, summary in (("temporal_holdout", primary), ("leave_one_dam_out", lodo)):
        for method, metrics in summary.items():
            tracking.log_experiment("forecasting-water-level", f"{protocol} / {method}",
                                    params={**params, "protocol": protocol, "method": method}, metrics=metrics,
                                    tags={"source": "train_forecaster", "protocol": protocol})
    served = f"{SERVED_VARIANT}_ensemble"
    mae = {k: round(v["mae"], 3) for k, v in primary[served].items()}
    base = {k: round(v["mae"], 3) for k, v in primary["persistence"].items()}
    version = registry.register_forecaster(
        MODEL_PATH, f"GRU ensemble ({SEEDS} seeds), +1/+2/+4 weeks. Test MAE (pts) {mae} vs no-change {base}.",
        {f"test_mae_{k}": v for k, v in mae.items()} | {f"baseline_mae_{k}": v for k, v in base.items()}
        | {"test_mae_mean": round(sum(mae.values()) / len(mae), 4)},
        metrics={"test_mae": mae, "persistence_mae": base})
    promoted = registry.promote_if_better(registry.FORECASTER, version, "test_mae_mean", higher_is_better=False)
    print(f"registered {registry.FORECASTER} v{version}; champion: {promoted}")


def run(csv_path=CSV_PATH, track=False):
    df = pd.read_csv(csv_path, parse_dates=["week"])
    stats = fm.compute_stats(df[df["week"] < TRAIN_END])
    full = fm.make_windows(df, stats, use_weather=True)
    no_weather = fm.make_windows(df, stats, use_weather=False)
    print(f"{len(full)} windows from {df['site'].nunique()} dams")

    print("\n== temporal holdout ==")
    preds, test, models = experiment(full, no_weather, SEEDS)
    primary = summarise(preds, test)
    print(f"test windows: {len(test)}  ({test.origin.min().astype('datetime64[D]')} to {test.origin.max().astype('datetime64[D]')})")
    print("MAE, percentage points:\n" + format_table(primary))
    print("MAE on event weeks (level moved >= 5 points):\n" + format_table(primary, "event_mae"))

    print("\n== leave-one-dam-out ==")
    folds = []
    for site in sorted(df["site"].unique()):
        fold_preds, fold_test, _ = experiment(full, no_weather, LODO_SEEDS, held_out=site)
        folds.append((fold_preds, fold_test))
        print(f"  held out {site}: {len(fold_test)} test windows", flush=True)
    lodo = summarise(*pooled(folds))
    print("MAE, percentage points:\n" + format_table(lodo))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    served = VARIANTS[SERVED_VARIANT]
    mae = lambda method: {k: round(v["mae"], 3) for k, v in primary[method].items()}
    torch.save({
        "variant": SERVED_VARIANT, "state_dicts": [m.state_dict() for m in models], "stats": stats,
        "n_features": full.x.shape[2] if served["weather"] else no_weather.x.shape[2],
        "n_future": served["n_future"], "window": fm.WINDOW, "horizons": list(fm.HORIZONS),
        "train_end": TRAIN_END, "val_end": VAL_END, "n_test": len(test),
        "test_mae": mae(f"{SERVED_VARIANT}_ensemble"), "persistence_mae": mae("persistence"),
        "linear_trend_mae": mae("linear_trend"),
    }, MODEL_PATH)

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "split": {"train_end": TRAIN_END, "val_end": VAL_END}, "seeds": SEEDS, "lodo_seeds": LODO_SEEDS,
            "n_windows": len(full), "n_test": len(test),
            "temporal_holdout": primary, "leave_one_dam_out": lodo,
        }, f, indent=2)
    print(f"\nsaved {MODEL_PATH} and {REPORT_PATH}")
    if track:
        log_run(primary, lodo, len(full), len(test))


if __name__ == "__main__":
    run(track=True)
