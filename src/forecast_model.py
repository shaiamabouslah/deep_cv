"""Deep-learning water-level forecaster: a small GRU that reads the last
WINDOW weeks of a dam (water area, observation freshness, rainfall,
evaporation, temperature, season) plus the rainfall expected over the next
two weeks, and predicts the change in water area at +1, +2 and +4 weeks.

It is meant to be compared, honestly, with simple baselines on the same
windows (persistence, the linear trend the app used before, a seasonal
average, ridge regression) — a deep model only earns its place if it beats
them. See src/train_forecaster.py and reports/forecast_eval.md.

No leakage by construction: missing weeks are carried forward from the last
real observation (never interpolated toward a future one) and an "age"
channel says how stale that value is; targets are real observations only.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from torch import nn

HORIZONS = (1, 2, 4)
WINDOW = 12
MAX_FFILL = 4
TREND_WEEKS = 8
FUTURE_WEEKS = 2
EVENT_THRESHOLD_PCT = 5.0
DELTA_SCALE = 10.0


@dataclass
class Windows:
    x: np.ndarray
    future: np.ndarray
    y: np.ndarray
    base: np.ndarray
    trend: np.ndarray
    origin: np.ndarray
    site: np.ndarray
    woy: np.ndarray

    def __len__(self):
        return len(self.base)

    def take(self, mask):
        return Windows(**{name: getattr(self, name)[mask] for name in self.__dataclass_fields__})


def _std(values):
    value = float(np.nanstd(values))
    return value if value > 1e-6 else 1.0


def compute_stats(df):
    rain = np.log1p(df["rain_mm"].clip(lower=0))
    return {
        "rain_mean": float(np.nanmean(rain)), "rain_std": _std(rain),
        "evap_mean": float(np.nanmean(df["evap_mm"])), "evap_std": _std(df["evap_mm"]),
        "temp_mean": float(np.nanmean(df["temp_c"])), "temp_std": _std(df["temp_c"]),
    }


def _zscore(values, mean, std):
    return np.nan_to_num((values - mean) / std, nan=0.0)


def _prepare(group, stats, use_weather):
    """Per-week feature matrix of one dam, built causally (see module docstring)."""
    group = group.sort_values("week").reset_index(drop=True)
    n = len(group)
    water = group["water_pct"].to_numpy(float)
    observed = ~np.isnan(water)
    filled = pd.Series(water).ffill(limit=MAX_FFILL).to_numpy()
    position = np.arange(n)
    last_seen = pd.Series(np.where(observed, position, np.nan)).ffill().to_numpy()
    age = np.where(np.isnan(last_seen), MAX_FFILL, np.minimum(position - last_seen, MAX_FFILL))

    weeks = group["week"]
    angle = 2 * np.pi * weeks.dt.isocalendar().week.to_numpy().astype(float) / 52.0
    rain = _zscore(np.log1p(group["rain_mm"].clip(lower=0).to_numpy(float)), stats["rain_mean"], stats["rain_std"])
    evap = _zscore(group["evap_mm"].to_numpy(float), stats["evap_mean"], stats["evap_std"])
    temp = _zscore(group["temp_c"].to_numpy(float), stats["temp_mean"], stats["temp_std"])

    columns = [np.nan_to_num(filled / 100.0), observed.astype(float), age / MAX_FFILL, np.sin(angle), np.cos(angle)]
    if use_weather:
        columns += [rain, evap, temp]
    features = np.stack(columns, axis=1).astype(np.float32)
    return {"n": n, "water": water, "observed": observed, "filled": filled, "weeks": weeks,
            "rain": rain, "features": features}


def latest_window(group, stats, use_weather=True):
    """The model input for the most recent week of `group`, which has no
    future targets yet. Returns (x[1, WINDOW, F], current level, week), or
    None when the last WINDOW weeks are not all backed by a recent real
    observation."""
    p = _prepare(group, stats, use_weather)
    if p["n"] < WINDOW or np.isnan(p["filled"][-WINDOW:]).any():
        return None
    return p["features"][-WINDOW:][None], float(p["filled"][-1]), p["weeks"].iloc[-1]


def _site_windows(group, stats, use_weather):
    p = _prepare(group, stats, use_weather)
    n, water, observed, filled = p["n"], p["water"], p["observed"], p["filled"]
    weeks, rain, features = p["weeks"], p["rain"], p["features"]

    out = {name: [] for name in ("x", "future", "y", "base", "trend", "origin", "woy")}
    for i in range(WINDOW - 1, n - FUTURE_WEEKS):
        window = slice(i - WINDOW + 1, i + 1)
        if np.isnan(filled[window]).any():
            continue
        targets = np.array([water[i + h] if i + h < n else np.nan for h in HORIZONS])
        if np.isnan(targets).all():
            continue

        recent = slice(max(0, i - TREND_WEEKS + 1), i + 1)
        points = np.where(observed[recent])[0]
        trend = np.full(len(HORIZONS), filled[i])
        if len(points) >= 3:
            slope = np.polyfit(points, water[recent][points], 1)[0]
            trend = np.clip(filled[i] + slope * np.array(HORIZONS), 0, 100)

        out["x"].append(features[window])
        out["future"].append([rain[i + 1], rain[i + 2]])
        out["y"].append(targets)
        out["base"].append(filled[i])
        out["trend"].append(trend)
        out["origin"].append(weeks[i])
        out["woy"].append(int(weeks[i].isocalendar().week))
    return out


def make_windows(df, stats, use_weather=True):
    parts, sites = [], []
    for name, group in df.groupby("site"):
        part = _site_windows(group, stats, use_weather)
        parts.append(part)
        sites += [name] * len(part["base"])
    n_features = 8 if use_weather else 5
    return Windows(
        x=np.concatenate([np.array(p["x"], dtype=np.float32).reshape(-1, WINDOW, n_features) for p in parts]),
        future=np.concatenate([np.array(p["future"], dtype=np.float32).reshape(-1, FUTURE_WEEKS) for p in parts]),
        y=np.concatenate([np.array(p["y"], dtype=float).reshape(-1, len(HORIZONS)) for p in parts]),
        base=np.concatenate([np.array(p["base"], dtype=float) for p in parts]),
        trend=np.concatenate([np.array(p["trend"], dtype=float).reshape(-1, len(HORIZONS)) for p in parts]),
        origin=np.concatenate([np.array(p["origin"], dtype="datetime64[ns]") for p in parts]),
        site=np.array(sites),
        woy=np.concatenate([np.array(p["woy"], dtype=int) for p in parts]),
    )


def split_by_time(windows, train_end, val_end):
    """Origins are split by date; train/val origins stop max(HORIZONS) weeks
    before the boundary so no training target falls inside the next period."""
    margin = np.timedelta64(7 * max(HORIZONS), "D")
    origin = windows.origin
    train = origin <= np.datetime64(train_end) - margin
    val = (origin > np.datetime64(train_end)) & (origin <= np.datetime64(val_end) - margin)
    test = origin > np.datetime64(val_end)
    return windows.take(train), windows.take(val), windows.take(test)


class GRUForecaster(nn.Module):
    def __init__(self, n_features, n_future, hidden=32):
        super().__init__()
        self.n_future = n_future
        self.gru = nn.GRU(n_features, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + n_future, 32), nn.ReLU(), nn.Linear(32, len(HORIZONS)))

    def forward(self, x, future):
        state = self.gru(x)[1][-1]
        return self.head(torch.cat([state, future[:, : self.n_future]], dim=1))


def _tensors(windows):
    base = torch.tensor(windows.base, dtype=torch.float32)
    target = torch.tensor((windows.y - windows.base[:, None]) / DELTA_SCALE, dtype=torch.float32)
    mask = ~torch.isnan(target)
    return torch.tensor(windows.x), torch.tensor(windows.future), base, torch.nan_to_num(target), mask


def predict_gru(model, windows):
    x, future, base, _, _ = _tensors(windows)
    model.eval()
    with torch.no_grad():
        delta = model(x, future) * DELTA_SCALE
    return np.clip(base[:, None].numpy() + delta.numpy(), 0, 100)


def masked_mae(pred, windows):
    errors = np.abs(pred - windows.y)
    return float(np.nanmean(errors)) if np.isfinite(errors).any() else float("nan")


def train_gru(train, val, seed, n_future=FUTURE_WEEKS, hidden=32, epochs=80, patience=12,
              lr=3e-3, batch_size=64, weight_decay=1e-4):
    torch.manual_seed(seed)
    model = GRUForecaster(train.x.shape[2], n_future, hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    x, future, _, target, mask = _tensors(train)
    best_state, best_val, best_epoch, stale = None, float("inf"), 0, 0

    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(x))
        for start in range(0, len(x), batch_size):
            batch = order[start:start + batch_size]
            diff = (model(x[batch], future[batch]) - target[batch]).abs() * mask[batch]
            loss = diff.sum() / mask[batch].sum().clamp(min=1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if len(val):
            score = masked_mae(predict_gru(model, val), val)
            if score < best_val - 1e-4:
                best_val, best_epoch, stale = score, epoch + 1, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch


def ridge_predict(train, test, alpha=10.0):
    flat = lambda w: np.concatenate([w.x.reshape(len(w), -1), w.future], axis=1)
    train_x, test_x = flat(train), flat(test)
    pred = np.zeros((len(test), len(HORIZONS)))
    for j in range(len(HORIZONS)):
        usable = ~np.isnan(train.y[:, j])
        model = Ridge(alpha=alpha).fit(train_x[usable], train.y[usable, j] - train.base[usable])
        pred[:, j] = np.clip(test.base + model.predict(test_x), 0, 100)
    return pred


def seasonal_predict(train, test, bandwidth=2):
    """Mean training change for the same time of year, whatever the weather."""
    delta = train.y - train.base[:, None]
    pred = np.zeros((len(test), len(HORIZONS)))
    for i, week in enumerate(test.woy):
        gap = np.abs(train.woy - week)
        near = np.minimum(gap, 52 - gap) <= bandwidth
        for j in range(len(HORIZONS)):
            usable = near & ~np.isnan(delta[:, j])
            pred[i, j] = test.base[i] + (delta[usable, j].mean() if usable.any() else 0.0)
    return np.clip(pred, 0, 100)


def persistence_predict(windows):
    return np.repeat(windows.base[:, None], len(HORIZONS), axis=1)


def score(pred, windows):
    """MAE in percentage points per horizon, overall and on event weeks
    (where the level really moved by EVENT_THRESHOLD_PCT or more)."""
    result = {}
    for j, horizon in enumerate(HORIZONS):
        known = ~np.isnan(windows.y[:, j])
        event = known & (np.abs(windows.y[:, j] - windows.base) >= EVENT_THRESHOLD_PCT)
        err = np.abs(pred[:, j] - windows.y[:, j])
        result[f"h{horizon}"] = {
            "mae": float(err[known].mean()) if known.any() else None, "n": int(known.sum()),
            "event_mae": float(err[event].mean()) if event.any() else None, "n_event": int(event.sum()),
        }
    return result
