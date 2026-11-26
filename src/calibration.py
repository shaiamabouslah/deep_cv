"""Turns the satellite surface index into an estimate of the official fill
rate, per dam, using the ONAGRI/DGRE bulletins (src/official_fill.py).

The surface index is "water area as % of the reservoir's historical maximum
extent"; the official rate is stored volume as % of capacity. They are
linked by the reservoir's own shape (its area-to-volume curve), so the
mapping is monotone but different for every dam: a one-size exponent or
the raw surface are wrong (see reports/calibration.md). Per dam it is fitted
as an isotonic (monotone non-decreasing) regression anchored at the
physical end points (empty = 0, full = 100) and stored as a lookup grid.

A calibrated rate is only offered when it clearly beats a naive constant
guess on periods the fit did not see (time-blocked cross-validation):
dams whose surface carries no usable signal are reported as not calibrated.

    python -m src.calibration
"""
import json
import os

import numpy as np
import pandas as pd

from src import official_fill

CALIBRATION_PATH = "data/official/calibration.json"
REPORT_PATH = "reports/calibration.md"
WEEKLY_PATH = "data/forecast/weekly.csv"
GRID = np.arange(0, 100.5, 0.5)
MIN_POINTS = 30
MIN_SKILL = 0.3
N_BLOCKS = 5


def pair_points(official, weekly):
    """Official bulletin points matched with the satellite measurement of the same week."""
    frame = pd.DataFrame(official)
    frame["date"] = pd.to_datetime(frame["date"])
    frame["week"] = frame["date"].dt.to_period("W-SUN").dt.start_time
    weekly = weekly.rename(columns={"site": "dam"})[["dam", "week", "water_pct"]]
    merged = frame.merge(weekly, on=["dam", "week"], how="left").dropna(subset=["water_pct"])
    return merged.rename(columns={"water_pct": "surface", "official_fill_pct": "official"})[
        ["dam", "date", "surface", "official"]
    ]


def fit_curve(surface, official):
    """Monotone curve sampled on GRID, anchored at (0, 0) and (100, 100)."""
    from sklearn.isotonic import IsotonicRegression

    x = np.concatenate([[0.0, 100.0], np.asarray(surface, float)])
    y = np.concatenate([[0.0, 100.0], np.asarray(official, float)])
    model = IsotonicRegression(y_min=0, y_max=100, out_of_bounds="clip").fit(x, y)
    return model.predict(GRID).tolist()


def apply_curve(curve, surface):
    return np.interp(np.clip(surface, 0, 100), GRID, curve)


def cross_validate(pairs, n_blocks=N_BLOCKS):
    """Each dam, each contiguous block of dates in turn: fit on the other
    blocks, score on the held-out one. Returns per-dam error of the
    calibration, of the raw surface and of a constant (training mean)."""
    dates = np.array(sorted(pairs["date"].unique()))
    blocks = np.array_split(dates, n_blocks)
    result = {}
    for dam, group in pairs.groupby("dam"):
        errors = {"calibrated": [], "raw": [], "constant": []}
        for held in blocks:
            test, train = group[group["date"].isin(held)], group[~group["date"].isin(held)]
            if len(test) == 0 or len(train) < MIN_POINTS // 2:
                continue
            curve = fit_curve(train["surface"], train["official"])
            errors["calibrated"] += list(np.abs(apply_curve(curve, test["surface"].values) - test["official"].values))
            errors["raw"] += list(np.abs(test["surface"].values - test["official"].values))
            errors["constant"] += list(np.abs(train["official"].mean() - test["official"].values))
        if errors["calibrated"]:
            mae = {k: float(np.mean(v)) for k, v in errors.items()}
            mae["skill"] = 1 - mae["calibrated"] / max(mae["constant"], 1e-9)
            mae["n_tested"] = len(errors["calibrated"])
            result[dam] = mae
    return result


def build(official_points, weekly, path=CALIBRATION_PATH):
    pairs = pair_points(official_points, weekly)
    cv = cross_validate(pairs)
    dams = {}
    for dam, group in pairs.groupby("dam"):
        stats = cv.get(dam)
        usable = bool(stats and len(group) >= MIN_POINTS and stats["skill"] >= MIN_SKILL)
        dams[dam] = {
            "usable": usable, "n_points": int(len(group)),
            "official_range": [float(group["official"].min()), float(group["official"].max())],
            "cv": stats,
            "typical_error_pts": round(stats["calibrated"], 1) if stats else None,
            "curve": fit_curve(group["surface"], group["official"]) if usable else None,
            "period": [str(group["date"].min().date()), str(group["date"].max().date())],
        }
    out = {"grid_step": 0.5, "min_skill": MIN_SKILL, "dams": dams}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    return out


def load(path=CALIBRATION_PATH):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def estimate_fill(dam, surface_pct, calibration):
    """{"fill_pct", "typical_error_pts", "n_points"} for a calibrated dam, else None."""
    entry = (calibration or {}).get("dams", {}).get(dam)
    if not entry or not entry["usable"] or surface_pct is None:
        return None
    return {
        "fill_pct": round(float(apply_curve(entry["curve"], surface_pct)), 1),
        "typical_error_pts": entry["typical_error_pts"], "n_points": entry["n_points"],
        "period": entry["period"],
    }


def report(calibration):
    lines = [
        "# Surface index to official fill rate", "",
        "Per-dam monotone calibration (isotonic regression anchored at empty = 0 and full = 100) fitted on the "
        "ONAGRI/DGRE daily bulletins (one per week, 2024 to October 2025) paired with the satellite measurement "
        "of the same week. Errors are in fill-rate points, measured on periods the fit did not see (5 contiguous "
        "time blocks, each dam predicted from its other blocks).", "",
        "| Dam | Points | Raw surface error | Constant-guess error | Calibrated error | Skill vs constant | Offered |",
        "|---|---|---|---|---|---|---|",
    ]
    for dam, e in calibration["dams"].items():
        cv = e["cv"] or {}
        lines.append(
            f"| {dam} | {e['n_points']} | {cv.get('raw', float('nan')):.1f} | {cv.get('constant', float('nan')):.1f} | "
            f"{cv.get('calibrated', float('nan')):.1f} | {cv.get('skill', float('nan')):.2f} | "
            f"{'yes' if e['usable'] else 'no'} |"
        )
    lines += ["", f"A dam is offered a calibrated rate only if it has at least {MIN_POINTS} points and a skill of at "
              f"least {MIN_SKILL} over the constant guess (the constant is the average official rate of the training "
              "blocks)."]
    return "\n".join(lines) + "\n"


def run(official_path=official_fill.POINTS_PATH, weekly_path=WEEKLY_PATH, path=CALIBRATION_PATH, report_path=REPORT_PATH,
        track=False):
    calibration = build(official_fill.load_points(official_path), pd.read_csv(weekly_path, parse_dates=["week"]), path)
    text = report(calibration)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    if track:
        from src import tracking

        for dam, entry in calibration["dams"].items():
            tracking.log_experiment(
                "fill-rate-calibration", dam, params={"dam": dam, "method": "isotonic regression anchored at (0,0),(100,100)",
                                                      "cv": f"{N_BLOCKS} contiguous time blocks"},
                metrics={"n_points": entry["n_points"], "usable": entry["usable"],
                         "typical_error_pts": entry["typical_error_pts"], "cv": entry["cv"] or {}},
                artifacts=[path], tags={"source": "calibration"})
    return calibration


if __name__ == "__main__":
    run(track=True)
