import numpy as np
import pandas as pd
import pytest

from src import calibration as cal


def _official_and_weekly(n=60, seed=0):
    """A dam whose fill rate is surface^2 (a convex area-volume curve) plus a
    dam whose surface says nothing about its fill rate."""
    rng = np.random.default_rng(seed)
    weeks = pd.date_range("2024-01-01", periods=n, freq="W-MON")
    surface = np.clip(40 + 35 * np.sin(np.linspace(0, 6, n)) + rng.normal(0, 1.5, n), 1, 99)
    informative = 100 * (surface / 100) ** 2
    noise_surface = rng.uniform(30, 50, n)
    flat_official = rng.uniform(3, 9, n)

    official, weekly = [], []
    for w, s, f, ns, fo in zip(weeks, surface, informative, noise_surface, flat_official):
        official += [
            {"dam": "Good", "date": str((w + pd.Timedelta(days=2)).date()), "official_fill_pct": float(f)},
            {"dam": "Flat", "date": str((w + pd.Timedelta(days=2)).date()), "official_fill_pct": float(fo)},
        ]
        weekly += [{"site": "Good", "week": w, "water_pct": float(s)}, {"site": "Flat", "week": w, "water_pct": float(ns)}]
    return official, pd.DataFrame(weekly)


def test_points_are_paired_with_the_satellite_measurement_of_the_same_week():
    official, weekly = _official_and_weekly()
    pairs = cal.pair_points(official, weekly)
    assert len(pairs) == 120
    row = pairs[(pairs.dam == "Good")].iloc[0]
    assert row["surface"] == weekly[(weekly.site == "Good")].iloc[0]["water_pct"]


def test_weeks_without_a_satellite_reading_are_dropped():
    official, weekly = _official_and_weekly()
    weekly.loc[weekly.index[:10], "water_pct"] = np.nan
    assert len(cal.pair_points(official, weekly)) == 110


def test_curve_is_monotone_anchored_and_interpolates():
    curve = cal.fit_curve([30, 50, 70], [5, 20, 55])
    assert all(b >= a for a, b in zip(curve, curve[1:]))
    assert cal.apply_curve(curve, 0) == 0 and cal.apply_curve(curve, 100) == 100
    assert 20 < cal.apply_curve(curve, 60) < 55
    assert cal.apply_curve(curve, 150) == 100


def test_informative_dam_is_calibrated_and_a_flat_dam_is_refused(tmp_path):
    official, weekly = _official_and_weekly()
    result = cal.build(official, weekly, str(tmp_path / "c.json"))

    assert result["dams"]["Good"]["usable"] is True
    assert result["dams"]["Good"]["typical_error_pts"] < 5
    assert result["dams"]["Flat"]["usable"] is False
    assert result["dams"]["Flat"]["curve"] is None


def test_calibration_beats_the_raw_surface_out_of_sample():
    official, weekly = _official_and_weekly()
    cv = cal.cross_validate(cal.pair_points(official, weekly))["Good"]
    assert cv["calibrated"] < cv["raw"] / 3 and cv["skill"] > 0.5


def test_estimate_fill_only_for_usable_dams(tmp_path):
    official, weekly = _official_and_weekly()
    calibration = cal.build(official, weekly, str(tmp_path / "c.json"))

    estimate = cal.estimate_fill("Good", 60.0, calibration)
    assert estimate["fill_pct"] == pytest.approx(36, abs=6)
    assert estimate["n_points"] == 60
    assert cal.estimate_fill("Flat", 40.0, calibration) is None
    assert cal.estimate_fill("Unknown dam", 40.0, calibration) is None
    assert cal.estimate_fill("Good", None, calibration) is None
    assert cal.estimate_fill("Good", 40.0, None) is None


def test_saved_calibration_reloads_and_missing_file_means_none(tmp_path):
    official, weekly = _official_and_weekly()
    path = str(tmp_path / "c.json")
    cal.build(official, weekly, path)
    assert cal.load(path)["dams"]["Good"]["usable"] is True
    assert cal.load(str(tmp_path / "missing.json")) is None


def test_report_lists_every_dam_and_the_decision(tmp_path):
    official, weekly = _official_and_weekly()
    text = cal.report(cal.build(official, weekly, str(tmp_path / "c.json")))
    good_row = next(line for line in text.splitlines() if line.startswith("| Good |"))
    flat_row = next(line for line in text.splitlines() if line.startswith("| Flat |"))
    assert good_row.endswith("| yes |") and flat_row.endswith("| no |")
