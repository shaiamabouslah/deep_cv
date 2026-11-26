import numpy as np
import pandas as pd
import pytest

from src import sar_gapfill as sg


def test_disc_has_the_right_area_and_is_centered():
    mask = sg.circle_mask(400, 30, 3000)
    area_m2 = mask.sum() * 30 * 30
    assert area_m2 == pytest.approx(np.pi * 3000 ** 2, rel=0.02)
    assert mask[200, 200] and not mask[0, 0]


def test_surface_pct_counts_only_water_inside_the_disc_on_valid_pixels():
    px, gsd = 100, 30
    water = np.ones((px, px), bool)
    valid = np.ones((px, px), bool)
    disc_area = sg.circle_mask(px, gsd, 1000).sum() * gsd * gsd
    assert sg.surface_pct(water, valid, 1000, disc_area * 2, px, gsd) == pytest.approx(50.0, abs=0.01)

    valid[:, :50] = False
    assert sg.surface_pct(water, valid, 1000, disc_area * 2, px, gsd) < 30


def test_surface_pct_is_capped_at_100():
    px = 60
    assert sg.surface_pct(np.ones((px, px), bool), np.ones((px, px), bool), 500, 1.0, px, 30) == 100.0


def test_evaluation_separates_dates_with_and_without_optical(tmp_path):
    from src import calibration as cal

    weeks = pd.date_range("2024-01-01", periods=60, freq="W-MON")
    surface = np.linspace(20, 80, 60)
    official = [{"dam": "D", "date": str((w + pd.Timedelta(days=2)).date()), "official_fill_pct": float(s * 0.5)}
                for w, s in zip(weeks, surface)]
    weekly = pd.DataFrame({"site": "D", "week": weeks, "water_pct": surface})
    weekly.loc[weekly.index[::5], "water_pct"] = np.nan
    calibration = cal.build(official, weekly, str(tmp_path / "c.json"))
    radar = pd.DataFrame({"dam": "D", "date": [o["date"] for o in official], "surface": surface + 1.0})

    result = sg.evaluate(radar, official, weekly, calibration)

    assert result["n_dam_dates"] == 60 and result["n_optical_missing"] == 12
    assert result["radar_vs_optical_surface"]["mae_pts"] == pytest.approx(1.0, abs=0.01)
    assert result["per_dam"]["D"]["fill_error_optical_curve_pts"] < 3
