import datetime

import numpy as np
import pandas as pd

from src.build_forecast_dataset import weekly_table


def _ms(day):
    return int(datetime.datetime.fromisoformat(day).replace(tzinfo=datetime.timezone.utc).timestamp() * 1000)


def _scene(day, water, visible, extent=1000.0):
    return {"time": _ms(day), "water": water, "visible": visible, "extent": extent}


def _weather(days, column, value):
    return [{"time": _ms(d), column: value} for d in days]


def _table(scenes):
    days = pd.date_range("2025-01-06", "2025-02-02").strftime("%Y-%m-%d").tolist()
    rain = _weather(days, "rain_mm", 2.0)
    climate = [{"time": _ms(d), "evap_mm": 1.5, "temp_c": 12.0} for d in days]
    return weekly_table("X", scenes, rain, climate, "2025-01-06", "2025-02-03")


def test_weeks_start_on_monday_and_cover_the_range():
    table = _table([])
    assert list(table["week"].dt.day_of_week.unique()) == [0]
    assert len(table) == 4


def test_water_pct_is_water_over_the_visible_part_of_the_extent():
    table = _table([_scene("2025-01-08", water=450.0, visible=900.0)])
    assert table.loc[0, "water_pct"] == 50.0
    assert table.loc[0, "visible_fraction"] == 0.9
    assert table["water_pct"].iloc[1:].isna().all()


def test_mostly_clouded_scenes_are_dropped_not_extrapolated():
    table = _table([_scene("2025-01-08", water=100.0, visible=600.0)])
    assert table["water_pct"].isna().all()


def test_best_scene_of_the_week_wins():
    table = _table([
        _scene("2025-01-07", water=100.0, visible=750.0),
        _scene("2025-01-09", water=480.0, visible=960.0),
    ])
    assert table.loc[0, "water_pct"] == 50.0


def test_sunday_belongs_to_the_week_that_started_the_previous_monday():
    table = _table([_scene("2025-01-12", water=300.0, visible=1000.0)])
    assert table.loc[0, "water_pct"] == 30.0
    assert np.isnan(table.loc[1, "water_pct"])


def test_weather_is_summed_or_averaged_per_week():
    table = _table([])
    assert table.loc[0, "rain_mm"] == 14.0
    assert table.loc[0, "evap_mm"] == 10.5
    assert table.loc[0, "temp_c"] == 12.0


def test_weeks_without_published_weather_stay_missing_instead_of_failing():
    table = weekly_table("X", [], [], [], "2025-01-06", "2025-02-03")
    assert len(table) == 4
    assert table[["rain_mm", "evap_mm", "temp_c"]].isna().all().all()


def test_water_pct_is_capped_at_100():
    table = _table([_scene("2025-01-08", water=2000.0, visible=1000.0)])
    assert table.loc[0, "water_pct"] == 100.0
