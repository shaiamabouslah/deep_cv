"""Builds the weekly dataset the deep-learning forecaster trains on: for
each monitored dam, its water area over time (the same cloud-aware
measurement as the live app: classified water pixels over the visible part
of the reservoir's historical maximum extent) plus the weather that drives
it (CHIRPS rainfall, ERA5-Land evaporation and temperature).

Scenes are processed server-side in Earth Engine (one request per dam and
year), not one round trip per scene. Weeks without a usable scene (cloud
cover, no pass) keep a missing water value — they are never filled in here,
so the training code can tell real observations from interpolated ones.

    python -m src.build_forecast_dataset
"""
import datetime
import os

import pandas as pd

from src.data import load_params
from src.gee_client import MIN_VISIBLE_FRACTION, _clear_mask, _ensure_initialized
from src.sites import TUNISIA_DAMS

OUT_PATH = "data/forecast/weekly.csv"
START = "2018-12-01"  # global Sentinel-2 L2A (with the SCL cloud band) coverage
SCALE_M = 10
WEATHER_SCALE_M = 5000


def _year_chunks(start, end):
    first = datetime.date.fromisoformat(start)
    last = datetime.date.fromisoformat(end)
    for year in range(first.year, last.year + 1):
        yield max(first, datetime.date(year, 1, 1)), min(last, datetime.date(year + 1, 1, 1))


def _split(lo, hi):
    mid = lo + (hi - lo) / 2
    return [(lo, mid), (mid, hi)]


def _features(collection, mapper, lo, hi):
    """Runs the per-image mapper server-side over [lo, hi); halves the window
    and retries if Earth Engine times out or returns too much at once."""
    import ee

    try:
        rows = collection.filterDate(str(lo), str(hi)).map(mapper).getInfo()["features"]
        return [f["properties"] for f in rows]
    except ee.ee_exception.EEException as exc:
        too_big = any(word in str(exc).lower() for word in ("timed out", "too many", "memory", "too large"))
        if not too_big or (hi - lo).days <= 8:
            raise
        return [row for a, b in _split(lo, hi) for row in _features(collection, mapper, a, b)]


def fetch_scene_rows(region, lo, hi):
    import ee

    focus = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("max_extent").unmask(0)
    area = ee.Image.pixelArea()

    def per_image(img):
        clear = _clear_mask(img)
        water = img.normalizedDifference(["B3", "B8"]).gt(0).And(clear).multiply(area).rename("water")
        visible = focus.multiply(clear).multiply(area).rename("visible")
        total = focus.multiply(area).rename("extent")
        sums = ee.Image.cat([water, visible, total]).reduceRegion(
            ee.Reducer.sum(), region, SCALE_M, maxPixels=1e9, bestEffort=True
        )
        return ee.Feature(None, {
            "time": img.get("system:time_start"),
            "water": sums.get("water"), "visible": sums.get("visible"), "extent": sums.get("extent"),
        })

    collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(region)
    return _features(collection, per_image, lo, hi)


def fetch_daily_weather(region, lo, hi):
    import ee

    chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").select("precipitation")
    era = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").select(["total_evaporation_sum", "temperature_2m"])

    def rain(img):
        v = img.reduceRegion(ee.Reducer.mean(), region, WEATHER_SCALE_M, bestEffort=True)
        return ee.Feature(None, {"time": img.get("system:time_start"), "rain_mm": v.get("precipitation")})

    def climate(img):
        v = img.reduceRegion(ee.Reducer.mean(), region.centroid(), 11132, bestEffort=True)
        return ee.Feature(None, {
            "time": img.get("system:time_start"),
            "evap_mm": ee.Number(v.get("total_evaporation_sum")).multiply(-1000),
            "temp_c": ee.Number(v.get("temperature_2m")).subtract(273.15),
        })

    rain_rows = _features(chirps, rain, lo, hi)
    climate_rows = _features(era, climate, lo, hi)
    return rain_rows, climate_rows


def _to_date(ms):
    return datetime.datetime.fromtimestamp(ms / 1000, datetime.timezone.utc).date()


def weekly_table(site, scene_rows, rain_rows, climate_rows, start, end):
    """One row per ISO week (Monday start): water_pct from the best scene of
    the week, weather summed/averaged over the week."""
    weeks = pd.date_range(start, end, freq="W-MON", inclusive="left")
    table = pd.DataFrame({"week": weeks})

    scenes = pd.DataFrame(scene_rows)
    if len(scenes):
        scenes = scenes.dropna(subset=["water", "visible", "extent"])
        scenes["date"] = pd.to_datetime([_to_date(t) for t in scenes["time"]])
        scenes["visible_fraction"] = scenes["visible"] / scenes["extent"].clip(lower=1)
        scenes = scenes[scenes["visible_fraction"] >= MIN_VISIBLE_FRACTION]
        scenes["water_pct"] = (scenes["water"] / scenes["visible"].clip(lower=1) * 100).clip(upper=100)
        scenes["week"] = scenes["date"].dt.to_period("W-SUN").dt.start_time
        best = scenes.sort_values("visible_fraction", ascending=False).groupby("week", as_index=False).first()
        table = table.merge(best[["week", "water_pct", "visible_fraction"]], on="week", how="left")
    else:
        table["water_pct"], table["visible_fraction"] = float("nan"), float("nan")

    def weekly(rows, column, how):
        if not rows:  # recent weeks can have no weather published yet
            return pd.DataFrame({"week": pd.Series(dtype="datetime64[ns]"), column: pd.Series(dtype=float)})
        frame = pd.DataFrame(rows)
        frame["date"] = pd.to_datetime([_to_date(t) for t in frame["time"]])
        frame["week"] = frame["date"].dt.to_period("W-SUN").dt.start_time
        return frame.groupby("week")[column].agg(how).reset_index()

    table = table.merge(weekly(rain_rows, "rain_mm", "sum"), on="week", how="left")
    table = table.merge(weekly(climate_rows, "evap_mm", "sum"), on="week", how="left")
    table = table.merge(weekly(climate_rows, "temp_c", "mean"), on="week", how="left")
    table.insert(0, "site", site)
    return table


def build(params_path="params.yaml", end=None):
    import ee

    _ensure_initialized(load_params(params_path)["gee"]["project"])
    end = end or str(datetime.date.today())
    frames = []
    for dam in TUNISIA_DAMS:
        region = ee.Geometry.Point([dam["lon"], dam["lat"]]).buffer(dam["buffer_m"])
        scene_rows, rain_rows, climate_rows = [], [], []
        for lo, hi in _year_chunks(START, end):
            scene_rows += fetch_scene_rows(region, lo, hi)
            print(f"{dam['name']} {lo.year}: {len(scene_rows)} scenes so far", flush=True)
        rain_rows, climate_rows = fetch_daily_weather(
            region, datetime.date.fromisoformat(START), datetime.date.fromisoformat(end)
        )
        frames.append(weekly_table(dam["name"], scene_rows, rain_rows, climate_rows, START, end))

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    table = pd.concat(frames, ignore_index=True)
    table.to_csv(OUT_PATH, index=False)
    observed = table.groupby("site")["water_pct"].apply(lambda s: s.notna().mean()).round(2).to_dict()
    print(f"saved {len(table)} rows to {OUT_PATH}; share of weeks with a usable scene: {observed}")


if __name__ == "__main__":
    build()
