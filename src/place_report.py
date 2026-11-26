"""Worldwide context for any place: search by name, local climate against its
own normal, and a long record of the water surface.

  - geocode: Open-Meteo geocoding (no key)
  - climate_summary: rain of the last 90 days against the same 90 days of the
    previous five years, evaporation (FAO ET0), balance, next-7-days rain;
    Open-Meteo forecast + archive (no key, global)
  - water_record: yearly water surface 1984-2021 from JRC Global Surface Water
    (Landsat) inside the water body's outline, as a percentage of its
    maximum extent. The record ends in 2021: the present state comes from
    Sentinel-2 (src/gee_client.py), not from here.
"""
import datetime

import httpx

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
WINDOW_DAYS = 90
NORMAL_YEARS = 5
MIN_VALID_FRACTION = 0.6
YEAR_RANGE = (1984, 2021)


def geocode(query, count=6, timeout=10.0):
    response = httpx.get(GEOCODE_URL, params={"name": query, "count": count, "language": "fr", "format": "json"}, timeout=timeout)
    response.raise_for_status()
    return [{"name": r.get("name"), "admin1": r.get("admin1"), "country": r.get("country"),
             "lat": r["latitude"], "lon": r["longitude"]} for r in response.json().get("results", [])]


def _sum(values):
    return sum(v for v in values if v is not None)


def climate_summary(lat, lon, today=None, timeout=20.0):
    today = today or datetime.date.today()
    recent = httpx.get(FORECAST_URL, params={
        "latitude": lat, "longitude": lon, "daily": "precipitation_sum,et0_fao_evapotranspiration",
        "past_days": WINDOW_DAYS, "forecast_days": 7, "timezone": "auto"}, timeout=timeout)
    recent.raise_for_status()
    daily = recent.json()["daily"]
    dates = daily["time"]
    split = len(dates) - 7
    precip = _sum(daily["precipitation_sum"][:split])
    et0 = _sum(daily["et0_fao_evapotranspiration"][:split])
    rain_next = _sum(daily["precipitation_sum"][split:])

    start_year = today.year - NORMAL_YEARS
    archive = httpx.get(ARCHIVE_URL, params={
        "latitude": lat, "longitude": lon, "start_date": f"{start_year}-01-01", "end_date": f"{today.year - 1}-12-31",
        "daily": "precipitation_sum", "timezone": "auto"}, timeout=timeout)
    archive.raise_for_status()
    series = archive.json()["daily"]
    window_start = (today - datetime.timedelta(days=WINDOW_DAYS)).timetuple().tm_yday
    window_end = today.timetuple().tm_yday
    totals = {}
    for day, value in zip(series["time"], series["precipitation_sum"]):
        d = datetime.date.fromisoformat(day)
        in_window = window_start <= d.timetuple().tm_yday <= window_end if window_start <= window_end else (
            d.timetuple().tm_yday >= window_start or d.timetuple().tm_yday <= window_end)
        if in_window and value is not None:
            totals[d.year] = totals.get(d.year, 0.0) + value
    normal = sum(totals.values()) / len(totals) if totals else None
    return {
        "window_days": WINDOW_DAYS, "precip_90d_mm": round(precip, 1), "et0_90d_mm": round(et0, 1),
        "balance_mm": round(precip - et0, 1), "rain_next_7d_mm": round(rain_next, 1),
        "precip_normal_90d_mm": round(normal, 1) if normal is not None else None,
        "anomaly_pct": round(precip / normal * 100, 0) if normal else None,
        "normal_years": len(totals),
    }


def water_record(polygon=None, lat=None, lon=None, buffer_m=3000, project=None, lake_id=None):
    """Yearly water surface inside the water body's outline, from JRC Global
    Surface Water YearlyHistory. With `lake_id` (HydroLAKES) the full outline
    is read inside Earth Engine, so even lakes too large to send to the
    browser are measured on their whole extent; otherwise the drawn polygon,
    otherwise a disc around the point."""
    from src.gee_client import _ensure_initialized
    from src.world_sites import HYDROLAKES

    _ensure_initialized(project)
    import ee

    if lake_id is not None:
        feature = ee.Feature(ee.FeatureCollection(HYDROLAKES).filter(ee.Filter.eq("Hylak_id", lake_id)).first())
        area_km2 = float(feature.get("Lake_area").getInfo() or 1.0)
        region = feature.geometry()
        if area_km2 < 300:
            region = region.buffer(500)
    else:
        region = ee.Geometry.Polygon([polygon]) if polygon else ee.Geometry.Point([lon, lat]).buffer(buffer_m)
        region = region.buffer(500)
        area_km2 = region.area(maxError=100).getInfo() / 1e6
    scale = 30 if area_km2 < 300 else 60 if area_km2 < 1500 else 120 if area_km2 < 5000 else 250
    pixel = ee.Image.pixelArea()
    maximum = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("max_extent").unmask(0).eq(1)
    collection = ee.ImageCollection("JRC/GSW1_4/YearlyHistory")
    years = list(range(YEAR_RANGE[0], YEAR_RANGE[1] + 1))
    bands = [pixel.updateMask(maximum).rename("extent")]
    for year in years:
        cls = collection.filter(ee.Filter.calendarRange(year, year, "year")).first().select("waterClass")
        bands.append(pixel.updateMask(cls.gte(2)).rename(f"w{year}"))
        bands.append(pixel.updateMask(cls.gte(1)).updateMask(maximum).rename(f"v{year}"))
    sums = ee.Image.cat(bands).reduceRegion(ee.Reducer.sum(), region, scale, maxPixels=1e9, bestEffort=True).getInfo()
    extent = sums.get("extent") or 0
    points = []
    for year in years:
        if extent and (sums.get(f"v{year}") or 0) / extent >= MIN_VALID_FRACTION:
            points.append({"year": year, "area_km2": round((sums.get(f"w{year}") or 0) / 1e6, 3)})
    if not points:
        return {"record": [], "max_km2": None, "years": None}
    peak = max(extent / 1e6, max(p["area_km2"] for p in points))
    for p in points:
        p["pct_of_max"] = round(p["area_km2"] / peak * 100, 1) if peak else None
    return {"record": points, "max_km2": round(peak, 3), "years": [points[0]["year"], points[-1]["year"]]}
