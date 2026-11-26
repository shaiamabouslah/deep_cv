"""Builds the SAR water-segmentation dataset: for each site, Sentinel-1
radar patches (VV, VH backscatter) paired with a water mask taken from a
clear Sentinel-2 image acquired within +-2 days.

Why this pairing: radar sees through clouds, but nobody has radar water
masks to learn from. The optical mask on a clear day is a free label, and
because the model is given only radar bands (never the optical bands the
label is computed from) there is no label leakage. Where optical is clouded
the learned radar model can still measure the water.

Patches are 30 m pixels (radar multilooked by averaging power, which is the
standard way to tame speckle) on a fixed grid in the site's UTM zone, so a
pixel means the same ground size everywhere. Cloud, shadow, snow/ice and
no-data pixels of the optical scene are marked invalid and ignored.

Output: data/sar/patches/<site>_<date>.npz  (vv, vh in dB as float16, water
and valid as uint8) and data/sar/manifest.json (kept small and committed;
pinned scene ids let the patches be rebuilt).

    python -m src.build_sar_dataset
"""
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from urllib.request import urlopen

import numpy as np

from src.data import load_params
from src.gee_client import _clear_mask, _ensure_initialized
from src.sites import TUNISIA_DAMS
from src.world_sites import load as load_world_sites

OUT_DIR = "data/sar"
WINDOW = ("2024-06-01", "2026-11-20")
PATCH_PX = 256
GSD_M = 30
MIN_VISIBLE = 0.9
MAX_PAIR_GAP_DAYS = 2
MIN_S1_COVERAGE = 0.98
MIN_SPACING_DAYS = 25
PER_SITE = {"world": 5, "tunisia": 10}
WORKERS = 4


def utm_crs(lon, lat):
    zone = int((lon + 180) // 6) + 1
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


def patch_grid(lon, lat, px=PATCH_PX, gsd=GSD_M):
    """UTM CRS and the affine transform of a px-by-px grid centered on (lon, lat)."""
    import ee

    crs = utm_crs(lon, lat)
    x, y = ee.Geometry.Point([lon, lat]).transform(crs, 1).coordinates().getInfo()
    half = px * gsd / 2
    return crs, [gsd, 0, round(x - half), 0, -gsd, round(y + half)]


def spread_dates(scenes, k, min_gap_days=MIN_SPACING_DAYS):
    """Up to k scenes, best-visible first, at least min_gap_days apart so the
    patches of a site cover different seasons instead of one wet week."""
    picked = []
    for s in sorted(scenes, key=lambda s: -s["visible"]):
        if all(abs(s["time"] - p["time"]) >= min_gap_days * 86400 * 1000 for p in picked):
            picked.append(s)
        if len(picked) == k:
            break
    return sorted(picked, key=lambda s: s["time"])


def optical_candidates(region, focus):
    """Clear Sentinel-2 scenes of the site, with how much of the water is visible."""
    import ee

    area = ee.Image.pixelArea()

    def score(img):
        sums = ee.Image.cat([
            focus.multiply(_clear_mask(img)).multiply(area).rename("c"), focus.multiply(area).rename("w"),
        ]).reduceRegion(ee.Reducer.sum(), region, 30, maxPixels=1e9, bestEffort=True)
        return ee.Feature(None, {
            "time": img.get("system:time_start"), "id": img.get("system:index"),
            "visible": ee.Number(sums.get("c")).divide(ee.Number(sums.get("w")).max(1)),
        })

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(region)
        .filterDate(*WINDOW).filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
    )
    rows = [f["properties"] for f in collection.map(score).getInfo()["features"]]
    return [r for r in rows if r.get("visible") is not None and r["visible"] >= MIN_VISIBLE]


def matching_radar(region, time_ms):
    """Closest Sentinel-1 IW scene (VV+VH) within the allowed gap that covers
    the whole patch, as an ee.Image, or None."""
    import ee

    t = ee.Date(time_ms)
    collection = (
        ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(region)
        .filterDate(t.advance(-MAX_PAIR_GAP_DAYS, "day"), t.advance(MAX_PAIR_GAP_DAYS + 1, "day"))
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    )

    def info(img):
        cover = img.select("VV").mask().reduceRegion(ee.Reducer.mean(), region, 100, bestEffort=True).get("VV")
        return ee.Feature(None, {
            "id": img.get("system:index"), "gap": ee.Number(img.get("system:time_start")).subtract(time_ms).abs(),
            "coverage": cover, "orbit": img.get("orbitProperties_pass"),
        })

    rows = [f["properties"] for f in collection.map(info).getInfo()["features"]]
    rows = [r for r in rows if (r.get("coverage") or 0) >= MIN_S1_COVERAGE]
    if not rows:
        return None
    best = min(rows, key=lambda r: r["gap"])
    return ee.Image(f"COPERNICUS/S1_GRD/{best['id']}"), best


def radar_image(s1_image, crs):
    """Bands vv, vh in dB on a 30 m grid of `crs`, multilooked by averaging
    power (not dB), the standard way to reduce speckle."""
    import ee

    linear = ee.Image(10).pow(s1_image.select(["VV", "VH"]).unmask(-35).divide(10))
    mean = linear.reduceResolution(ee.Reducer.mean(), bestEffort=True, maxPixels=1024).reproject(crs, None, GSD_M)
    return mean.log10().multiply(10).rename(["vv", "vh"])


def patch_image(s2_id, s1_image, crs):
    """Bands vv, vh (dB, 30 m multilook), water, valid (optical label), on `crs`."""
    import ee

    optical = ee.Image(f"COPERNICUS/S2_SR_HARMONIZED/{s2_id}")
    clear = _clear_mask(optical).And(optical.select("SCL").neq(11))
    water = optical.normalizedDifference(["B3", "B8"]).gt(0).rename("water")
    label = ee.Image.cat([water, clear.rename("valid")]).toFloat()
    coarse = label.reduceResolution(ee.Reducer.mean(), bestEffort=True, maxPixels=1024).reproject(crs, None, GSD_M)
    return ee.Image.cat([
        radar_image(s1_image, crs), coarse.select("water").gt(0.5).rename("water"),
        coarse.select("valid").gte(0.9).rename("valid"),
    ]).toFloat()


def download_patch(image, crs, transform, px=PATCH_PX, names=("vv", "vh", "water", "valid")):
    url = image.getDownloadURL({
        "format": "NPY", "crs": crs, "crs_transform": transform, "dimensions": f"{px}x{px}",
    })
    with urlopen(url, timeout=240) as response:
        data = np.load(io.BytesIO(response.read()))
    return {name: data[name].astype(np.float32) for name in names}


def build_site(site, kind, out_dir=OUT_DIR):
    import ee

    lat, lon = site["lat"], site["lon"]
    crs, transform = patch_grid(lon, lat)
    half = PATCH_PX * GSD_M / 2
    region = ee.Geometry.Point([lon, lat]).buffer(half * 1.5).bounds()
    focus = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("max_extent").unmask(0)

    rows, skipped = [], 0
    for scene in spread_dates(optical_candidates(region, focus), PER_SITE[kind] * 3):
        if len(rows) == PER_SITE[kind]:
            break
        pair = matching_radar(region, scene["time"])
        if pair is None:
            skipped += 1
            continue
        s1_image, s1 = pair
        date = datetime.fromtimestamp(scene["time"] / 1000, timezone.utc).date().isoformat()
        name = f"{site['id']}_{date}.npz"
        patch = download_patch(patch_image(scene["id"], s1_image, crs), crs, transform)
        np.savez_compressed(
            os.path.join(out_dir, "patches", name), vv=patch["vv"].astype(np.float16),
            vh=patch["vh"].astype(np.float16), water=patch["water"].astype(np.uint8),
            valid=patch["valid"].astype(np.uint8),
        )
        rows.append({
            "file": name, "site_id": site["id"], "site": site["name"], "kind": kind,
            "country": site["country"], "continent": site.get("continent"), "biome_name": site.get("biome_name"),
            "s2_date": date, "s2_id": scene["id"], "s1_id": s1["id"], "gap_hours": round(s1["gap"] / 3.6e6, 1),
            "orbit": s1["orbit"], "optical_visible": round(scene["visible"], 3),
            "water_share": round(float((patch["water"] * patch["valid"]).sum() / max(patch["valid"].sum(), 1)), 4),
        })
    return rows


def all_sites():
    world = [{**s, "lat": s["center_lat"], "lon": s["center_lon"]} for s in load_world_sites()]
    tunisia = [{"id": f"tn-{d['id']}", "name": d["name"], "country": "Tunisia", "continent": "Africa",
                "biome_name": "Mediterranean", "lat": d["lat"], "lon": d["lon"]} for d in TUNISIA_DAMS]
    return [(s, "world") for s in world] + [(s, "tunisia") for s in tunisia]


def build(params_path="params.yaml", out_dir=OUT_DIR):
    _ensure_initialized(load_params(params_path)["gee"]["project"])
    os.makedirs(os.path.join(out_dir, "patches"), exist_ok=True)

    def run(item):
        site, kind = item
        try:
            return build_site(site, kind, out_dir)
        except Exception as exc:
            print("failed", site["id"], str(exc)[:100], flush=True)
            return []

    with ThreadPoolExecutor(WORKERS) as pool:
        manifest = [row for rows in pool.map(run, all_sites()) for row in rows]

    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"{len(manifest)} radar/optical pairs from {len({r['site_id'] for r in manifest})} sites")


if __name__ == "__main__":
    build()
