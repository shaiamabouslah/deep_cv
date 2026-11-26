"""Builds the world benchmark: one Sentinel-2 true-color tile per site of
data/world/sites.json, with an NDWI water mask from the same scene, in the
same format as data/eval/ (255 water, 0 land, 128 ignore) so src/evaluate.py
scores it unchanged.

Images are chosen in the warm / ice-free season of each site's hemisphere,
because a frozen lake reads as "no water" in NDWI and would poison the
labels. Snow, ice, cloud, shadow and no-data are all marked "ignore".
Sites that stay mostly hidden (persistent cloud) are skipped and listed in
the manifest's `skipped`; that is a bias of the benchmark (the cloudiest
tropical sites are under-represented) and is reported, not hidden.

    python -m src.build_world_set
"""
import io
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from src.data import load_params
from src.gee_client import _clear_mask, _clearest_image, _ensure_initialized
from src.preprocess import nodata_mask
from src.world_sites import load as load_sites

OUT_DIR = "data/world"
WINDOW = ("2024-11-01", "2026-11-26")
MIN_VISIBLE = 0.9
RANK_SCALE_M = 80
MAX_TILE_CLOUD = 40
THUMB_PX = 512
WORKERS = 4


def warm_months(lat):
    """Calendar months with ice-free lakes and little snow for this latitude."""
    if lat >= 50:
        return (6, 9)
    if lat >= 35:
        return (5, 9)
    if lat <= -35:
        return (11, 12)
    return (1, 12)


def _download(image, region):
    url = image.getThumbURL({"region": region, "dimensions": THUMB_PX, "format": "png"})
    with urllib.request.urlopen(url, timeout=180) as response:
        return Image.open(io.BytesIO(response.read()))


def build_site(site, out_dir=OUT_DIR):
    import ee

    region = (
        ee.Geometry.Point([site["center_lon"], site["center_lat"]])
        .buffer(site["side_km"] * 500).bounds()
    )
    focus = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("max_extent").unmask(0)
    try:
        image, visible, date = _clearest_image(
            region, WINDOW[0], WINDOW[1], focus=focus, months=warm_months(site["center_lat"]),
            scale=RANK_SCALE_M, max_cloud=MAX_TILE_CLOUD,
        )
    except ValueError:
        return {"skipped": site["id"], "reason": "no scene in the warm season"}
    if visible < MIN_VISIBLE:
        return {"skipped": site["id"], "reason": f"water only {visible:.0%} visible at best"}

    scl = image.select("SCL")
    clear = _clear_mask(image).And(scl.neq(11))
    water = image.normalizedDifference(["B3", "B8"]).gt(0)
    label = water.multiply(255).where(clear.eq(0), 128).unmask(128)

    rgb = _download(image.visualize(bands=["B4", "B3", "B2"], min=0, max=3000), region).convert("RGB")
    mask = np.array(_download(label.visualize(min=0, max=255), region).convert("L"))
    mask[nodata_mask(np.array(rgb))] = 128

    name = f"{site['id']}.png"
    rgb.save(os.path.join(out_dir, "images", name))
    Image.fromarray(mask).save(os.path.join(out_dir, "masks", name))
    return {
        "file": name, "site": site["name"], "site_id": site["id"], "country": site["country"],
        "continent": site["continent"], "biome": site["biome"], "biome_name": site["biome_name"],
        "type": site["type"], "area_km2": site["area_km2"], "side_km": site["side_km"],
        "season": f"months {warm_months(site['center_lat'])[0]}-{warm_months(site['center_lat'])[1]}",
        "date": date, "scene_id": image.get("scene_id").getInfo(), "reservoir_visible": round(visible, 3),
    }


def build(params_path="params.yaml", out_dir=OUT_DIR):
    _ensure_initialized(load_params(params_path)["gee"]["project"])
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "masks"), exist_ok=True)
    sites = load_sites(os.path.join(out_dir, "sites.json"))

    def run(site):
        try:
            return build_site(site, out_dir)
        except Exception as exc:  # one bad site must not lose the other hundred
            return {"skipped": site["id"], "reason": f"error: {str(exc)[:80]}"}

    with ThreadPoolExecutor(WORKERS) as pool:
        results = list(pool.map(run, sites))

    manifest = [r for r in results if "file" in r]
    skipped = [r for r in results if "skipped" in r]
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    with open(os.path.join(out_dir, "skipped.json"), "w", encoding="utf-8") as f:
        json.dump(skipped, f, ensure_ascii=False, indent=1)
    print(f"{len(manifest)} tiles built, {len(skipped)} sites skipped")
    for r in skipped[:10]:
        print("  skipped", r)


if __name__ == "__main__":
    build()
