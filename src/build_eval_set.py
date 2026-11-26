"""Builds the real evaluation set for the segmentation model: Sentinel-2
true-color tiles of each monitored dam, at three seasons, with a water mask
from NDWI on the same scene (the same labelling rule as the Kaggle training
masks; the model only sees RGB, so NDWI is an independent reference).

Mask values: 255 water, 0 land, 128 ignore (cloud, shadow or no data — they
are neither water nor land and must not count for or against the model).

Run once, then commit data/eval/ so evaluation is offline and reproducible:
    python -m src.build_eval_set
"""
import io
import json
import os
import urllib.request

import numpy as np
from PIL import Image

from src.data import load_params
from src.gee_client import _clear_mask, _clearest_image, _ensure_initialized
from src.preprocess import nodata_mask
from src.sites import TUNISIA_DAMS

OUT_DIR = "data/eval"
SEASONS = {
    "wet": ("2025-12-01", "2026-02-28"),
    "spring": ("2026-04-01", "2026-05-31"),
    "dry": ("2026-08-01", "2026-10-31"),
}
MIN_VISIBLE = 0.9
THUMB_PX = 512


def _download(image, region):
    url = image.getThumbURL({"region": region, "dimensions": THUMB_PX, "format": "png"})
    with urllib.request.urlopen(url, timeout=120) as response:
        return Image.open(io.BytesIO(response.read()))


def build(params_path="params.yaml"):
    import ee

    _ensure_initialized(load_params(params_path)["gee"]["project"])
    os.makedirs(os.path.join(OUT_DIR, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "masks"), exist_ok=True)
    jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("max_extent").unmask(0)

    manifest = []
    for dam in TUNISIA_DAMS:
        region = ee.Geometry.Point([dam["lon"], dam["lat"]]).buffer(dam["buffer_m"] * 1.3).bounds()
        for season, (start, end) in SEASONS.items():
            image, visible, date = _clearest_image(region, start, end, focus=jrc)
            if visible < MIN_VISIBLE:
                print(f"skip {dam['name']} {season}: reservoir only {visible:.0%} visible")
                continue

            water = image.normalizedDifference(["B3", "B8"]).gt(0)
            label = water.multiply(255).where(_clear_mask(image).eq(0), 128).unmask(128)
            rgb = _download(image.visualize(bands=["B4", "B3", "B2"], min=0, max=3000), region).convert("RGB")
            mask = np.array(_download(label.visualize(min=0, max=255), region).convert("L"))
            # Outside the scene footprint the thumbnail is transparent (reads
            # as 0 = land); mark it "ignore" with the same rule as serving.
            mask[nodata_mask(np.array(rgb))] = 128
            mask = Image.fromarray(mask)

            name = f"{dam['id']}_{season}.png"
            rgb.save(os.path.join(OUT_DIR, "images", name))
            mask.save(os.path.join(OUT_DIR, "masks", name))
            manifest.append({"file": name, "site": dam["name"], "season": season,
                             "date": date, "reservoir_visible": round(visible, 3)})
            print(f"saved {name} ({date}, visible {visible:.0%})")

    with open(os.path.join(OUT_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    build()
