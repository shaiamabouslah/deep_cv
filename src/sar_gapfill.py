"""Can radar alone estimate a dam's official fill rate, including the weeks
when clouds hide it from optical imagery?

For each Tunisian dam and each official bulletin date: take the Sentinel-1
radar scene closest in time (within +-3 days), segment the water, measure
the water inside the dam's analysis circle as % of its historical maximum
extent (the same definition as the optical measurement, with no cloud
correction because radar has none), convert it with the per-dam calibration
and compare with the official figure. Two conversions are compared: the
curve fitted on OPTICAL measurements, and a curve fitted directly on the
radar ones (radar has its own bias).

    python -m src.sar_gapfill fetch      # download radar patches (Earth Engine)
    python -m src.sar_gapfill evaluate   # score
"""
import json
import os
import sys

import numpy as np
import pandas as pd

GAP_DIR = "data/sar_gap"
PATCH_PX = 400
GSD_M = 30
MAX_GAP_DAYS = 3
WEEKLY_PATH = "data/forecast/weekly.csv"
REPORT_PATH = "reports/sar_gapfill.json"


def circle_mask(px, gsd, radius_m):
    """True inside a disc of radius_m centered on the patch."""
    axis = (np.arange(px) - (px - 1) / 2) * gsd
    xx, yy = np.meshgrid(axis, axis)
    return (xx ** 2 + yy ** 2) <= radius_m ** 2


def surface_pct(mask, valid_radar, radius_m, max_area_m2, px=PATCH_PX, gsd=GSD_M):
    """Water area inside the analysis disc as % of the maximum extent."""
    inside = circle_mask(px, gsd, radius_m) & valid_radar
    return min(100.0, float((mask & inside).sum()) * gsd * gsd / max_area_m2 * 100)


def fetch(official_points, dams, out_dir=GAP_DIR, project=None, workers=4):
    from concurrent.futures import ThreadPoolExecutor

    import ee

    from src.build_sar_dataset import download_patch, patch_grid, radar_image
    from src.gee_client import _ensure_initialized, fetch_max_extent_area_m2

    _ensure_initialized(project)
    os.makedirs(os.path.join(out_dir, "patches"), exist_ok=True)
    max_area = {d["name"]: fetch_max_extent_area_m2(ee.Geometry.Point([d["lon"], d["lat"]]).buffer(d["buffer_m"]))
                for d in dams}

    jobs = sorted({(p["dam"], p["date"]) for p in official_points})
    by_name = {d["name"]: d for d in dams}

    def one(job):
        name, day = job
        dam = by_name[name]
        path = os.path.join(out_dir, "patches", f"{name.replace(' ', '_').replace(chr(39), '')}_{day}.npz")
        if os.path.exists(path):
            return {"dam": name, "date": day, "file": os.path.basename(path)}
        try:
            crs, transform = patch_grid(dam["lon"], dam["lat"], PATCH_PX, GSD_M)
            half = PATCH_PX * GSD_M / 2
            region = ee.Geometry.Point([dam["lon"], dam["lat"]]).buffer(half * 1.4).bounds()
            when = ee.Date(day)
            scenes = (
                ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(region)
                .filterDate(when.advance(-MAX_GAP_DAYS, "day"), when.advance(MAX_GAP_DAYS + 1, "day"))
                .filter(ee.Filter.eq("instrumentMode", "IW"))
                .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
                .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            )

            def info(img):
                cover = img.select("VV").mask().reduceRegion(ee.Reducer.mean(), region, 100, bestEffort=True).get("VV")
                return ee.Feature(None, {"id": img.get("system:index"), "coverage": cover,
                                         "gap": ee.Number(img.get("system:time_start")).subtract(when.millis()).abs()})

            rows = [f["properties"] for f in scenes.map(info).getInfo()["features"]]
            rows = [r for r in rows if (r.get("coverage") or 0) >= 0.98]
            if not rows:
                return None
            best = min(rows, key=lambda r: r["gap"])
            image = radar_image(ee.Image(f"COPERNICUS/S1_GRD/{best['id']}"), crs)
            patch = download_patch(image, crs, transform, PATCH_PX, ("vv", "vh"))
            np.savez_compressed(path, vv=patch["vv"].astype(np.float16), vh=patch["vh"].astype(np.float16))
            return {"dam": name, "date": day, "file": os.path.basename(path), "s1_id": best["id"],
                    "gap_hours": round(best["gap"] / 3.6e6, 1)}
        except Exception as exc:
            print("failed", name, day, str(exc)[:80], flush=True)
            return None

    with ThreadPoolExecutor(workers) as pool:
        manifest = [m for m in pool.map(one, jobs) if m]
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"max_area_m2": max_area, "patches": manifest}, f, indent=1)
    print(f"{len(manifest)} radar patches for {len(jobs)} dam-dates")
    return manifest


def radar_surface_table(predict_mask, official_points, dams, out_dir=GAP_DIR):
    """Rows (dam, date, radar surface %) for every downloaded patch."""
    with open(os.path.join(out_dir, "manifest.json"), encoding="utf-8") as f:
        meta = json.load(f)
    by_name = {d["name"]: d for d in dams}
    rows = []
    for item in meta["patches"]:
        data = np.load(os.path.join(out_dir, "patches", item["file"]))
        vv, vh = data["vv"].astype(np.float32), data["vh"].astype(np.float32)
        valid = np.isfinite(vv) & (vv > -34)
        mask = predict_mask(vv, vh)
        dam = by_name[item["dam"]]
        rows.append({"dam": item["dam"], "date": item["date"],
                     "surface": surface_pct(mask, valid, dam["buffer_m"], meta["max_area_m2"][item["dam"]])})
    return pd.DataFrame(rows)


def evaluate(radar_table, official_points, weekly, calibration):
    """Radar vs optical surface, and radar-based fill rate vs official."""
    from src import calibration as cal

    official = pd.DataFrame(official_points)
    official["date"] = pd.to_datetime(official["date"])
    radar = radar_table.assign(date=pd.to_datetime(radar_table["date"]))
    joined = official.merge(radar, on=["dam", "date"], how="inner").rename(columns={"official_fill_pct": "official"})

    optical = cal.pair_points(official_points, weekly).rename(columns={"surface": "optical"})
    joined = joined.merge(optical[["dam", "date", "optical"]], on=["dam", "date"], how="left")
    joined["has_optical"] = joined["optical"].notna()

    out = {"n_dam_dates": int(len(joined)), "n_with_optical": int(joined["has_optical"].sum()),
           "n_optical_missing": int((~joined["has_optical"]).sum()), "per_dam": {}}
    both = joined[joined["has_optical"]]
    out["radar_vs_optical_surface"] = {
        "mae_pts": round(float(np.abs(both["surface"] - both["optical"]).mean()), 2) if len(both) else None,
        "correlation": round(float(np.corrcoef(both["surface"], both["optical"])[0, 1]), 3) if len(both) > 2 else None,
    }

    for dam, group in joined.groupby("dam"):
        entry = calibration["dams"].get(dam)
        if not entry or not entry["usable"]:
            continue
        optical_curve = cal.apply_curve(entry["curve"], group["surface"].values)
        refit = cal.cross_validate(group.rename(columns={"surface": "_r", "official": "_o"})
                                   .assign(surface=group["surface"], official=group["official"]))
        out["per_dam"][dam] = {
            "n": int(len(group)),
            "fill_error_optical_curve_pts": round(float(np.abs(optical_curve - group["official"].values).mean()), 2),
            "fill_error_radar_refit_cv_pts": round(refit[dam]["calibrated"], 2) if dam in refit else None,
            "optical_cv_error_pts": entry["typical_error_pts"],
        }
    return out


def threshold_predictor(threshold_db=-17.5):
    """The classical baseline: median-filtered VV below a cut-off."""
    from scipy import ndimage

    return lambda vv, vh: ndimage.median_filter(vv, size=3) < threshold_db


def unet_predictor(path="models/sar_unet.pt", threshold=0.5):
    """The trained radar U-Net, applied to patches of any size (padded to a
    multiple of 32 by reflection, then cropped back)."""
    import torch

    from src import sar_model as sm

    checkpoint = torch.load(path, map_location="cpu")
    model = sm.build_model(checkpoint["encoder"], pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    def predict(vv, vh):
        x = torch.tensor(sm.to_input(vv, vh))[None]
        h, w = x.shape[-2:]
        pad_h, pad_w = (-h) % 32, (-w) % 32
        x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        with torch.no_grad():
            prob = torch.sigmoid(model(x))[0, 0, :h, :w].numpy()
        return prob > threshold

    return predict


def run_evaluate(predictors, out_path=REPORT_PATH):
    from src import calibration as cal
    from src import official_fill
    from src.sites import TUNISIA_DAMS

    official = official_fill.load_points()
    weekly = pd.read_csv(WEEKLY_PATH, parse_dates=["week"])
    calibration = cal.load()
    report = {}
    for name, predictor in predictors.items():
        table = radar_surface_table(predictor, official, TUNISIA_DAMS)
        report[name] = evaluate(table, official, weekly, calibration)
        print(name, json.dumps(report[name], indent=1), flush=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    return report


if __name__ == "__main__":
    from src import official_fill
    from src.data import load_params
    from src.sites import TUNISIA_DAMS

    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "fetch":
        fetch(official_fill.load_points(), TUNISIA_DAMS, project=load_params()["gee"]["project"])
    elif command == "evaluate":
        predictors = {"threshold_vv_median": threshold_predictor()}
        if os.path.exists("models/sar_unet.pt"):
            predictors["unet"] = unet_predictor()
        run_evaluate(predictors)
