"""Scores segmentation checkpoints on the world benchmark (data/world/) and
breaks the result down by biome, continent and water-body type, to show
where a model built and validated on Tunisian dams fails elsewhere.

Overlap is pooled within each group (total intersection over total union),
not averaged per tile, so a group is not dominated by tiles with almost no
water. Tiles where the model is wrong by a lot are listed explicitly.

    python -m src.world_benchmark
"""
import json
import os

import numpy as np

from src.evaluate import score_checkpoint

WORLD_DIR = "data/world"
MODELS = {
    "served": "models/model.pt",
    "previous (v1)": "models/model_previous.pt",
    "v2 (Kaggle retrain)": "models/model_v2.pt",
}
OUT_JSON = "reports/world_benchmark.json"
TILES_JSON = "reports/world_tiles.json"
BAD_TILE_IOU = 0.3
BOOTSTRAP = 2000


MIN_VALID_FRACTION = 0.5
MIN_LABEL_WATER_PCT = 0.5


def tile_quality(row):
    """Why a tile cannot be trusted as a test of the model, or None. The
    labels come from NDWI plus Sentinel-2's own cloud/snow flags, so a tile
    that is mostly snow, flagged invalid, or has almost no labelled water
    is noise: a handful of pixels would decide its score."""
    if row.get("valid_fraction", 1.0) < MIN_VALID_FRACTION:
        return "mostly ignored (snow, cloud or no data)"
    if row["true_water_pct"] < MIN_LABEL_WATER_PCT:
        return "almost no labelled water"
    return None


def pooled(rows, bootstrap=0, seed=0):
    tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))
    out = {
        "n": len(rows),
        "iou": round(tp / max(tp + fp + fn, 1), 3),
        "precision": round(tp / max(tp + fp, 1), 3),
        "recall": round(tp / max(tp + fn, 1), 3),
        "water_pct_error": round(float(np.mean([abs(r["pred_water_pct"] - r["true_water_pct"]) for r in rows])), 2),
    }
    if bootstrap and len(rows) > 1:
        rng = np.random.default_rng(seed)
        counts = np.array([[r["tp"], r["fp"], r["fn"]] for r in rows])
        ious = []
        for _ in range(bootstrap):
            t, f, n = counts[rng.integers(0, len(rows), len(rows))].sum(axis=0)
            ious.append(t / max(t + f + n, 1))
        out["iou_ci95"] = [round(float(np.percentile(ious, 2.5)), 3), round(float(np.percentile(ious, 97.5)), 3)]
    return out


def by_group(per_image, key, bootstrap=0):
    groups = {}
    for row in per_image:
        groups.setdefault(row[key], []).append(row)
    return {name: pooled(rows, bootstrap) for name, rows in sorted(groups.items(), key=lambda kv: -len(kv[1]))}


def tile_iou(row):
    union = row["tp"] + row["fp"] + row["fn"]
    return row["tp"] / union if union else 1.0


def worst_tiles(per_image, n=8):
    ranked = sorted((r for r in per_image if r["true_water_pct"] + r["pred_water_pct"] > 1), key=tile_iou)
    return [{"site": r["site"], "country": r["country"], "biome": r["biome_name"],
             "iou": round(tile_iou(r), 2), "true_pct": r["true_water_pct"], "pred_pct": r["pred_water_pct"]}
            for r in ranked[:n]]


def run(world_dir=WORLD_DIR, models=None, out_json=OUT_JSON, tiles_json=TILES_JSON, track=False):
    results, tiles_by_model = {}, {}
    for label, path in (models or MODELS).items():
        if not os.path.exists(path):
            continue
        metrics, per_image = score_checkpoint(path, world_dir)
        clean = [r for r in per_image if tile_quality(r) is None]
        excluded = {}
        for r in per_image:
            reason = tile_quality(r)
            if reason:
                excluded[reason] = excluded.get(reason, 0) + 1
        results[label] = {
            "all_tiles": pooled(per_image),
            "clean": {
                "overall": pooled(clean, BOOTSTRAP),
                "by_biome": by_group(clean, "biome_name", BOOTSTRAP),
                "by_continent": by_group(clean, "continent", BOOTSTRAP),
                "by_type": by_group(clean, "type", BOOTSTRAP),
            },
            "excluded_tiles": excluded,
            "failures": sum(tile_iou(r) < BAD_TILE_IOU for r in clean),
            "worst_tiles": worst_tiles(clean),
        }
        print(label, "| all tiles:", results[label]["all_tiles"]["iou"], "| clean:", results[label]["clean"]["overall"],
              "| excluded:", excluded, "| clean tiles below IoU", BAD_TILE_IOU, ":", results[label]["failures"])
        tiles_by_model[label] = per_image
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    if tiles_by_model:
        write_map_tiles(tiles_by_model, world_dir, tiles_json)
    if track:
        from src import tracking

        for label, result in results.items():
            clean = result["clean"]
            tracking.log_experiment(
                "world-benchmark", f"benchmark {label}", params={"model": label, "tiles_dir": world_dir},
                metrics={"all_tiles": result["all_tiles"], "overall": clean["overall"], "failures": result["failures"],
                         "by_type": clean["by_type"]}, artifacts=[out_json], tags={"source": "world_benchmark"})
    return results


def write_map_tiles(tiles_by_model, world_dir=WORLD_DIR, path=TILES_JSON):
    """One record per benchmark site for the dashboard's world map."""
    with open(os.path.join(world_dir, "sites.json"), encoding="utf-8") as f:
        sites = {s["id"]: s for s in json.load(f)}
    served = next(iter(tiles_by_model))
    records = []
    for row in tiles_by_model[served]:
        site = sites[row["site_id"]]
        entry = {
            "id": row["site_id"], "name": site["name"], "country": site["country"], "biome": site["biome_name"],
            "type": site["type"], "area_km2": site["area_km2"], "lat": site["center_lat"], "lon": site["center_lon"],
            "quality_issue": tile_quality(row), "true_water_pct": row["true_water_pct"], "iou": {},
        }
        for model, rows in tiles_by_model.items():
            match = next(r for r in rows if r["site_id"] == row["site_id"])
            entry["iou"][model] = round(tile_iou(match), 3)
        records.append(entry)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"served_model": served, "tiles": records}, f, ensure_ascii=False)
    return records


if __name__ == "__main__":
    run(track=True)
