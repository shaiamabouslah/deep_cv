"""Worldwide set of lakes and reservoirs for the geographic benchmark.

Sites are drawn at random, per biome, from HydroLAKES (Messager et al. 2016:
1.4 million lakes and reservoirs, with area and type) so that no region is
hand-picked, and capped at two per country for geographic spread. Biome comes
from the RESOLVE Ecoregions 2017 map. The point is to test models built on
Tunisian dams where they have never been: tropical forest, boreal, desert,
monsoon, alpine.

The list is saved to data/world/sites.json (fixed seed) so every later step
uses exactly the same sites.

    python -m src.world_sites
"""
import json
import math
import os

from src.data import load_params
from src.gee_client import _ensure_initialized

HYDROLAKES = "projects/sat-io/open-datasets/HydroLakes/lake_poly_v10"
ECOREGIONS = "RESOLVE/ECOREGIONS/2017"
SITES_PATH = "data/world/sites.json"
MIN_AREA_KM2, MAX_AREA_KM2 = 3, 300
MIN_SIDE_KM, MAX_SIDE_KM = 4.0, 24.0
LAKE_TYPES = {1: "lake", 2: "reservoir"}

BIOMES = {
    1: "Tropical moist forest", 2: "Tropical dry forest", 3: "Tropical conifer forest",
    4: "Temperate broadleaf forest", 5: "Temperate conifer forest", 6: "Boreal forest",
    7: "Tropical savanna", 8: "Temperate grassland", 9: "Flooded grassland", 10: "Montane grassland",
    11: "Tundra", 12: "Mediterranean", 13: "Desert", 14: "Mangrove",
}


def tile_side_km(width_km, height_km):
    """Ground size of the tile: the lake fits with a margin of land around it."""
    return float(min(max(1.8 * max(width_km, height_km), MIN_SIDE_KM), MAX_SIDE_KM))


def bounds_km(west, south, east, north):
    mid_lat = math.radians((south + north) / 2)
    return (east - west) * 111.32 * math.cos(mid_lat), (north - south) * 110.57


def pick_per_biome(candidates, per_biome, max_per_country=2):
    """`candidates` are already in random order. Keeps the first ones per
    biome, at most `max_per_country` from the same country."""
    chosen, per_country = [], {}
    counts = {}
    for site in candidates:
        biome = site["biome"]
        if biome not in BIOMES or counts.get(biome, 0) >= per_biome:
            continue
        key = (biome, site["country"])
        if per_country.get(key, 0) >= max_per_country:
            continue
        per_country[key] = per_country.get(key, 0) + 1
        counts[biome] = counts.get(biome, 0) + 1
        chosen.append(site)
    return chosen


def sample_sites(per_biome=8, seed=7, candidates=3000, project=None, exclude=(), slices=1):
    """`exclude` = Hylak ids that must not be drawn (e.g. the benchmark sites,
    when drawing a disjoint training set). `slices` > 1 splits the candidate
    draw into several smaller Earth Engine requests (slices=1 reproduces the
    original benchmark draw exactly)."""
    import ee

    _ensure_initialized(project)
    eligible = ee.FeatureCollection(HYDROLAKES).filter(ee.Filter.And(
        ee.Filter.gte("Lake_area", MIN_AREA_KM2), ee.Filter.lte("Lake_area", MAX_AREA_KM2),
        ee.Filter.inList("Lake_type", list(LAKE_TYPES)),
    ))
    randomized = eligible.randomColumn("rand", seed)
    props = ["Hylak_id", "Lake_name", "Country", "Continent", "Lake_type", "Lake_area", "rand"]
    biome_map = ee.FeatureCollection(ECOREGIONS).reduceToImage(["BIOME_NUM"], ee.Reducer.first()).rename("biome")

    def tag(lakes):
        centroids = lakes.map(lambda f: ee.Feature(f.geometry().centroid(100), f.toDictionary(props)))
        return biome_map.reduceRegions(centroids, ee.Reducer.first().setOutputs(["biome"]), 1000).getInfo()["features"]

    if slices == 1:
        tagged = tag(randomized.sort("rand").limit(candidates))
    else:
        # One request for thousands of lakes times out: ask for slices of the random order.
        tagged = []
        for i in range(slices):
            part = randomized.filter(ee.Filter.And(ee.Filter.gte("rand", i / slices), ee.Filter.lt("rand", (i + 1) / slices)))
            tagged += tag(part.sort("rand").limit(candidates // slices))

    rows = []
    for f in tagged:
        p, (lon, lat) = f["properties"], f["geometry"]["coordinates"]
        if p.get("first") is None and p.get("biome") is None:
            continue
        rows.append({
            "id": int(p["Hylak_id"]), "name": p.get("Lake_name") or f"lake {int(p['Hylak_id'])}",
            "country": p["Country"], "continent": p["Continent"], "type": LAKE_TYPES[int(p["Lake_type"])],
            "area_km2": round(float(p["Lake_area"]), 1), "lat": lat, "lon": lon,
            "biome": int(p.get("biome") if p.get("biome") is not None else p["first"]), "rand": p["rand"],
        })
    excluded = {int(i) for i in exclude}
    rows = [r for r in rows if r["id"] not in excluded]
    rows.sort(key=lambda r: r["rand"])
    chosen = pick_per_biome(rows, per_biome, max_per_country=max(2, per_biome // 4))

    ids = [s["id"] for s in chosen]
    shapes = eligible.filter(ee.Filter.inList("Hylak_id", ids)).map(
        lambda f: ee.Feature(None, {"id": f.get("Hylak_id"), "box": f.geometry().bounds().coordinates()})
    ).getInfo()["features"]
    boxes = {int(f["properties"]["id"]): f["properties"]["box"][0] for f in shapes}

    sites = []
    for s in chosen:
        ring = boxes[s["id"]]
        lons, lats = [c[0] for c in ring], [c[1] for c in ring]
        width, height = bounds_km(min(lons), min(lats), max(lons), max(lats))
        sites.append({
            **{k: v for k, v in s.items() if k != "rand"},
            "biome_name": BIOMES[s["biome"]],
            "center_lat": (min(lats) + max(lats)) / 2, "center_lon": (min(lons) + max(lons)) / 2,
            "side_km": round(tile_side_km(width, height), 1),
        })
    return sites


def build(path=SITES_PATH, params_path="params.yaml"):
    sites = sample_sites(project=load_params(params_path)["gee"]["project"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sites, f, ensure_ascii=False, indent=1)
    by_biome = {}
    for s in sites:
        by_biome[s["biome_name"]] = by_biome.get(s["biome_name"], 0) + 1
    print(f"{len(sites)} sites, {len({s['country'] for s in sites})} countries, "
          f"{len({s['continent'] for s in sites})} continents")
    print(by_biome)
    return sites


def load(path=SITES_PATH):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    build()
