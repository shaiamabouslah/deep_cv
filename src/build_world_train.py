"""Builds a worldwide TRAINING set of tiles, disjoint from the benchmark
(data/world/): different lakes and reservoirs, drawn the same way (random
per biome from HydroLAKES), rendered and labelled exactly like the
benchmark tiles so a model trained on them sees the imagery it will be
judged on. Output: data/world_train/.

    python -m src.build_world_train
"""
import json
import os

from src import build_world_set, world_sites
from src.data import load_params

TRAIN_DIR = "data/world_train"
PER_BIOME = 28
SEED = 11
CANDIDATES = 8000
SLICES = 8


def build(params_path="params.yaml"):
    benchmark_ids = {s["id"] for s in world_sites.load(world_sites.SITES_PATH)}
    sites = world_sites.sample_sites(
        per_biome=PER_BIOME, seed=SEED, candidates=CANDIDATES,
        project=load_params(params_path)["gee"]["project"], exclude=benchmark_ids, slices=SLICES,
    )
    assert not benchmark_ids & {s["id"] for s in sites}
    os.makedirs(TRAIN_DIR, exist_ok=True)
    with open(os.path.join(TRAIN_DIR, "sites.json"), "w", encoding="utf-8") as f:
        json.dump(sites, f, ensure_ascii=False, indent=1)
    print(f"{len(sites)} training sites in {len({s['country'] for s in sites})} countries", flush=True)
    build_world_set.build(params_path, TRAIN_DIR)


if __name__ == "__main__":
    build()
