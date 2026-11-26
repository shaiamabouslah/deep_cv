import json

import numpy as np
import pytest
import torch
from PIL import Image

from src import evaluate as evaluate_module
from src import world_benchmark as wb
from src import world_sites as ws


def _row(biome, tp, fp, fn, true_pct=10.0, pred_pct=10.0, site="S"):
    return {"biome_name": biome, "continent": "X", "type": "lake", "tp": tp, "fp": fp, "fn": fn,
            "true_water_pct": true_pct, "pred_water_pct": pred_pct, "site": site, "country": "C"}


def test_group_overlap_is_pooled_not_averaged_per_tile():
    rows = [_row("A", tp=900, fp=0, fn=100), _row("A", tp=0, fp=10, fn=0, true_pct=0.0, pred_pct=1.0)]
    pooled = wb.pooled(rows)
    assert pooled["iou"] == round(900 / 1010, 3)
    assert pooled["n"] == 2


def test_by_group_splits_on_the_requested_key():
    rows = [_row("Desert", 10, 0, 0), _row("Desert", 10, 0, 0), _row("Tundra", 5, 5, 0)]
    groups = wb.by_group(rows, "biome_name")
    assert list(groups) == ["Desert", "Tundra"]
    assert groups["Desert"]["n"] == 2 and groups["Tundra"]["iou"] == 0.5


def test_worst_tiles_ranks_the_failures_and_skips_empty_tiles():
    rows = [_row("A", 100, 0, 0, site="good"), _row("B", 10, 80, 10, site="bad"),
            _row("C", 0, 0, 0, true_pct=0.0, pred_pct=0.0, site="empty")]
    worst = wb.worst_tiles(rows)
    assert [w["site"] for w in worst] == ["bad", "good"]


def test_score_checkpoint_keeps_per_tile_counts(tmp_path, monkeypatch):
    size = 32
    eval_dir = tmp_path / "eval"
    (eval_dir / "images").mkdir(parents=True)
    (eval_dir / "masks").mkdir()
    Image.new("RGB", (size, size), (100, 120, 110)).save(eval_dir / "images" / "t.png")
    label = np.zeros((size, size), np.uint8)
    label[:, : size // 2] = 255
    Image.fromarray(label).save(eval_dir / "masks" / "t.png")
    (eval_dir / "manifest.json").write_text(json.dumps([{"file": "t.png", "site": "X"}]))
    monkeypatch.setattr(evaluate_module, "load_model", lambda path: (lambda x: torch.full((1, 1, size, size), 10.0), size, 0.5))

    metrics, per_image = evaluate_module.score_checkpoint("ignored.pt", str(eval_dir))

    assert per_image[0]["tp"] == size * size // 2
    assert per_image[0]["fp"] == size * size // 2
    assert per_image[0]["fn"] == 0
    assert metrics["iou"] == 0.5


def test_tile_side_keeps_the_lake_inside_with_margin_and_stays_bounded():
    assert ws.tile_side_km(1.0, 1.0) == ws.MIN_SIDE_KM
    assert ws.tile_side_km(5.0, 3.0) == pytest.approx(9.0)
    assert ws.tile_side_km(100.0, 40.0) == ws.MAX_SIDE_KM


def test_pick_per_biome_caps_per_country_and_per_biome():
    candidates = [{"biome": 13, "country": "Chad"}] * 5 + [{"biome": 13, "country": "Mali"}] * 5
    chosen = ws.pick_per_biome(candidates, per_biome=3, max_per_country=2)
    assert len(chosen) == 3
    assert sum(c["country"] == "Chad" for c in chosen) == 2


def test_pick_per_biome_ignores_non_biome_classes():
    assert ws.pick_per_biome([{"biome": 98, "country": "X"}, {"biome": 99, "country": "Y"}], 5) == []
