import json

import numpy as np
import pytest
import torch
from PIL import Image

from src import finetune_world as fw
from src import promote


def _write_tile(directory, name, seed, water_cols=slice(10, 40), bright=1.0):
    rng = np.random.default_rng(seed)
    size = 96
    label = np.zeros((size, size), np.uint8)
    label[:, water_cols] = 255
    rgb = np.where(label[..., None] == 255, [30, 60, 90], [120, 140, 80]).astype(float) * bright
    rgb = np.clip(rgb + rng.normal(0, 4, (size, size, 3)), 5, 255).astype(np.uint8)
    (directory / "images").mkdir(parents=True, exist_ok=True)
    (directory / "masks").mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(directory / "images" / name)
    Image.fromarray(label).save(directory / "masks" / name)


def _tiles(tmp_path, n=10):
    entries = []
    for i in range(n):
        _write_tile(tmp_path, f"s{i}.png", i, water_cols=slice(8 + i, 40 + i))
        entries.append({"file": f"s{i}.png", "site_id": f"site{i}"})
    (tmp_path / "manifest.json").write_text(json.dumps(entries))
    return entries


def test_tiles_that_are_mostly_ignored_are_not_used_for_training(tmp_path):
    _tiles(tmp_path, 3)
    label = np.full((96, 96), 128, np.uint8)
    Image.fromarray(label).save(tmp_path / "masks" / "s1.png")
    assert [e["file"] for e in fw.usable_entries(str(tmp_path))] == ["s0.png", "s2.png"]


def test_validation_split_keeps_each_site_in_one_side():
    entries = [{"file": f"{i}.png", "site_id": f"site{i // 2}"} for i in range(20)]
    train, val = fw.split_by_site(entries, val_frac=0.2)
    assert not {e["site_id"] for e in train} & {e["site_id"] for e in val}
    assert len(train) + len(val) == 20 and val


def test_color_jitter_stays_in_range_and_changes_the_image():
    x = torch.rand(3, 32, 32)
    out = fw.color_jitter(x)
    assert out.min() >= 0 and out.max() <= 1
    assert not torch.allclose(out, x)


def test_dataset_returns_a_model_sized_normalized_sample_with_a_valid_mask(tmp_path):
    entries = _tiles(tmp_path, 2)
    x, y, m = fw.TileDataset(str(tmp_path), entries, augment=True)[0]
    assert x.shape == (3, fw.IMAGE_SIZE, fw.IMAGE_SIZE) and y.shape == m.shape == (1, fw.IMAGE_SIZE, fw.IMAGE_SIZE)
    assert set(y.unique().tolist()) <= {0.0, 1.0}


def test_finetune_improves_on_a_random_start_and_returns_a_threshold(tmp_path):
    entries = _tiles(tmp_path, 10)
    train, val = entries[:8], entries[8:]
    model, threshold, curve = fw.finetune(train, val, str(tmp_path), base_path=None, epochs=12, batch_size=2,
                                          lr=3e-4, log=lambda *_: None)
    assert len(curve) == 12 and max(curve) > 0.5
    assert threshold in (0.3, 0.4, 0.5, 0.6, 0.7)


def _scores(tunisia, world):
    return {"tunisia_iou": tunisia, "world": {"iou": world}}


@pytest.mark.parametrize("candidate, expected", [
    (_scores(0.72, 0.50), True),
    (_scores(0.71, 0.48), True),
    (_scores(0.60, 0.70), False),
    (_scores(0.73, 0.46), False),
])
def test_gate_requires_no_home_regression_and_a_world_gain(candidate, expected):
    promoted, rules = promote.decide(_scores(0.72, 0.45), candidate)
    assert promoted is expected and set(rules) == {"no regression in Tunisia", "gain on the world benchmark"}


def test_promotion_only_replaces_the_live_model_when_applied_and_all_rules_hold(tmp_path):
    live, cand = tmp_path / "model.pt", tmp_path / "candidate.pt"
    live.write_bytes(b"live")
    cand.write_bytes(b"candidate")
    scores = {str(live): _scores(0.72, 0.45), str(cand): _scores(0.73, 0.60)}
    evaluator = lambda path: scores[path]

    dry = promote.run(str(cand), apply=False, live_path=str(live), report_path=str(tmp_path / "r.json"), evaluator=evaluator)
    assert dry["would_promote"] is True and dry["promoted"] is False and live.read_bytes() == b"live"

    done = promote.run(str(cand), apply=True, live_path=str(live), report_path=str(tmp_path / "r.json"), evaluator=evaluator)
    assert done["promoted"] is True and live.read_bytes() == b"candidate"
    assert (tmp_path / "model_previous.pt").read_bytes() == b"live"


def test_a_failing_candidate_never_touches_the_live_model(tmp_path):
    live, cand = tmp_path / "model.pt", tmp_path / "candidate.pt"
    live.write_bytes(b"live")
    cand.write_bytes(b"candidate")
    scores = {str(live): _scores(0.72, 0.45), str(cand): _scores(0.50, 0.80)}

    report = promote.run(str(cand), apply=True, live_path=str(live), report_path=str(tmp_path / "r.json"),
                         evaluator=lambda path: scores[path])

    assert report["promoted"] is False and live.read_bytes() == b"live"
    assert not (tmp_path / "model_previous.pt").exists()
