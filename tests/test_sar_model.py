import numpy as np
import torch

from src import sar_model as sm


def _write_patch(directory, name, seed, dark_rows=slice(10, 40), invalid_cols=None):
    """Synthetic radar patch: a dark band is water (VV ~ -22 dB), the rest is
    bright land (-9 dB), both with speckle."""
    rng = np.random.default_rng(seed)
    size = 64
    water = np.zeros((size, size), np.uint8)
    water[dark_rows, :] = 1
    vv = np.where(water == 1, -22.0, -9.0) + rng.normal(0, 1.5, (size, size))
    vh = np.where(water == 1, -29.0, -16.0) + rng.normal(0, 1.5, (size, size))
    valid = np.ones((size, size), np.uint8)
    if invalid_cols is not None:
        valid[:, invalid_cols] = 0
    (directory / "patches").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / "patches" / name, vv=vv.astype(np.float16), vh=vh.astype(np.float16),
                        water=water, valid=valid)
    return {"file": name}


def _manifest():
    rows = []
    for i in range(20):
        rows.append({"file": f"w{i}.npz", "site_id": f"site{i // 2}", "kind": "world"})
    for i in range(3):
        rows.append({"file": f"t{i}.npz", "site_id": f"tn{i}", "kind": "tunisia"})
    return rows


def test_split_keeps_every_site_in_one_split_and_tunisia_out_of_training():
    splits = sm.split_sites(_manifest(), seed=1)
    sites = {name: {r["site_id"] for r in rows} for name, rows in splits.items()}

    assert not (sites["train"] & sites["val"]) and not (sites["train"] & sites["test_world"])
    assert not (sites["val"] & sites["test_world"])
    assert all(r["kind"] == "world" for r in splits["train"])
    assert {r["site_id"] for r in splits["test_tunisia"]} == {"tn0", "tn1", "tn2"}
    assert len(splits["train"]) + len(splits["val"]) + len(splits["test_world"]) == 20


def test_input_has_three_bounded_channels():
    x = sm.to_input(np.full((8, 8), -12.0, np.float32), np.full((8, 8), -19.0, np.float32))
    assert x.shape == (3, 8, 8)
    assert np.isfinite(x).all() and np.abs(x).max() < 10


def test_masked_loss_ignores_invalid_pixels():
    logits = torch.zeros(1, 1, 4, 4)
    target = torch.zeros(1, 1, 4, 4)
    valid = torch.ones(1, 1, 4, 4)
    valid[..., 2:] = 0
    wrong_but_ignored = target.clone()
    wrong_but_ignored[..., 2:] = 1
    assert masked_equal(sm.masked_loss(logits, target, valid), sm.masked_loss(logits, wrong_but_ignored, valid))


def masked_equal(a, b):
    return torch.isclose(a, b)


def test_scores_count_only_valid_pixels():
    pred = np.array([[1, 1], [0, 0]])
    truth = np.array([[1, 0], [1, 0]])
    valid = np.array([[1, 1], [1, 0]])
    assert sm.counts(pred, truth, valid) == (1, 1, 1)
    assert sm.scores((1, 1, 1))["iou"] == round(1 / 3, 4)


def test_threshold_baseline_finds_the_water_band(tmp_path):
    rows = [_write_patch(tmp_path, f"p{i}.npz", i) for i in range(4)]
    baseline = sm.ThresholdBaseline("vv").fit(rows, str(tmp_path))

    assert -22 < baseline.threshold < -9
    assert sm.evaluate_baseline(baseline, rows, str(tmp_path))["iou"] > 0.9


def test_threshold_baseline_does_not_score_invalid_pixels(tmp_path):
    rows = [_write_patch(tmp_path, "p0.npz", 0, invalid_cols=slice(0, 32))]
    baseline = sm.ThresholdBaseline("vv").fit(rows, str(tmp_path))
    pred = baseline.predict(*sm.load_patch(tmp_path / "patches" / "p0.npz")[:2])
    water, valid = sm.load_patch(tmp_path / "patches" / "p0.npz")[2:]
    assert sum(sm.counts(pred, water, valid)) < pred.size * 0.6


def test_gbm_baseline_learns_the_band(tmp_path):
    rows = [_write_patch(tmp_path, f"p{i}.npz", i) for i in range(4)]
    baseline = sm.GbmBaseline(max_pixels=5000).fit(rows[:3], str(tmp_path))
    assert sm.evaluate_baseline(baseline, rows[3:], str(tmp_path))["iou"] > 0.9


def test_unet_trains_on_masked_radar_patches_and_beats_chance(tmp_path):
    rows = [_write_patch(tmp_path, f"p{i}.npz", i, dark_rows=slice(8 + i, 36 + i)) for i in range(8)]
    model, history = sm.train(rows[:6], rows[6:], epochs=6, batch_size=2, sar_dir=str(tmp_path),
                              pretrained=False, log=lambda *_: None)

    assert len(history) == 6 and max(history) > 0.7
    assert sm.evaluate_model(model, rows[6:], str(tmp_path))["iou"] > 0.7
