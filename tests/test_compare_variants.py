import json

from src import compare_variants as cv


def test_render_markdown_lists_every_variant_and_the_data_curve():
    rows = [
        {"variant": "scratch", "kaggle_test_iou": 0.71, "best_epoch": 18, "dam_iou": 0.6,
         "dam_precision": 0.7, "dam_recall": 0.8, "dam_water_pct_mae": 2.5},
        {"variant": "finetune", "kaggle_test_iou": 0.83, "best_epoch": 12, "dam_iou": 0.74,
         "dam_precision": 0.8, "dam_recall": 0.85, "dam_water_pct_mae": 1.1},
    ]
    fractions = {"scratch": {"0.1": 0.4, "1.0": 0.71}, "finetune": {"0.1": 0.7, "1.0": 0.83}}

    text = cv.render_markdown(rows, fractions)

    assert "From scratch (random init)" in text and "ImageNet, fine-tuned" in text
    assert "0.830" in text and "| 10% | 100% |" in text.replace("| Variant | ", "| ")


def test_compare_scores_each_checkpoint_and_merges_kaggle_numbers(tmp_path, monkeypatch):
    variants = tmp_path / "variants"
    variants.mkdir()
    for name in ("scratch", "finetune"):
        (variants / f"model_{name}.pt").write_bytes(b"")
    kaggle = tmp_path / "kaggle.json"
    kaggle.write_text(json.dumps({
        "scratch@1.0": {"test_iou": 0.7, "best_epoch": 15},
        "finetune@1.0": {"test_iou": 0.8, "best_epoch": 9},
        "finetune@0.1": {"test_iou": 0.6}, "scratch@0.1": {"test_iou": 0.3},
    }))
    fake = {"threshold": 0.5, "iou": 0.5, "precision": 0.6, "recall": 0.7, "water_pct_mae": 1.0}
    monkeypatch.setattr(cv, "score_checkpoint", lambda path: (fake, []))

    rows = cv.compare(str(variants), str(kaggle), str(tmp_path / "a.json"), str(tmp_path / "a.md"))

    assert [r["variant"] for r in rows] == ["finetune", "scratch"]
    assert rows[0]["kaggle_test_iou"] == 0.8 and rows[0]["best_epoch"] == 9
    saved = json.loads((tmp_path / "a.json").read_text())
    assert saved["data_fraction_test_iou"]["scratch"]["0.1"] == 0.3
    assert (tmp_path / "a.md").exists()


def test_compare_without_kaggle_file_still_reports_dam_scores(tmp_path, monkeypatch):
    variants = tmp_path / "variants"
    variants.mkdir()
    (variants / "model_scratch.pt").write_bytes(b"")
    fake = {"threshold": 0.5, "iou": 0.5, "precision": 0.6, "recall": 0.7, "water_pct_mae": 1.0}
    monkeypatch.setattr(cv, "score_checkpoint", lambda path: (fake, []))

    rows = cv.compare(str(variants), str(tmp_path / "missing.json"), str(tmp_path / "a.json"), str(tmp_path / "a.md"))

    assert rows[0]["kaggle_test_iou"] is None and rows[0]["dam_iou"] == 0.5
