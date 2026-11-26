import json

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from src import evaluate as evaluate_module

SIZE = 32


def _write_params(tmp_path, model_path):
    params = {"model": {"model_path": str(model_path)}, "paths": {"metrics_out": str(tmp_path / "metrics.json")}}
    params_path = tmp_path / "params.yaml"
    params_path.write_text(yaml.safe_dump(params))
    return str(params_path)


def _write_eval_set(tmp_path):
    eval_dir = tmp_path / "eval"
    (eval_dir / "images").mkdir(parents=True)
    (eval_dir / "masks").mkdir()
    Image.new("RGB", (SIZE, SIZE), (100, 120, 110)).save(eval_dir / "images" / "t.png")
    label = np.zeros((SIZE, SIZE), np.uint8)
    label[:, : SIZE // 2] = 255
    label[: SIZE // 2, SIZE // 2:] = 128
    Image.fromarray(label).save(eval_dir / "masks" / "t.png")
    (eval_dir / "manifest.json").write_text(json.dumps([{"file": "t.png", "site": "X"}]))
    return str(eval_dir)


def test_evaluate_scores_only_clear_pixels(tmp_path, monkeypatch):
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"")
    predicts_all_water = lambda x: torch.full((1, 1, SIZE, SIZE), 10.0)
    monkeypatch.setattr(evaluate_module, "load_model", lambda path: (predicts_all_water, SIZE, 0.5))

    metrics = evaluate_module.evaluate(
        _write_params(tmp_path, model_path), _write_eval_set(tmp_path), str(tmp_path / "per_image.json")
    )

    # Left half water (correct), bottom-right land (false positive),
    # top-right cloud (ignored): 2 correct quarters out of 3 scored.
    assert metrics["iou"] == round(2 / 3, 4)
    assert metrics["precision"] == round(2 / 3, 4)
    assert metrics["recall"] == 1.0
    assert metrics["water_pct_mae"] == pytest.approx(100 - 200 / 3, abs=0.01)


def test_evaluate_skips_without_model(tmp_path):
    metrics = evaluate_module.evaluate(
        _write_params(tmp_path, tmp_path / "missing.pt"), _write_eval_set(tmp_path), str(tmp_path / "p.json")
    )
    assert metrics["status"] == "skipped"
