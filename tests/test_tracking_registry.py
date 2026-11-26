import pytest
import torch

from src import registry, tracking
from src.model import build_segmentation_model


@pytest.fixture()
def uri(tmp_path, monkeypatch):
    monkeypatch.setattr(tracking, "ARTIFACT_DIR", str(tmp_path / "artifacts"))
    return f"sqlite:///{tmp_path / 'mlflow.db'}"


def _checkpoint(path, threshold=0.5):
    model = build_segmentation_model("resnet18", encoder_weights=None)
    torch.save({"state_dict": model.state_dict(), "encoder_name": "resnet18", "image_size": 64,
                "threshold": threshold}, path)
    return str(path)


def test_flatten_keeps_numbers_only_and_makes_valid_metric_names():
    flat = tracking.flatten({"a": {"b": 1, "c": True}, "d": "text", "e": float("nan"), "f g": 2.5, "h": {"i:j": 3}})
    assert flat == {"a.b": 1.0, "a.c": 1.0, "f g": 2.5, "h.i_j": 3.0}


def test_git_info_reports_a_commit_and_dirty_flag():
    info = tracking.git_info()
    assert set(info) == {"git_commit", "git_branch", "git_dirty"} and info["git_dirty"] in {"true", "false"}


def test_logged_experiment_has_params_metrics_curves_and_code_version(uri, tmp_path):
    from mlflow import MlflowClient

    artifact = tmp_path / "report.json"
    artifact.write_text("{}")
    run_id = tracking.log_experiment(
        "unit-test", "run-1", params={"lr": 0.001}, metrics={"iou": {"val": 0.8}},
        curves={"val_iou": [0.5, 0.6, 0.7]}, artifacts=[str(artifact)], tags={"stage": "test"}, tracking_uri=uri,
    )
    client = MlflowClient(tracking_uri=uri)
    run = client.get_run(run_id)

    assert run.data.params["lr"] == "0.001"
    assert run.data.metrics["iou.val"] == 0.8
    assert [m.value for m in client.get_metric_history(run_id, "val_iou")] == [0.5, 0.6, 0.7]
    assert run.data.tags["stage"] == "test" and "git_commit" in run.data.tags
    assert [a.path for a in client.list_artifacts(run_id)] == ["report.json"]


def test_tracking_failure_never_breaks_the_caller(monkeypatch):
    monkeypatch.setattr(tracking, "configure", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("store down")))
    assert tracking.log_experiment("x", "y", metrics={"a": 1}) is None


def test_registry_versions_aliases_and_rollback(uri, tmp_path):
    v1 = registry.register_segmentation(_checkpoint(tmp_path / "a.pt"), "first", {"tunisia_iou": 0.72},
                                        alias="champion", tracking_uri=uri)
    v2 = registry.register_segmentation(_checkpoint(tmp_path / "b.pt", 0.3), "second", {"tunisia_iou": 0.76},
                                        tracking_uri=uri)
    assert (v1, v2) == (1, 2)
    assert registry.alias_version(registry.SEGMENTATION, "champion", uri) == 1

    registry.promote_segmentation(v2, previous_champion=v1, tracking_uri=uri)
    assert registry.alias_version(registry.SEGMENTATION, "champion", uri) == 2
    assert registry.alias_version(registry.SEGMENTATION, "previous", uri) == 1

    registry.reject_segmentation(1, tracking_uri=uri)
    history = registry.lineage(registry.SEGMENTATION, uri)
    assert history[0]["aliases"] == ["previous", "rejected"] and history[1]["aliases"] == ["champion"]
    assert history[1]["tags"]["tunisia_iou"] == "0.76" and history[1]["description"] == "second"


def test_a_registered_model_can_be_loaded_back_by_its_alias(uri, tmp_path):
    import mlflow

    registry.register_segmentation(_checkpoint(tmp_path / "a.pt"), "d", {}, alias="champion", tracking_uri=uri)
    mlflow.set_tracking_uri(uri)
    model = mlflow.pytorch.load_model(f"models:/{registry.SEGMENTATION}@champion")
    assert model(torch.zeros(1, 3, 64, 64)).shape == (1, 1, 64, 64)


def test_promote_if_better_only_replaces_the_champion_when_the_metric_improves(uri, tmp_path):
    v1 = registry.register_segmentation(_checkpoint(tmp_path / "a.pt"), "a", {"iou": 0.7}, tracking_uri=uri)
    v2 = registry.register_segmentation(_checkpoint(tmp_path / "b.pt"), "b", {"iou": 0.6}, tracking_uri=uri)
    v3 = registry.register_segmentation(_checkpoint(tmp_path / "c.pt"), "c", {"iou": 0.8}, tracking_uri=uri)

    assert registry.promote_if_better(registry.SEGMENTATION, v1, "iou", uri) is True
    assert registry.promote_if_better(registry.SEGMENTATION, v2, "iou", uri) is False
    assert registry.alias_version(registry.SEGMENTATION, "champion", uri) == 1
    assert registry.promote_if_better(registry.SEGMENTATION, v3, "iou", uri) is True
    assert registry.alias_version(registry.SEGMENTATION, "champion", uri) == 3
    assert registry.alias_version(registry.SEGMENTATION, "previous", uri) == 1


def test_promote_if_better_can_prefer_lower_values(uri, tmp_path):
    v1 = registry.register_segmentation(_checkpoint(tmp_path / "a.pt"), "a", {"mae": 5.0}, tracking_uri=uri)
    v2 = registry.register_segmentation(_checkpoint(tmp_path / "b.pt"), "b", {"mae": 3.0}, tracking_uri=uri)
    registry.promote_if_better(registry.SEGMENTATION, v1, "mae", uri, higher_is_better=False)
    assert registry.promote_if_better(registry.SEGMENTATION, v2, "mae", uri, higher_is_better=False) is True
    assert registry.alias_version(registry.SEGMENTATION, "champion", uri) == 2
