"""MLflow Model Registry: one registered model per kind of model, with
versions, descriptive tags and aliases that say which version plays which
role.

  water-segmentation       U-Net for the image-upload path
      alias `champion`  = the version served by the API
      alias `previous`  = the version it replaced (instant rollback target)
      alias `rejected`  = a candidate that failed the promotion gate
  sar-water-segmentation   radar U-Net
  water-forecaster         GRU ensemble (an MLflow pyfunc model that wraps
                           the checkpoint, so it is loadable by alias too)

Aliases, not the deprecated stages, are MLflow's current way to manage a
model's lifecycle; a version's tags hold the metrics it was promoted on.
"""
import os

from src import tracking

SEGMENTATION = "water-segmentation"
SAR = "sar-water-segmentation"
FORECASTER = "water-forecaster"
EXPERIMENT = "model-registry"


def _client(tracking_uri=None):
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(tracking_uri or tracking.TRACKING_URI)
    return mlflow, MlflowClient()


def set_alias(name, alias, version, tracking_uri=None):
    _, client = _client(tracking_uri)
    client.set_registered_model_alias(name, alias, str(version))


def alias_version(name, alias, tracking_uri=None):
    """Version number an alias points to, or None."""
    _, client = _client(tracking_uri)
    try:
        return int(client.get_model_version_by_alias(name, alias).version)
    except Exception:
        return None


def load_radar(checkpoint_path):
    import torch

    from src.sar_model import build_model

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = build_model(checkpoint["encoder"], pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), None, None


def register_segmentation(checkpoint_path, description, tags, metrics=None, alias=None, name=SEGMENTATION,
                          tracking_uri=None, loader=None):
    """Logs a segmentation checkpoint as an MLflow PyTorch model and registers
    it as a new version of `name`. Returns the version number. `loader`
    returns (model, image_size, threshold); the default reads an optical
    checkpoint."""
    if loader is None:
        from src.model import load_model as loader

    model, image_size, threshold = loader(checkpoint_path)
    mlflow = tracking.configure(EXPERIMENT, tracking_uri)
    with mlflow.start_run(run_name=f"register {os.path.basename(checkpoint_path)}") as run:
        mlflow.set_tags({**tracking.git_info(), "checkpoint": checkpoint_path})
        mlflow.log_params({"checkpoint": checkpoint_path, **({"image_size": image_size, "threshold": threshold}
                                                             if image_size is not None else {})})
        for key, value in tracking.flatten(metrics or {}).items():
            mlflow.log_metric(key, value)
        mlflow.pytorch.log_model(model, name="model", serialization_format="pickle", registered_model_name=name)
    return _finish_registration(name, run.info.run_id, description, tags, alias, tracking_uri)


def _finish_registration(name, run_id, description, tags, alias, tracking_uri):
    _, client = _client(tracking_uri)
    versions = [v for v in client.search_model_versions(f"name='{name}'") if v.run_id == run_id]
    version = max(int(v.version) for v in versions)
    client.update_model_version(name, str(version), description=description)
    for key, value in (tags or {}).items():
        client.set_model_version_tag(name, str(version), tracking.metric_name(key), str(value))
    if alias:
        client.set_registered_model_alias(name, alias, str(version))
    return version


def make_forecaster_pyfunc():
    """pyfunc wrapper around the forecaster checkpoint (built lazily so
    importing this module does not require MLflow's pyfunc machinery)."""
    import mlflow.pyfunc

    class ForecasterModel(mlflow.pyfunc.PythonModel):
        def load_context(self, context):
            from src import forecast_dl

            self.bundle = forecast_dl.load_bundle(context.artifacts["checkpoint"])

        def predict(self, context, model_input, params=None):
            from src import forecast_dl

            return forecast_dl.predict(self.bundle, model_input)

    return ForecasterModel()


def register_forecaster(checkpoint_path, description, tags, metrics=None, alias=None, tracking_uri=None):
    import mlflow.pyfunc

    mlflow = tracking.configure(EXPERIMENT, tracking_uri)
    with mlflow.start_run(run_name=f"register {os.path.basename(checkpoint_path)}") as run:
        mlflow.set_tags({**tracking.git_info(), "checkpoint": checkpoint_path})
        for key, value in tracking.flatten(metrics or {}).items():
            mlflow.log_metric(key, value)
        mlflow.pyfunc.log_model(
            name="model", python_model=make_forecaster_pyfunc(), artifacts={"checkpoint": checkpoint_path},
            registered_model_name=FORECASTER, pip_requirements=["torch", "numpy", "pandas"],
        )
    return _finish_registration(FORECASTER, run.info.run_id, description, tags, alias, tracking_uri)


def promote_segmentation(candidate_version, previous_champion=None, name=SEGMENTATION, tracking_uri=None):
    """The new champion replaces the old one; the old one is kept as `previous`."""
    if previous_champion is not None:
        set_alias(name, "previous", previous_champion, tracking_uri)
    set_alias(name, "champion", candidate_version, tracking_uri)


def reject_segmentation(candidate_version, name=SEGMENTATION, tracking_uri=None):
    set_alias(name, "rejected", candidate_version, tracking_uri)


def promote_if_better(name, version, tag, tracking_uri=None, higher_is_better=True):
    """Makes `version` the champion if its numeric tag `tag` beats the current
    champion's (or there is none; lower wins when `higher_is_better` is False); the old champion becomes `previous`.
    Returns True if promoted."""
    _, client = _client(tracking_uri)
    champion = alias_version(name, "champion", tracking_uri)
    new = float(client.get_model_version(name, str(version)).tags.get(tag, "nan"))
    if champion is None:
        set_alias(name, "champion", version, tracking_uri)
        return True
    old = float(client.get_model_version(name, str(champion)).tags.get(tag, "nan"))
    if (new > old) if higher_is_better else (new < old):
        promote_segmentation(version, previous_champion=champion, name=name, tracking_uri=tracking_uri)
        return True
    return False


def lineage(name, tracking_uri=None):
    """Plain-dict history of a registered model: versions, aliases, tags."""
    _, client = _client(tracking_uri)
    by_version = {}
    for alias, version in client.get_registered_model(name).aliases.items():
        by_version.setdefault(int(version), []).append(alias)
    out = []
    for v in sorted(client.search_model_versions(f"name='{name}'"), key=lambda v: int(v.version)):
        out.append({"version": int(v.version), "aliases": sorted(by_version.get(int(v.version), [])),
                    "tags": dict(v.tags), "description": v.description, "run_id": v.run_id})
    return out
