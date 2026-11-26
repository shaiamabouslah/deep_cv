"""MLflow experiment tracking for every training / evaluation script.

All runs go to one local SQLite store (`mlflow.db`, browse it with
`mlflow ui --backend-store-uri sqlite:///mlflow.db`). Each run carries the
git commit, branch and whether the working tree had uncommitted changes, so
a result can always be tied back to the code (and, through dvc.lock, the data)
that produced it. Set MLFLOW_TRACKING_URI to log somewhere else (a server,
DagsHub...).

`log_experiment` logs a finished experiment in one call (params, metrics,
per-epoch curves as step series, artifacts). Tracking never breaks a
training script: if MLflow is unavailable the failure is reported and the
script's own outputs are untouched.
"""
import json
import os
import re
import subprocess

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
ARTIFACT_DIR = os.environ.get("MLFLOW_ARTIFACT_DIR", "mlartifacts")
_INVALID = re.compile(r"[^0-9A-Za-z_\-\. /]")


def metric_name(name):
    """MLflow metric names allow letters, digits and _ - . / and space."""
    return _INVALID.sub("_", str(name))[:240]


def flatten(values, prefix=""):
    """{'a': {'b': 1.0}} -> {'a.b': 1.0}; keeps only numbers."""
    flat = {}
    for key, value in values.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten(value, name))
        elif isinstance(value, bool):
            flat[name] = float(value)
        elif isinstance(value, (int, float)) and value == value:
            flat[name] = float(value)
    return {metric_name(k): v for k, v in flat.items()}


def git_info():
    def run(*args):
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return ""

    return {
        "git_commit": run("rev-parse", "--short", "HEAD") or "unknown",
        "git_branch": run("rev-parse", "--abbrev-ref", "HEAD") or "unknown",
        "git_dirty": str(bool(run("status", "--porcelain"))).lower(),
    }


def configure(experiment, tracking_uri=None):
    import mlflow

    mlflow.set_tracking_uri(tracking_uri or TRACKING_URI)
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=os.path.join(ARTIFACT_DIR, metric_name(experiment)))
    mlflow.set_experiment(experiment)
    return mlflow


def log_experiment(experiment, run_name, params=None, metrics=None, curves=None, artifacts=None, tags=None,
                   tracking_uri=None, nested=False):
    """Logs one finished experiment. `curves` = {name: [value per step]}.
    Returns the run id, or None if tracking failed (never raises)."""
    try:
        mlflow = configure(experiment, tracking_uri)
        with mlflow.start_run(run_name=run_name, nested=nested) as run:
            mlflow.set_tags({**git_info(), **(tags or {})})
            if params:
                mlflow.log_params({metric_name(k): str(v)[:500] for k, v in params.items()})
            for name, value in flatten(metrics or {}).items():
                mlflow.log_metric(name, value)
            for name, series in (curves or {}).items():
                for step, value in enumerate(series):
                    mlflow.log_metric(metric_name(name), float(value), step=step)
            for path in artifacts or []:
                if os.path.isdir(path):
                    mlflow.log_artifacts(path)
                elif os.path.exists(path):
                    mlflow.log_artifact(path)
            return run.info.run_id
    except Exception as exc:
        print(f"[tracking] MLflow logging skipped: {str(exc)[:120]}")
        return None


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
