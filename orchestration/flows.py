"""Prefect orchestration of the project's pipelines.

Runs in its own virtualenv (.venv-orchestrator, requirements-orchestration.txt)
because Prefect pins web-framework versions the serving app does not use. Each
task launches a project command with the project's own interpreter, so the
orchestrator never imports torch or the app code.

Error handling, by layer:
  - a command that exits non-zero raises CommandFailed with the tail of its
    stderr, so the task fails visibly instead of hiding the error
  - network tasks (Earth Engine, downloads) retry with growing delays
    (RETRY_DELAYS); transient failures are absorbed, persistent ones surface
  - every task has a timeout, so a hung job cannot block the pipeline
  - on_failure / on_completion hooks append to reports/orchestration.log, so a
    failed run leaves a trace even when nobody is watching the UI
  - in the continuous-training flow a candidate model that fails the gate
    leaves the live model untouched and the flow ends COMPLETED with the
    decision recorded (a rejected model is a normal outcome, not an error)

    prefect server start                           # UI at http://127.0.0.1:4200
    python orchestration/flows.py training         # one full training run
    python orchestration/flows.py continuous       # monitoring -> retrain if needed
    python orchestration/flows.py resilience       # retries and failure hook demo
    python orchestration/deploy.py                 # scheduled deployments
"""
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

from prefect import flow, get_run_logger, task
from prefect.cache_policies import NO_CACHE

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_BIN = "Scripts" if os.name == "nt" else "bin"
PROJECT_PYTHON = os.environ.get("PROJECT_PYTHON", os.path.join(ROOT, ".venv", _BIN, "python.exe" if os.name == "nt" else "python"))
LOG_PATH = os.path.join(ROOT, "reports", "orchestration.log")
RETRY_DELAYS = [10, 60, 300]


class CommandFailed(RuntimeError):
    pass


def trace(message):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}  {message}\n")


def notify_failure(task_or_flow, run, state):
    trace(f"FAILED  {run.name}: {state.message}")


def notify_completion(task_or_flow, run, state):
    trace(f"OK      {run.name}")


def run_command(args, timeout):
    """Runs `python -m ...` in the project; returns stdout, raises with the stderr tail on failure."""
    logger = get_run_logger()
    env = {**os.environ, "MLFLOW_DISABLE_AGENT_HINT": "1", "PYTHONIOENCODING": "utf-8"}
    logger.info("running: %s", " ".join(args))
    done = subprocess.run([PROJECT_PYTHON, *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout, env=env)
    if done.returncode != 0:
        raise CommandFailed(f"{' '.join(args)} exited with {done.returncode}: {done.stderr.strip()[-600:]}")
    return done.stdout


common = dict(on_failure=[notify_failure], on_completion=[notify_completion], cache_policy=NO_CACHE)


@task(name="refresh weekly data", retries=3, retry_delay_seconds=RETRY_DELAYS, timeout_seconds=1800, **common)
def refresh_data():
    return run_command(["-m", "src.build_forecast_dataset"], timeout=1800)


@task(name="validate data", retries=0, timeout_seconds=120, **common)
def validate_data(path="data/forecast/weekly.csv", min_rows=500):
    import csv

    full = os.path.join(ROOT, path)
    if not os.path.exists(full):
        raise CommandFailed(f"{path} is missing")
    with open(full, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) < min_rows:
        raise CommandFailed(f"{path} has {len(rows)} rows, expected at least {min_rows}")
    if not {"site", "week", "water_pct"} <= set(rows[0]):
        raise CommandFailed(f"{path} lacks the expected columns")
    return len(rows)


@task(name="train forecaster", retries=1, retry_delay_seconds=30, timeout_seconds=3600, **common)
def train_forecaster():
    return run_command(["-m", "src.train_forecaster"], timeout=3600)


@task(name="calibrate fill rate", retries=1, retry_delay_seconds=15, timeout_seconds=600, **common)
def calibrate():
    return run_command(["-m", "src.calibration"], timeout=600)


@task(name="evaluate served model", retries=0, timeout_seconds=900, **common)
def evaluate_model():
    return run_command(["-m", "src.evaluate"], timeout=900)


@task(name="world benchmark", retries=0, timeout_seconds=1800, **common)
def world_benchmark():
    return run_command(["-m", "src.world_benchmark"], timeout=1800)


@task(name="version data and models (DVC)", retries=2, retry_delay_seconds=10, timeout_seconds=1800, **common)
def version_artifacts():
    """Records the new models and reports in dvc.lock and pushes them to the DVC remote."""
    run_command(["-m", "dvc", "commit", "-f"], timeout=900)
    run_command(["-m", "dvc", "push"], timeout=1800)


@task(name="check monitoring", retries=2, retry_delay_seconds=10, timeout_seconds=300, **common)
def check_monitoring():
    run_command(["-m", "src.monitoring", "check"], timeout=300)
    with open(os.path.join(ROOT, "reports", "monitoring.json"), encoding="utf-8") as f:
        return json.load(f)


@task(name="fine-tune segmentation candidate", retries=0, timeout_seconds=7200, **common)
def finetune_candidate(epochs=12, out="models/model_candidate.pt"):
    run_command(["-m", "src.finetune_world", "--epochs", str(epochs), "--out", out], timeout=7200)
    return out


@task(name="promotion gate", retries=0, timeout_seconds=1800, **common)
def promotion_gate(candidate, apply=True):
    args = ["-m", "src.promote", candidate] + (["--apply"] if apply else [])
    run_command(args, timeout=1800)
    with open(os.path.join(ROOT, "reports", "promotion.json"), encoding="utf-8") as f:
        return json.load(f)


@task(name="flaky step (demo)", retries=3, retry_delay_seconds=[1, 2, 4], **common)
def flaky_step(state_file, failures_before_success=2):
    """Fails a fixed number of times, then succeeds: shows retries in the UI."""
    logger = get_run_logger()
    count = int(open(state_file).read()) if os.path.exists(state_file) else 0
    with open(state_file, "w") as f:
        f.write(str(count + 1))
    if count < failures_before_success:
        logger.warning("attempt %d fails (simulated transient error)", count + 1)
        raise ConnectionError(f"simulated transient failure #{count + 1}")
    logger.info("attempt %d succeeds", count + 1)
    return count + 1


@task(name="always failing step (demo)", retries=1, retry_delay_seconds=1, **common)
def broken_step():
    raise CommandFailed("simulated permanent failure: the failure hook records it")


@flow(name="training-pipeline", on_failure=[notify_failure], on_completion=[notify_completion])
def training_pipeline(refresh=False):
    """Data (optionally refreshed) -> validation -> forecaster -> calibration -> evaluation -> benchmark -> DVC versioning."""
    if refresh:
        refresh_data()
    rows = validate_data()
    train_forecaster()
    calibrate()
    evaluate_model()
    world_benchmark()
    version_artifacts()
    return {"weekly_rows": rows}


@flow(name="continuous-training", on_failure=[notify_failure], on_completion=[notify_completion])
def continuous_training(force=False, retrain_segmentation=False, refresh=False):
    """Monitoring decides: when inputs drift or the forecaster stops beating
    the no-change baseline (or `force`), retrain. The segmentation candidate
    only replaces the live model if the promotion gate accepts it."""
    logger = get_run_logger()
    report = check_monitoring()
    logger.info("drift: %s | forecaster: %s", report["data_drift"]["status"], report["forecast_performance"]["status"])
    if not (report["retrain_needed"] or force):
        logger.info("no retraining needed")
        return {"retrained": False, "reasons": []}
    reasons = report["reasons"] or ["forced run"]
    logger.info("retraining because: %s", "; ".join(reasons))
    outcome = {"retrained": True, "reasons": reasons}
    training_pipeline(refresh=refresh)
    if retrain_segmentation:
        candidate = finetune_candidate()
        decision = promotion_gate(candidate, apply=True)
        outcome["segmentation"] = {"promoted": decision["promoted"], "rules": decision["rules"]}
        logger.info("segmentation candidate %s", "PROMOTED" if decision["promoted"] else "rejected by the gate")
    return outcome


@flow(name="resilience-demo", on_failure=[notify_failure], on_completion=[notify_completion])
def resilience_demo(state_file=None):
    """Two tasks: one recovers after two retries, one fails for good (the
    flow is then marked failed and the hook writes to reports/orchestration.log)."""
    state_file = state_file or os.path.join(ROOT, "reports", ".flaky_attempts")
    if os.path.exists(state_file):
        os.remove(state_file)
    attempts = flaky_step(state_file)
    get_run_logger().info("recovered after %d attempts", attempts)
    broken_step()


FLOWS = {"training": training_pipeline, "continuous": continuous_training, "resilience": resilience_demo}

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "training"
    extra = {"force": True} if name == "continuous" and "--force" in sys.argv else {}
    FLOWS[name](**extra)
