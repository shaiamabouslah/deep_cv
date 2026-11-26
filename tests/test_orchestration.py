import json
import os
import sys

import pytest

pytest.importorskip("prefect")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "orchestration"))

import flows  # noqa: E402
from prefect.testing.utilities import prefect_test_harness  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def harness():
    with prefect_test_harness():
        yield


@pytest.fixture()
def workdir(tmp_path, monkeypatch):
    (tmp_path / "reports").mkdir()
    monkeypatch.setattr(flows, "ROOT", str(tmp_path))
    monkeypatch.setattr(flows, "LOG_PATH", str(tmp_path / "reports" / "orchestration.log"))
    return tmp_path


def _monitoring_report(retrain, reasons=()):
    return {"data_drift": {"status": "alert" if retrain else "ok"}, "forecast_performance": {"status": "ok"},
            "retrain_needed": retrain, "reasons": list(reasons)}


def _fake_commands(monkeypatch, workdir, report, calls, promotion=None):
    def run_command(args, timeout):
        calls.append(args[1])
        if args[1] == "src.monitoring":
            (workdir / "reports" / "monitoring.json").write_text(json.dumps(report))
        if args[1] == "src.promote":
            (workdir / "reports" / "promotion.json").write_text(json.dumps(promotion))
        return ""

    monkeypatch.setattr(flows, "run_command", run_command)
    (workdir / "data" / "forecast").mkdir(parents=True)
    rows = "site,week,water_pct\n" + "\n".join(f"A,2025-01-{i % 28 + 1:02d},{i}" for i in range(600))
    (workdir / "data" / "forecast" / "weekly.csv").write_text(rows)


def test_a_transient_failure_is_retried_until_it_succeeds(workdir):
    @flows.flow
    def only_the_flaky_step():
        return flows.flaky_step(str(workdir / "attempts"), failures_before_success=2)

    assert only_the_flaky_step() == 3


def test_a_permanent_failure_fails_the_flow_and_leaves_a_trace(workdir):
    with pytest.raises(flows.CommandFailed):
        flows.resilience_demo(state_file=str(workdir / "attempts"))
    log = (workdir / "reports" / "orchestration.log").read_text()
    assert "FAILED  always failing step" in log and "OK      flaky step" in log


def test_a_failing_command_raises_with_the_end_of_its_stderr(workdir, monkeypatch):
    monkeypatch.setattr(flows, "PROJECT_PYTHON", sys.executable)

    @flows.flow
    def probe():
        return flows.run_command(["-c", "import sys; sys.stderr.write('boom from the child'); sys.exit(3)"], timeout=30)

    with pytest.raises(flows.CommandFailed, match="exited with 3: boom from the child"):
        probe()


def test_a_successful_command_returns_its_output(workdir, monkeypatch):
    monkeypatch.setattr(flows, "PROJECT_PYTHON", sys.executable)

    @flows.flow
    def probe():
        return flows.run_command(["-c", "print('hello')"], timeout=30)

    assert probe().strip() == "hello"


def test_no_retraining_when_monitoring_is_clean(workdir, monkeypatch):
    calls = []
    _fake_commands(monkeypatch, workdir, _monitoring_report(False), calls)
    outcome = flows.continuous_training()
    assert outcome == {"retrained": False, "reasons": []} and calls == ["src.monitoring"]


def test_drift_triggers_the_training_pipeline(workdir, monkeypatch):
    calls = []
    _fake_commands(monkeypatch, workdir, _monitoring_report(True, ["input data drift: brightness"]), calls)
    outcome = flows.continuous_training()
    assert outcome["retrained"] is True and outcome["reasons"] == ["input data drift: brightness"]
    assert calls == ["src.monitoring", "src.train_forecaster", "src.calibration", "src.evaluate", "src.world_benchmark",
                     "dvc", "dvc"]


def test_a_candidate_rejected_by_the_gate_is_a_normal_outcome(workdir, monkeypatch):
    calls = []
    promotion = {"promoted": False, "would_promote": False, "rules": {"no regression in Tunisia": True,
                                                                      "gain on the world benchmark": False}}
    _fake_commands(monkeypatch, workdir, _monitoring_report(True, ["forecaster no longer beats the no-change baseline"]),
                   calls, promotion)
    outcome = flows.continuous_training(retrain_segmentation=True)
    assert outcome["segmentation"]["promoted"] is False
    assert calls[-2:] == ["src.finetune_world", "src.promote"]


def test_invalid_data_stops_the_pipeline_before_any_training(workdir, monkeypatch):
    calls = []
    _fake_commands(monkeypatch, workdir, _monitoring_report(True, ["x"]), calls)
    (workdir / "data" / "forecast" / "weekly.csv").write_text("site,week,water_pct\nA,2025-01-01,5\n")
    with pytest.raises(flows.CommandFailed, match="expected at least"):
        flows.training_pipeline()
    assert "src.train_forecaster" not in calls
