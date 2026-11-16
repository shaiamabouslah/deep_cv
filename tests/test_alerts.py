import os

from src.alerts import check_and_record_alert, get_active_alerts, get_history, record_reading


def test_record_and_get_history(tmp_path):
    db_path = str(tmp_path / "history.db")
    record_reading(db_path, "Sidi Salem", 80.0)
    record_reading(db_path, "Sidi Salem", 78.0)
    history = get_history(db_path, "Sidi Salem")
    assert len(history) == 2
    assert history[0]["water_area_pct"] == 80.0


def test_no_alert_without_enough_history(tmp_path):
    db_path = str(tmp_path / "history.db")
    record_reading(db_path, "Sidi Salem", 80.0)
    alert = check_and_record_alert(db_path, "Sidi Salem", 10.0, drop_threshold_pct=20)
    assert alert is None


def test_alert_triggered_on_significant_drop(tmp_path):
    db_path = str(tmp_path / "history.db")
    record_reading(db_path, "Sidi Salem", 80.0)
    record_reading(db_path, "Sidi Salem", 82.0)
    alert = check_and_record_alert(db_path, "Sidi Salem", 50.0, drop_threshold_pct=20)
    assert alert is not None
    assert alert["site"] == "Sidi Salem"
    assert get_active_alerts(db_path)[0]["site"] == "Sidi Salem"


def test_no_alert_when_drop_below_threshold(tmp_path):
    db_path = str(tmp_path / "history.db")
    record_reading(db_path, "Sidi Salem", 80.0)
    record_reading(db_path, "Sidi Salem", 80.0)
    alert = check_and_record_alert(db_path, "Sidi Salem", 75.0, drop_threshold_pct=20)
    assert alert is None
