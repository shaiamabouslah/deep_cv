from src.alerts import (
    check_and_record_alert,
    classify_alert_level,
    get_active_alerts,
    get_highest_alert_level,
    get_history,
    get_previous_ndvi,
    record_reading,
)


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
    alert = check_and_record_alert(db_path, "Sidi Salem", 10.0)
    assert alert is None


def test_alert_triggered_on_significant_drop(tmp_path):
    db_path = str(tmp_path / "history.db")
    record_reading(db_path, "Sidi Salem", 80.0)
    record_reading(db_path, "Sidi Salem", 82.0)
    alert = check_and_record_alert(db_path, "Sidi Salem", 50.0)
    assert alert is not None
    assert alert["site"] == "Sidi Salem"
    assert alert["level"] == "critique"
    assert get_active_alerts(db_path)[0]["site"] == "Sidi Salem"


def test_no_alert_when_drop_below_lowest_tier(tmp_path):
    db_path = str(tmp_path / "history.db")
    record_reading(db_path, "Sidi Salem", 80.0)
    record_reading(db_path, "Sidi Salem", 80.0)
    alert = check_and_record_alert(db_path, "Sidi Salem", 75.0)
    assert alert is None


def test_classify_alert_level_tiers():
    assert classify_alert_level(5) is None
    assert classify_alert_level(12) == "vigilance"
    assert classify_alert_level(25) == "alerte"
    assert classify_alert_level(40) == "critique"


def test_record_reading_with_ndvi_and_source(tmp_path):
    db_path = str(tmp_path / "history.db")
    record_reading(db_path, "Sidi Salem", 80.0, ndvi=0.3, source="live")
    record_reading(db_path, "Sidi Salem", 60.0, ndvi=0.1, source="backfill")
    history = get_history(db_path, "Sidi Salem")
    assert history[0]["ndvi"] == 0.3
    assert history[1]["source"] == "backfill"


def test_get_previous_ndvi_returns_latest_non_null(tmp_path):
    db_path = str(tmp_path / "history.db")
    assert get_previous_ndvi(db_path, "Sidi Salem") is None
    record_reading(db_path, "Sidi Salem", 80.0, ndvi=0.3)
    record_reading(db_path, "Sidi Salem", 70.0, ndvi=0.2)
    assert get_previous_ndvi(db_path, "Sidi Salem") == 0.2


def test_get_highest_alert_level(tmp_path):
    db_path = str(tmp_path / "history.db")
    record_reading(db_path, "Sidi Salem", 80.0)
    record_reading(db_path, "Sidi Salem", 82.0)
    assert get_highest_alert_level(db_path, "Sidi Salem") is None
    check_and_record_alert(db_path, "Sidi Salem", 50.0)
    assert get_highest_alert_level(db_path, "Sidi Salem") == "critique"


def test_record_reading_with_explicit_timestamp(tmp_path):
    db_path = str(tmp_path / "history.db")
    record_reading(db_path, "Sidi Salem", 80.0, timestamp="2026-01-15")
    history = get_history(db_path, "Sidi Salem")
    assert history[0]["timestamp"] == "2026-01-15"
