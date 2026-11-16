"""Historical storage of water-area readings per monitored site, and the
alerting rule: flag a site when its latest reading drops significantly
below its own historical average (e.g. a dam emptying)."""
import os
import sqlite3
from datetime import datetime, timezone


def _connect(db_path):
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            site TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            water_area_pct REAL NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            site TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            water_area_pct REAL NOT NULL,
            baseline_pct REAL NOT NULL,
            drop_pct REAL NOT NULL
        )
    """)
    return conn


def record_reading(db_path, site, water_area_pct):
    conn = _connect(db_path)
    timestamp = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO readings (site, timestamp, water_area_pct) VALUES (?, ?, ?)",
        (site, timestamp, water_area_pct),
    )
    conn.commit()
    conn.close()
    return timestamp


def get_history(db_path, site):
    conn = _connect(db_path)
    rows = conn.execute(
        "SELECT timestamp, water_area_pct FROM readings WHERE site = ? ORDER BY timestamp",
        (site,),
    ).fetchall()
    conn.close()
    return [{"timestamp": t, "water_area_pct": w} for t, w in rows]


def check_and_record_alert(db_path, site, water_area_pct, drop_threshold_pct):
    """Compares the new reading to the site's prior average. If it is more
    than `drop_threshold_pct` percent below baseline, records and returns
    an alert. Returns None if there isn't enough history yet, or no drop."""
    history = get_history(db_path, site)
    if len(history) < 2:
        return None

    baseline = sum(h["water_area_pct"] for h in history) / len(history)
    if baseline <= 0:
        return None

    drop_pct = (baseline - water_area_pct) / baseline * 100
    if drop_pct < drop_threshold_pct:
        return None

    conn = _connect(db_path)
    timestamp = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO alerts (site, timestamp, water_area_pct, baseline_pct, drop_pct) "
        "VALUES (?, ?, ?, ?, ?)",
        (site, timestamp, water_area_pct, baseline, drop_pct),
    )
    conn.commit()
    conn.close()
    return {
        "site": site,
        "timestamp": timestamp,
        "water_area_pct": water_area_pct,
        "baseline_pct": baseline,
        "drop_pct": drop_pct,
    }


def get_latest_reading(db_path, site):
    conn = _connect(db_path)
    row = conn.execute(
        "SELECT timestamp, water_area_pct FROM readings WHERE site = ? "
        "ORDER BY timestamp DESC LIMIT 1",
        (site,),
    ).fetchone()
    conn.close()
    if row is None:
        return None
    return {"timestamp": row[0], "water_area_pct": row[1]}


def has_active_alert(db_path, site):
    conn = _connect(db_path)
    row = conn.execute(
        "SELECT 1 FROM alerts WHERE site = ? ORDER BY timestamp DESC LIMIT 1",
        (site,),
    ).fetchone()
    conn.close()
    return row is not None


def get_active_alerts(db_path, limit=50):
    conn = _connect(db_path)
    rows = conn.execute(
        "SELECT site, timestamp, water_area_pct, baseline_pct, drop_pct "
        "FROM alerts ORDER BY timestamp DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [
        {
            "site": site,
            "timestamp": ts,
            "water_area_pct": w,
            "baseline_pct": b,
            "drop_pct": d,
        }
        for site, ts, w, b, d in rows
    ]
