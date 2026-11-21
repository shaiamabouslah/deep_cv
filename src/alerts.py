"""Historical storage of water-area (and, when available, NDVI) readings
per monitored site, and the tiered alerting rule: flag a site when its
latest reading drops significantly below its own historical average (e.g.
a dam emptying), at one of three severity levels — the same idea as a
weather vigilance system (vigilance / alerte / critique) rather than a
single binary on/off alert."""
import os
import sqlite3
from datetime import datetime, timezone

# (minimum drop %, level), checked from the most severe down.
ALERT_TIERS = [
    (35, "critique"),
    (20, "alerte"),
    (10, "vigilance"),
]


def classify_alert_level(drop_pct, tiers=ALERT_TIERS):
    for threshold, level in tiers:
        if drop_pct >= threshold:
            return level
    return None


def _connect(db_path):
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            site TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            water_area_pct REAL NOT NULL,
            ndvi REAL,
            source TEXT DEFAULT 'live'
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            site TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            water_area_pct REAL NOT NULL,
            baseline_pct REAL NOT NULL,
            drop_pct REAL NOT NULL,
            level TEXT
        )
    """)
    # Migration for databases created before ndvi/source/level existed.
    existing_reading_cols = {row[1] for row in conn.execute("PRAGMA table_info(readings)")}
    if "ndvi" not in existing_reading_cols:
        conn.execute("ALTER TABLE readings ADD COLUMN ndvi REAL")
    if "source" not in existing_reading_cols:
        conn.execute("ALTER TABLE readings ADD COLUMN source TEXT DEFAULT 'live'")
    existing_alert_cols = {row[1] for row in conn.execute("PRAGMA table_info(alerts)")}
    if "level" not in existing_alert_cols:
        conn.execute("ALTER TABLE alerts ADD COLUMN level TEXT")
    conn.commit()
    return conn


def record_reading(db_path, site, water_area_pct, ndvi=None, source="live", timestamp=None):
    conn = _connect(db_path)
    timestamp = timestamp or datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO readings (site, timestamp, water_area_pct, ndvi, source) VALUES (?, ?, ?, ?, ?)",
        (site, timestamp, water_area_pct, ndvi, source),
    )
    conn.commit()
    conn.close()
    return timestamp


def get_history(db_path, site):
    conn = _connect(db_path)
    rows = conn.execute(
        "SELECT timestamp, water_area_pct, ndvi, source FROM readings WHERE site = ? "
        "ORDER BY timestamp, id",
        (site,),
    ).fetchall()
    conn.close()
    return [
        {"timestamp": t, "water_area_pct": w, "ndvi": n, "source": s}
        for t, w, n, s in rows
    ]


def check_and_record_alert(db_path, site, water_area_pct, tiers=ALERT_TIERS):
    """Compares the new reading to the site's prior average and classifies
    the drop into a severity tier (vigilance/alerte/critique). Returns None
    if there isn't enough history yet, or the drop doesn't reach the
    lowest tier."""
    history = get_history(db_path, site)
    if len(history) < 2:
        return None

    baseline = sum(h["water_area_pct"] for h in history) / len(history)
    if baseline <= 0:
        return None

    drop_pct = (baseline - water_area_pct) / baseline * 100
    level = classify_alert_level(drop_pct, tiers)
    if level is None:
        return None

    conn = _connect(db_path)
    timestamp = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO alerts (site, timestamp, water_area_pct, baseline_pct, drop_pct, level) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (site, timestamp, water_area_pct, baseline, drop_pct, level),
    )
    conn.commit()
    conn.close()
    return {
        "site": site,
        "timestamp": timestamp,
        "water_area_pct": water_area_pct,
        "baseline_pct": baseline,
        "drop_pct": drop_pct,
        "level": level,
    }


def get_latest_reading(db_path, site):
    conn = _connect(db_path)
    row = conn.execute(
        "SELECT timestamp, water_area_pct, ndvi FROM readings WHERE site = ? "
        "ORDER BY timestamp DESC, id DESC LIMIT 1",
        (site,),
    ).fetchone()
    conn.close()
    if row is None:
        return None
    return {"timestamp": row[0], "water_area_pct": row[1], "ndvi": row[2]}


def get_previous_ndvi(db_path, site):
    """Most recent NDVI reading for this site, excluding the current one
    being written — used by the water-balance diagnostic."""
    conn = _connect(db_path)
    rows = conn.execute(
        "SELECT ndvi FROM readings WHERE site = ? AND ndvi IS NOT NULL "
        "ORDER BY timestamp DESC, id DESC LIMIT 1",
        (site,),
    ).fetchone()
    conn.close()
    return rows[0] if rows else None


def has_active_alert(db_path, site):
    conn = _connect(db_path)
    row = conn.execute(
        "SELECT 1 FROM alerts WHERE site = ? ORDER BY timestamp DESC, id DESC LIMIT 1",
        (site,),
    ).fetchone()
    conn.close()
    return row is not None


def get_highest_alert_level(db_path, site):
    """Most severe level among this site's currently active alerts (there's
    no 'resolve' step yet, so this is simply the most recent alert's level)."""
    conn = _connect(db_path)
    row = conn.execute(
        "SELECT level FROM alerts WHERE site = ? ORDER BY timestamp DESC, id DESC LIMIT 1",
        (site,),
    ).fetchone()
    conn.close()
    return row[0] if row else None


def get_active_alerts(db_path, limit=50):
    conn = _connect(db_path)
    rows = conn.execute(
        "SELECT site, timestamp, water_area_pct, baseline_pct, drop_pct, level "
        "FROM alerts ORDER BY timestamp DESC, id DESC LIMIT ?",
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
            "level": level,
        }
        for site, ts, w, b, d, level in rows
    ]
