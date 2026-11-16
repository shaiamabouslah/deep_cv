"""Stores user-added monitoring sites (drawn/clicked on the map), on top of
the seed list in sites.py. Lets the platform grow beyond the initial 6 dams
without code changes."""
import os
import sqlite3
from datetime import datetime, timezone


def _connect(db_path):
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS custom_sites (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            lat REAL NOT NULL,
            lon REAL NOT NULL,
            polygon TEXT,
            created_at TEXT NOT NULL
        )
    """)
    return conn


def add_custom_site(db_path, name, lat, lon, polygon=None):
    conn = _connect(db_path)
    try:
        conn.execute(
            "INSERT INTO custom_sites (name, lat, lon, polygon, created_at) VALUES (?, ?, ?, ?, ?)",
            (name, lat, lon, polygon, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        conn.close()
        return {"id": name, "name": name, "lat": lat, "lon": lon, "name_taken": False}
    except sqlite3.IntegrityError:
        existing = conn.execute(
            "SELECT lat, lon FROM custom_sites WHERE name = ?", (name,)
        ).fetchone()
        conn.close()
        existing_lat, existing_lon = existing
        return {"id": name, "name": name, "lat": existing_lat, "lon": existing_lon, "name_taken": True}


def get_custom_sites(db_path):
    conn = _connect(db_path)
    rows = conn.execute("SELECT name, lat, lon FROM custom_sites ORDER BY created_at").fetchall()
    conn.close()
    return [{"id": name, "name": name, "lat": lat, "lon": lon, "custom": True} for name, lat, lon in rows]
