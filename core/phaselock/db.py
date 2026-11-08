"""SQLite library index.

One row per track holds the searchable/sortable facts (BPM, key, loudness).
Heavy per-sample data (beatgrids, energy curves, waveform peaks) lives in a
JSON sidecar on disk (see sidecar.py), keyed by content fingerprint.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS tracks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT UNIQUE NOT NULL,
    file_size INTEGER NOT NULL,
    mtime REAL NOT NULL,
    duration_s REAL,
    samplerate INTEGER,
    channels INTEGER,
    bpm REAL,
    bpm_source TEXT,
    beat_offset_s REAL,
    downbeat_offset_s REAL,
    key_root TEXT,
    key_mode TEXT,
    key_name TEXT,
    camelot TEXT,
    loudness_lufs REAL,
    energy_mean REAL,
    key_override INTEGER NOT NULL DEFAULT 0,
    analysis_status TEXT NOT NULL DEFAULT 'pending',
    error TEXT,
    analyzer_version TEXT,
    analyzed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_tracks_bpm ON tracks(bpm);
CREATE INDEX IF NOT EXISTS idx_tracks_camelot ON tracks(camelot);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open (and migrate) the library database."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn
