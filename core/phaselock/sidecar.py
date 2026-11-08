"""Sidecar cache: per-track analysis curves and grids as JSON on disk.

Keyed by a cheap content fingerprint (path|size|mtime), so re-analysis of an
unchanged file reuses the same sidecar and a changed file gets a fresh one.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

# Fields every sidecar carries; curves are decimated to keep files small.
ENERGY_HOP_S = 0.5
WAVEFORM_BINS = 2000


def fingerprint(path: str | Path, size: int, mtime: float) -> str:
    raw = f"{Path(path).resolve()}|{size}|{mtime}".encode("utf-8", "surrogateescape")
    return hashlib.sha1(raw).hexdigest()


def sidecar_path(cache_dir: str | Path, fp: str) -> Path:
    d = Path(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{fp}.json"


def load(cache_dir: str | Path, fp: str) -> dict | None:
    p = sidecar_path(cache_dir, fp)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def save(cache_dir: str | Path, fp: str, data: dict) -> None:
    p = sidecar_path(cache_dir, fp)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(p)
