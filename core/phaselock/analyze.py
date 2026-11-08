"""Track analysis orchestration: file → DB row + JSON sidecar."""

from __future__ import annotations

import datetime as _dt
import sqlite3
from pathlib import Path

from . import __version__, sidecar
from .analysis import energy as energy_mod
from .analysis import key as key_mod
from .analysis import tempo as tempo_mod
from .analysis import waveform as waveform_mod
from . import decode


def analyze_file(conn: sqlite3.Connection, cache_dir: str | Path, path: str | Path,
                 force: bool = False) -> dict:
    """Analyze one file, upsert its row, save its sidecar. Returns the row.

    Unchanged files (same size+mtime, status ok) are skipped unless force.
    A pre-existing human key override survives re-analysis.
    """
    p = Path(path)
    size, mtime = p.stat().st_size, p.stat().st_mtime
    row = conn.execute(
        "SELECT * FROM tracks WHERE path = ?", (str(p.resolve()),)
    ).fetchone()

    if (not force and row is not None and row["analysis_status"] == "ok"
            and row["file_size"] == size and abs(row["mtime"] - mtime) < 1e-6):
        return dict(row)

    fp = sidecar.fingerprint(p, size, mtime)
    result = {
        "path": str(p.resolve()), "file_size": size, "mtime": mtime,
        "analysis_status": "ok", "error": None,
        "analyzer_version": __version__,
        "analyzed_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "key_override": row["key_override"] if row is not None else 0,
    }

    try:
        duration_s, sr, ch = decode.probe(p)
        y = decode.load_mono(p)

        tempo = tempo_mod.analyze(y, decode.ANALYSIS_SR, path=p)
        chroma = key_mod.chroma(y, decode.ANALYSIS_SR)
        key = key_mod.analyze(chroma)
        if result["key_override"]:
            key = _overridden_key(row)
        ener = energy_mod.analyze(y, decode.ANALYSIS_SR)
        wave = waveform_mod.analyze(y)

        result.update({
            "duration_s": round(duration_s, 3), "samplerate": sr, "channels": ch,
            "bpm": tempo["bpm"], "bpm_source": tempo["bpm_source"],
            "beat_offset_s": tempo["beat_offset_s"],
            "downbeat_offset_s": tempo["downbeat_offset_s"],
            "key_root": key["key_root"], "key_mode": key["key_mode"],
            "key_name": key["key_name"], "camelot": key["camelot"],
            "loudness_lufs": ener["loudness_lufs"],
            "energy_mean": (
                round(sum(ener["energy_curve_db"]) / len(ener["energy_curve_db"]), 2)
                if ener["energy_curve_db"] else None
            ),
        })

        sidecar.save(cache_dir, fp, {
            "version": __version__,
            "fingerprint": fp,
            "beats": tempo["beats"],
            "downbeat_phase": tempo["downbeat_phase"],
            "grid": tempo["grid"],
            "key_chroma": [round(float(v), 5) for v in chroma],
            "key_candidates": key["candidates"],
            "energy_curve_db": ener["energy_curve_db"],
            "novelty": ener["novelty"],
            "waveform": wave,
        })
    except Exception as e:  # noqa: BLE001 - record the failure, keep scanning
        result.update({"analysis_status": "error", "error": f"{type(e).__name__}: {e}"})

    _upsert(conn, result)
    return result


def _overridden_key(row) -> dict:
    name = row["key_name"] or ""
    parts = name.split()
    camelot = row["camelot"]
    return {
        "key_root": parts[0] if parts else None,
        "key_mode": parts[1] if len(parts) > 1 else None,
        "key_name": name, "camelot": camelot,
    }


def _upsert(conn: sqlite3.Connection, r: dict) -> None:
    cols = [c for c in r.keys()]
    # Keep schema columns only.
    valid = {row[1] for row in conn.execute("PRAGMA table_info(tracks)")}
    cols = [c for c in cols if c in valid]
    placeholders = ", ".join("?" for _ in cols)
    updates = ", ".join(f"{c} = excluded.{c}" for c in cols if c != "path")
    conn.execute(
        f"INSERT INTO tracks ({', '.join(cols)}) VALUES ({placeholders}) "
        f"ON CONFLICT(path) DO UPDATE SET {updates}",
        [r[c] for c in cols],
    )
    conn.commit()


def set_key_override(conn: sqlite3.Connection, path: str | Path, key_name: str) -> bool:
    """Human key correction; wins over every future re-analysis."""
    camelot = key_mod.camelot_of(key_name)
    if camelot is None:
        return False
    parts = key_name.split()
    conn.execute(
        "UPDATE tracks SET key_override = 1, key_name = ?, key_root = ?, "
        "key_mode = ?, camelot = ? WHERE path = ?",
        (key_name, parts[0], parts[1], camelot, str(Path(path).resolve())),
    )
    conn.commit()
    return True
