"""Demucs stem separation with an on-disk cache.

stems_cache/<fingerprint>/{drums,bass,vocals,other}.flac + manifest.json.
A manifest is stale when the source file's size/mtime changed — stale caches
are reported, never silently reused.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import numpy as np

MODEL = "htdemucs"
STEM_NAMES = ("drums", "bass", "vocals", "other")


def stems_dir(cache_root: str | Path, fp: str) -> Path:
    return Path(cache_root) / fp


def manifest_path(cache_root: str | Path, fp: str) -> Path:
    return stems_dir(cache_root, fp) / "manifest.json"


def is_stale(cache_root: str | Path, fp: str, size: int, mtime: float) -> bool:
    """True when no manifest, missing stems, or source changed."""
    import json

    d = stems_dir(cache_root, fp)
    mp = manifest_path(cache_root, fp)
    if not mp.exists():
        return True
    try:
        m = json.loads(mp.read_text())
    except (json.JSONDecodeError, OSError):
        return True
    if m.get("source_size") != size or abs(m.get("source_mtime", 0) - mtime) > 1e-6:
        return True
    return not all((d / f"{s}.flac").exists() for s in m.get("stems", ()))


def load_stems(cache_root: str | Path, fp: str,
               names: tuple[str, ...] = STEM_NAMES) -> dict[str, np.ndarray] | None:
    """Load cached stems (float32, native sr) or None if missing/stale."""
    import soundfile as sf

    d = stems_dir(cache_root, fp)
    out = {}
    for name in names:
        p = d / f"{name}.flac"
        if not p.exists():
            return None
        y, sr = sf.read(str(p), always_2d=True)
        out[name] = y.T  # (channels, n)
    out["sr"] = sr
    return out


def separate(path: str | Path, cache_root: str | Path, fp: str,
             size: int, mtime: float, device: str | None = None) -> Path:
    """Separate one file into the cache; returns the stems dir.

    Raises RuntimeError if demucs is unavailable or separation fails.
    """
    import json

    import soundfile as sf

    try:
        import torch
        from demucs.api import Separator
    except ImportError as e:
        raise RuntimeError(
            "demucs is not installed — `pip install demucs`"
        ) from e

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    sep = Separator(model=MODEL, device=device, progress=False)
    # returns (original_wave, {stem_name: tensor})
    _, stems_out = sep.separate_audio_file(Path(path))
    if not isinstance(stems_out, dict) or not stems_out:
        raise RuntimeError(f"demucs failed on {path}: {stems_out}")
    sr = int(sep.samplerate)

    d = stems_dir(cache_root, fp)
    d.mkdir(parents=True, exist_ok=True)
    for name, sig in stems_out.items():
        arr = sig.detach().cpu().numpy() if hasattr(sig, "detach") else np.asarray(sig)
        sf.write(str(d / f"{name}.flac"), arr.T, sr)
    names = list(stems_out.keys())

    manifest = {
        "version": 1,
        "model": MODEL,
        "device": str(device),
        "source": str(Path(path).resolve()),
        "source_size": size,
        "source_mtime": mtime,
        "sr": sr,
        "stems": names,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    manifest_path(cache_root, fp).write_text(json.dumps(manifest, indent=2))
    return d


def fingerprint_for(conn, path: str | Path, cache_root_dummy=None) -> str | None:
    """DB-aware fingerprint lookup for a track path."""
    from . import sidecar

    row = conn.execute(
        "SELECT file_size, mtime FROM tracks WHERE path = ?", (str(Path(path).resolve()),)
    ).fetchone()
    if row is None:
        return None
    return sidecar.fingerprint(path, row["file_size"], row["mtime"])
