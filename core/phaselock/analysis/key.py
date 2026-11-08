"""Musical key detection → Camelot (Open Key) notation.

Krumhansl–Schmuckler profiles correlated against an aggregated chroma —
textbook algorithm, no training data, good enough that human overrides
(key_override in the DB) only matter for edge cases.
"""

from __future__ import annotations

import numpy as np

# KS key profiles, C-rooted, indexed by pitch class.
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

PC_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Open Key / Camelot wheel: pitch class → code, per mode.
CAMELOT_MINOR = {"A": "8A", "E": "9A", "B": "10A", "F#": "11A", "C#": "12A",
                 "G#": "1A", "D#": "2A", "A#": "3A", "F": "4A", "C": "5A",
                 "G": "6A", "D": "7A"}
CAMELOT_MAJOR = {"C": "8B", "G": "9B", "D": "10B", "A": "11B", "E": "12B",
                 "B": "1B", "F#": "2B", "C#": "3B", "G#": "4B", "D#": "5B",
                 "A#": "6B", "F": "7B"}


def _rotate(profile: np.ndarray, root: int) -> np.ndarray:
    return np.roll(profile, root)


def chroma(y: np.ndarray, sr: int) -> np.ndarray:
    """Aggregate chroma vector (12,) — mean CQT chroma over the track."""
    import librosa

    # Long tracks: analyze the loudest middle 120 s; intros/outros are sparse.
    max_samples = sr * 120
    if y.size > max_samples:
        start = (y.size - max_samples) // 2
        y = y[start : start + max_samples]
    c = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=1024)
    return c.mean(axis=1)


def analyze(chroma_vec: np.ndarray) -> dict:
    """Best key for an aggregate chroma vector, plus runner-ups.

    Keys: key_root, key_mode, key_name, camelot, confidence (corr, 0..1-ish),
    candidates (top 3).
    """
    c = np.asarray(chroma_vec, dtype=float)
    c = c / (np.linalg.norm(c) + 1e-12)

    best = None
    scored = []
    for mode, profile in (("major", _MAJOR), ("minor", _MINOR)):
        for root in range(12):
            p = _rotate(profile, root)
            p = p / (np.linalg.norm(p) + 1e-12)
            corr = float(np.dot(c, p))
            scored.append((corr, PC_NAMES[root], mode))
            if best is None or corr > best[0]:
                best = (corr, PC_NAMES[root], mode)

    scored.sort(reverse=True)
    corr, root, mode = best
    camelot = (CAMELOT_MINOR if mode == "minor" else CAMELOT_MAJOR)[root]

    return {
        "key_root": root,
        "key_mode": mode,
        "key_name": f"{root} {mode}",
        "camelot": camelot,
        "confidence": round(corr, 4),
        "candidates": [
            {"key": f"{r} {m}", "corr": round(cr, 4)} for cr, r, m in scored[:3]
        ],
    }


def camelot_of(key_name: str) -> str | None:
    """'A minor' / 'C major' → '8A' / '8B'. For human overrides."""
    parts = key_name.strip().split()
    if len(parts) != 2 or parts[1] not in ("major", "minor"):
        return None
    root, mode = parts
    root = root[0].upper() + root[1:].lower()
    flats = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}
    root = flats.get(root, root)
    table = CAMELOT_MINOR if mode == "minor" else CAMELOT_MAJOR
    return table.get(root)
