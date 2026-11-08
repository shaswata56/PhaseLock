"""Energy: loudness curve, integrated LUFS, onset novelty.

The momentary RMS curve (dB, 0.5 s hop) is what Phase 2 reads to shape set
energy arcs; integrated LUFS is what Phase 3 uses for gain matching between
tracks. Novelty (normalized onset strength) is stored for drop detection.
"""

from __future__ import annotations

import numpy as np

HOP_S = 0.5


def analyze(y: np.ndarray, sr: int, onset_env: np.ndarray | None = None) -> dict:
    """Return energy facts + decimated curves.

    Keys: loudness_lufs, energy_mean_db, energy_curve (list[float], dB per
    HOP_S), novelty (list[float] 0..1 per HOP_S).
    """
    frame = int(sr * HOP_S)
    n = max(1, y.size // frame)
    trimmed = y[: n * frame].reshape(n, frame)
    rms = np.sqrt((trimmed.astype(np.float64) ** 2).mean(axis=1) + 1e-12)
    db = 20.0 * np.log10(rms + 1e-12)
    db = np.clip(db, -60.0, 0.0)

    loudness_lufs = _integrated_lufs(y, sr)

    if onset_env is None:
        import librosa

        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    novelty = None
    if onset_env is not None and onset_env.size:
        hop_len = y.size / onset_env.size  # samples per onset frame
        idx = (np.arange(n) * frame / hop_len).astype(int)
        idx = np.clip(idx, 0, onset_env.size - 1)
        nov = onset_env[idx]
        mx = nov.max()
        novelty = (nov / mx).tolist() if mx > 0 else [0.0] * n

    return {
        "loudness_lufs": loudness_lufs,
        "energy_curve_db": [round(float(v), 2) for v in db],
        "novelty": [round(float(v), 4) for v in novelty] if novelty else None,
    }


def _integrated_lufs(y: np.ndarray, sr: int) -> float | None:
    """K-weighted integrated loudness via pyloudnorm; None if it errors."""
    try:
        import pyloudnorm as pyln

        meter = pyln.Meter(sr)
        v = float(meter.integrated_loudness(y.astype(np.float64)))
        return round(v, 2) if np.isfinite(v) else None
    except Exception:
        return None
