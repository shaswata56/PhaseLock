"""Waveform peaks for display — min/max per bin, WAVEFORM_BINS total."""

from __future__ import annotations

import numpy as np


def analyze(y: np.ndarray, bins: int = 2000) -> dict:
    n = min(bins, max(1, y.size))
    pad = (-y.size) % n
    if pad:
        y = np.concatenate([y, np.zeros(pad, dtype=y.dtype)])
    frames = y.reshape(n, -1)
    return {
        "bins": n,
        "min": [round(float(v), 4) for v in frames.min(axis=1)],
        "max": [round(float(v), 4) for v in frames.max(axis=1)],
    }
