"""Local tempo + join alignment for real-world (drifting, humanized) tracks.

The global beatgrid assumes constant tempo; real recordings drift. These
helpers estimate the LOCAL beat period around a point (onset-envelope
autocorrelation with parabolic interpolation) and the beat-phase offset
between an outgoing tail and an incoming head (onset-envelope
cross-correlation), which the renderer uses to phase-lock the join where it
actually matters instead of trusting the global grid.
"""

from __future__ import annotations

import numpy as np

HOP = 512


def _onset_env(y: np.ndarray, sr: int, fmax: int | None = None) -> np.ndarray:
    import librosa

    return librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP, fmax=fmax)


def _sharpen(oe: np.ndarray) -> np.ndarray:
    """Keep only strong onsets: quasi-periodic envelopes otherwise correlate
    at every beat phase and sub-beat alignment becomes a coin flip."""
    x = np.clip(oe - np.median(oe), 0.0, None)
    m = x.max()
    return x / m if m > 0 else x


def local_beat_period(y: np.ndarray, sr: int, center_s: float,
                      bpm_hint: float, window_s: float = 12.0) -> float | None:
    """Beat period (seconds) near center_s, or None if too aperiodic."""
    a = max(0, int((center_s - window_s) * sr))
    b = min(y.size, int((center_s + window_s) * sr))
    if b - a < int(window_s * sr):
        return None
    oe = _onset_env(y[a:b], sr)
    return _acf_period(oe, sr, 60.0 / bpm_hint * 0.94, 60.0 / bpm_hint * 1.06)


def _acf_period(oe: np.ndarray, sr: int, lo_s: float, hi_s: float) -> float | None:
    x = oe - oe.mean()
    n = x.size
    if n < 8:
        return None
    size = 1 << int(np.ceil(np.log2(2 * n)))
    spec = np.fft.rfft(x, size)
    acf = np.fft.irfft(spec * np.conj(spec), size)[:n]
    if acf[0] <= 0:
        return None
    acf /= acf[0]
    lo = max(1, int(lo_s * sr / HOP))
    hi = min(n - 2, int(hi_s * sr / HOP))
    if hi <= lo:
        return None
    lag = lo + int(np.argmax(acf[lo : hi + 1]))
    if acf[lag] < 0.05:
        return None
    d = acf[lag - 1] - 2 * acf[lag] + acf[lag + 1]
    shift = np.clip(0.5 * (acf[lag - 1] - acf[lag + 1]) / d, -0.5, 0.5) \
        if abs(d) > 1e-12 else 0.0
    return (lag + float(shift)) * HOP / sr


def join_offset(out_tail: np.ndarray, in_head: np.ndarray, sr: int,
                period_hint: float, max_beats: float = 0.45
                ) -> tuple[float, float]:
    """Beat-phase correction δ (seconds) for the incoming deck.

    Kick-band (≤300 Hz) spike-sharpened onset envelopes, cross-correlated
    over ±max_beats·period with parabolic sub-frame refinement. Returns
    (δ, confidence) — low confidence means δ should be ignored. Positive δ
    delays the incoming deck.
    """
    # kick band: the kick marks the beat most reliably in EDM
    oe_a = _sharpen(_onset_env(out_tail, sr, fmax=300))
    oe_b = _sharpen(_onset_env(in_head, sr, fmax=300))
    if oe_a.size < 16 or oe_b.size < 16:
        return 0.0, 0.0
    n = min(oe_a.size, oe_b.size)
    max_lag = int(max_beats * period_hint * sr / HOP)
    if n <= max_lag + 16:
        return 0.0, 0.0
    m = n - max_lag
    corrs = np.empty(2 * max_lag + 1)
    for i, lag in enumerate(range(-max_lag, max_lag + 1)):
        if lag >= 0:
            corrs[i] = np.dot(oe_a[lag : lag + m], oe_b[:m])
        else:
            corrs[i] = np.dot(oe_a[:m], oe_b[-lag : -lag + m])
    best = int(np.argmax(corrs))
    # parabolic refinement to sub-frame precision
    delta_frames = float(best - max_lag)
    if 0 < best < len(corrs) - 1:
        d = corrs[best - 1] - 2 * corrs[best] + corrs[best + 1]
        if abs(d) > 1e-12:
            delta_frames += float(np.clip(
                0.5 * (corrs[best - 1] - corrs[best + 1]) / d, -1.0, 1.0))
    # confidence: peak sharpness relative to the mean over ±1 beat
    span = max(3, int(period_hint * sr / HOP))
    lo = max(0, best - span)
    hi = min(len(corrs), best + span + 1)
    neighborhood = np.delete(corrs[lo:hi], best - lo)
    conf = float((corrs[best] - neighborhood.mean())
                 / (corrs[best] + 1e-12)) if corrs[best] > 0 else 0.0
    return delta_frames * HOP / sr, float(np.clip(conf, 0.0, 1.0))
