"""Tempo, beatgrid and downbeat estimation.

Primary backend is madmom's RNN+DBN tracker when importable; librosa's
dynamic-programming tracker is the built-in fallback. Detected beats carry
frame-grid quantization (~±1% at the 512 hop), so BPM is re-derived by
regressing beat times against integer beat indices — the snapping noise
averages out. The result is cross-checked against the onset-envelope
autocorrelation period; a mismatch (e.g. a 4/3 metrical-level lock) triggers
a rebuild of the grid as a uniform click-train fit, which is exact for
constant-tempo electronic music. The downbeat phase is the beat position
(mod 4) with the strongest harmonic-band onset — in EDM the chords mark "1".
"""

from __future__ import annotations

import numpy as np

HOP = 512
# Fold half/double-time estimates into a plausible electronic-music range.
BPM_FOLD = (90.0, 180.0)
# Tracked vs ACF period disagreement that triggers a grid rebuild.
LEVEL_TOLERANCE = 0.03
# Below this normalized ACF peak the envelope is too aperiodic to trust.
ACF_MIN_PEAK = 0.05


def _fold_bpm(bpm: float) -> float:
    while bpm < BPM_FOLD[0]:
        bpm *= 2
    while bpm > BPM_FOLD[1]:
        bpm /= 2
    return bpm


def _onset_env(y: np.ndarray, sr: int, hop: int = HOP) -> np.ndarray:
    import librosa

    return librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)


def _madmom_beats(path) -> np.ndarray | None:
    try:
        from madmom.features.beats import (
            DBNBeatTrackingProcessor,
            RNNBeatProcessor,
        )
    except Exception:
        return None
    try:
        act = RNNBeatProcessor()(str(path))
        beats = DBNBeatTrackingProcessor(fps=100)(act)
        return np.asarray(beats, dtype=float)
    except Exception:
        return None


def _librosa_beats(oe: np.ndarray, sr: int, hop: int = HOP) -> np.ndarray:
    import librosa

    _, beat_frames = librosa.beat.beat_track(
        onset_envelope=oe, sr=sr, hop_length=hop, trim=False
    )
    return np.asarray(
        librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop), dtype=float
    )


def _refine(beats: np.ndarray) -> tuple[float, float]:
    """Least-squares (offset, period) over integer beat indices.

    Indices are assigned by rounding each interval to a whole number of beats,
    so occasional dropped or spurious beats don't skew the fit. The regression
    over hundreds of beats recovers sub-frame period precision.
    """
    p0 = float(np.median(np.diff(beats)))
    idx = [0]
    for d in np.diff(beats):
        idx.append(idx[-1] + max(1, int(round(d / p0))))
    slope, intercept = np.polyfit(np.asarray(idx, dtype=float), beats, 1)
    return float(intercept), float(slope)


def _acf_period(oe: np.ndarray, sr: int, hop: int = HOP) -> float | None:
    """Pulse period from onset-envelope autocorrelation, parabolic-interpolated.

    The search window (0.33–0.67 s) matches BPM_FOLD and excludes 8th-note
    lags, so for 4/4 material it locks onto the quarter-note pulse.
    """
    x = oe - oe.mean()
    n = x.size
    size = 1 << int(np.ceil(np.log2(2 * n)))
    spec = np.fft.rfft(x, size)
    acf = np.fft.irfft(spec * np.conj(spec), size)[:n]
    if acf[0] <= 0:
        return None
    acf = acf / acf[0]
    lo = int(0.33 * sr / hop)
    hi = int(0.67 * sr / hop)
    if hi + 1 >= n:
        return None
    lag = lo + int(np.argmax(acf[lo : hi + 1]))
    peak = acf[lag]
    if peak < ACF_MIN_PEAK or lag == 0 or lag + 1 >= n:
        return None
    denom = acf[lag - 1] - 2 * acf[lag] + acf[lag + 1]
    shift = 0.5 * (acf[lag - 1] - acf[lag + 1]) / denom if abs(denom) > 1e-12 else 0.0
    shift = float(np.clip(shift, -0.5, 0.5))
    return (lag + shift) * hop / sr


def _uniform_grid(oe: np.ndarray, sr: int, hop: int, period: float,
                  duration_s: float) -> np.ndarray:
    """Best-phase uniform click grid at `period` over the track."""
    n_k = int(duration_s / period)
    offsets = np.linspace(0, period, 200, endpoint=False)
    ks = np.arange(n_k)
    times = offsets[None, :] + ks[:, None] * period  # (n_k, n_off)
    frame_times = np.arange(oe.size) * hop / sr
    scores = np.interp(times, frame_times, oe).mean(axis=0)
    off = float(offsets[int(np.argmax(scores))])
    return off + ks * period


def _downbeat_phase(beats: np.ndarray, y: np.ndarray, sr: int, hop: int = HOP) -> int | None:
    """Beat index mod 4 that is most likely the downbeat ('1')."""
    import librosa

    if len(beats) < 8:
        return None
    # Harmonic band only: the kick hits every beat, chords/melody mark the bar.
    harm = librosa.onset.onset_strength(
        y=y, sr=sr, hop_length=hop, fmin=250, fmax=4000
    )
    frames = np.clip(np.round(beats * sr / hop).astype(int), 0, len(harm) - 1)
    strengths = harm[frames]
    scores = [strengths[p::4].mean() if len(strengths[p::4]) else 0.0 for p in range(4)]
    return int(np.argmax(scores))


def analyze(y: np.ndarray, sr: int, path=None) -> dict:
    """Return tempo facts + beatgrid for one track.

    Keys: bpm, bpm_source, beat_offset_s, downbeat_offset_s, beats (list[float]),
    downbeat_phase (int|None), grid ('tracked'|'rebuilt').
    """
    oe = _onset_env(y, sr)
    beats = None
    source = "librosa"
    if path is not None:
        beats = _madmom_beats(path)
        if beats is not None and len(beats) >= 8:
            source = "madmom"
        else:
            beats = None
    if beats is None:
        beats = _librosa_beats(oe, sr)

    if len(beats) < 4:
        return {
            "bpm": None, "bpm_source": source, "beat_offset_s": None,
            "downbeat_offset_s": None, "beats": [], "downbeat_phase": None,
            "grid": None,
        }

    grid = "tracked"
    _, p_tracked = _refine(beats)
    p_acf = _acf_period(oe, sr)
    if p_acf and p_tracked > 0 and abs(p_tracked / p_acf - 1) > LEVEL_TOLERANCE:
        rebuilt = _uniform_grid(oe, sr, HOP, p_acf, y.size / sr)
        if len(rebuilt) >= 8:
            beats = rebuilt
            grid = "rebuilt"

    offset, period = _refine(beats)
    bpm = _fold_bpm(60.0 / period)
    phase = _downbeat_phase(beats, y, sr)
    downbeat_offset = float(beats[phase]) if phase is not None else None

    return {
        "bpm": round(bpm, 2),
        "bpm_source": source,
        "beat_offset_s": float(beats[0]),
        "downbeat_offset_s": downbeat_offset,
        "beats": [round(float(b), 4) for b in beats],
        "downbeat_phase": phase,
        "grid": grid,
    }
