"""Mix renderer: set JSON → one continuous, phase-locked mix.

Timeline construction (all sample-exact):
  - deck k plays its [in, out] region, time-warped by WSOLA at rate mix_bpm
    for the solo part and the NEXT deck's mix_bpm for its final transition
    beats — so both decks run at one tempo during the overlap;
  - the next deck starts exactly at the previous deck's rate-change sample,
    a downbeat landing on a downbeat;
  - the transition DSP replaces the summed overlap (equal-power fades,
    Linkwitz-Riley bass swaps on the beat, swept high-pass exits);
  - per-deck LUFS gain riding, then a look-ahead limiter at −1 dBFS.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d
from scipy.signal import butter, sosfiltfilt

from .. import decode, sidecar
from .wsola import HOP, wsola

from pathlib import Path

TARGET_LUFS = -14.0
CEILING = 10.0 ** (-1.0 / 20.0)   # −1 dBFS
XOVER_HZ = 110.0
GAIN_LIMIT_DB = 12.0


def _lufs_gain(lufs: float | None) -> float:
    if lufs is None:
        return 1.0
    g = 10.0 ** ((TARGET_LUFS - lufs) / 20.0)
    return float(np.clip(g, 10 ** (-GAIN_LIMIT_DB / 20), 10 ** (GAIN_LIMIT_DB / 20)))


def _low_band(x: np.ndarray, sr: int) -> np.ndarray:
    sos = butter(4, XOVER_HZ, btype="low", fs=sr, output="sos")
    if x.ndim == 1:
        return sosfiltfilt(sos, x)
    return np.stack([sosfiltfilt(sos, ch) for ch in x])


def _eq_power(l: int) -> tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, np.pi / 2, l)
    return np.cos(theta), np.sin(theta)


def _fade(x: np.ndarray, n: int, fade_in: bool) -> None:
    n = min(n, x.shape[-1])
    r = np.linspace(0.0, 1.0, n)
    if x.ndim == 1:
        x[:n] *= r if fade_in else r[::-1]
        return
    if fade_in:
        x[:, :n] *= r
    else:
        x[:, -n:] *= r[::-1]


def _sweep_hp(x: np.ndarray, sr: int, n_chunks: int,
              f_lo: float = 100.0, f_hi: float = 8000.0) -> np.ndarray:
    """High-pass sweep across the overlap: filter per beat, 10 ms splices."""
    l = x.shape[-1]
    out = np.zeros_like(x, dtype=np.float64)
    edges = np.linspace(0, l, n_chunks + 1).astype(int)
    xf = int(sr * 0.01)
    for j, (a, b) in enumerate(zip(edges, edges[1:])):
        f = f_lo * (f_hi / f_lo) ** (j / max(1, n_chunks - 1))
        sos = butter(4, min(f, sr / 2 - 100), btype="high", fs=sr, output="sos")
        pa, pb = max(0, a - xf), min(l, b + xf)
        seg = x[..., pa:pb]
        filt = np.stack([sosfiltfilt(sos, ch) for ch in seg]) if x.ndim == 2 \
            else sosfiltfilt(sos, seg)
        # splice with a short crossfade against whatever is already there
        n = pb - pa
        env = np.ones(n)
        na = min(xf, n)
        if pa == 0:
            env[:na] = np.linspace(0, 1, na)
        else:
            env[:na] = np.linspace(0, 1, na)
        if pb == l:
            env[-na:] = np.linspace(1, 0, na) if n > na else env[-na:]
        out[..., pa:pb] = out[..., pa:pb] * (1 - env) + filt * env
    return out


def apply_transition(a: np.ndarray, b: np.ndarray, kind: str, sr: int,
                     beats: int) -> np.ndarray:
    """Mix the overlap of outgoing (a) and incoming (b) decks."""
    l = min(a.shape[-1], b.shape[-1])
    if l % 2:
        l -= 1
    a, b = a[..., :l].astype(np.float64), b[..., :l].astype(np.float64)
    ca, cb = _eq_power(l)
    if a.ndim == 2:
        ca, cb = ca[None, :], cb[None, :]
    mid = l // 2  # on a beat: even beat counts keep the midpoint on-grid

    if kind == "filter_sweep":
        swept = _sweep_hp(a, sr, beats)
        return swept * ca + b * cb

    a_low, b_low = _low_band(a, sr), _low_band(b, sr)
    a_high, b_high = a - a_low, b - b_low
    if kind == "bass_swap":
        low = np.concatenate([a_low[..., :mid], b_low[..., mid:]], axis=-1)
    else:  # blend — lows complete the swap by the midpoint
        half = mid
        la = np.concatenate([np.linspace(1, 0, half), np.zeros(l - half)])
        lb = np.concatenate([np.linspace(0, 1, half), np.ones(l - half)])
        if a.ndim == 2:
            la, lb = la[None, :], lb[None, :]
        low = a_low * la + b_low * lb
    return low + a_high * ca + b_high * cb


def _limiter(x: np.ndarray, sr: int) -> np.ndarray:
    look = max(1, int(sr * 0.005))
    rel = int(sr * 0.06)
    peak = np.abs(x).max(axis=0)
    env = maximum_filter1d(peak, look)
    g = np.minimum(1.0, CEILING / np.maximum(env, 1e-9))
    g = minimum_filter1d(g, rel)
    return x * g[None, :]


def render_set(set_dict: dict, cache_dir, out_wav: str, sr: int = 44100,
               conn=None, progress=None) -> dict:
    """Render a set (as produced by planning.planner.plan_set) to a WAV."""
    tracks = set_dict["tracks"]
    decks, starts, ovs = [], [], []

    def report(msg: str) -> None:
        if progress:
            progress(msg)

    for k, t in enumerate(tracks):
        y = decode.load_stereo(t["path"], sr)
        sc = _sidecar_for(t, conn, cache_dir)
        beats = sc["beats"] if sc else None
        T = t.get("transition_out", {}).get("beats", 0)
        in_i = _sample_of(beats, t, "in_s", sr)
        out_i = _sample_of(beats, t, "out_s", sr)
        if T and beats:
            split_i = int(round(beats[t["out_beat"] - T] * sr))
        else:
            split_i = out_i
        r_solo = t["bpm"] / t["mix_bpm"]
        regions = [(in_i, split_i, r_solo)]
        if T and k + 1 < len(tracks):
            r_ov = t["bpm"] / tracks[k + 1]["mix_bpm"]
            regions.append((split_i, out_i, r_ov))

        ref = y.mean(axis=0)
        chans, frames = wsola(ref, [y[0], y[1]], regions)
        deck = np.stack(chans)
        deck *= _lufs_gain(t.get("lufs"))
        _fade(deck, int(sr * 0.002), True)
        _fade(deck, int(sr * 0.01), False)
        decks.append(deck)

        if k == 0:
            starts.append(0)
        else:
            starts.append(ovs[-1]["start"])
        if T and k + 1 < len(tracks):
            ov_start = starts[k] + frames[0] * HOP
            ov_len = deck.shape[-1] - frames[0] * HOP
            ovs.append({
                "k": k, "start": ov_start, "len": ov_len, "beats": T,
                "type": t["transition_out"]["type"],
                "bpm": tracks[k + 1]["mix_bpm"],
            })
        report(f"deck {k + 1}/{len(tracks)}: {Path(t['path']).name}")

    total = starts[-1] + decks[-1].shape[-1]
    master = np.zeros((2, total))
    for start, deck in zip(starts, decks):
        master[:, start : start + deck.shape[-1]] += deck

    for ov in ovs:
        a_deck, b_deck = decks[ov["k"]], decks[ov["k"] + 1]
        a0 = ov["start"] - starts[ov["k"]]
        l = min(ov["len"], a_deck.shape[-1] - a0, b_deck.shape[-1])
        a_seg = a_deck[:, a0 : a0 + l]
        b_seg = b_deck[:, :l]
        master[:, ov["start"] : ov["start"] + l] = apply_transition(
            a_seg, b_seg, ov["type"], sr, ov["beats"]
        )
        report(f"transition {ov['k'] + 1}: {ov['type']} over {ov['beats']} beats")

    master = _limiter(master, sr)
    master = np.clip(master, -CEILING, CEILING).astype(np.float32)

    import soundfile as sf

    Path(out_wav).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav), master.T, sr, subtype="PCM_24")
    return {
        "out": str(out_wav), "sr": sr, "duration_s": round(total / sr, 2),
        "transitions": [{"type": o["type"], "start_s": round(o["start"] / sr, 2)}
                        for o in ovs],
    }


def _sample_of(beats: list | None, t: dict, key: str, sr: int) -> int:
    if beats and t.get("in_beat") is not None:
        idx = t["in_beat"] if key == "in_s" else t["out_beat"]
        return int(round(beats[idx] * sr))
    return int(round(t[key] * sr))


def _sidecar_for(t: dict, conn, cache_dir):
    if conn is None:
        return None
    row = conn.execute(
        "SELECT file_size, mtime FROM tracks WHERE path = ?", (t["path"],)
    ).fetchone()
    if row is None:
        return None
    fp = sidecar.fingerprint(t["path"], row["file_size"], row["mtime"])
    return sidecar.load(cache_dir, fp)
