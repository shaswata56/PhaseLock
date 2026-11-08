"""WSOLA time-stretch — single pass, piecewise-constant rate.

Fixed synthesis hop with a continuous analysis cursor keeps the time-warp
continuous across rate changes, so beat times map to the planned timeline
sample-exactly: the phase-lock at the heart of the mixer. Correlation
offsets are computed on a mono reference and applied to every channel.

`regions` are contiguous [(start_sample, end_sample, ratio), ...] where
ratio = output_length / input_length = native_bpm / mix_bpm.
"""

from __future__ import annotations

import numpy as np

FRAME = 2048
HOP = FRAME // 2
SEARCH = 441          # ±10 ms correlation search
COARSE = 16           # coarse search step, then refined ±COARSE


def wsola(ref: np.ndarray, channels: list[np.ndarray],
          regions: list[tuple[int, int, float]]) -> tuple[list[np.ndarray], list[int]]:
    """Stretch `channels` over `regions`. Returns (channels, frames_per_region).

    Deck sample length ≈ sum(frames)*HOP + HOP; the output time of each
    region boundary is frames_before*HOP exactly — use it for placement.
    """
    frames_per_region = [int(round((e - s) * r / HOP)) for s, e, r in regions]
    total_frames = sum(frames_per_region)
    out_len = total_frames * HOP + FRAME
    win = np.hanning(FRAME + 1)[:FRAME]
    outs = [np.zeros(out_len, dtype=np.float64) for _ in channels]
    wsum = np.zeros(out_len)

    pad = FRAME + SEARCH + 4
    ref_p = np.pad(ref.astype(np.float64), (pad, pad))
    ch_p = [np.pad(c.astype(np.float64), (pad, pad)) for c in channels]

    c = float(regions[0][0]) + pad
    template: np.ndarray | None = None
    f = 0
    for (s, _e, r), n_frames in zip(regions, frames_per_region):
        if f > 0:  # snap the cursor to the region boundary exactly
            c = float(s) + pad
        for _ in range(n_frames):
            base = int(round(c))
            delta = _search_delta(ref_p, base, template)
            a = base + delta
            o = f * HOP
            for ci, ch in enumerate(ch_p):
                outs[ci][o : o + FRAME] += ch[a : a + FRAME] * win
            wsum[o : o + FRAME] += win
            template = ref_p[a + HOP : a + HOP + HOP].copy()
            c += HOP / r
            f += 1

    norm = np.where(wsum > 1e-3, 1.0 / np.maximum(wsum, 1e-9), 0.0)
    outs = [o * norm for o in outs]
    return outs, frames_per_region


def _search_delta(ref_p: np.ndarray, base: int,
                  template: np.ndarray | None) -> int:
    if template is None:
        return 0
    best, delta = -np.inf, 0
    for d in range(-SEARCH, SEARCH + 1, COARSE):
        seg = ref_p[base + d : base + d + HOP]
        if seg.size < HOP:
            break
        sc = float(np.dot(seg, template))
        if sc > best:
            best, delta = sc, d
    for d in range(delta - COARSE, delta + COARSE + 1):
        seg = ref_p[base + d : base + d + HOP]
        if seg.size < HOP:
            break
        sc = float(np.dot(seg, template))
        if sc > best:
            best, delta = sc, d
    return delta
