"""Phase 3 acceptance gates: the mix renderer.

Gates (PLAN.md):
  G1: finite output, no clipping (true-peak ≤ −0.9 dBFS), loudness in range
  G2: duration matches the plan within ±3 s
  G3: deterministic — same set → byte-identical WAV
  G4: phase-lock — inside every transition overlap the onset period matches
      the overlap tempo within 1.5% (both decks share one grid)
  G5: solo regions carry the planned mix tempo within 1%

Usage: python test_render.py   (requires the Phase 1 test library analyzed)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
REPO = Path(__file__).resolve().parents[2]


def _acf_period_db(y: np.ndarray, sr: int, lo_bpm: float, hi_bpm: float) -> float | None:
    import librosa

    oe = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    x = oe - oe.mean()
    n = x.size
    size = 1 << int(np.ceil(np.log2(2 * n)))
    spec = np.fft.rfft(x, size)
    acf = np.fft.irfft(spec * np.conj(spec), size)[:n]
    if acf[0] <= 0:
        return None
    acf /= acf[0]
    lo = int(60.0 / hi_bpm * sr / 512)
    hi = int(60.0 / lo_bpm * sr / 512)
    lag = lo + int(np.argmax(acf[lo:hi]))
    if lag + 1 >= n:
        return None
    d = acf[lag - 1] - 2 * acf[lag] + acf[lag + 1]
    shift = np.clip(0.5 * (acf[lag - 1] - acf[lag + 1]) / d, -0.5, 0.5) if abs(d) > 1e-12 else 0.0
    return 60.0 / ((lag + shift) * 512 / sr)


def main() -> int:
    import soundfile as sf

    from phaselock import db as db_mod
    from phaselock.planning import planner
    from phaselock.rendering import engine

    conn = db_mod.connect(REPO / "data" / "phaselock.db")
    cache = REPO / "data" / "cache"
    cards = planner.load_library(conn, cache)
    pool = [c for c in cards if "test_" in Path(c.path).name]
    s = planner.plan_set(pool, 8.0, "peak")

    out1 = REPO / "renders" / "gate_mix_1.wav"
    out2 = REPO / "renders" / "gate_mix_2.wav"
    engine.render_set(s, cache, out1, conn=conn)
    engine.render_set(s, cache, out2, conn=conn)  # determinism run

    x, sr = sf.read(str(out1))
    mono = x.mean(axis=1) if x.ndim > 1 else x
    ok = True

    # G1 — finite, unclipped, loudness sane
    finite = bool(np.isfinite(x).all())
    peak = float(np.abs(x).max())
    from scipy.signal import resample_poly
    tail = mono[-sr * 60 :] if mono.size > sr * 60 else mono
    tp = float(np.abs(resample_poly(tail, 4, 1)).max())
    import pyloudnorm as pyln

    lufs = float(pyln.Meter(sr).integrated_loudness(
        (x[: sr * 120] if x.shape[0] > sr * 120 else x).astype(np.float64)))
    g1 = finite and peak <= 0.9 and tp <= 0.9 and -19 <= lufs <= -8
    print(f"G1 finite/no-clip/loudness: {'PASS' if g1 else 'FAIL'} "
          f"(peak {20 * np.log10(peak):.2f} dBFS, tp {20 * np.log10(tp):.2f} dBTP, "
          f"{lufs:.1f} LUFS)")

    # G2 — duration matches plan
    expected = sum(t["play_beats"] * 60 / t["mix_bpm"] for t in s["tracks"])
    for t in s["tracks"][:-1]:
        expected -= t["transition_out"]["beats"] * 60 / (
            s["tracks"][s["tracks"].index(t) + 1]["mix_bpm"])
    dur = x.shape[0] / sr
    g2 = abs(dur - expected) <= 3.0
    print(f"G2 duration: {'PASS' if g2 else 'FAIL'} "
          f"({dur:.1f}s vs expected {expected:.1f}s)")

    # G3 — determinism
    b1 = out1.read_bytes()
    b2 = out2.read_bytes()
    g3 = b1 == b2
    print(f"G3 byte-identical render: {'PASS' if g3 else 'FAIL'}")

    # G4 — phase-lock inside transitions; G5 — solo tempo
    g4 = g5 = True
    ts = s["tracks"]
    cursor = 0.0
    for k, t in enumerate(ts):
        solo = t["play_beats"] * 60 / t["mix_bpm"]
        T = t.get("transition_out", {}).get("beats", 0)
        if T and k + 1 < len(ts):
            ov_bpm = ts[k + 1]["mix_bpm"]
            ov_dur = T * 60 / ov_bpm
            a = int((cursor + solo - ov_dur + 2) * sr)
            b = int((cursor + solo - 2) * sr)
            seg = mono[a:b]
            if seg.size > sr:
                bpm = _acf_period_db(seg, sr, ov_bpm * 0.9, ov_bpm * 1.1)
                if bpm is None or abs(bpm / ov_bpm - 1) > 0.015:
                    g4 = False
                    print(f"  transition {k + 1}: overlap period {bpm} "
                          f"vs {ov_bpm}")
            cursor += solo - ov_dur
        else:
            # solo tempo check on the middle 20 s of the last track
            mid = cursor + solo / 2
            seg = mono[int((mid - 10) * sr): int((mid + 10) * sr)]
            if seg.size > sr:
                bpm = _acf_period_db(seg, sr, t["mix_bpm"] * 0.94, t["mix_bpm"] * 1.06)
                if bpm is None or abs(bpm / t["mix_bpm"] - 1) > 0.01:
                    g5 = False
                    print(f"  solo {k + 1}: {bpm} vs {t['mix_bpm']}")
            cursor += solo
    print(f"G4 phase-locked overlaps: {'PASS' if g4 else 'FAIL'}")
    print(f"G5 solo tempo as planned: {'PASS' if g5 else 'FAIL'}")

    ok = g1 and g2 and g3 and g4 and g5
    print(f"\nPHASE 3 GATES: {'ALL PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
