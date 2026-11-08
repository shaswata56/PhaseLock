"""Phase 4 acceptance gates: Demucs stems + stem bass-swap transitions.

Gates (PLAN.md):
  G1: separation produces a manifest + 4 stems; fresh check passes
  G2: staleness: changed source size/mtime (simulated) → stale
  G3: a bass_swap between different-key tracks renders from real stems
      (engine reports the stem swap) — finite, unclipped output
  G4: without stems the same set renders via the crossover fallback

Usage: python test_stems.py   (needs `pip install demucs`; first run
downloads the htdemucs weights)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
REPO = Path(__file__).resolve().parents[2]


def main() -> int:
    import soundfile as sf

    from phaselock import analyze as analyze_mod
    from phaselock import db as db_mod, stems as stems_mod
    from phaselock.planning import planner
    from phaselock.rendering import engine

    try:
        import demucs  # noqa: F401
    except ImportError:
        print("FAIL: demucs not installed")
        return 1

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gen_test_tracks import SR, render_track

    # two short tracks, wheel-adjacent keys (8A / 7A)
    tmp = REPO / "library" / "stem_test"
    tmp.mkdir(parents=True, exist_ok=True)
    files = []
    for name, bpm, root, mode, bars in [("stem_a", 126.0, 9, "minor", 40),
                                        ("stem_b", 128.0, 2, "minor", 40)]:
        p = tmp / f"{name}.wav"
        sf.write(str(p), render_track(bpm, root, mode, bars), SR)
        files.append(p)

    conn = db_mod.connect(REPO / "data" / "phaselock.db")
    cache = REPO / "data" / "cache"
    root_stems = cache / "stems"
    fps = []
    for p in files:
        r = analyze_mod.analyze_file(conn, cache, p, force=True)
        assert r["analysis_status"] == "ok", r["error"]
        fp = stems_mod.fingerprint_for(conn, p)
        if stems_mod.is_stale(root_stems, fp, p.stat().st_size, p.stat().st_mtime):
            print(f"separating {p.name} (cpu — this is the slow part)…")
            stems_mod.separate(p, root_stems, fp, p.stat().st_size,
                               p.stat().st_mtime)
        fps.append(fp)

    # G1 — manifest + stems present, freshness flips
    g1 = True
    for p, fp in zip(files, fps):
        d = stems_mod.stems_dir(root_stems, fp)
        have = all((d / f"{s}.flac").exists()
                   for s in stems_mod.STEM_NAMES)
        fresh = not stems_mod.is_stale(root_stems, fp, p.stat().st_size,
                                       p.stat().st_mtime)
        g1 &= have and fresh
    print(f"G1 separation cached + fresh: {'PASS' if g1 else 'FAIL'}")

    # G2 — staleness on simulated source change
    p0, fp0 = files[0], fps[0]
    g2 = stems_mod.is_stale(root_stems, fp0, p0.stat().st_size + 999,
                            p0.stat().st_mtime)
    print(f"G2 stale on source change: {'PASS' if g2 else 'FAIL'}")

    # plan a 2-track set, force a bass_swap transition
    cards = planner.load_library(conn, cache)
    pool = [c for c in cards if str(c.path) in {str(f) for f in files}]
    s = planner.plan_set(pool, 3.0, "auto", pool_filter=False)
    s["tracks"][0]["transition_out"]["type"] = "bass_swap"

    # G3 — stem swap renders
    out3 = REPO / "renders" / "gate_stems.wav"
    info = engine.render_set(s, cache, out3, conn=conn, use_stems=True)
    x, sr = sf.read(str(out3))
    g3 = (info["stem_transitions"] == 1 and np.isfinite(x).all()
          and float(np.abs(x).max()) <= 0.9)
    print(f"G3 stem bass-swap render: {'PASS' if g3 else 'FAIL'} "
          f"(swaps={info['stem_transitions']}, "
          f"peak={20 * np.log10(np.abs(x).max()):.2f} dBFS)")

    # G4 — fallback without stems
    out4 = REPO / "renders" / "gate_stems_fallback.wav"
    info4 = engine.render_set(s, cache, out4, conn=conn, use_stems=False)
    x4, _ = sf.read(str(out4))
    g4 = (info4["stem_transitions"] == 0 and np.isfinite(x4).all()
          and float(np.abs(x4).max()) <= 0.9)
    print(f"G4 crossover fallback: {'PASS' if g4 else 'FAIL'}")

    ok = g1 and g2 and g3 and g4
    print(f"\nPHASE 4 GATES: {'ALL PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
