"""Manual-build gates: any track with any track, any lengths.

G1: clashing pair builds (warning present, not refused) and renders
G2: per-track minutes respected (play length ≈ requested, phrase-rounded)
G3: extreme tempo gap clamps the chain instead of breaking

Usage: python test_build.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
REPO = Path(__file__).resolve().parents[2]


def main() -> int:
    import soundfile as sf

    from phaselock import db as db_mod
    from phaselock.planning import camelot, planner
    from phaselock.rendering import engine

    conn = db_mod.connect(REPO / "data" / "phaselock.db")
    cache = REPO / "data" / "cache"
    cards = planner.load_library(conn, cache)
    by_stem = {Path(c.path).stem: c for c in cards}

    # G1 — clash builds with a warning
    a = by_stem["test_01"]           # 8A
    b = by_stem["test_05"]           # 5A — clash vs 8A
    assert camelot.compat(a.camelot, b.camelot) == 0.0
    s = planner.build_manual_set(cards, [a.path, b.path], minutes=[1.5, 1.5])
    g1 = any("clash" in w for w in s["warnings"]) and len(s["tracks"]) == 2
    out = REPO / "renders" / "gate_manual.wav"
    info = engine.render_set(s, cache, out, conn=conn)
    x, _ = sf.read(str(out))
    g1 = g1 and np.isfinite(x).all() and float(np.abs(x).max()) <= 0.9
    print(f"G1 clash builds + renders: {'PASS' if g1 else 'FAIL'} "
          f"(warnings={len(s['warnings'])}, demoted={len(info['demotions'])})")

    # G2 — minutes respected (±1 phrase)
    s2 = planner.build_manual_set(cards, [a.path, b.path], minutes=[2.0, 1.0])
    t0, t1 = s2["tracks"]
    want0 = 2.0 * 60 * t0["mix_bpm"] / 60   # beats in 2 minutes
    want1 = 1.0 * 60 * t1["mix_bpm"] / 60
    g2 = (abs(t0["play_beats"] - want0) <= 16
          and abs(t1["play_beats"] - want1) <= 16
          and t0["play_beats"] > t1["play_beats"])
    print(f"G2 per-track minutes: {'PASS' if g2 else 'FAIL'} "
          f"(play {t0['play_beats']}b vs ~{want0:.0f}b, "
          f"{t1['play_beats']}b vs ~{want1:.0f}b)")

    # G3 — extreme tempo gap clamps
    c = by_stem.get("test_09")       # 133.8
    slow = by_stem["test_08"]        # 120.2 → gap ~11%
    s3 = planner.build_manual_set(cards, [slow.path, c.path], minutes=[1.5, 1.5])
    t = s3["tracks"][1]
    g3 = abs(t["mix_bpm"] - slow.bpm) / slow.bpm <= 0.061 \
        and any("stretch" in w or "gap" in w for w in s3["warnings"])
    print(f"G3 extreme gap clamps: {'PASS' if g3 else 'FAIL'} "
          f"(join at {t['mix_bpm']}, stretch ×{t['stretch']})")

    ok = g1 and g2 and g3
    print(f"\nMANUAL BUILD GATES: {'ALL PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
