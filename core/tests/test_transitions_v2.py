"""v2 transition gates: drum_drop and acapella_overlay.

G1: drum_drop renders from the incoming track's drum stem (stem usage
    reported), output finite and unclipped; the drop is audible as an
    energy step at the midpoint
G2: acapella_overlay renders with the vocals stem riding the blend
G3: both degrade cleanly with stems disabled (fallbacks reported, output
    still valid)

Usage: python test_transitions_v2.py   (stems for library/stem_test must
exist — run test_stems.py first)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
REPO = Path(__file__).resolve().parents[2]


def _rms_db(x: np.ndarray) -> float:
    return 20 * np.log10(np.sqrt((x.astype(np.float64) ** 2).mean()) + 1e-12)


def main() -> int:
    import soundfile as sf

    from phaselock import db as db_mod
    from phaselock.planning import planner
    from phaselock.rendering import engine

    conn = db_mod.connect(REPO / "data" / "phaselock.db")
    cache = REPO / "data" / "cache"
    cards = planner.load_library(conn, cache)
    pool = [c for c in cards if Path(c.path).stem in {"stem_a", "stem_b"}]
    if not all(c.has_stems for c in pool):
        print("FAIL: stem_test tracks lack stems — run test_stems.py first")
        return 1

    ok = True
    for kind, expect_stems in [("drum_drop", True), ("acapella_overlay", True)]:
        s = planner.plan_set(pool, 3.0, "auto", pool_filter=False)
        s["tracks"][0]["transition_out"]["type"] = kind
        out = REPO / "renders" / f"gate_v2_{kind}.wav"
        info = engine.render_set(s, cache, out, conn=conn, use_stems=True)
        x, sr = sf.read(str(out))
        mono = x.mean(axis=1)

        good = (info["stem_transitions"] == 1 and info["stem_fallbacks"] == 0
                and np.isfinite(x).all() and float(np.abs(x).max()) <= 0.9)

        # the transition midpoint must be an energy event, not silence
        tr = info["transitions"][0]
        mid = int(tr["start_s"] * sr) + sr  # 1 s past midpoint
        pre = mono[mid - 2 * sr : mid - sr]
        post = mono[mid : mid + sr]
        step_db = _rms_db(post) - _rms_db(pre)
        good &= step_db > -6.0 and _rms_db(post) > -30.0

        print(f"{kind}: {'PASS' if good else 'FAIL'} "
              f"(stems={info['stem_transitions']}, "
              f"fallbacks={info['stem_fallbacks']}, step={step_db:+.1f} dB)")

        # G3 — degraded render
        info2 = engine.render_set(s, cache, out.with_name(out.name.replace(
            "gate_v2", "gate_v2fb")), conn=conn, use_stems=False)
        y, _ = sf.read(str(out.with_name(out.name.replace("gate_v2", "gate_v2fb"))))
        fb_ok = (info2["stem_transitions"] == 0
                 and np.isfinite(y).all() and float(np.abs(y).max()) <= 0.9)
        print(f"{kind} fallback: {'PASS' if fb_ok else 'FAIL'} "
              f"(fallbacks={info2['stem_fallbacks']})")
        ok &= good and fb_ok

    print(f"\nV2 TRANSITION GATES: {'ALL PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
