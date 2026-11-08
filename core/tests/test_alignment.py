"""Alignment gates: measured join correction for drifting/grid-error tracks.

G1: grid-error correction — a constant-tempo incoming deck whose in-point is
    off by 100 ms gets measured and corrected; residual < 25 ms, no demotion
G2: drifting material renders safely — a (124.5→127.5) deck joins cleanly:
    either residual < 25 ms or the join is demoted with a reason
G3: determinism holds with alignment active (constant pair)

Usage: python test_alignment.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
REPO = Path(__file__).resolve().parents[2]


def main() -> int:
    import soundfile as sf

    from gen_test_tracks import SR, render_track
    from phaselock import analyze as analyze_mod
    from phaselock import db as db_mod
    from phaselock.planning import planner
    from phaselock.rendering import engine

    tmp = REPO / "library" / "align_test"
    tmp.mkdir(parents=True, exist_ok=True)
    files = []
    for name, bpm, root, mode in [
        ("align_a", 126.0, 9, "minor"),          # A minor, constant
        ("align_b", 128.0, 2, "minor"),          # D minor, constant
        ("align_drift", (124.5, 127.5), 9, "minor"),  # drifting A minor
    ]:
        p = tmp / f"{name}.wav"
        sf.write(str(p), render_track(bpm, root, mode, 56), SR)
        files.append(p)

    conn = db_mod.connect(REPO / "data" / "phaselock.db")
    cache = REPO / "data" / "cache"
    for p in files:
        r = analyze_mod.analyze_file(conn, cache, p, force=True)
        assert r["analysis_status"] == "ok", r["error"]

    cards = planner.load_library(conn, cache)
    pool = [c for c in cards if str(c.path) in {str(f) for f in files}]

    def build(a_name: str, b_name: str):
        sub = [c for c in pool if Path(c.path).stem in (a_name, b_name)]
        s = planner.plan_set(sub, 2.5, "auto", pool_filter=False)
        names = [Path(t["path"]).stem for t in s["tracks"]]
        if names != [a_name, b_name]:
            s["tracks"] = list(reversed(s["tracks"]))
        s["tracks"][0]["transition_out"] = {"type": "blend", "beats": 16}
        return s

    # G1 — grid error corrected: incoming's in-point off by 100 ms
    s = build("align_a", "align_b")
    s["tracks"][1]["in_beat"] = None        # force the renderer onto in_s
    s["tracks"][1]["in_s"] = round(s["tracks"][1]["in_s"] + 0.100, 3)
    out = REPO / "renders" / "gate_align_1.wav"
    info = engine.render_set(s, cache, out, conn=conn)
    j = info["alignment"][0]
    g1 = (abs(j["delta_ms"]) >= 65 and abs(j["residual_ms"]) <= 25
          and not info["demotions"])
    print(f"G1 grid-error correction: {'PASS' if g1 else 'FAIL'} "
          f"(δ={j['delta_ms']}ms resid={j['residual_ms']}ms conf={j['conf']} "
          f"demoted={info['demotions']})")

    # G2 — drifting deck joins safely
    s = build("align_drift", "align_a")
    out2 = REPO / "renders" / "gate_align_2.wav"
    info2 = engine.render_set(s, cache, out2, conn=conn)
    j2 = info2["alignment"][0]
    demoted_this = any(d["join"] == 1 for d in info2["demotions"])
    g2 = abs(j2["residual_ms"]) <= 25 or demoted_this
    x2, _ = sf.read(str(out2))
    g2 = g2 and np.isfinite(x2).all() and float(np.abs(x2).max()) <= 0.9
    print(f"G2 drifting deck safe join: {'PASS' if g2 else 'FAIL'} "
          f"(δ={j2['delta_ms']}ms resid={j2['residual_ms']}ms "
          f"demoted={demoted_this})")

    # G3 — determinism with alignment (clean constant pair)
    s = build("align_a", "align_b")
    o3, o4 = REPO / "renders" / "gate_align_3.wav", REPO / "renders" / "gate_align_4.wav"
    engine.render_set(s, cache, o3, conn=conn)
    engine.render_set(s, cache, o4, conn=conn)
    g3 = o3.read_bytes() == o4.read_bytes()
    print(f"G3 determinism with alignment: {'PASS' if g3 else 'FAIL'}")

    ok = g1 and g2 and g3
    print(f"\nALIGNMENT GATES: {'ALL PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
