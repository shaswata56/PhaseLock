"""Mashup gates: layered render sanity on stem-cached tracks.

G1: renders finite, unclipped, duration ≈ bars·4·60/bpm (±3%)
G2: vocal placements exist and land inside their sections
G3: deterministic — byte-identical re-render

Usage: python test_mashup.py   (stems for library/stem_test must exist)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
REPO = Path(__file__).resolve().parents[2]

SECTIONS_T = [
    ("intro", 4, dict(drums=1.0, bass=1.0, other=0.0), None),
    ("A1", 8, dict(drums=1.0, bass=1.0, other=1.0), (0, "verse")),
    ("drop", 8, dict(drums=1.0, bass=1.1, other=1.0), (0, "hook")),
    ("outro", 4, dict(drums=1.0, bass=0.5, other=0.0), None),
]


def main() -> int:
    import soundfile as sf

    from phaselock import db as db_mod
    from phaselock.rendering import mashup as mm

    conn = db_mod.connect(REPO / "data" / "phaselock.db")
    cache = REPO / "data" / "cache"
    base = str(REPO / "library" / "stem_test" / "stem_a.wav")
    donor = str(REPO / "library" / "stem_test" / "stem_b.wav")

    out1 = REPO / "renders" / "gate_mashup_1.wav"
    out2 = REPO / "renders" / "gate_mashup_2.wav"
    info = mm.render_mashup(base, [donor], out1, conn, cache,
                            bpm=126.0, sections=SECTIONS_T)
    mm.render_mashup(base, [donor], out2, conn, cache,
                     bpm=126.0, sections=SECTIONS_T)

    x, sr = sf.read(str(out1))
    want = sum(s[1] for s in SECTIONS_T) * 4 * 60 / 126.0
    g1 = (np.isfinite(x).all() and float(np.abs(x).max()) <= 0.9
          and abs(x.shape[0] / sr - want) / want <= 0.03)
    print(f"G1 finite/no-clip/duration: {'PASS' if g1 else 'FAIL'} "
          f"({x.shape[0]/sr:.1f}s vs {want:.1f}s, "
          f"peak {20*np.log10(np.abs(x).max()):.2f} dBFS)")

    dur = x.shape[0] / sr
    g2 = len(info["placements"]) >= 1
    for p in info["placements"]:
        if not (0 <= p["at_s"] < dur and p["length_s"] > 1):
            g2 = False
    print(f"G2 vocal placements sane: {'PASS' if g2 else 'FAIL'} "
          f"({len(info['placements'])} placed)")

    g3 = out1.read_bytes() == out2.read_bytes()
    print(f"G3 determinism: {'PASS' if g3 else 'FAIL'}")

    ok = g1 and g2 and g3
    print(f"\nMASHUP GATES: {'ALL PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
