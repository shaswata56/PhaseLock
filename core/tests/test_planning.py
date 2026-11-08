"""Phase 2 acceptance gates: Camelot graph, set planning.

Gates (PLAN.md):
  G1: zero key clashes, no tempo drift > 3%, no stretch > 4% (validate_set)
  G2: in/out points land on the 16-beat phrase grid anchored to downbeats
  G3: two different vibes produce meaningfully different sets
  G4: planner is deterministic

Usage: python test_planning.py   (requires the Phase 1 test library analyzed)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO = Path(__file__).resolve().parents[2]


def main() -> int:
    from phaselock import db as db_mod
    from phaselock.planning import camelot, planner

    # --- Camelot unit checks
    cases = [
        ("8A", "8A", 1.0), ("8A", "8B", 1.0),      # same key, relative
        ("8A", "9A", 0.9), ("8A", "9B", 0.9), ("8A", "7B", 0.9),
        ("8A", "11B", 0.55), ("12A", "3B", 0.55),  # same tonic M↔m
        ("8A", "5A", 0.0), ("8A", "10B", 0.0), ("8A", "2A", 0.0),
    ]
    unit_fail = [f"{a}/{b} → {camelot.compat(a, b)} != {want}"
                 for a, b, want in cases
                 if abs(camelot.compat(a, b) - want) > 1e-9]
    print(f"camelot unit: {len(cases) - len(unit_fail)}/{len(cases)} "
          f"{'PASS' if not unit_fail else 'FAIL'}")
    for u in unit_fail:
        print(f"  {u}")

    conn = db_mod.connect(REPO / "data" / "phaselock.db")
    cache = REPO / "data" / "cache"
    cards = planner.load_library(conn, cache)
    pool = [c for c in cards if "test_" in Path(c.path).name]
    print(f"library: {len(pool)} synthetic cards")
    if len(pool) < 4:
        print("FAIL: analyze the synthetic test library first")
        return 1

    # --- G1: harmonic/tempo constraints
    s = planner.plan_set(pool, 12.0, "peak")
    errors = planner.validate_set(s)
    g1 = not errors
    print(f"G1 no clashes / drift / stretch: {'PASS' if g1 else 'FAIL'}")
    for e in errors:
        print(f"  {e}")

    # --- G2: phrase alignment of in/out points
    g2 = True
    by_path = {c.path: c for c in pool}
    for t in s["tracks"]:
        c = by_path[t["path"]]
        p = c.downbeat_phase
        if p is None:
            g2 = False
            print(f"  {c.path}: no downbeat phase")
            continue
        for idx in (t["in_beat"], t["out_beat"]):
            if (idx - p) % planner.PHRASE_BEATS != 0 or idx < p:
                g2 = False
                print(f"  {Path(c.path).name}: beat {idx} off phrase grid "
                      f"(phase {p})")
    print(f"G2 phrase-aligned in/out: {'PASS' if g2 else 'FAIL'}")

    # --- G3: vibes differ
    s2 = planner.plan_set(pool, 12.0, "warmup")
    order1 = [Path(t["path"]).name for t in s["tracks"]]
    order2 = [Path(t["path"]).name for t in s2["tracks"]]
    g3 = order1 != order2
    print(f"G3 vibes produce different sets: {'PASS' if g3 else 'FAIL'}")
    print(f"  peak:   {order1}")
    print(f"  warmup: {order2}")

    # --- G4: determinism (ignore timestamps)
    s3 = planner.plan_set(pool, 12.0, "peak")
    g4 = s3["tracks"] == s["tracks"]
    print(f"G4 deterministic planning: {'PASS' if g4 else 'FAIL'}")

    ok = not unit_fail and g1 and g2 and g3 and g4
    print(f"\nPHASE 2 GATES: {'ALL PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
