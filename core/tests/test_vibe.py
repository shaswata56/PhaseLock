"""Phase 6 gates: vibe DSL + optional LLM ordering hook.

G1: DSL directives parse — 'end NNN', 'bpm A-B', modifiers
G2: two vibe specs → different sets (carry-over sanity with the DSL)
G3: LLM hook (fake command via PHASELOCK_LLM_CMD) orders the set and the
   planner honors it after compatibility repair
G4: hook failure/absence falls back to built-in search silently
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
REPO = Path(__file__).resolve().parents[2]


def main() -> int:
    from phaselock import db as db_mod
    from phaselock.planning import planner

    # G1 — DSL parsing
    v1 = planner.resolve_vibe("dark end 128")
    v2 = planner.resolve_vibe("melodic bpm 122-127")
    v3 = planner.resolve_vibe("peak hype")
    g1 = (v1["bpm"][1] == 128.0 and v2["bpm"] == (122.0, 127.0)
          and v3["peak"] == 1.0 and v3["peak_at"] == 0.55)
    print(f"G1 vibe DSL parsing: {'PASS' if g1 else 'FAIL'} "
          f"({v1['bpm']}, {v2['bpm']}, {v3['peak']}/{v3['peak_at']})")

    conn = db_mod.connect(REPO / "data" / "phaselock.db")
    cache = REPO / "data" / "cache"
    cards = planner.load_library(conn, cache)
    pool = [c for c in cards if "test_" in Path(c.path).name]

    # G2 — DSL steers the output: disjoint bpm windows must pick
    # different tracks (overlapping windows may legitimately converge)
    a = planner.plan_set(pool, 12.0, "dark bpm 129-134")
    b = planner.plan_set(pool, 12.0, "warmup bpm 118-124")
    g2 = [t["path"] for t in a["tracks"]] != [t["path"] for t in b["tracks"]]
    print(f"G2 DSL steers selection: {'PASS' if g2 else 'FAIL'}")

    # G3 — LLM hook honored (reverse library order via a fake hook script)
    hook = REPO / "core" / "tests" / "_fake_llm_hook.py"
    hook.write_text(
        "import json, sys\n"
        "req = json.load(sys.stdin)\n"
        "paths = [t['path'] for t in req['library']][::-1]\n"
        "print(json.dumps(paths))\n"
    )
    import os

    venv_py = REPO / ".venv" / "Scripts" / "python.exe"
    os.environ["PHASELOCK_LLM_CMD"] = f'"{venv_py}" "{hook}"'
    lib = [{"path": c.path, "bpm": c.bpm} for c in pool]
    order = planner.llm_order(lib, "test")
    want = [c.path for c in pool][::-1]
    s = planner.plan_set(pool, 12.0, "auto", order=order)
    got = [t["path"] for t in s["tracks"]]
    repaired = planner._compat_repair([c for c in pool][::-1])
    g3 = order == want and got == [c.path for c in repaired] \
        and not planner.validate_set(s)
    print(f"G3 LLM hook honored + repaired: {'PASS' if g3 else 'FAIL'} "
          f"(hook {len(order)}, set {len(got)}, clashes 0)")

    # G4 — absent hook falls back
    del os.environ["PHASELOCK_LLM_CMD"]
    s2 = planner.plan_set(pool, 12.0, "auto")
    g4 = len(s2["tracks"]) >= 2 and not planner.validate_set(s2)
    print(f"G4 fallback without hook: {'PASS' if g4 else 'FAIL'}")

    ok = g1 and g2 and g3 and g4
    print(f"\nPHASE 6 GATES: {'ALL PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
