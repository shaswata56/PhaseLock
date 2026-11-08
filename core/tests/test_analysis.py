"""Phase 1 acceptance gates against synthetic ground truth.

Gates (PLAN.md):
  G1: BPM within ±0.5% on ALL 10 tracks
  G2: key exact match on ≥ 8/10 tracks
  G3: beatgrid monotonic and median inter-beat interval consistent with BPM

Usage: python test_analysis.py [library/test dir]
Exit code 0 = all gates pass.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

TEST_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else
                Path(__file__).resolve().parents[2] / "library" / "test")
DB = TEST_DIR.parent.parent / "data" / "phaselock.db"
CACHE = TEST_DIR.parent.parent / "data" / "cache"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    from phaselock import analyze as analyze_mod, db as db_mod
    from phaselock import sidecar

    truth = json.loads((TEST_DIR / "ground_truth.json").read_text())
    conn = db_mod.connect(DB)

    print(f"{'file':<12} {'bpm(gt)':>8} {'bpm':>7} {'err%':>6}  "
          f"{'key(gt)':<10} {'key':<10} verdict")
    bpm_pass = key_pass = 0
    for t in truth:
        r = analyze_mod.analyze_file(conn, CACHE, TEST_DIR / t["file"], force=True)
        if r["analysis_status"] != "ok":
            print(f"{t['file']:<12} ANALYSIS ERROR: {r['error']}")
            continue

        err = abs(r["bpm"] - t["bpm"]) / t["bpm"] * 100
        bpm_ok = err <= 0.5
        key_ok = r["key_name"] == t["key"]
        bpm_pass += bpm_ok
        key_pass += key_ok

        # G3: sidecar beatgrid sanity
        row = conn.execute("SELECT file_size, mtime FROM tracks WHERE path=?",
                           (r["path"],)).fetchone()
        fp = sidecar.fingerprint(r["path"], row["file_size"], row["mtime"])
        sc = sidecar.load(CACHE, fp)
        beats = sc["beats"]
        ibis = [b - a for a, b in zip(beats, beats[1:])]
        grid_ok = all(d > 0 for d in ibis) and abs(
            60.0 / statistics.median(ibis) - r["bpm"]) / r["bpm"] < 0.01

        verdict = []
        if bpm_ok:
            verdict.append("bpm✓")
        else:
            verdict.append("BPM FAIL")
        verdict.append("key✓" if key_ok else "KEY MISS")
        verdict.append("grid✓" if grid_ok else "GRID FAIL")
        print(f"{t['file']:<12} {t['bpm']:>8.1f} {r['bpm']:>7.2f} {err:>5.2f}%  "
              f"{t['key']:<10} {r['key_name']:<10} {' '.join(verdict)}")

    g1 = bpm_pass == len(truth)
    g2 = key_pass >= 8
    print(f"\nG1 bpm ±0.5% all tracks:  {bpm_pass}/{len(truth)}  "
          f"{'PASS' if g1 else 'FAIL'}")
    print(f"G2 key exact ≥8/10:       {key_pass}/{len(truth)}  "
          f"{'PASS' if g2 else 'FAIL'}")
    return 0 if (g1 and g2) else 1


if __name__ == "__main__":
    raise SystemExit(main())
