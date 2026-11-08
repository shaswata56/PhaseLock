"""Diagnose a set's transitions: drift + join phase risk.

For every transition: local beat periods at the join (vs global BPM → drift
%), and the beat-phase offset δ the renderer would need to correct. High
drift or large |δ| = the transitions that sound flams/mushy.

Usage: python diagnose_mix.py [set.json]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
REPO = Path(__file__).resolve().parents[2]

DRIFT_BAD = 0.006      # 0.6% local-vs-global = grid mistrust
DELTA_BAD_MS = 40.0


def main() -> int:
    from phaselock import decode
    from phaselock.rendering.align import join_offset, local_beat_period

    set_path = sys.argv[1] if len(sys.argv) > 1 \
        else str(REPO / "data" / "sets" / "showcase_real.json")
    s = json.loads(Path(set_path).read_text())
    ts = s["tracks"]

    print(f"{'#':>2} {'outgoing → incoming':<44} {'type':<16} "
          f"{'driftA%':>7} {'driftB%':>7} {'δms':>6} {'conf':>5}  risk")
    risky = 0
    for k in range(len(ts) - 1):
        a, b = ts[k], ts[k + 1]
        kind = a.get("transition_out", {}).get("type", "?")
        ya = decode.load_mono(a["path"]).astype(np.float64) if (np := __import__("numpy")) else None
        yb = decode.load_mono(b["path"]).astype(np.float64)
        P_a = local_beat_period(ya, decode.ANALYSIS_SR, a["out_s"], a["bpm"])
        P_b = local_beat_period(yb, decode.ANALYSIS_SR, b["in_s"], b["bpm"])
        if P_a is None or P_b is None:
            print(f"{k+1:>2} {'(aperiodic window)':<44} {kind:<16}  —")
            risky += 1
            continue
        drift_a = abs(P_a - 60 / a["bpm"]) / (60 / a["bpm"])
        drift_b = abs(P_b - 60 / b["bpm"]) / (60 / b["bpm"])
        tail = ya[max(0, int((a["out_s"] - 16) * decode.ANALYSIS_SR)):int(a["out_s"] * decode.ANALYSIS_SR)]
        head = yb[int(b["in_s"] * decode.ANALYSIS_SR):int((b["in_s"] + 16) * decode.ANALYSIS_SR)]
        delta, conf = join_offset(tail, head, decode.ANALYSIS_SR, (P_a + P_b) / 2)
        risk = []
        if max(drift_a, drift_b) > DRIFT_BAD:
            risk.append("DRIFT")
        if abs(delta) * 1000 > DELTA_BAD_MS:
            risk.append("PHASE")
        if risk:
            risky += 1
        label = f"{Path(a['path']).stem[:20]} → {Path(b['path']).stem[:20]}"
        print(f"{k+1:>2} {label:<44} {kind:<16} {drift_a*100:>6.2f} "
              f"{drift_b*100:>6.2f} {delta*1000:>6.1f} {conf:>5.2f}  "
              f"{'+'.join(risk) or 'ok'}")
    print(f"\n{risky}/{len(ts)-1} transitions carry drift/phase risk")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
