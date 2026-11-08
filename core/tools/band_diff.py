"""Aggregate per-15s band means for two tracks, side by side."""
import sys, subprocess
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core"))

def scan_rows(path):
    out = subprocess.run([sys.executable, "data/ref/scan.py", path],
                         capture_output=True, text=True).stdout.splitlines()
    rows = []
    for ln in out[1:]:
        p = ln.split()
        if len(p) >= 5 and p[0].rstrip('s').isdigit():
            rows.append((int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])))
    return rows

def blocks(rows, bs=15):
    res = []
    for b0 in range(0, rows[-1][0] + 2, bs):
        sel = [r for r in rows if b0 <= r[0] < b0 + bs]
        if not sel:
            continue
        a = np.array([[r[1], r[2], r[3], r[4]] for r in sel])
        res.append((b0, *a.mean(axis=0)))
    return res

refs = blocks(scan_rows(sys.argv[1]))
mine = blocks(scan_rows(sys.argv[2]))
print(f"{'blk':>4} | {'REF':^27} | {'MINE':^27}")
print(f"{'':>4} | {'sub':>6}{'low':>7}{'voc':>7}{'hi':>7} | {'sub':>6}{'low':>7}{'voc':>7}{'hi':>7}")
for i in range(max(len(refs), len(mine))):
    r = refs[i] if i < len(refs) else None
    m = mine[i] if i < len(mine) else None
    rs = f"{r[1]:6.1f}{r[2]:7.1f}{r[3]:7.1f}{r[4]:7.1f}" if r else " " * 27
    ms = f"{m[1]:6.1f}{m[2]:7.1f}{m[3]:7.1f}{m[4]:7.1f}" if m else " " * 27
    print(f"{(i*15):>4} | {rs} | {ms}")
