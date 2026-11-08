"""Camelot (Open Key) compatibility graph for harmonic mixing.

Score semantics:
  1.0  same key, or relative major/minor (same tonic, e.g. 8A/8B)
  0.9  adjacent wheel position (±1), either mode — the workhorse transition
  0.55 same tonic, mode switch (minor ↔ major, wheel distance 3) — the
       "energy boost" move; allowed but never required
  0.0  everything else — a key clash; the planner may not link these
"""

from __future__ import annotations


def parse(camelot: str) -> tuple[int, str]:
    """'8A' → (8, 'A'). Raises ValueError on malformed codes."""
    c = camelot.strip().upper()
    n, mode = c[:-1], c[-1:]
    if mode not in ("A", "B") or not n.isdigit() or not 1 <= int(n) <= 12:
        raise ValueError(f"bad camelot code: {camelot!r}")
    return int(n), mode


def compat(a: str, b: str) -> float:
    """Harmonic compatibility score, 0..1 (0 = clash)."""
    na, ma = parse(a)
    nb, mb = parse(b)
    d = abs(na - nb)
    d = min(d, 12 - d)  # wheel distance
    if d == 0:
        return 1.0
    if d == 1:
        return 0.9
    if d == 3 and ma != mb:
        return 0.55  # same tonic minor↔major (e.g. 8A ↔ 11B is d=3; 8A↔8B is d=0)
    return 0.0


def compatible(a: str, b: str, min_score: float = 0.55) -> bool:
    return compat(a, b) >= min_score
