"""Set generation: energy arcs, Camelot-chained ordering, phrase points.

Deterministic beam search over the analyzed library. Every adjacent pair must
be harmonically compatible (compat > 0), tempo drift is bounded (each deck
stretches ≤ 4% from native, running tempo moves ≤ 3% per transition), and
in/out points land on phrase boundaries (16-beat grid anchored to the
downbeat phase) so Phase 3 can phase-lock transitions by construction.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field

import numpy as np

from .. import sidecar
from .camelot import compat, parse

SET_VERSION = 1
MAX_STRETCH = 0.04      # deck may deviate this far from its native BPM
MAX_DRIFT = 0.03        # running mix BPM may move this much per transition
TRANSITION_BEATS = 32   # standard 8-bar blend
PHRASE_BEATS = 16       # 4 bars — the phrase grid
DEFAULT_TRACK_MIN = 3.0

VIBES: dict[str, dict] = {
    "auto":    dict(bpm=(118, 140), start=0.35, peak=1.00, peak_at=0.70, end=0.55, minor_bias=0.0),
    "peak":    dict(bpm=(126, 140), start=0.70, peak=1.00, peak_at=0.60, end=0.80, minor_bias=0.1),
    "warmup":  dict(bpm=(118, 128), start=0.20, peak=0.85, peak_at=0.75, end=0.45, minor_bias=0.0),
    "melodic": dict(bpm=(120, 128), start=0.30, peak=0.85, peak_at=0.65, end=0.50, minor_bias=0.4),
    "dark":    dict(bpm=(122, 132), start=0.35, peak=0.95, peak_at=0.70, end=0.55, minor_bias=0.6),
    "closing": dict(bpm=(118, 126), start=0.60, peak=0.75, peak_at=0.35, end=0.25, minor_bias=0.2),
}


def resolve_vibe(spec: str | None) -> dict:
    """Keyword vibe → arc/bpm parameters. Unknown words fall back to 'auto'."""
    base = dict(VIBES["auto"])
    if not spec:
        return base
    words = [w.lower() for w in spec.replace(",", " ").split()]
    used = False
    for w in words:
        if w in VIBES:
            base = dict(VIBES[w])
            used = True
    return base


@dataclass
class TrackCard:
    path: str
    bpm: float
    key_name: str
    camelot: str
    duration_s: float
    lufs: float | None
    energy_mean: float | None
    beats: list[float] = field(default_factory=list)
    downbeat_phase: int | None = None
    energy: float = 0.5  # percentile among planned library, set by load

    @property
    def beat_sec(self) -> float:
        return 60.0 / self.bpm

    def phrase_indices(self) -> list[int]:
        """Beat indices that start a 4-bar phrase (anchored to downbeat)."""
        if self.downbeat_phase is None or len(self.beats) < 8:
            return []
        p = self.downbeat_phase
        return [i for i in range(len(self.beats))
                if (i - p) % PHRASE_BEATS == 0 and i >= p]


def load_library(conn, cache_dir) -> list[TrackCard]:
    """TrackCards for every analyzed track; sidecar curves joined in."""
    rows = conn.execute(
        "SELECT path, file_size, mtime, bpm, key_name, camelot, duration_s, "
        "loudness_lufs, energy_mean FROM tracks "
        "WHERE analysis_status='ok' AND bpm IS NOT NULL ORDER BY path"
    ).fetchall()
    cards = []
    for r in rows:
        fp = sidecar.fingerprint(r["path"], r["file_size"], r["mtime"])
        sc = sidecar.load(cache_dir, fp) or {}
        cards.append(TrackCard(
            path=r["path"], bpm=r["bpm"], key_name=r["key_name"],
            camelot=r["camelot"], duration_s=r["duration_s"],
            lufs=r["loudness_lufs"], energy_mean=r["energy_mean"],
            beats=sc.get("beats", []), downbeat_phase=sc.get("downbeat_phase"),
        ))
    _assign_energy_percentiles(cards)
    return cards


def _assign_energy_percentiles(cards: list[TrackCard]) -> None:
    vals = np.array([c.energy_mean if c.energy_mean is not None else -30.0
                     for c in cards])
    if len(cards) > 1:
        order = vals.argsort()
        ranks = np.empty(len(cards), dtype=float)
        ranks[order] = np.arange(len(cards))
        for i, c in enumerate(cards):
            c.energy = float(ranks[i] / (len(cards) - 1))
    else:
        for c in cards:
            c.energy = 0.5


def _arc(n: int, v: dict) -> list[float]:
    """Target energy per slot: rise → peak → land."""
    pk = max(1, min(n - 1, round((n - 1) * v["peak_at"])))
    out = []
    for k in range(n):
        if k < pk:
            out.append(v["start"] + (v["peak"] - v["start"]) * k / pk)
        elif k == pk:
            out.append(v["peak"])
        else:
            span = max(1, n - 1 - pk)
            out.append(v["peak"] + (v["end"] - v["peak"]) * (k - pk) / span)
    return out


def _pick_in_out(card: TrackCard, play_sec: float, tail_beats: int = 8):
    """Phrase-aligned in/out beat indices for a desired play length."""
    phrases = card.phrase_indices()
    if not phrases:
        return None, None, []
    # skip the first ~2 phrases (intro), keep at least a phrase + tail after in
    in_idx = next((i for i in phrases if i >= 2 * PHRASE_BEATS), phrases[0])
    last = len(card.beats) - 1 - tail_beats
    play_beats = max(PHRASE_BEATS, round(play_sec / card.beat_sec / PHRASE_BEATS) * PHRASE_BEATS)
    out_idx = in_idx + play_beats
    while out_idx > last and out_idx - PHRASE_BEATS > in_idx:
        out_idx -= PHRASE_BEATS
    if out_idx - in_idx < PHRASE_BEATS:
        return None, None, []
    return in_idx, out_idx, phrases


def _transition_type(cur: TrackCard, nxt: TrackCard, score: float) -> str:
    if score >= 1.0 and nxt.energy >= cur.energy:
        return "bass_swap"
    if nxt.energy > cur.energy + 0.25:
        return "filter_sweep"
    return "blend"


def plan_set(cards: list[TrackCard], duration_min: float, vibe: str | None = None,
             pool_filter: bool = True) -> dict:
    """Plan a set. Returns the set dict consumed by the renderer."""
    v = resolve_vibe(vibe)
    lo, hi = v["bpm"]
    pool = [c for c in cards if lo <= c.bpm <= hi] if pool_filter else list(cards)
    if not pool:  # progressively widen until non-empty
        for widen in (3, 6, 12, 100):
            pool = [c for c in cards if lo - widen <= c.bpm <= hi + widen]
            if pool:
                break
    if len(pool) < 2:
        raise ValueError("need at least 2 analyzable tracks in range")

    n = int(np.clip(round(duration_min / DEFAULT_TRACK_MIN), 2, len(pool)))
    arc = _arc(n, v)
    play_sec = duration_min * 60.0 / n

    def opener_score(c: TrackCard) -> float:
        fit = 1.0 - abs(c.energy - arc[0])
        center = (lo + hi) / 2
        bpm_fit = 1.0 - min(1.0, abs(c.bpm - center) / 10.0)
        minor = v["minor_bias"] if "minor" in (c.key_name or "") else 0.0
        return round(fit + 0.5 * bpm_fit + minor, 6)

    def step_score(cur: TrackCard | None, cand: TrackCard, slot: int) -> float | None:
        if cur is None:
            return opener_score(cand)
        score = compat(cur.camelot, cand.camelot)
        if score <= 0:
            return None
        drift = abs(cand.bpm - cur.bpm) / cur.bpm
        if drift > 0.06:
            return None
        energy_fit = 1.0 - abs(cand.energy - arc[slot])
        bpm_pen = (drift / 0.05) ** 2
        rising = 0.3 if (slot <= _arc_peak(n, v) and cand.bpm >= cur.bpm) else 0.0
        return round(score + energy_fit - 1.5 * bpm_pen + rising, 6)

    # beam search, deterministic
    beam: list[tuple[list[TrackCard], float]] = []
    for c in sorted(pool, key=lambda c: c.path):
        beam.append(([c], step_score(None, c, 0)))
    beam.sort(key=lambda st: (-st[1], st[0][0].path))
    beam = beam[:6]

    best: tuple[list[TrackCard], float] | None = None
    for slot in range(1, n):
        nxt: list[tuple[list[TrackCard], float]] = []
        for seq, total in beam:
            cur = seq[-1]
            used = {c.path for c in seq}
            for cand in pool:
                if cand.path in used:
                    continue
                sc = step_score(cur, cand, slot)
                if sc is None:
                    continue
                nxt.append((seq + [cand], total + sc))
        if not nxt:  # dead end: fall back to best partial set
            break
        nxt.sort(key=lambda st: (-st[1], st[0][-1].path))
        beam = nxt[:6]
        if len(beam[0][0]) == n:
            best = beam[0]
    if best is None:
        best = max(beam, key=lambda st: (len(st[0]), st[1]))
    seq = best[0]

    # tempo chain + per-track details
    tracks_out, running = [], seq[0].bpm
    for k, c in enumerate(seq):
        if k == 0:
            mix_bpm = c.bpm
        else:
            blend = 0.5 * running + 0.5 * c.bpm
            mix_bpm = float(np.clip(
                blend,
                max(c.bpm * (1 - MAX_STRETCH), running * (1 - MAX_DRIFT)),
                min(c.bpm * (1 + MAX_STRETCH), running * (1 + MAX_DRIFT)),
            ))
        running = mix_bpm
        play = play_sec * (1.15 if k == 0 else 1.0)  # opener breathes a little
        in_i, out_i, _ = _pick_in_out(c, play)
        if in_i is None:
            raise ValueError(f"no usable phrase window in {c.path}")
        entry = {
            "path": c.path,
            "bpm": round(c.bpm, 2),
            "key": c.key_name,
            "camelot": c.camelot,
            "energy": round(c.energy, 3),
            "mix_bpm": round(mix_bpm, 2),
            "stretch": round(mix_bpm / c.bpm, 4),
            "in_beat": in_i,
            "out_beat": out_i,
            "in_s": round(c.beats[in_i], 3),
            "out_s": round(c.beats[out_i], 3),
            "play_beats": out_i - in_i,
        }
        if k < len(seq) - 1:
            nxt_card = seq[k + 1]
            entry["transition_out"] = {
                "type": _transition_type(c, nxt_card, compat(c.camelot, nxt_card.camelot)),
                "beats": TRANSITION_BEATS,
            }
        tracks_out.append(entry)

    return {
        "version": SET_VERSION,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "duration_target_min": duration_min,
        "vibe": (vibe or "auto"),
        "arc": [round(a, 3) for a in arc],
        "tracks": tracks_out,
    }


def _arc_peak(n: int, v: dict) -> int:
    return max(1, min(n - 1, round((n - 1) * v["peak_at"])))


def validate_set(s: dict) -> list[str]:
    """Gate checks used by tests: no key clashes, bounded drift/stretch."""
    errors = []
    ts = s["tracks"]
    for a, b in zip(ts, ts[1:]):
        if compat(a["camelot"], b["camelot"]) <= 0:
            errors.append(f"key clash {a['camelot']}→{b['camelot']}")
        drift = abs(b["mix_bpm"] - a["mix_bpm"]) / a["mix_bpm"]
        if drift > MAX_DRIFT + 1e-6:
            errors.append(f"tempo drift {drift:.1%} at {a['path']}")
    for t in ts:
        if abs(t["stretch"] - 1.0) > MAX_STRETCH + 1e-6:
            errors.append(f"stretch out of range on {t['path']}")
    return errors
