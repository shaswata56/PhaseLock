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
from pathlib import Path

import numpy as np

from .. import sidecar
from .camelot import compat, parse

SET_VERSION = 1
MAX_STRETCH = 0.04      # deck may deviate this far from its native BPM
MAX_DRIFT = 0.03        # running mix BPM may move this much per transition
PHRASE_BEATS = 16       # 4 bars — the phrase grid
DEFAULT_TRACK_MIN = 3.0
# Short blends expose less double-content time; builds need the long runway.
BLEND_BEATS = 16
BUILD_BEATS = 32

VIBES: dict[str, dict] = {
    "auto":    dict(bpm=(118, 140), start=0.35, peak=1.00, peak_at=0.70, end=0.55, minor_bias=0.0),
    "peak":    dict(bpm=(126, 140), start=0.70, peak=1.00, peak_at=0.60, end=0.80, minor_bias=0.1),
    "warmup":  dict(bpm=(118, 128), start=0.20, peak=0.85, peak_at=0.75, end=0.45, minor_bias=0.0),
    "melodic": dict(bpm=(120, 128), start=0.30, peak=0.85, peak_at=0.65, end=0.50, minor_bias=0.4),
    "dark":    dict(bpm=(122, 132), start=0.35, peak=0.95, peak_at=0.70, end=0.55, minor_bias=0.6),
    "closing": dict(bpm=(118, 126), start=0.60, peak=0.75, peak_at=0.35, end=0.25, minor_bias=0.2),
}


def resolve_vibe(spec: str | None) -> dict:
    """Vibe DSL → arc/bpm parameters.

    Preset keyword (auto/peak/warmup/melodic/dark/closing, last one wins),
    then directives: 'bpm A-B', 'end NNN', 'start NNN', and modifiers
    'hype' (peak 1.0, higher opening), 'chill' (lower arc), 'minor'/
    'major' (tonality bias). Unknown words are ignored, so free-text
    prompts work — the LLM layer (optional) only refines track choice.
    """
    base = dict(VIBES["auto"])
    if not spec:
        return base
    import re

    words = [w.lower() for w in re.split(r"[,\s]+", spec.strip()) if w]
    i = 0
    while i < len(words):
        w = words[i]
        if w in VIBES:
            base = dict(VIBES[w])
        elif w == "bpm" and i + 1 < len(words) and "-" in words[i + 1]:
            a, b = words[i + 1].split("-", 1)
            if a.replace(".", "").isdigit() and b.replace(".", "").isdigit():
                base["bpm"] = (float(a), float(b))
            i += 1
        elif w in ("end", "start") and i + 1 < len(words) \
                and words[i + 1].replace(".", "").isdigit():
            lo, hi = base["bpm"]
            v = float(words[i + 1])
            base["bpm"] = (lo, v) if w == "end" else (v, hi)
            i += 1
        elif w == "hype":
            base.update(start=min(1.0, base["start"] + 0.2), peak=1.0,
                        peak_at=0.55)
        elif w in ("chill", "soft"):
            base.update(start=max(0.1, base["start"] - 0.15),
                        peak=min(base["peak"], 0.75),
                        end=max(0.2, base["end"] - 0.1))
        elif w == "minor":
            base["minor_bias"] = 0.6
        elif w == "major":
            base["minor_bias"] = 0.0
        i += 1
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
    has_stems: bool = False

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
    """TrackCards for every analyzed track; sidecar curves + stems state."""
    from .. import stems as stems_mod

    rows = conn.execute(
        "SELECT path, file_size, mtime, bpm, key_name, camelot, duration_s, "
        "loudness_lufs, energy_mean FROM tracks "
        "WHERE analysis_status='ok' AND bpm IS NOT NULL ORDER BY path"
    ).fetchall()
    stems_root = Path(cache_dir) / "stems" if cache_dir else None
    cards = []
    for r in rows:
        fp = sidecar.fingerprint(r["path"], r["file_size"], r["mtime"])
        sc = sidecar.load(cache_dir, fp) or {}
        has_stems = False
        if stems_root is not None:
            has_stems = not stems_mod.is_stale(
                stems_root, fp, r["file_size"], r["mtime"])
        cards.append(TrackCard(
            path=r["path"], bpm=r["bpm"], key_name=r["key_name"],
            camelot=r["camelot"], duration_s=r["duration_s"],
            lufs=r["loudness_lufs"], energy_mean=r["energy_mean"],
            beats=sc.get("beats", []), downbeat_phase=sc.get("downbeat_phase"),
            has_stems=has_stems,
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
    """Pick the transition move. Stem moves need stems on the incoming side
    (the renderer degrades gracefully when they're missing)."""
    de = nxt.energy - cur.energy
    if score >= 0.9 and nxt.has_stems and de >= 0.2:
        return "drum_drop"          # drum build over the outgoing, then drop
    if score >= 0.9 and nxt.has_stems and abs(de) < 0.15:
        return "acapella_overlay"   # vocal ride over a level blend
    if score >= 1.0 and de >= 0:
        return "bass_swap"
    if de > 0.25:
        return "filter_sweep"
    return "blend"


def llm_order(library: list[dict], prompt: str) -> list[str]:
    """Optional external LLM hook: PHASELOCK_LLM_CMD.

    The command receives {"library": [...cards], "prompt": str} on stdin and
    prints a JSON array of paths (the desired order). Any failure returns []
    and planning falls back to the built-in search — the LLM refines, never
    breaks, the pipeline.
    """
    import json
    import os
    import subprocess

    cmd = os.environ.get("PHASELOCK_LLM_CMD")
    if not cmd:
        return []
    try:
        p = subprocess.run(
            cmd, input=json.dumps({"library": library, "prompt": prompt}),
            capture_output=True, text=True, shell=True, timeout=180,
        )
        lines = [ln for ln in p.stdout.strip().splitlines() if ln.strip()]
        arr = json.loads(lines[-1]) if lines else []
        return [x for x in arr if isinstance(x, str)]
    except Exception:
        return []


def _compat_repair(seq: list[TrackCard]) -> list[TrackCard]:
    """Drop tracks that would clash with the previous kept one."""
    out: list[TrackCard] = []
    for c in seq:
        if out:
            drift = abs(c.bpm - out[-1].bpm) / out[-1].bpm
            if compat(out[-1].camelot, c.camelot) <= 0 or drift > 0.06:
                continue
        out.append(c)
    return out


def plan_set(cards: list[TrackCard], duration_min: float, vibe: str | None = None,
             pool_filter: bool = True, order: list[str] | None = None) -> dict:
    """Plan a set. Returns the set dict consumed by the renderer.

    `order` (e.g. from the LLM hook) pins the sequence; incompatible tracks
    are dropped rather than allowed to clash.
    """
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

    forced = None
    if order:
        by_path = {c.path: c for c in pool}
        forced = _compat_repair([by_path[p] for p in order if p in by_path])
        if len(forced) < 2:
            forced = None

    n = len(forced) if forced else int(
        np.clip(round(duration_min / DEFAULT_TRACK_MIN), 2, len(pool)))
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

    if forced:
        best = (forced, 0.0)
    else:
        # beam search, deterministic
        beam: list[tuple[list[TrackCard], float]] = []
        for c in sorted(pool, key=lambda c: c.path):
            beam.append(([c], step_score(None, c, 0)))
        beam.sort(key=lambda st: (-st[1], st[0][0].path))
        beam = beam[:6]

        best = None
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
            "lufs": c.lufs,
            "in_beat": in_i,
            "out_beat": out_i,
            "in_s": round(c.beats[in_i], 3),
            "out_s": round(c.beats[out_i], 3),
            "play_beats": out_i - in_i,
        }
        if k < len(seq) - 1:
            nxt_card = seq[k + 1]
            ttype = _transition_type(c, nxt_card, compat(c.camelot, nxt_card.camelot))
            entry["transition_out"] = {
                "type": ttype,
                "beats": BUILD_BEATS if ttype == "drum_drop" else BLEND_BEATS,
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
