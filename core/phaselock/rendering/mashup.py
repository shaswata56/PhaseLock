"""Mashup renderer: one instrumental bed + guest vocal layers.

Unlike the DJ-set renderer (sequential decks), a mashup stacks: the base
track's drums/bass/other stems form a constant-tempo bed, and phrase-locked
windows of DONOR vocals ride on top, chosen automatically by vocal energy.
Section structure (intro / A / build / drop / break / outro) is cut on the
base's warped phrase grid; every vocal entry lands downbeat-on-downbeat by
construction. All stems of a track share one WSOLA delta sequence, so bed
and donor layers stay sample-coherent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import decode, sidecar, stems as stems_mod
from .engine import CEILING, _limiter
from .wsola import wsola

SR = 44100
XFADE_S = 0.03
VOCAL_OVER_DB = 3.0     # vocals sit this far above the bed's RMS

# (name, bars, bed gains {drums,bass,other}, vocal slot)
# vocal slot: (donor_index, "verse"|"hook") or None. "both" = top two donors.
# Gains are real mixing moves: drops hit ~+4.5 dB in the bass, builds pull
# the sub back and ramp the tops (tension), breaks clear the floor.
SECTIONS = [
    ("intro",  8, dict(drums=1.0,  bass=1.0,  other=0.0), None),
    ("A1",    16, dict(drums=1.0,  bass=1.0,  other=1.0), (0, "verse")),
    ("A2",    16, dict(drums=1.0,  bass=1.0,  other=1.0), (1, "verse")),
    ("build",  8, dict(drums=1.0,  bass=0.8,  other=0.85, ramp=True), None),
    ("drop1", 16, dict(drums=1.15, bass=1.7,  other=1.05), (0, "hook")),
    ("break",  8, dict(drums=0.0,  bass=0.0,  other=1.0), (1, "verse")),
    ("A3",    16, dict(drums=1.0,  bass=1.0,  other=1.0), (0, "verse")),
    ("build2", 8, dict(drums=1.0,  bass=0.8,  other=0.85, ramp=True), None),
    ("drop2", 16, dict(drums=1.15, bass=1.7,  other=1.05), (1, "hook")),
    ("break2", 8, dict(drums=0.0,  bass=0.0,  other=1.0), None),
    ("build3", 8, dict(drums=1.0,  bass=0.8,  other=0.85, ramp=True), None),
    ("drop3", 16, dict(drums=1.15, bass=1.7,  other=1.05), (0, "hook")),
    ("outro", 16, dict(drums=1.0,  bass=0.7,  other=0.3), None),
]

OTHER_HP_HZ = 90.0   # melody stem leaves sub space to the bass stem


def _warp(channels: list[np.ndarray], ratio: float, n_out: int,
          src_start: int = 0) -> np.ndarray:
    """Warp stem channels at a constant ratio to n_out samples.

    src_start maps to output t=0 — warping from an anchor phrase keeps the
    output grid phrase-aligned. If the source is shorter than the target,
    the bed loops back to the anchor (phrase-anchored wrap, as mashups do).
    """
    src_len = min(c.size for c in channels)
    regions: list[tuple[int, int, float]] = []
    out_left = n_out
    s = src_start
    guard = 0
    while out_left > 0 and guard < 64:
        guard += 1
        span = int(out_left / ratio) + SR
        e = min(src_len - SR, s + span)
        if e <= s + SR // 2:  # degenerate pass (anchor too near the end)
            s = max(0, src_start % max(1, src_len - SR))
            e = min(src_len - SR, s + span)
            if e <= s + SR // 2:
                break
        regions.append((s, e, ratio))
        out_left -= int((e - s) * ratio)
        s = src_start if e >= src_len - SR else e

    ref = channels[0]
    if len(channels) > 1:
        ref = sum(channels) / len(channels)
    outs, _, _ = wsola(ref.astype(np.float64), [c.astype(np.float64) for c in channels],
                       regions)
    out = np.stack(outs)
    if out.shape[-1] < n_out:  # pad if the source ran short
        pad = np.zeros((out.shape[0], n_out - out.shape[-1]))
        out = np.concatenate([out, pad], axis=-1)
    return out[:, :n_out]


def _phrase_starts(beats: list[float], phase: int | None, rho: float,
                   upto: float) -> np.ndarray:
    """Warped phrase start times (16-beat grid anchored to the downbeat)."""
    if not beats or phase is None:
        return np.arange(0.0, upto, 16 * 60 / 130.0)  # uniform fallback
    p = phase
    out = [beats[i] * rho for i in range(len(beats))
           if (i - p) % 16 == 0 and i >= p and beats[i] * rho < upto]
    return np.asarray(out)


def _rms_curve(x: np.ndarray, hop_s: float = 0.25) -> np.ndarray:
    frame = int(hop_s * SR)
    n = max(1, x.shape[-1] // frame)
    y = x[..., : n * frame]
    if y.ndim == 2:
        y = y.mean(axis=0)
    y = y[: n * frame].reshape(n, frame)
    return np.sqrt((y.astype(np.float64) ** 2).mean(axis=1) + 1e-12)


class _VocalPicker:
    """Energy-ranked, phrase-anchored vocal windows from a donor's stems.

    All times are in the donor's ORIGINAL time base; the caller maps the
    chosen window through the warp ratio.
    """

    def __init__(self, voc: np.ndarray, beats, phase):
        self.voc = voc
        self.rms = _rms_curve(voc)
        self.hop = 0.25
        self.starts = _phrase_starts(beats, phase, 1.0, 1e9)

    def best(self, length_s: float, want_hook: bool, used: set[int]) -> tuple[float, float]:
        n_need = int(length_s / self.hop)
        if n_need >= self.rms.size or self.starts.size == 0:
            return 0.0, min(length_s, self.voc.shape[-1] / SR)
        scores = []
        for idx, t in enumerate(self.starts):
            i0 = int(t / self.hop)
            i1 = i0 + n_need
            if i1 >= self.rms.size:
                continue
            seg = self.rms[i0:i1]
            # hooks want sustained energy; verses tolerate dynamics
            score = (np.median(seg) if want_hook else seg.mean()) \
                * (1.0 - 0.5 * float((seg < self.rms.mean() * 0.25).mean()))
            scores.append((score, idx, t))
        if not scores:
            return 0.0, min(length_s, self.voc.shape[-1] / SR)
        scores.sort(reverse=True)
        for score, idx, t in scores:
            if idx not in used or not used:
                return t, length_s
        return scores[0][2], length_s


def render_mashup(base_path: str, donor_paths: list[str], out_wav: str,
                  conn, cache_dir, bpm: float | None = None,
                  sections: list | None = None, progress=None) -> dict:
    sections = sections or SECTIONS

    def report(msg: str) -> None:
        if progress:
            progress(msg)

    cache_root = Path(cache_dir) / "stems"

    def stems_of(path: str):
        fp = stems_mod.fingerprint_for(conn, path)
        if fp is None:
            raise ValueError(f"not in library: {path}")
        row = conn.execute("SELECT file_size, mtime, bpm FROM tracks WHERE path=?",
                           (str(Path(path).resolve()),)).fetchone()
        if stems_mod.is_stale(cache_root, fp, row["file_size"], row["mtime"]):
            raise ValueError(f"no stems cached for {Path(path).name} — "
                             f"run `phaselock stems`")
        st = stems_mod.load_stems(cache_root, fp)
        side = _sidecar(conn, cache_dir, path)
        return st, row["bpm"], side

    def _sidecar(conn, cache_dir, path):
        row = conn.execute("SELECT file_size, mtime FROM tracks WHERE path=?",
                           (str(Path(path).resolve()),)).fetchone()
        fp = sidecar.fingerprint(path, row["file_size"], row["mtime"])
        return sidecar.load(cache_dir, fp) or {}

    base_st, base_bpm, base_side = stems_of(base_path)
    donors = [stems_of(p) for p in donor_paths]
    mix_bpm = bpm or base_bpm

    total_bars = sum(sec[1] for sec in sections)
    n_out = int(total_bars * 4 * 60 / mix_bpm * SR) + SR

    rho_b = base_bpm / mix_bpm
    # anchor the bed at the base's third phrase: output t=0 = a clean phrase
    beats_b = base_side.get("beats", [])
    phase_b = base_side.get("downbeat_phase")
    anchor = 0
    if beats_b and phase_b is not None:
        phrase_idx = [i for i in range(len(beats_b))
                      if (i - phase_b) % 16 == 0 and i >= phase_b]
        if len(phrase_idx) > 2:
            anchor = int(round(beats_b[phrase_idx[2]] * SR))
    # uniform section grid on the mix tempo (sections are whole phrases,
    # and the anchor guarantees phrase alignment of the bed content)
    bounds = _section_bounds(sections, beats_b, phase_b, rho_b, n_out, mix_bpm)
    picks = _plan_bed_windows(base_st, base_side, sections)
    bed = _mix_sections(base_st, picks, rho_b, sections, bounds)
    report(f"bed: {Path(base_path).name} @ {mix_bpm:.1f} bpm, "
           f"{len(sections)} sections, {total_bars} bars "
           f"(drops on the source's biggest moments)")

    # guest vocals
    placements = []
    used: dict[int, set[int]] = {i: set() for i in range(len(donors))}
    for si, (name, bars, _g, slot) in enumerate(sections):
        if slot is None:
            continue
        di, kind = slot
        st, d_bpm, d_side = donors[di]
        if "vocals" not in st:
            continue
        t0, t1 = bounds[si], bounds[si + 1]
        length_s = t1 - t0
        rho_d = d_bpm / mix_bpm
        picker = _VocalPicker(st["vocals"], d_side.get("beats", []),
                              d_side.get("downbeat_phase"))
        win_t, win_len = picker.best(length_s, kind == "hook", used[di])
        # mark used windows (±1 phrase) so sections differ
        for k, pt in enumerate(picker.starts):
            if abs(pt - win_t) < 1e-3:
                used[di].add(k)
                used[di].add(k + 1)
                used[di].add(k - 1)
        # warp ONLY the chosen window: output t=0 = the donor's phrase start
        voc_seg = _warp(list(st["vocals"]), rho_d, int(length_s * SR) + 1,
                        src_start=int(round(win_t * SR)))
        _edge_fades(voc_seg)
        # vocal sits VOCAL_OVER_DB above the local bed RMS
        bed_rms = float(np.sqrt((bed[:, int(t0 * SR):int(t1 * SR)] ** 2
                                 ).mean() + 1e-12))
        voc_rms = float(np.sqrt((voc_seg ** 2).mean() + 1e-12))
        g = bed_rms / max(voc_rms, 1e-9) * 10 ** (VOCAL_OVER_DB / 20)
        a0 = int(t0 * SR)
        bed[:, a0 : a0 + voc_seg.shape[-1]] += voc_seg * g
        placements.append({
            "section": name, "donor": Path(donor_paths[di]).name,
            "kind": kind, "at_s": round(t0, 2), "length_s": round(length_s, 2),
        })
        report(f"vocals: {Path(donor_paths[di]).name} {kind} in {name} "
               f"@ {t0:.0f}s (gain {g:.2f})")

    master = _limiter(bed, SR)
    master = np.clip(master, -CEILING, CEILING).astype(np.float32)
    Path(out_wav).parent.mkdir(parents=True, exist_ok=True)
    import soundfile as sf

    sf.write(str(out_wav), master.T, SR, subtype="PCM_24")
    return {
        "out": str(out_wav), "bpm": round(mix_bpm, 2),
        "duration_s": round(master.shape[-1] / SR, 2),
        "bars": total_bars, "placements": placements,
        "sections": [s[0] for s in sections],
    }


def _section_bounds(sections, beats, phase, rho: float, n_out: int, bpm: float):
    """Uniform section bounds on the mix tempo, output t=0 = anchor phrase.

    Sections are whole (8/16-bar) phrases, so every boundary sits on the
    shared phrase grid; beats/phase are accepted for signature stability.
    """
    p = 60.0 / bpm
    bounds = [0.0]
    for _name, bars, *_ in sections[:-1]:
        bounds.append(bounds[-1] + bars * 4 * p)
    bounds.append(min(bounds[-1] + sections[-1][1] * 4 * p, (n_out - 10) / SR))
    return bounds


def _plan_bed_windows(base_st, base_side, sections) -> list[float]:
    """Pick which part of the base each section reads from.

    A continuous read would put our drops on the source's breakdowns.
    Instead: rank the base's phrases by sub energy; the mashup's drops get
    the source's biggest moments, A-sections get verses, breaks get
    breakdowns. Deterministic, phrase-anchored.
    """
    beats = base_side.get("beats", [])
    phase = base_side.get("downbeat_phase")
    if not beats or phase is None:
        return [0.0] * len(sections)
    phrases = [beats[i] for i in range(len(beats))
               if (i - phase) % 16 == 0 and i >= phase]
    bass = base_st["bass"].mean(axis=0) if base_st["bass"].ndim > 1 \
        else base_st["bass"]
    sr = SR
    total = bass.size / sr

    def score(t0: float, L: float) -> float:
        a, b = int(t0 * sr), min(int((t0 + L) * sr), bass.size)
        if b - a < sr // 2:
            return -99.0
        return float(np.sqrt((bass[a:b] ** 2).mean()))

    # candidate runs: (score, start) per needed window length
    needed = [(name, bars) for name, bars, _g, _s in sections]
    phrase_len = (phrases[1] - phrases[0]) if len(phrases) > 1 else 7.4
    cand: dict[float, list[tuple[float, float]]] = {}
    for _, bars in needed:
        L = bars // 4 * phrase_len
        if L not in cand:
            runs = []
            for t0 in phrases:
                if t0 + L <= total + 0.5:
                    runs.append((score(t0, L), t0))
            runs.sort(key=lambda r: (-r[0], r[1]))
            cand[L] = runs

    def take(lengths: list[float], exclude: list[tuple[float, float]],
             top: bool, count: int) -> list[float]:
        """Grab `count` non-overlapping runs, best (or worst) first."""
        out = []
        for L in lengths:
            pool = cand.get(L, [])
            ordered = pool if top else pool[::-1]
            for sc, t0 in ordered:
                if len(out) >= count:
                    break
                if any(abs(t0 - e0) < max(L, eL) - phrase_len * 0.99
                       for e0, eL in exclude):
                    continue
                out.append(t0)
                exclude.append((t0, L))
                break
            if len(out) >= count:
                break
        while len(out) < count:
            out.append(phrases[0])
        return out

    excl: list[tuple[float, float]] = []
    drop_L = [16 // 4 * phrase_len] * 3
    a_L = [16 // 4 * phrase_len] * 3
    small_L = [8 // 4 * phrase_len] * 6
    outro_L = [16 // 4 * phrase_len]

    drops = take(drop_L, excl, top=True, count=3)
    asecs = take(a_L, excl, top=True, count=3)
    smalls = take(small_L, excl, top=False, count=6)
    outros = take(outro_L, excl, top=False, count=1)

    # map sections to picks by role
    picks: list[float] = []
    di = ai = si_ = oi = 0
    for name, bars, _g, _s in sections:
        if name.startswith("drop"):
            picks.append(drops[di]); di += 1
        elif name.startswith("A"):
            picks.append(asecs[ai]); ai += 1
        elif name == "outro":
            picks.append(outros[oi]); oi += 1
        else:  # intro / build / break
            picks.append(smalls[si_]); si_ += 1
    return picks


def _mix_sections(base_st, picks, rho_b, sections, bounds) -> np.ndarray:
    """Per-section: warp each stem from its picked source window and mix
    with per-stem section gains + envelopes (30 ms edges, build ramps)."""
    from scipy.signal import butter, sosfiltfilt

    sr = SR
    n_out = int(bounds[-1] * sr) + SR
    bed = np.zeros((2, n_out), dtype=np.float64)
    xf = max(1, int(XFADE_S * sr))
    # melody stem leaves the sub to the bass stem (pre-warp, once)
    hp = butter(4, OTHER_HP_HZ, btype="high", fs=sr, output="sos")
    other_src = np.stack([sosfiltfilt(hp, ch) for ch in base_st["other"]])
    stems = {"drums": base_st["drums"], "bass": base_st["bass"],
             "other": other_src}
    for si, (name, bars, gains, slot) in enumerate(sections):
        t0, t1 = bounds[si], bounds[si + 1]
        i0, n = int(t0 * sr), int(t1 * sr) - int(t0 * sr)
        if n <= 0:
            continue
        env = np.ones(n)
        if name == "intro":
            env = np.linspace(0.0, 1.0, n)
        elif name == "outro":
            env = np.linspace(1.0, 0.0, n)
        env[:xf] *= np.linspace(0.0, 1.0, xf)
        env[-xf:] *= np.linspace(1.0, 0.0, xf)
        src_start = int(round(picks[si] * sr))
        for sname, src in stems.items():
            w = _warp(list(src), rho_b, n + SR, src_start=src_start)[:, :n]
            g_env = env * float(gains.get(sname, 1.0))
            if gains.get("ramp") and sname in ("drums", "other"):
                rise = np.linspace(0.0, 1.0, n) ** 1.4
                g_env = g_env * (0.82 + 0.18 * rise)
            bed[:, i0 : i0 + n] += w * g_env
    return bed


def _edge_fades(seg: np.ndarray) -> None:
    n = seg.shape[-1]
    xf = min(max(1, int(0.05 * SR)), n // 2)
    seg[:, :xf] *= np.linspace(0, 1, xf)
    seg[:, -xf:] *= np.linspace(1, 0, xf)
