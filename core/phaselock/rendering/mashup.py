"""Mashup renderer: one instrumental bed + one continuous guest topline.

Architecture is reference-driven (pro mashups of the Million Voices x Calling
type): a two-act arc, not a patchwork —
  intro (grooving from bar 1, no dead air) -> verse groove with the topline
  -> sub-notched build w/ noise riser -> drop 1 (medium) -> ONE breakdown
  (beat strips out, the vocal HOLDS — tension) -> build -> final drop (the
  single biggest moment, topline callback) -> quick outro, cut on the phrase.

The bed is the base track's stems, phrase-sampled: source phrases are ranked
by sub+brightness energy and the FINAL section gets the source's biggest
window, drop 1 the next, groove the next; the breakdown gets the quietest.
The topline is the donor's hottest consecutive phrase run, consumed in order
(the song's own verse->chorus progression), then rewound for the callback.

Hard, downbeat-quantized section cuts (5 ms de-click, not blended), 16-beat
phrase anchoring throughout, all stems of a track share one WSOLA delta
sequence so bed and topline stay sample-coherent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import decode, sidecar, stems as stems_mod
from .engine import CEILING, _limiter
from .wsola import wsola

SR = 44100
VOCAL_OVER_DB = 3.0     # topline sits this far above the local bed RMS
BED_EDGE_S = 0.005      # de-click at bed section boundaries (hard cuts)

# (name, bars, bed gains {drums,bass,other}, topline slot)
# slot: (donor_index, role) with role in:
#   "run"      — consume the donor's hot phrase run sequentially
#   "solo"     — the run's strongest phrase, held over the stripped bed
#   "callback" — rewind to the run start (the payoff reprise over the max drop)
# "other_in" fades the melody stem in across the intro; "ramp"/"riser" shape
# builds (energy ramp + white-noise riser, sub notched via bass gain).
SECTIONS = [
    ("intro",  4, dict(drums=1.0, bass=1.0, other=1.0, other_in=True), None),
    ("groove", 16, dict(drums=1.0, bass=1.0, other=1.0), (0, "run")),
    ("build1", 8, dict(drums=1.0, bass=0.55, other=0.9, ramp=True,
                       riser=True), None),
    ("drop1", 24, dict(drums=1.1, bass=1.45, other=1.05), (0, "run")),
    ("break",  8, dict(drums=0.15, bass=0.0, other=0.95), (0, "solo")),
    ("build2", 8, dict(drums=1.0, bass=0.5, other=0.9, ramp=True,
                       riser=True), None),
    ("final", 32, dict(drums=1.15, bass=1.75, other=1.1), (0, "callback")),
    ("outro",  4, dict(drums=1.0, bass=0.85, other=0.35), None),
]

OTHER_HP_HZ = 90.0   # melody stem leaves sub space to the bass stem

# Reference contour (dB relative to the final drop's sub level, measured off
# the Million Voices x Calling reference): verse groove sits ~10 under the
# final drop, builds RISE to ~5.5 under (tension into the impact), drop 1 is
# a groove lift (-9), the breakdown is a deep natural valley. Enforced with
# a post-limiter feedback pass — the limiter eats a pre-set ladder unevenly.
SECTION_TARGET_DB = {
    "final": 0.0, "drop1": -8.5, "groove": -12.0, "intro": -10.0,
    "build1": -5.5, "build2": -5.5,
}


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


class _VocalRun:
    """The donor's hottest consecutive phrase run, consumed in order.

    All times are in the donor's ORIGINAL time base. The run preserves the
    donor's own verse->chorus progression instead of scattering energy-max
    patches, so the topline reads as one performance.
    """

    def __init__(self, voc: np.ndarray, beats, phase):
        self.voc = voc
        self.rms = _rms_curve(voc)
        self.hop = 0.25
        self.starts = list(_phrase_starts(beats, phase, 1.0, 1e9))
        self.cursor = 0  # phrase index into the run
        self.run: list[int] = self._best_run()
        if not self.run:
            self.run = [0]

    def _phrase_score(self, i: int) -> float:
        if i + 1 < len(self.starts):
            L = (self.starts[i + 1] - self.starts[i]) / self.hop
        else:
            L = 16 * 60 / 128.0 / self.hop
        i0 = int(self.starts[i] / self.hop)
        i1 = min(i0 + max(1, int(L)), self.rms.size)
        if i0 >= self.rms.size:
            return -99.0
        return float(self.rms[i0:i1].mean())

    def _best_run(self, want=(12, 10, 8, 6, 4)) -> list[int]:
        n = len(self.starts)
        for k in want:
            if k > n:
                continue
            scores = np.array([self._phrase_score(i) for i in range(n)])
            best_k, best_s = 0, -1e18
            for i0 in range(n - k + 1):
                s = float(scores[i0:i0 + k].mean())
                if s > best_s:
                    best_s, best_k = s, i0
            return list(range(best_k, best_k + k))
        return []

    def take(self, n_phrases: int) -> tuple[float, float]:
        """Next n_phrases of the run as (start_s, length_s) in donor time."""
        k = len(self.run)
        i0 = self.run[self.cursor % k]
        # span actual phrase boundaries (uniform grid may drift vs beats)
        s0 = self.starts[i0]
        last = self.run[min(self.cursor + n_phrases, k - 1) % k]
        end = (self.starts[last + 1] if last + 1 < len(self.starts)
               else s0 + n_phrases * 16 * 60 / 128.0)
        length = max(2.0, end - s0)
        # if the run is shorter than the ask, stretch the window and let
        # _warp's phrase-anchored loop cover the remainder
        length = max(length, n_phrases * 16 * 60 / 128.0 * 0.98)
        self.cursor += n_phrases
        return s0, length

    def solo(self) -> tuple[float, float]:
        """The run's strongest phrase (2 phrases long for coverage)."""
        k = len(self.run)
        scores = [self._phrase_score(self.run[i]) for i in range(k)]
        j = int(np.argmax(scores))
        i0 = self.run[j]
        s0 = self.starts[i0]
        end = (self.starts[i0 + 2] if i0 + 2 < len(self.starts)
               else s0 + 2 * 16 * 60 / 128.0)
        return s0, max(2.0, end - s0)

    def best_entry(self) -> int:
        """Index (into the run) of the phrase with the sharpest real onset:
        quiet before, strong and sustained after. The run's first phrase can
        start mid-verse (singing already underway) — entries must begin on
        a breath."""
        best_i, best_score = 0, -1e9
        for k, i in enumerate(self.run):
            t = self.starts[i]
            i0 = max(0, int((t - 0.35) * SR / self.hop / 0.25 * 0.25))
            a = int((t - 0.35) / self.hop)
            b = int(t / self.hop)
            c = int((t + 0.6) / self.hop)
            if a < 0 or c >= self.rms.size:
                continue
            pre = float(self.rms[a:b].mean())
            post = float(self.rms[b:c].mean())
            if post < self.rms.mean() * 0.5:
                continue  # not a strong phrase head
            score = 10 * np.log10((post + 1e-18) / (pre + 1e-18))
            if score > best_score:
                best_score, best_i = score, k
        return best_i

    def rewind(self) -> None:
        self.cursor = 0


def _riser(n: int, bpm: float, sr: int = SR, seed: int = 777) -> np.ndarray:
    """White-noise riser: band-limited swell rising into the section end,
    cut dead on the downbeat it lands on."""
    from scipy.signal import butter, sosfiltfilt

    rng = np.random.default_rng(seed)
    nz = rng.standard_normal(n)
    sos = butter(2, [1500 / (sr / 2), 8000 / (sr / 2)], btype="band",
                 output="sos")
    nz = sosfiltfilt(sos, nz)
    env = np.exp(np.linspace(0.0, 5.5, n))
    env /= env[-1]
    tail = int(0.025 * sr)
    env[-tail:] *= np.linspace(1.0, 0.0, tail)
    nz *= env
    # slight inter-channel decorrelation; the wrapped tail is already env=0
    right = np.roll(nz, int(0.011 * sr))
    return np.stack([nz, right])


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
    # uniform section grid on the mix tempo (sections are whole phrases and
    # bed windows are phrase-anchored at their own source phrases)
    beats_b = base_side.get("beats", [])
    phase_b = base_side.get("downbeat_phase")
    bounds = _section_bounds(sections, beats_b, phase_b, rho_b, n_out, mix_bpm)
    picks = _plan_bed_windows(base_st, base_side, sections)
    bed = _mix_sections(base_st, picks, rho_b, sections, bounds, mix_bpm)
    report(f"bed: {Path(base_path).name} @ {mix_bpm:.1f} bpm, "
           f"{len(sections)} sections, {total_bars} bars "
           f"(final drop on the source's biggest moment)")

    # guest topline: one hot run per donor, consumed in order
    placements = []
    runs: dict[int, _VocalRun] = {}
    global_rms = float(np.sqrt((bed ** 2).mean() + 1e-12))
    for si, (name, bars, _g, slot) in enumerate(sections):
        if slot is None:
            continue
        di, role = slot
        st, d_bpm, d_side = donors[di]
        if "vocals" not in st:
            continue
        if di not in runs:
            runs[di] = _VocalRun(st["vocals"], d_side.get("beats", []),
                                 d_side.get("downbeat_phase"))
        run = runs[di]
        n_phrases = max(1, bars // 4)
        if role == "callback":
            run.rewind()
        if role == "solo":
            win_t, win_len = run.solo()
        else:  # "run" / "callback" consume the run in order
            win_t, win_len = run.take(n_phrases)
        t0, t1 = bounds[si], bounds[si + 1]
        length_s = t1 - t0
        rho_d = d_bpm / mix_bpm
        voc_seg = _warp(list(st["vocals"]), rho_d, int(length_s * SR) + 1,
                        src_start=int(round(win_t * SR)))
        _edge_fades(voc_seg)
        # vocal rides VOCAL_OVER_DB above the local bed RMS; in stripped
        # sections the reference level holds (never duck with the bed)
        seg_bed = bed[:, int(t0 * SR):int(t1 * SR)]
        bed_rms = float(np.sqrt((seg_bed ** 2).mean() + 1e-12))
        bed_rms = max(bed_rms, 0.7 * global_rms)
        voc_rms = float(np.sqrt((voc_seg ** 2).mean() + 1e-12))
        g = bed_rms / max(voc_rms, 1e-9) * 10 ** (VOCAL_OVER_DB / 20)
        a0 = int(t0 * SR)
        bed[:, a0 : a0 + voc_seg.shape[-1]] += voc_seg * g
        placements.append({
            "section": name, "donor": Path(donor_paths[di]).name,
            "role": role, "at_s": round(t0, 2), "length_s": round(length_s, 2),
        })
        report(f"topline: {Path(donor_paths[di]).name} {role} in {name} "
               f"@ {t0:.0f}s (gain {g:.2f})")

    master = _limiter(bed[:, :n_out], SR)
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
    """Uniform section bounds on the mix tempo, output t=0 = bar 1.

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

    Phrases are ranked by combined sub energy (bass stem) and brightness
    (drum-stem HF) — drops must hit hard AND stay crisp. The FINAL section
    gets the single best window, drop1 the next, groove the next; the
    breakdown and outro get the quietest. Deterministic, phrase-anchored.
    """
    from scipy.signal import butter, sosfiltfilt

    beats = base_side.get("beats", [])
    phase = base_side.get("downbeat_phase")
    if not beats or phase is None:
        return [0.0] * len(sections)
    phrases = [beats[i] for i in range(len(beats))
               if (i - phase) % 16 == 0 and i >= phase]
    bass = base_st["bass"].mean(axis=0) if base_st["bass"].ndim > 1 \
        else base_st["bass"]
    drums = base_st["drums"].mean(axis=0) if base_st["drums"].ndim > 1 \
        else base_st["drums"]
    sos_hf = butter(2, [5000 / (SR / 2), min(14000 / (SR / 2), 0.99)],
                    btype="band", output="sos")
    drums_hf = sosfiltfilt(sos_hf, drums)
    sr = SR
    total = bass.size / sr

    def score(t0: float, L: float) -> float:
        a, b = int(t0 * sr), min(int((t0 + L) * sr), bass.size)
        if b - a < sr // 2:
            return -99.0
        sub = float(np.sqrt((bass[a:b] ** 2).mean()))
        hi = float(np.sqrt((drums_hf[a:b] ** 2).mean()))
        # consistency: a drop window must not open on its own build-up or
        # hide a breakdown inside — effective sub is dragged toward the
        # window's QUIETEST quarter
        blocks = np.array_split(bass[a:b], 8)
        bl_rms = np.array([float(np.sqrt((bl ** 2).mean() + 1e-18))
                           for bl in blocks if bl.size])
        sub_eff = float(np.sqrt(sub * (bl_rms.min() if bl_rms.size else sub)))
        return 0.7 * np.log(sub_eff + 1e-9) + 0.3 * np.log(hi + 1e-9)

    phrase_len = (phrases[1] - phrases[0]) if len(phrases) > 1 else 7.4

    def candidates(L: float, need_strong_open: bool = False) -> list[tuple[float, float]]:
        runs, filtered = [], []
        for t0 in phrases:
            if t0 + L <= total + 0.5:
                sc = score(t0, L)
                runs.append((sc, t0))
                if need_strong_open:
                    a, b = int(t0 * sr), min(int((t0 + L / 4) * sr), bass.size)
                    q0 = float(np.sqrt((bass[a:b] ** 2).mean() + 1e-18))                         if b > a else 0.0
                    mean = float(np.sqrt((bass[int(t0 * sr):
                                               min(int((t0 + L) * sr), bass.size)] ** 2).mean() + 1e-18))
                    if mean > 0 and q0 >= mean * 10 ** (-6 / 20):
                        filtered.append((sc, t0))
        pool = filtered if (filtered or not runs) else runs
        pool.sort(key=lambda r: (-r[0], r[1]))
        return pool

    excl: list[tuple[float, float]] = []

    def free(t0: float, L: float) -> bool:
        return not any(abs(t0 - e0) < max(L, eL) - phrase_len * 0.99
                       for e0, eL in excl)

    def grab(name: str, bars: int, from_top: bool) -> float:
        L = max(1, bars // 4) * phrase_len
        pool = candidates(L, need_strong_open=bars >= 16)
        ordered = pool if from_top else pool[::-1]
        for _sc, t0 in ordered:
            if free(t0, L):
                excl.append((t0, L))
                return t0
        return phrases[0]

    picks: dict[int, float] = {}
    order = [i for i, s in enumerate(sections)]
    # priority: the arc's peaks first, valleys last, middles wherever left
    def tier(name: str) -> int:
        if name == "final":
            return 0
        if name.startswith("drop"):
            return 1
        if name == "groove" or name.startswith("A"):
            return 2
        if name == "outro":
            return 3   # bottom, taken before break so break gets the true low
        if name == "break":
            return 4
        return 5       # intro / builds — mid tier
    for si in sorted(order, key=lambda i: tier(sections[i][0])):
        name, bars, _g, _s = sections[si]
        t = tier(name)
        picks[si] = grab(name, bars, from_top=(t <= 2))
    return [picks[i] for i in range(len(sections))]


def _mix_sections(base_st, picks, rho_b, sections, bounds,
                  bpm: float) -> np.ndarray:
    """Per-section: warp each stem from its picked source window and mix
    with per-stem section gains. Hard cuts (5 ms de-click), intro opens
    already grooving, builds ramp + riser, outro fades only over its last
    bar, then the section ends."""
    from scipy.signal import butter, sosfiltfilt

    sr = SR
    n_out = int(bounds[-1] * sr) + SR
    bed = np.zeros((2, n_out), dtype=np.float64)
    edge = max(1, int(BED_EDGE_S * sr))
    # melody stem leaves the sub to the bass stem (pre-warp, once)
    hp = butter(4, OTHER_HP_HZ, btype="high", fs=sr, output="sos")
    other_src = np.stack([sosfiltfilt(hp, ch) for ch in base_st["other"]])
    stems = {"drums": base_st["drums"], "bass": base_st["bass"],
             "other": other_src}
    lp = butter(4, 120.0, btype="low", fs=sr, output="sos")

    def sub_rms(x: np.ndarray) -> float:
        m = sosfiltfilt(lp, x.mean(axis=0))
        return float(np.sqrt((m ** 2).mean() + 1e-18))

    secs: list[tuple[int, int, np.ndarray, str]] = []
    for si, (name, bars, gains, slot) in enumerate(sections):
        t0, t1 = bounds[si], bounds[si + 1]
        i0, n = int(t0 * sr), int(t1 * sr) - int(t0 * sr)
        if n <= 0:
            continue
        env = np.ones(n)
        if name == "outro":  # hold, then down only over the last bar
            b4 = int(4 * 60 / bpm * sr)
            env[-b4:] *= np.cos(np.linspace(0.0, np.pi / 2, b4)) ** 2
        env[:edge] *= np.linspace(0.0, 1.0, edge)
        env[-edge:] *= np.linspace(1.0, 0.0, edge)
        src_start = int(round(picks[si] * sr))
        sec = np.zeros((2, n), dtype=np.float64)
        for sname, src in stems.items():
            g = float(gains.get(sname, 1.0))
            if name == "intro" and gains.get("other_in") and sname == "other":
                g *= float(np.mean(env))  # melody enters across the intro
            w = _warp(list(src), rho_b, n + SR, src_start=src_start)[:, :n]
            g_env = env * g
            if gains.get("ramp") and sname in ("drums", "other"):
                rise = np.linspace(0.0, 1.0, n) ** 1.4
                g_env = g_env * (0.82 + 0.18 * rise)
            sec += w * g_env
        if gains.get("riser"):  # white-noise swell into the next downbeat
            r = _riser(n, bpm, sr, seed=777 + si)
            scale = 0.5 * float(np.sqrt((sec ** 2).mean() + 1e-12)) \
                / max(float(np.sqrt((r ** 2).mean() + 1e-12)), 1e-9)
            sec += r * scale
        secs.append((i0, n, sec, name))

    # enforce the arc. The limiter eats a pre-set ladder unevenly (bass-
    # heavy sections get crushed more), so: limit, MEASURE the ladder on the
    # limited master, correct each section, re-limit. Sections keep their
    # own arrays (warps already cached), so the feedback pass is cheap.
    def assemble() -> np.ndarray:
        out = np.zeros_like(bed)
        for i0, n, sec, _name in secs:
            out[:, i0 : i0 + n] += sec
        return out

    anchor_db = None
    for _i0, _n, sec, name in secs:
        if name == "final":
            anchor_db = 20 * np.log10(sub_rms(sec))
            break
    if anchor_db is not None and np.isfinite(anchor_db):
        for i0, n, sec, name in secs:
            tgt = SECTION_TARGET_DB.get(name)
            if tgt is None:
                continue
            cur_db = 20 * np.log10(sub_rms(sec))
            sec *= float(10 ** ((anchor_db + tgt - cur_db) / 20.0))
    for _pass in range(1):
        master = _limiter(assemble(), sr)
        ref_db = None
        for i0, n, _sec, name in secs:
            if name == "final":
                ref_db = 20 * np.log10(sub_rms(master[:, i0:i0 + n]))
                break
        if ref_db is None:
            break
        for i0, n, sec, name in secs:
            tgt = SECTION_TARGET_DB.get(name)
            if tgt is None:
                continue
            want_db = ref_db + tgt
            cur_db = 20 * np.log10(sub_rms(master[:, i0:i0 + n]))
            corr = 10 ** ((want_db - cur_db) / 20.0)
            sec *= float(np.clip(corr, 10 ** (-5 / 20), 10 ** (5 / 20)))
    return assemble()


def _edge_fades(seg: np.ndarray) -> None:
    n = seg.shape[-1]
    if n < 8:
        return
    xf = min(max(1, int(0.05 * SR)), n // 2)
    seg[:, :xf] *= np.linspace(0, 1, xf)
    seg[:, -xf:] *= np.linspace(1, 0, xf)
