"""Reference-mode mashup renderer: assemble the arc from the source's parts.

Pro mashup-edit technique, driven by the Million Voices x Calling reference:
the instrumental's own phrases are cut and REORDERED into the reference arc
— its riser is the build, its biggest drop is the FINAL (biggest-is-last),
its quietest window is the ONE breakdown — and the topline rides the result.
Every section is a continuous native-rate slice of the source (zero WSOLA on
the bed: transients stay pristine), cut phrase-aligned with 5 ms de-clicks.

Sound integration (the gap the vision diff exposed): beat-locked sidechain
pump re-applied to the bed's low band (gently on the topline), topline glued
with HP + presence EQ + a synthesized plate reverb, bed ducked under active
vocal phrases.

Personal-use mashups of owned, locally-stored tracks only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import sidecar, stems as stems_mod
from .engine import CEILING, _limiter
from .mashup import _VocalRun, _edge_fades, _warp

SR = 44100
OTHER_HP_HZ = 90.0      # melody stem leaves the sub to the bass stem
VOCAL_OVER_DB = 2.5     # topline above the local bed RMS
PUMP_DEPTH = 0.32       # low-band duck under each kick
PUMP_DEPTH_VOC = 0.15   # gentle pump on the topline
PUMP_TAU_S = 0.135      # pump recovery time constant
DUCK_DB = 1.7           # bed dip under active vocal phrases
PLATE_WET_DB = -13.0    # reverb wet level rel. dry topline
EDGE_S = 0.005          # de-click at phrase-aligned hard cuts


# ------------------------------------------------------------------ helpers

def _bar_times(beats: list[float], phase: int | None) -> np.ndarray:
    if not beats or phase is None:
        return np.asarray([])
    return np.asarray([beats[i] for i in range(len(beats))
                       if (i - phase) % 4 == 0 and i >= phase])


def _bar_sub_db(bars: np.ndarray, mono: np.ndarray) -> np.ndarray:
    """Raw sub-band dB per bar — the number listening measures."""
    from scipy.signal import butter, sosfiltfilt

    lp = butter(4, 120.0, btype="low", fs=SR, output="sos")
    m = sosfiltfilt(lp, mono)
    vals = []
    for i in range(len(bars)):
        a = int(bars[i] * SR)
        b = int((bars[i + 1] if i + 1 < len(bars) else bars[i] + 1.9) * SR)
        b = min(b, m.size)
        vals.append(10 * np.log10(float((m[a:b] ** 2).mean()) + 1e-18)
                    if b > a else -120.0)
    return np.asarray(vals)


def extract_parts(sub_db: np.ndarray) -> dict | None:
    """Assign the source's own phrases to roles in the reference arc.

    final  = the loudest sustained run (whole run, capped at 32 bars),
    build  = the 8 bars the source itself uses to lead into that run,
    drop1  = the next-loudest non-overlapping run (capped at 16 bars),
    groove = a medium-energy 16-bar window (fallback: drop1 softened),
    break  = the final's own chords with the floor pulled (beat-strip) —
             the classic tension move, and always musical.
    Parts map role -> (bars, mode) with mode in {"", "soft", "strip"}.
    """
    n = len(sub_db)
    if n < 40:
        return None
    peak = float(np.max(sub_db))

    def runs_at(thresh: float) -> list[tuple[int, int]]:
        mask = sub_db > thresh
        out, st = [], None
        for i, v in enumerate(mask):
            if v and st is None:
                st = i
            elif not v and st is not None:
                out.append((st, i))
                st = None
        if st is not None:
            out.append((st, n))
        return [r for r in out if r[1] - r[0] >= 8]

    runs = runs_at(peak - 8.0) or runs_at(peak - 10.0)
    if not runs:
        return None
    means = {r: float(np.mean(sub_db[r[0]:r[1]])) for r in runs}

    def best_sub(run: tuple[int, int], cap: int) -> tuple[int, int]:
        if run[1] - run[0] <= cap:
            return run
        best, bs = None, run[0]
        for i in range(run[0], run[1] - cap + 1):
            m = float(np.mean(sub_db[i:i + cap]))
            if best is None or m > best:
                best, bs = m, i
        return (bs, bs + cap)

    fin_run = max(runs, key=lambda r: means[r])
    final = best_sub(fin_run, 32)
    build = (max(0, final[0] - 8), final[0])

    taken = [final, build]

    def free(a: int, b: int) -> bool:
        return not any(a < t[1] + 1 and b > t[0] - 1 for t in taken)

    others = [r for r in runs if free(r[0], r[1])]
    drop1 = best_sub(max(others, key=lambda r: means[r]), 16) if others         else (max(0, final[0] - 24), max(0, final[0] - 8))
    taken.append(drop1)

    # groove: medium-energy window clear of every run; if the grid is too
    # crowded, drop1's own slice softened is the fallback groove
    cands = []
    for i in range(0, n - 15, 2):
        if not free(i, i + 16):
            continue
        if any(i < r[1] and i + 16 > r[0] for r in runs):
            continue
        cands.append((float(np.mean(sub_db[i:i + 16])), i))
    if cands and max(cands)[0] > peak - 22.0:  # a real medium window exists
        m, i = min(cands, key=lambda c: abs(c[0] - (peak - 12.0)))
        groove = ((i, i + 16), "")
    else:
        groove = (drop1, "soft")
    return {"groove": groove, "build": (build, ""),
            "drop1": (drop1, ""), "break": (final, "strip"),
            "final": (final, "")}


def _pump_gain(beats_rel: list[float], n: int, depth: float,
               tau: float = PUMP_TAU_S) -> np.ndarray:
    """1 - depth*exp(-(t-t_k)/tau) max-combined over beats."""
    g = np.zeros(n)
    win = int(0.32 * SR)
    tt = np.arange(win) / SR
    seg = np.exp(-tt / tau)
    for t in beats_rel:
        i0 = int(t * SR)
        if i0 < 0 or i0 >= n:
            continue
        e = min(win, n - i0)
        g[i0:i0 + e] = np.maximum(g[i0:i0 + e], seg[:e])
    return 1.0 - depth * g


def _plate_ir(sr: int = SR, seed: int = 4242) -> np.ndarray:
    """Short bright plate: decaying stereo noise, band-limited 350-7k."""
    from scipy.signal import butter, sosfiltfilt

    n = int(1.25 * sr)
    rng = np.random.default_rng(seed)
    ir = rng.standard_normal((2, n))
    ir *= np.exp(-np.arange(n) / (0.30 * sr))
    sos_hp = butter(2, 350 / (sr / 2), btype="high", output="sos")
    sos_lp = butter(2, 7000 / (sr / 2), btype="low", output="sos")
    for ch in range(2):
        ir[ch] = sosfiltfilt(sos_lp, sosfiltfilt(sos_hp, ir[ch]))
    # L2 (energy) normalization: convolution scales loudness by the IR's
    # L2 norm — RMS-normalizing a 1.25 s IR makes wet ~235x the dry signal
    ir /= max(float(np.sqrt((ir ** 2).sum())), 1e-12)
    return ir


def _glue_vocal(v: np.ndarray, ir: np.ndarray) -> np.ndarray:
    """HP + presence + plate reverb — the topline joins the bed's space."""
    from scipy.signal import butter, fftconvolve, sosfiltfilt

    out = np.zeros_like(v, dtype=np.float64)
    sos_hp = butter(2, 80.0, btype="high", fs=SR, output="sos")
    sos_pk = butter(2, [2800 / (SR / 2), 3600 / (SR / 2)], btype="band",
                    output="sos")
    pk_gain = 10 ** (2.5 / 20.0) - 1.0
    for ch in range(v.shape[0]):
        dry = sosfiltfilt(sos_hp, v[ch])
        out[ch] = dry + sosfiltfilt(sos_pk, dry) * pk_gain
    wet = np.stack([fftconvolve(out[ch], ir[ch])[: out.shape[1]]
                    for ch in range(out.shape[0])])
    return out + wet * 10 ** (PLATE_WET_DB / 20.0)


def _activity(x: np.ndarray) -> np.ndarray:
    """0..1 envelope of vocal activity (50 ms RMS, 300 ms smoothing)."""
    hop = int(0.05 * SR)
    n = x.shape[-1] // hop
    r = np.sqrt((x[:, : n * hop].reshape(x.shape[0], n, hop) ** 2
                 ).mean(axis=(0, 2)) + 1e-18)
    r = np.convolve(r, np.ones(6) / 6.0, mode="same")
    e = np.repeat(np.clip(r / (np.percentile(r, 95) + 1e-18), 0.0, 1.0), hop)
    if e.size < x.shape[-1]:
        e = np.concatenate([e, np.full(x.shape[-1] - e.size, e[-1])])
    return e


def _declick(x: np.ndarray) -> None:
    edge = max(1, int(EDGE_S * SR))
    n = x.shape[-1]
    if n < 2 * edge:
        return
    x[:, :edge] *= np.linspace(0.0, 1.0, edge)
    x[:, -edge:] *= np.linspace(1.0, 0.0, edge)


# ------------------------------------------------------------------ render

def render_mashup_ref(base_path: str, donor_path: str, out_wav: str,
                      conn, cache_dir, progress=None) -> dict:
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
        srow = conn.execute("SELECT file_size, mtime FROM tracks WHERE path=?",
                            (str(Path(path).resolve()),)).fetchone()
        sfp = sidecar.fingerprint(path, srow["file_size"], srow["mtime"])
        side_d = sidecar.load(cache_dir, sfp) or {}
        return st, row["bpm"], side_d

    base_st, base_bpm, base_side = stems_of(base_path)
    don_st, don_bpm, don_side = stems_of(donor_path)

    from scipy.signal import butter, sosfiltfilt

    beats = list(base_side.get("beats", []))
    phase = base_side.get("downbeat_phase")
    bars = _bar_times(beats, phase)
    if bars.size < 40:
        raise ValueError("base track too short / no beat grid")

    def _mono(stem):
        x = base_st[stem]
        return x.mean(axis=0) if x.ndim > 1 else x

    sub_db = _bar_sub_db(bars, (_mono("drums") + _mono("bass")) / 2.0)
    parts = extract_parts(sub_db)
    if parts is None:
        raise ValueError("base contour unreadable")
    for k, (v, mode) in parts.items():
        if v is not None:
            report(f"part {k}: bars {v[0]}-{v[1]} "
                   f"(src {bars[v[0]]:.0f}-"
                   f"{bars[min(v[1], len(bars) - 1)]:.0f}s{', ' + mode if mode else ''})")

    # ---- arc assembly: continuous native-rate slices, phrase-aligned cuts
    sos_hp = butter(4, OTHER_HP_HZ, btype="high", fs=SR, output="sos")
    bar_s = 4 * 60.0 / base_bpm

    seq = [("groove", parts["groove"], "run", 4),
           ("build", parts["build"], None, 0),
           ("drop1", parts["drop1"], "run", 4),
           ("break", parts["break"], "solo", 2),
           ("build", parts["build"], None, 0),
           ("final", parts["final"], "callback", 6)]

    stems_all = {k: np.stack(list(base_st[k])).astype(np.float64)
                 for k in ("drums", "bass", "other")}
    min_len = min(v.shape[-1] for v in stems_all.values())

    run = _VocalRun(don_st["vocals"], don_side.get("beats", []),
                    don_side.get("downbeat_phase"))
    rho_d = don_bpm / base_bpm

    secs: list[dict] = []
    out_t = 0.0
    for role, (part, mode), voc_role, n_phr in seq:
        if part is None and role == "build":
            part = (max(0, parts["drop1"][0][0] - 8), parts["drop1"][0][0])
            mode = ""
        if part is None:
            continue
        a_s = float(bars[part[0]])
        b_s = float(bars[min(part[1], len(bars) - 1)])
        i0, i1 = int(a_s * SR), min(int(b_s * SR), min_len)
        n = i1 - i0
        if n <= 0:
            continue
        secs.append(dict(role=role, i0=i0, n=n, out0=0, mode=mode,
                         a_s=a_s, b_s=b_s, voc_role=voc_role, n_phr=n_phr,
                         out_a=0.0, out_b=0.0))
    # exact integer output grid (float accumulation drifts a sample)
    n_out, cursor = 0, 0
    for s_ in secs:
        s_["out0"] = cursor
        s_["out_a"] = cursor / SR
        s_["out_b"] = (cursor + s_["n"]) / SR
        cursor += s_["n"]
        n_out += s_["n"]

    # mix each stem plane section-by-section (native rate, hard cuts)
    bed = np.zeros((2, n_out), dtype=np.float64)
    voc_mix = np.zeros((2, n_out), dtype=np.float64)
    for s in secs:
        i0, n, o0, mode = s["i0"], s["n"], s["out0"], s["mode"]
        stem_g = {"drums": 0.12, "bass": 0.0} if mode == "strip" else {}
        sec_g = 10 ** (-5.0 / 20.0) if mode == "soft" else 1.0
        for name in ("drums", "bass", "other"):
            seg = stems_all[name][:, i0:i0 + n].copy() * stem_g.get(name, 1.0)
            if name == "other":
                seg = np.stack([sosfiltfilt(sos_hp, c) for c in seg])
            _declick(seg)
            bed[:, o0:o0 + n] += seg * sec_g

    # ---- topline placement (donor's hot run, roles per section)
    placements = []
    for s in secs:
        voc_role = s["voc_role"]
        if not voc_role:
            continue
        if voc_role == "callback":
            run.rewind()
        win_t, _L = (run.solo() if voc_role == "solo"
                     else run.take(s["n_phr"]))
        length_s = s["n"] / SR
        if abs(rho_d - 1.0) < 0.002:
            src = int(win_t * SR)
            seg = np.stack([c[src:src + int(length_s * SR)]
                            for c in don_st["vocals"]]).astype(np.float64)
            if seg.shape[-1] < int(length_s * SR):
                seg = np.concatenate(
                    [seg, np.zeros((2, int(length_s * SR) - seg.shape[-1]))],
                    axis=-1)
        else:
            seg = _warp(list(don_st["vocals"]), rho_d,
                        int(length_s * SR) + 1,
                        src_start=int(round(win_t * SR)))
        _edge_fades(seg)
        o0, n = s["out0"], s["n"]
        bed_rms = float(np.sqrt((bed[:, o0:o0 + n] ** 2).mean() + 1e-18))
        vr = float(np.sqrt((seg ** 2).mean() + 1e-18))
        g = bed_rms / max(vr, 1e-9) * 10 ** (VOCAL_OVER_DB / 20)
        m = min(n, seg.shape[-1])
        voc_mix[:, o0:o0 + m] += seg[:, :m] * g
        placements.append({"role": voc_role, "section": s["role"],
                           "at_s": round(s["out_a"], 2),
                           "length_s": round(length_s, 2)})
        report(f"topline {voc_role} over {s['role']} @ {s['out_a']:.0f}s "
               f"({length_s:.0f}s, gain {g:.2f})")

    # ---- pump (each slice's own beats, mapped into output time) + glue
    pump = np.ones(n_out)
    for s in secs:
        rel = [t - s["a_s"] for t in beats if s["a_s"] <= t < s["b_s"]]
        p = _pump_gain(rel, s["n"], PUMP_DEPTH)
        o0, n = s["out0"], s["n"]
        pump[o0:o0 + n] = p[:n]

    voc_mix = _glue_vocal(voc_mix, _plate_ir())
    pump_v = 1.0 - (1.0 - pump) * (PUMP_DEPTH_VOC / PUMP_DEPTH)
    voc_mix *= pump_v
    act = _activity(voc_mix)
    duck = 1.0 - (1.0 - 10 ** (-DUCK_DB / 20.0)) * act

    # pump the bed's low band hard, the upper band gently
    sos_lo = butter(4, 200.0, btype="low", fs=SR, output="sos")
    bed_lo = np.stack([sosfiltfilt(sos_lo, c) for c in bed])
    bed_hi = bed - bed_lo
    bed = bed_lo * (pump * duck) + bed_hi * (np.sqrt(pump) * duck)

    mix = bed + voc_mix
    lp2 = butter(4, 120.0, btype="low", fs=SR, output="sos")
    def sec_sub(a, b):
        seg = sosfiltfilt(lp2, mix[:, int(a * SR):int(b * SR)].mean(axis=0))
        return 10 * np.log10(float((seg ** 2).mean()) + 1e-18)
    # reference ladder (dB under the final): verse ~-10, drop1 ~-9,
    # builds ~-3 rising in. Down-only, capped at 6 dB, applied post-mix.
    LADDER = {"groove": -10.0, "drop1": -9.0, "build": -3.0, "break": None}
    fin_span = next((s for s in secs if s["role"] == "final"), None)
    if fin_span:
        f_db = sec_sub(fin_span["out_a"], fin_span["out_b"])
        seen: set[str] = set()
        for s_ in secs:
            role = s_["role"]
            tgt = LADDER.get(role)
            if role in seen or tgt is None or role == "final":
                continue
            seen.add(role)
            d = sec_sub(s_["out_a"], s_["out_b"])
            want = f_db + tgt
            if d > want:  # pull DOWN toward the ladder, never push up
                trim = min(6.0, d - want)
                o0, n = s_["out0"], s_["n"]
                mix[:, o0:o0 + n] *= 10 ** (-trim / 20.0)
                bed[:, o0:o0 + n] *= 10 ** (-trim / 20.0)
    # master loudness: push into the limiter for reference-density output
    master = _limiter(mix * 10 ** (4.0 / 20.0), SR)
    master = np.clip(master, -CEILING, CEILING).astype(np.float32)

    Path(out_wav).parent.mkdir(parents=True, exist_ok=True)
    import soundfile as sf

    sf.write(str(out_wav), master.T, SR, subtype="PCM_24")
    return {
        "out": str(out_wav), "bpm": round(base_bpm, 2),
        "duration_s": round(master.shape[-1] / SR, 2),
        "structure": {s["role"]: [round(s["out_a"], 1), round(s["out_b"], 1)]
                      for s in secs},
        "placements": placements,
    }
