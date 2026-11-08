"""Mix renderer: set JSON → one continuous, phase-locked mix.

Timeline construction (all sample-exact):
  - deck k plays its [in, out] region, time-warped by WSOLA at rate mix_bpm
    for the solo part and the NEXT deck's mix_bpm for its final transition
    beats — so both decks run at one tempo during the overlap;
  - the next deck starts exactly at the previous deck's rate-change sample,
    a downbeat landing on a downbeat;
  - the transition DSP replaces the summed overlap (equal-power fades,
    Linkwitz-Riley bass swaps on the beat, swept high-pass exits);
  - per-deck LUFS gain riding, then a look-ahead limiter at −1 dBFS.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d
from scipy.signal import butter, sosfiltfilt

from .. import decode, sidecar
from .wsola import HOP, wsola

from pathlib import Path

TARGET_LUFS = -14.0
CEILING = 10.0 ** (-1.0 / 20.0)   # −1 dBFS
XOVER_HZ = 110.0
GAIN_LIMIT_DB = 12.0


def _lufs_gain(lufs: float | None) -> float:
    if lufs is None:
        return 1.0
    g = 10.0 ** ((TARGET_LUFS - lufs) / 20.0)
    return float(np.clip(g, 10 ** (-GAIN_LIMIT_DB / 20), 10 ** (GAIN_LIMIT_DB / 20)))


def _low_band(x: np.ndarray, sr: int) -> np.ndarray:
    sos = butter(4, XOVER_HZ, btype="low", fs=sr, output="sos")
    if x.ndim == 1:
        return sosfiltfilt(sos, x)
    return np.stack([sosfiltfilt(sos, ch) for ch in x])


def _eq_power(l: int) -> tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, np.pi / 2, l)
    return np.cos(theta), np.sin(theta)


def _fade(x: np.ndarray, n: int, fade_in: bool) -> None:
    n = min(n, x.shape[-1])
    r = np.linspace(0.0, 1.0, n)
    if x.ndim == 1:
        x[:n] *= r if fade_in else r[::-1]
        return
    if fade_in:
        x[:, :n] *= r
    else:
        x[:, -n:] *= r[::-1]


def _sweep_hp(x: np.ndarray, sr: int, n_chunks: int,
              f_lo: float = 100.0, f_hi: float = 8000.0) -> np.ndarray:
    """High-pass sweep across the overlap: filter per beat, 10 ms splices."""
    l = x.shape[-1]
    out = np.zeros_like(x, dtype=np.float64)
    edges = np.linspace(0, l, n_chunks + 1).astype(int)
    xf = int(sr * 0.01)
    for j, (a, b) in enumerate(zip(edges, edges[1:])):
        f = f_lo * (f_hi / f_lo) ** (j / max(1, n_chunks - 1))
        sos = butter(4, min(f, sr / 2 - 100), btype="high", fs=sr, output="sos")
        pa, pb = max(0, a - xf), min(l, b + xf)
        seg = x[..., pa:pb]
        filt = np.stack([sosfiltfilt(sos, ch) for ch in seg]) if x.ndim == 2 \
            else sosfiltfilt(sos, seg)
        # splice with a short crossfade against whatever is already there
        n = pb - pa
        env = np.ones(n)
        na = min(xf, n)
        if pa == 0:
            env[:na] = np.linspace(0, 1, na)
        else:
            env[:na] = np.linspace(0, 1, na)
        if pb == l:
            env[-na:] = np.linspace(1, 0, na) if n > na else env[-na:]
        out[..., pa:pb] = out[..., pa:pb] * (1 - env) + filt * env
    return out


def apply_transition(a: np.ndarray, b: np.ndarray, kind: str, sr: int,
                     beats: int, bass_a: np.ndarray | None = None,
                     bass_b: np.ndarray | None = None,
                     drums_b: np.ndarray | None = None,
                     vocals_b: np.ndarray | None = None) -> np.ndarray:
    """Mix the overlap of outgoing (a) and incoming (b) decks.

    Stem decks (bass_a/bass_b/drums_b/vocals_b) share the exact time-warp
    of their full decks. drum_drop needs drums_b (incoming), acapella_overlay
    wants vocals_b; both degrade gracefully to non-stem moves.
    """
    l = min(a.shape[-1], b.shape[-1])
    for stem in (bass_a, bass_b, drums_b, vocals_b):
        if stem is not None:
            l = min(l, stem.shape[-1])
    if l % 2:
        l -= 1
    a, b = a[..., :l].astype(np.float64), b[..., :l].astype(np.float64)
    ca, cb = _eq_power(l)
    if a.ndim == 2:
        ca, cb = ca[None, :], cb[None, :]
    mid = l // 2  # on a beat: even beat counts keep the midpoint on-grid

    if kind == "drum_drop":
        if drums_b is None:
            kind = "filter_sweep"  # graceful degrade
        else:
            # DJ-style build: high-pass the outgoing wider and wider while
            # the incoming drums rise, then the drop — incoming full slams
            # in on the downbeat, outgoing dies.
            swept = _sweep_hp(a, sr, beats, f_lo=150.0, f_hi=2000.0)
            ramp = np.linspace(0.0, 1.0, mid)
            if a.ndim == 2:
                ramp = ramp[None, :]
            pre = swept[..., :mid] + drums_b[..., :mid].astype(np.float64) * ramp
            slam = b[..., mid:].copy()
            atk = max(1, int(0.003 * sr))
            env = np.linspace(0.0, 1.0, atk)
            if slam.ndim == 2:
                env = env[None, :]
            slam[..., :atk] *= env
            return np.concatenate([pre, slam], axis=-1)

    if kind == "filter_sweep":
        swept = _sweep_hp(a, sr, beats)
        return swept * ca + b * cb

    a_high, b_high = a - _low_band(a, sr), b - _low_band(b, sr)
    if kind == "acapella_overlay":
        base = _blend_body(a, b, a_high, b_high, ca, cb, l, mid, sr,
                           bass_a, bass_b)
        if vocals_b is None:
            return base
        # vocal rides the pre-mid window: gentler boost, blend ducks under it
        w = np.zeros(l)
        w[:mid] = np.sin(np.linspace(0.0, np.pi, mid))
        duck = np.ones(l)
        duck[:mid] = 1.0 - 0.3 * np.sin(np.linspace(0.0, np.pi, mid))
        if base.ndim == 2:
            w = w[None, :]
            duck = duck[None, :]
        return base * duck + vocals_b[..., :l].astype(np.float64) * w * 1.25

    if kind == "bass_swap":
        if bass_a is not None and bass_b is not None:
            ba = _low_band(bass_a[..., :l].astype(np.float64), sr)
            bb = _low_band(bass_b[..., :l].astype(np.float64), sr)
        else:
            ba, bb = _low_band(a, sr), _low_band(b, sr)
        low = np.concatenate([ba[..., :mid], bb[..., mid:]], axis=-1)
        return low + a_high * ca + b_high * cb

    # blend
    return _blend_body(a, b, a_high, b_high, ca, cb, l, mid, sr, bass_a, bass_b)


def _blend_body(a, b, a_high, b_high, ca, cb, l, mid, sr,
                bass_a, bass_b) -> np.ndarray:
    """Equal-power blend; lows swap by the midpoint (stems if available)."""
    if bass_a is not None and bass_b is not None:
        la_full = _low_band(bass_a[..., :l].astype(np.float64), sr)
        lb_full = _low_band(bass_b[..., :l].astype(np.float64), sr)
    else:
        la_full, lb_full = _low_band(a, sr), _low_band(b, sr)
    half = mid
    la = np.concatenate([np.linspace(1, 0, half), np.zeros(l - half)])
    lb = np.concatenate([np.linspace(0, 1, half), np.ones(l - half)])
    if a.ndim == 2:
        la, lb = la[None, :], lb[None, :]
    low = la_full * la + lb_full * lb
    return low + a_high * ca + b_high * cb


def _limiter(x: np.ndarray, sr: int) -> np.ndarray:
    look = max(1, int(sr * 0.005))
    rel = int(sr * 0.06)
    peak = np.abs(x).max(axis=0)
    env = maximum_filter1d(peak, look)
    g = np.minimum(1.0, CEILING / np.maximum(env, 1e-9))
    g = minimum_filter1d(g, rel)
    return x * g[None, :]


def render_set(set_dict: dict, cache_dir, out_wav: str, sr: int = 44100,
               conn=None, progress=None, use_stems: bool = True) -> dict:
    """Render a set (as produced by planning.planner.plan_set) to a WAV.

    Real recordings drift and humanize against their global beatgrid, so
    every overlap is rate-adapted to the LOCAL tempo at the join (both decks
    run at the join BPM, each stretched by its own local estimate) and the
    incoming deck is phase-shifted by the measured onset-alignment offset
    (±0.45 beat). Joins that stay misaligned — or tracks that drift hard —
    are demoted to masking transitions (filter_sweep) instead of tight ones.
    """
    from .. import stems as stems_mod
    from .align import join_offset, local_beat_period

    tracks = set_dict["tracks"]
    decks, stem_decks, starts, ovs = [], [], [], []
    drifts: list[dict] = []
    align_info, demotions = [], []
    DRIFT_TIGHT = 0.006   # above this, tight moves are unsafe
    DRIFT_ANY = 0.012     # above this, only masking transitions

    def report(msg: str) -> None:
        if progress:
            progress(msg)

    for k, t in enumerate(tracks):
        y = decode.load_stereo(t["path"], sr)
        sc = _sidecar_for(t, conn, cache_dir)
        beats = sc["beats"] if sc else None
        T_prev = (tracks[k - 1].get("transition_out", {}).get("beats", 0)
                  if k > 0 else 0)
        T_out = t.get("transition_out", {}).get("beats", 0)
        kind = t.get("transition_out", {}).get("type", "")
        in_smp = _sample_of(beats, t, "in_s", sr)
        out_smp = _sample_of(beats, t, "out_s", sr)

        ref = y.mean(axis=0)
        p_glob = 60.0 / t["bpm"]
        p_head = local_beat_period(ref, sr, in_smp / sr, t["bpm"]) or p_glob
        p_tail = local_beat_period(ref, sr, out_smp / sr, t["bpm"]) or p_glob
        drift_head = abs(p_head - p_glob) / p_glob
        drift_tail = abs(p_tail - p_glob) / p_glob

        # region plan: [head-overlap?] [middle] [tail-overlap?]
        # overlap regions target T beats of timeline at the join BPM while
        # spanning the deck's LOCAL T beats — no intra-overlap drift.
        r_mid = t["bpm"] / t["mix_bpm"]
        regions: list[tuple[int, int, float]] = []
        cursor = in_smp
        if T_prev and k > 0:
            join_dur = T_prev * 60.0 / t["mix_bpm"]
            head_end = min(out_smp - 1, in_smp + int(T_prev * p_head * sr))
            if head_end > cursor:
                regions.append((cursor, head_end,
                                join_dur / max(1e-6, (head_end - cursor) / sr)))
                cursor = head_end
        if T_out and k + 1 < len(tracks):
            r_join_out = t["bpm"] / tracks[k + 1]["mix_bpm"]
            split = max(cursor + 1, out_smp - int(T_out * p_tail * sr))
            join_dur = T_out * 60.0 / tracks[k + 1]["mix_bpm"]
            if split > cursor:
                regions.append((cursor, split, r_mid))
            regions.append((split, out_smp,
                            join_dur / max(1e-6, (out_smp - split) / sr)))
        else:
            regions.append((cursor, out_smp, r_mid))

        chans, frames, deltas = wsola(ref, [y[0], y[1]], regions)
        deck = np.stack(chans)
        gain = _lufs_gain(t.get("lufs"))
        deck *= gain
        _fade(deck, int(sr * 0.002), True)
        _fade(deck, int(sr * 0.01), False)
        decks.append(deck)

        # stem decks (same warping) for adjacent transitions
        needed: set[str] = set()
        if "bass_swap" in (kind, tracks[k - 1].get("transition_out", {}).get("type", "")
                           if k > 0 else ""):
            needed.add("bass")
        prev_kind = (tracks[k - 1].get("transition_out", {}).get("type", "")
                     if k > 0 else "")
        if prev_kind == "drum_drop":
            needed.add("drums")
        if prev_kind == "acapella_overlay":
            needed.add("vocals")
        st_decks: dict[str, np.ndarray] = {}
        if use_stems and conn is not None and needed:
            fp = stems_mod.fingerprint_for(conn, t["path"])
            if fp:
                st = stems_mod.load_stems(
                    Path(cache_dir) / "stems", fp, names=tuple(needed))
                if st is not None and abs(st["sr"] - sr) < 1:
                    for sname in needed:
                        s = st[sname].astype(np.float32)
                        if s.ndim == 1:
                            s = np.stack([s, s])
                        sch, _, _ = wsola(ref, [s[0], s[1]], regions, deltas=deltas)
                        st_decks[sname] = np.stack(sch) * gain
        stem_decks.append(st_decks)

        # solo end = start of this deck's tail-overlap region (timeline)
        solo_end = sum(frames[:-1]) * HOP if (T_out and k + 1 < len(tracks)) \
            else deck.shape[-1]
        drifts.append({"head": drift_head, "tail": drift_tail})
        if k == 0:
            starts.append(0)
        else:
            starts.append(ovs[-1]["start"])
        if T_out and k + 1 < len(tracks):
            ovs.append({
                "k": k, "start": starts[k] + solo_end,
                "len": deck.shape[-1] - solo_end,
                "beats": T_out, "type": kind,
                "bpm": tracks[k + 1]["mix_bpm"],
            })
        report(f"deck {k + 1}/{len(tracks)}: {Path(t['path']).name}"
               + (f" (+{','.join(st_decks)})" if st_decks else ""))

    # --- join alignment: measure δ per join, shift incoming, verify ---
    for j, ov in enumerate(ovs):
        k = ov["k"]
        join_p = 60.0 / ov["bpm"]
        join_drift = max(drifts[k]["tail"], drifts[k + 1]["head"])
        a0 = ov["start"] - starts[k]
        look = min(int(20 * sr), decks[k].shape[-1] - a0, decks[k + 1].shape[-1])
        delta = conf = resid = 0.0
        if look > int(5 * sr):
            tail = decks[k][:, a0 : a0 + look].mean(axis=0)
            head = decks[k + 1][:, :look].mean(axis=0)
            delta, conf = join_offset(tail, head, sr, join_p)
            if conf < 0.15:
                delta = 0.0
            shift = int(round(delta * sr))
            if shift:
                # everything at or after the incoming deck moves with it;
                # the overlap window moves too, so the incoming's window
                # content is unchanged — the outgoing tail re-slices.
                for m in range(k + 1, len(starts)):
                    starts[m] += shift
                for later in ovs[j:]:
                    later["start"] += shift
                a0 += shift
                tail2 = decks[k][:, a0 : a0 + look].mean(axis=0)
                resid, rconf = join_offset(tail2, head, sr, join_p)
                if rconf < 0.15:
                    resid = 0.0
        align_info.append({"join": k + 1, "delta_ms": round(delta * 1000, 1),
                           "conf": round(conf, 2),
                           "residual_ms": round(resid * 1000, 1)})

        # demote tight moves that reality can't support
        why = None
        if ov["type"] in ("drum_drop", "bass_swap", "acapella_overlay") \
                and join_drift > DRIFT_TIGHT:
            why = f"drift {join_drift:.1%}"
        elif abs(resid) > 0.25 * join_p:
            why = f"residual {resid*1000:.0f}ms"
        elif ov["type"] == "blend" and join_drift > DRIFT_ANY:
            why = f"drift {join_drift:.1%}"
        if why:
            demotions.append({"join": k + 1, "from": ov["type"],
                              "to": "filter_sweep", "why": why})
            ov["type"] = "filter_sweep"
        if align_info[-1]["delta_ms"]:
            report(f"join {k + 1}: aligned by {delta*1000:.0f} ms "
                   f"(conf {conf:.2f})")

    total = starts[-1] + decks[-1].shape[-1]
    master = np.zeros((2, total))
    for start, deck in zip(starts, decks):
        master[:, start : start + deck.shape[-1]] += deck

    stem_transitions, fallbacks = 0, 0
    for ov in ovs:
        a_deck, b_deck = decks[ov["k"]], decks[ov["k"] + 1]
        a_st, b_st = stem_decks[ov["k"]], stem_decks[ov["k"] + 1]
        a0 = ov["start"] - starts[ov["k"]]
        l = min(ov["len"], a_deck.shape[-1] - a0, b_deck.shape[-1])
        if l <= 0:
            continue
        if l % 2:
            l -= 1  # apply_transition keeps windows even (beat-midpoint math)
        a_seg = a_deck[:, a0 : a0 + l]
        b_seg = b_deck[:, :l]
        used_stems = bool(a_st or b_st)
        if used_stems:
            stem_transitions += 1
        if ov["type"] == "drum_drop" and "drums" not in b_st:
            fallbacks += 1
        if ov["type"] == "acapella_overlay" and "vocals" not in b_st:
            fallbacks += 1
        master[:, ov["start"] : ov["start"] + l] = apply_transition(
            a_seg, b_seg, ov["type"], sr, ov["beats"],
            bass_a=(a_st.get("bass")[:, a0 : a0 + l] if "bass" in a_st else None),
            bass_b=(b_st.get("bass")[:, :l] if "bass" in b_st else None),
            drums_b=(b_st.get("drums")[:, :l] if "drums" in b_st else None),
            vocals_b=(b_st.get("vocals")[:, :l] if "vocals" in b_st else None),
        )
        report(f"transition {ov['k'] + 1}: {ov['type']} over {ov['beats']} beats")

    master = _limiter(master, sr)
    master = np.clip(master, -CEILING, CEILING).astype(np.float32)

    import soundfile as sf

    Path(out_wav).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav), master.T, sr, subtype="PCM_24")
    return {
        "out": str(out_wav), "sr": sr, "duration_s": round(total / sr, 2),
        "stem_transitions": stem_transitions,
        "stem_fallbacks": fallbacks,
        "alignment": align_info,
        "demotions": demotions,
        "transitions": [{"type": o["type"], "start_s": round(o["start"] / sr, 2)}
                        for o in ovs],
    }


def _sample_of(beats: list | None, t: dict, key: str, sr: int) -> int:
    if beats and t.get("in_beat") is not None:
        idx = t["in_beat"] if key == "in_s" else t["out_beat"]
        return int(round(beats[idx] * sr))
    return int(round(t[key] * sr))


def _sidecar_for(t: dict, conn, cache_dir):
    if conn is None:
        return None
    row = conn.execute(
        "SELECT file_size, mtime FROM tracks WHERE path = ?", (t["path"],)
    ).fetchone()
    if row is None:
        return None
    fp = sidecar.fingerprint(t["path"], row["file_size"], row["mtime"])
    return sidecar.load(cache_dir, fp)
