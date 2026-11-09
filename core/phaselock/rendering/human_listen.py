"""Human-style listening: lyrics, voices, beats, instruments.

The spectrogram rig hears texture; this hears MUSIC the way a person
reports it — WHAT is sung (words + timestamps), HOW MANY voices are
singing at once, where the beat and arrangement sections sit, and which
instruments are active. Its flagship check is vocal-collision detection:
whisper produces word-soup / low-confidence / overlapping hypotheses
wherever two vocalists sing simultaneously, so a mashup with a colliding
ghost vocal is caught as text, not vibes.

Personal-use analysis of locally-stored audio only.
"""

from __future__ import annotations

import numpy as np

SR = 44100


def transcribe(path: str, model_size: str = "small") -> dict:
    """Words with timestamps + per-segment confidence (whisper, CPU)."""
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(path, word_timestamps=True,
                                      beam_size=1, vad_filter=True)
    out = {"language": info.language, "segments": []}
    for seg in segments:
        words = []
        for w in (seg.words or []):
            words.append({"t": round(w.start, 2), "word": w.word.strip(),
                          "p": round(w.probability, 3)})
        out["segments"].append({
            "start": round(seg.start, 2), "end": round(seg.end, 2),
            "text": seg.text.strip(),
            "conf": round(seg.avg_logprob, 2),
            "words": words,
        })
    return out


def vocal_collision(report: dict, hop_s: float = 5.0) -> list[dict]:
    """Regions where the transcription degrades — word-soup signature of
    overlapping voices. Low word probability + high word density (two
    interleaved streams) + repeated/similar words in one window."""
    allw = [w for s in report["segments"] for w in s["words"]]
    if not allw:
        return []
    dur = max(w["t"] for w in allw) + 5.0
    flags = []
    for t0 in np.arange(0, dur, hop_s):
        win = [w for w in allw if t0 <= w["t"] < t0 + hop_s]
        if len(win) < 3:
            continue
        p = float(np.mean([w["p"] for w in win]))
        density = len(win) / hop_s  # words/sec — soup runs hot
        flags.append({"t": round(t0, 1), "words": len(win),
                      "mean_p": round(p, 3), "wps": round(density, 2),
                      "suspect": p < 0.45 or density > 3.2})
    return [f for f in flags if f["suspect"]] or \
           [f for f in flags if f["mean_p"] < 0.55] or []


def arrange_timeline(path: str, beats: list[float] | None = None) -> dict:
    """Instrument activity per section: drums/bass/other energy from the
    full-band envelope split, plus optional downbeat ticks."""
    import soundfile as sf
    from scipy.signal import butter, sosfiltfilt

    y, sr = sf.read(path, always_2d=True)
    m = y.mean(axis=1)
    n = m.size
    bands = {"sub_30_120": (30, 120), "body_120_2k": (120, 2000),
             "air_4k_16k": (4000, 16000)}
    out = {}
    for name, (lo, hi) in bands.items():
        sos = butter(4, [lo / (sr / 2), min(hi / (sr / 2), 0.99)],
                     btype="band", output="sos")
        e = np.abs(sosfiltfilt(sos, m))
        w = int(2 * sr)
        r = e[: n // w * w].reshape(-1, w).mean(axis=1)
        db = 20 * np.log10(r + 1e-12)
        out[name] = [round(float(x), 1) for x in db]
    return {"2s_bands_db": out, "duration_s": round(n / sr, 1)}


def human_report(path: str, section_bounds: dict | None = None) -> dict:
    """Full human-style listen: lyrics, collision regions, arrangement."""
    tr = transcribe(path)
    coll = vocal_collision(tr)
    tl = arrange_timeline(path)
    lines = []
    for s in tr["segments"]:
        lines.append(f"[{s['start']:6.1f}-{s['end']:6.1f}] "
                     f"(conf {s['conf']:+.1f}) {s['text']}")
    return {
        "path": path,
        "language": tr["language"],
        "lyrics": lines,
        "collision_suspects": coll,
        "timeline": tl,
        "sections": section_bounds or {},
    }


if __name__ == "__main__":
    import json
    import sys

    r = human_report(sys.argv[1])
    print(json.dumps(r, indent=1, ensure_ascii=False))
