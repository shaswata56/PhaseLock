"""Generate 10 synthetic EDM tracks with exact ground truth.

Synthesis is deliberately EDM-canonical: 4-on-the-floor kick, offbeat hats,
root/fifth bass, chord stabs marking the bar (downbeat marker for the
detector), a breakdown section for energy-curve variation. Ground truth BPM
and key are exact by construction, so the acceptance gates measure the
analyzer, not the material. Regenerates deterministically (seeded).

Usage: python gen_test_tracks.py [output_dir]
Writes: <out>/test_NN.wav + <out>/ground_truth.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SR = 44100
SEED = 20261108
PC_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]
MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]

# (name, bpm, root_pc, mode, bars) — close BPM pairs + one non-integer tempo
# to prove sub-BPM precision, mixed keys, two major-key tracks.
TRACKS = [
    ("test_01", 126.0, 9, "minor", 88),   # A minor
    ("test_02", 124.0, 4, "minor", 88),   # E minor
    ("test_03", 125.5, 2, "minor", 84),   # D minor — non-integer BPM
    ("test_04", 128.0, 7, "minor", 88),   # G minor
    ("test_05", 130.0, 0, "minor", 88),   # C minor
    ("test_06", 122.0, 5, "minor", 84),   # F minor
    ("test_07", 132.0, 10, "minor", 88),  # A# (Bb) minor
    ("test_08", 120.0, 9, "major", 88),   # A major
    ("test_09", 134.0, 5, "major", 84),   # F major
    ("test_10", 127.0, 11, "minor", 88),  # B minor
]


def midi_hz(midi: float) -> float:
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


def add(buf: np.ndarray, start_s: float, sig: np.ndarray) -> None:
    i = int(start_s * SR)
    j = min(i + sig.size, buf.size)
    if i < buf.size:
        buf[i:j] += sig[: j - i]


def kick(dur: float = 0.28) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    freq = 42.0 + 110.0 * np.exp(-t * 38.0)
    amp = np.exp(-t * 16.0)
    return (np.sin(2 * np.pi * np.cumsum(freq) / SR) * amp).astype(np.float64)


def hat(dur: float = 0.05, amp: float = 0.12) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    n = int(dur * SR)
    noise = rng.standard_normal(n)
    noise = np.diff(noise, prepend=0.0)  # crude highpass
    env = np.exp(-np.arange(n) / SR * 90.0)
    return noise * env * amp


def tone(freq: float, dur: float, amp: float, harmonics=(1.0,)) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    sig = np.zeros(n)
    for k, h in enumerate(harmonics, start=1):
        sig += h * np.sin(2 * np.pi * freq * k * t)
    env = np.minimum(1.0, t / 0.004) * np.exp(-t * (6.0 / max(dur, 0.05)))
    return sig / max(1e-9, np.abs(sig).max()) * amp


def render_track(bpm, root_pc: int, mode: str, bars: int) -> np.ndarray:
    """Render a synthetic EDM track. `bpm` may be a float or (start, end) —
    a tuple produces a continuously drifting tempo (live-recording proxy)."""
    total_beats = bars * 4
    if isinstance(bpm, tuple):
        b0, b1 = bpm
        _bt = [0.0]
        for i in range(total_beats):
            cur = b0 + (b1 - b0) * i / total_beats
            _bt.append(_bt[-1] + 60.0 / cur)
        _bt = np.asarray(_bt)

        def t_of(pos: float) -> float:
            i = int(pos)
            frac = pos - i
            i = min(i, total_beats - 1)
            return float(_bt[i] + (_bt[i + 1] - _bt[i]) * frac)

        duration = float(_bt[-1])
    else:
        beat = 60.0 / bpm
        t_of = lambda pos: pos * beat  # noqa: E731
        duration = total_beats * beat

    buf = np.zeros(int(duration * SR) + SR, dtype=np.float64)  # tail pad

    scale = MINOR_SCALE if mode == "minor" else MAJOR_SCALE
    third = 3 if mode == "minor" else 4
    root_midi = 36 + root_pc                       # bass root, C2-ish octave
    pad_midi = 60 + root_pc                        # chords/arp octave

    k = kick()
    h = hat()
    triad_rel = [0, third, 7]   # chord intervals above the root

    for bar in range(bars):
        t0 = t_of(bar * 4)
        breakdown = bars * 2 // 3 <= bar < bars * 2 // 3 + 8
        intro = bar < 8

        for b in range(4):                          # four on the floor
            if not breakdown:
                add(buf, t_of(bar * 4 + b), k)
            # offbeat hats everywhere but the intro
            if not intro:
                add(buf, t_of(bar * 4 + b + 0.5), h)

        if not breakdown:                           # bass: root/fifth 8ths
            for e in range(8):
                step = scale[0] if e % 4 != 3 else 7
                f = midi_hz(root_midi + (step if step <= 5 else 12))
                add(buf, t_of(bar * 4 + e * 0.5),
                    tone(f, 0.16, 0.5, harmonics=(1.0, 0.33, 0.15)))

        # chord stabs mark the bar: strong on the 1, weaker on the 3
        stab_amp = 0.6 if breakdown else 0.45
        for rel in triad_rel:
            add(buf, t0, tone(midi_hz(pad_midi + rel), 0.30,
                              stab_amp, harmonics=(1.0, 0.2)))
        if not breakdown:
            for rel in triad_rel:                   # same triad, octave up
                add(buf, t_of(bar * 4 + 2), tone(midi_hz(pad_midi + 12 + rel),
                                                 0.22, 0.25, harmonics=(1.0, 0.2)))

        if not intro and not breakdown and bar % 2 == 1:   # sparse arp
            for s in range(8):
                pc = scale[(s * 2) % len(scale)]
                add(buf, t_of(bar * 4 + s * 0.5),
                    tone(midi_hz(pad_midi + 12 + pc), 0.10, 0.10))

    peak = np.abs(buf).max()
    return (buf / peak * 0.9).astype(np.float32)


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else
               Path(__file__).resolve().parents[2] / "library" / "test")
    out.mkdir(parents=True, exist_ok=True)
    import soundfile as sf

    truth = []
    for name, bpm, root_pc, mode, bars in TRACKS:
        y = render_track(bpm, root_pc, mode, bars)
        f = out / f"{name}.wav"
        sf.write(str(f), y, SR)
        key = f"{PC_NAMES[root_pc]} {mode}"
        truth.append({"file": f.name, "bpm": bpm, "key": key})
        print(f"{f.name}: {bpm} BPM, {key}, {y.size / SR:.1f}s")

    (out / "ground_truth.json").write_text(json.dumps(truth, indent=2))
    print(f"ground truth → {out / 'ground_truth.json'}")


if __name__ == "__main__":
    main()
