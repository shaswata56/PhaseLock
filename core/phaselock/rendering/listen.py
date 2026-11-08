"""The listening rig: render audio as spectrograms the agent can SEE.

I can't hear, but I can read images. This module turns audio into the
visuals a mastering engineer would stare at — log-frequency spectrogram,
waveform with onsets, band energies — plus machine-readable listening
notes: clicks, flammed transients, section/band balance. It exists so
"verified" means listened-to, not just measured.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SR = 44100


def listening_notes(y: np.ndarray, sr: int) -> dict:
    """Machine notes on what a section 'sounds like' — flags, not verdicts."""
    import librosa

    mono = y.mean(axis=0) if y.ndim > 1 else y
    notes: dict = {"duration_s": round(mono.size / sr, 1)}

    # clicks: sample-to-sample jumps no music makes
    d = np.abs(np.diff(mono))
    thresh = max(0.25, float(np.quantile(d, 0.9999)) * 3)
    idx = np.where(d > thresh)[0]
    notes["clicks"] = [round(float(i) / sr, 3) for i in idx[:12]]
    notes["click_count"] = int(idx.size)

    # flammed transients: onset peak pairs 25-160 ms apart (double kicks)
    oe = librosa.onset.onset_strength(y=mono, sr=sr, hop_length=512)
    peaks = librosa.util.peak_pick(oe, pre_max=6, post_max=6, pre_avg=12,
                                   post_avg=12, delta=0.4, wait=8)
    times = peaks * 512 / sr
    flams = []
    for a, b in zip(times, times[1:]):
        if 0.025 <= b - a <= 0.16:
            flams.append((round(float(a), 2), round(float(b), 2)))
    notes["flam_pairs"] = flams[:10]
    notes["flam_count"] = len(flams)

    # band balance (rough ears: where does the energy live?)
    S = np.abs(librosa.stft(mono, n_fft=2048)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    def band(lo, hi):
        m = (freqs >= lo) & (freqs < hi)
        return float(10 * np.log10(S[m].mean() + 1e-12))
    notes["bands_db"] = {
        "sub_30_120": round(band(30, 120), 1),
        "lowmid_120_500": round(band(120, 500), 1),
        "vocal_500_4k": round(band(500, 4000), 1),
        "high_4k_16k": round(band(4000, 16000), 1),
    }
    return notes


def render_view(path: str, out_png: str, start_s: float = 0.0,
                dur_s: float | None = None, title: str | None = None
                ) -> dict:
    """Render spectrogram + waveform + band view of a window to PNG.

    Returns listening notes for the same window.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import librosa
    import librosa.display

    y, sr = librosa.load(path, sr=SR, mono=False,
                         offset=start_s, duration=dur_s)
    mono = y.mean(axis=0) if y.ndim > 1 else y

    S = librosa.feature.melspectrogram(y=mono, sr=sr, n_fft=2048,
                                       hop_length=512, n_mels=128,
                                       fmax=16000, power=2.0)
    Sdb = librosa.power_to_db(S, ref=np.max)
    oe = librosa.onset.onset_strength(y=mono, sr=sr, hop_length=512)
    onsets = librosa.onset.onset_detect(onset_envelope=oe, sr=sr,
                                        hop_length=512, units="time",
                                        backtrack=False)
    tempo, beats = librosa.beat.beat_track(onset_envelope=oe, sr=sr,
                                           hop_length=512, trim=False)
    beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=512)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    Sfull = np.abs(librosa.stft(mono, n_fft=2048)) ** 2

    def band_db(lo, hi):
        m = (freqs >= lo) & (freqs < hi)
        tdb = 10 * np.log10(Sfull[m].mean(axis=0) + 1e-12)
        # decimate for plotting
        k = max(1, tdb.size // 800)
        return tdb[:: k]

    fig, axes = plt.subplots(
        4, 1, figsize=(15, 11), sharex=True,
        gridspec_kw={"height_ratios": [6, 2, 1.4, 1.4]},
    )
    fig.suptitle(title or Path(path).name, fontsize=13, color="white")

    ax = axes[0]
    librosa.display.specshow(Sdb, sr=sr, hop_length=512, x_axis="time",
                             y_axis="mel", ax=ax, cmap="magma")
    ax.set_ylabel("mel freq")
    for bt in beat_times[::4]:
        ax.axvline(bt, color="cyan", alpha=0.25, lw=0.6)

    ax = axes[1]
    t = np.arange(mono.size) / sr
    k = max(1, mono.size // 6000)
    ax.plot(t[::k], mono[::k], lw=0.4, color="lime")
    for ot in onsets:
        ax.axvline(ot, color="orange", alpha=0.5, lw=0.8)
    ax.set_ylabel("wave")

    ax = axes[2]
    ax.plot(band_db(30, 120), color="red", label="sub")
    ax.plot(band_db(500, 4000), color="dodgerblue", label="vocal band")
    ax.legend(loc="upper right", fontsize=7)

    ax = axes[3]
    ax.plot(band_db(4000, 16000), color="violet", label="highs")
    ax.plot(band_db(120, 500), color="orange", label="low-mid")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_xlabel("time (s)")

    for ax in axes:
        ax.set_facecolor("#0a0a0a")
        ax.tick_params(colors="gray", labelsize=7)
        for sp in ax.spines.values():
            sp.set_color("#333")
    fig.patch.set_facecolor("#111111")
    for ax in axes:
        ax.title.set_color("white")
        ax.xaxis.label.set_color("gray")
        ax.yaxis.label.set_color("gray")
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)

    return listening_notes(y, sr)
