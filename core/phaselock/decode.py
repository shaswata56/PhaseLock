"""Audio loading.

All analysis runs on mono float32 at a fixed rate (44.1 kHz) so curves and
beatgrids share one time base. soundfile handles wav/flac/mp3 via bundled
libsndfile; anything else (m4a/aac/ogg/opus) falls back to an ffmpeg pipe.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

ANALYSIS_SR = 44100
AUDIO_EXTS = {".mp3", ".flac", ".wav", ".aiff", ".aif", ".m4a", ".aac", ".ogg", ".opus"}

# librosa is imported lazily so `phaselock --help` stays fast.


class DecodeError(Exception):
    pass


def probe(path: str | Path) -> tuple[float, int, int]:
    """Return (duration_s, samplerate, channels) without decoding the file."""
    import soundfile as sf

    p = Path(path)
    if p.suffix.lower() in {".mp3", ".flac", ".wav", ".aiff", ".aif"}:
        info = sf.info(str(p))
        return float(info.duration), info.samplerate, info.channels
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=sample_rate,channels:format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(p)],
        capture_output=True, text=True, check=True,
    )
    try:
        sr, ch, dur = (x.strip() for x in out.stdout.strip().splitlines())
        return float(dur), int(sr), int(ch)
    except ValueError as e:
        raise DecodeError(f"ffprobe could not parse {p}: {e}") from e


def load_mono(path: str | Path, sr: int = ANALYSIS_SR) -> np.ndarray:
    """Decode to mono float32 at `sr`. Raises DecodeError on failure."""
    y, _ = _load(path, sr, mono=True)
    if y.size == 0:
        raise DecodeError(f"empty audio: {path}")
    return y


def load_stereo(path: str | Path, sr: int = ANALYSIS_SR) -> np.ndarray:
    """Decode to (2, n) float32 at `sr` (mono sources are duplicated)."""
    y, _ = _load(path, sr, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y])
    elif y.shape[0] > 2:
        y = y[:2]
    if y.size == 0:
        raise DecodeError(f"empty audio: {path}")
    return y.astype(np.float32)


def _load(path: str | Path, sr: int, mono: bool) -> tuple[np.ndarray, int]:
    import librosa

    p = Path(path)
    if p.suffix.lower() in {".m4a", ".aac", ".ogg", ".opus"}:
        # librosa/soundfile can't read these on Windows; pipe through ffmpeg.
        cmd = ["ffmpeg", "-v", "error", "-i", str(p), "-f", "f32le",
               "-acodec", "pcm_f32le", "-ac", "1" if mono else "2",
               "-ar", str(sr), "-"]
        raw = subprocess.run(cmd, capture_output=True, check=True).stdout
        if len(raw) % 4:
            raw = raw[: len(raw) - (len(raw) % 4)]
        y = np.frombuffer(raw, dtype=np.float32)
        return (y, sr)
    try:
        return librosa.load(str(p), sr=sr, mono=mono, res_type="soxr_hq")
    except Exception as e:  # noqa: BLE001 - surface a single failure type
        raise DecodeError(f"could not decode {p}: {e}") from e


def scan_audio_files(root: str | Path, recursive: bool = True) -> list[Path]:
    """All supported audio files under `root`, sorted for stable runs."""
    p = Path(root)
    it = p.rglob("*") if recursive else p.glob("*")
    return sorted(f for f in it if f.is_file() and f.suffix.lower() in AUDIO_EXTS)
