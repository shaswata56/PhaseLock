"""Structure scan: per-2s band energies in dB + novelty → section table."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core"))
import numpy as np
import soundfile as sf

path = sys.argv[1]
y, sr = sf.read(path, always_2d=True)
if y.shape[1] == 1:
    y = np.tile(y, (1, 2))
y = y.T.mean(axis=0)  # mono

def band_db(sig, sr, lo, hi, win_s=2.0):
    n = int(win_s * sr)
    m = len(sig) // n
    out = []
    for i in range(m):
        seg = sig[i*n:(i+1)*n]
        sp = np.fft.rfft(seg * np.hanning(n))
        freqs = np.fft.rfftfreq(n, 1/sr)
        mask = (freqs >= lo) & (freqs < hi)
        out.append(10*np.log10(np.mean(np.abs(sp[mask])**2) + 1e-12))
    return np.array(out)

sub = band_db(y, sr, 30, 120)
low = band_db(y, sr, 120, 500)
voc = band_db(y, sr, 500, 4000)
hi  = band_db(y, sr, 4000, 16000)

print(f"{Path(path).name}: {len(y)/sr:.1f}s @ {sr}Hz")
print("t(s)   sub   low   voc   high | profile")
for i in range(len(sub)):
    t = i * 2
    tot = sub[i] + low[i]
    tag = ""
    if sub[i] > 25: tag += " SUB-ON"
    if voc[i] > 14: tag += " VOCAL"
    if hi[i] > 6: tag += " HISS/HAT"
    print(f"{t:4d}  {sub[i]:5.1f} {low[i]:5.1f} {voc[i]:5.1f} {hi[i]:5.1f} |{tag}")
