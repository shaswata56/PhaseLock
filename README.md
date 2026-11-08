# PhaseLock

**A personal, local-first AI DJ for EDM.** It listens to your library, plans the
journey, and plays the set — beat-matched, phrase-aware, stem-deep.

> A phase-locked loop is the circuit that syncs two oscillating signals until
> they hold one beat. DJing is the same act, performed by hand on two records.
> PhaseLock performs it on your library.

## What it does (when complete)

1. **Knows your library** — BPM, key (Camelot), energy curve, beatgrid,
   downbeats, waveform, per track. Computed once, cached forever.
2. **Plans the set** — orders tracks into a harmonic, energy-arc journey with
   transition points already chosen.
3. **Plays the set** — renders fully mixed sessions offline: beat-matched,
   phrase-aware transitions (bass swaps on the downbeat, EQ/filter automation).
4. **Takes tracks apart** — precomputed stems (Demucs) enable acapella
   overlays, bass swaps, drum-only drops.
5. **Has taste** — natural-language set curation ("90 min dark melodic techno,
   land at 126 BPM").

## Principles

- **Assemble, don't train.** Every hard ML piece is a pre-trained open-source
  model (Demucs, madmom, Essentia, librosa). The novelty is transition craft —
  which is code we control.
- **EDM-first.** Rigid 4/4, steady tempo, predictable 16/32-bar phrases,
  dramatic energy contours. The genre does half the mixing logic for us.
- **Local-first, $0 infra.** Your files, your machine, SQLite, no backend
  required. GPU time only for batch stem separation (optional, pennies).
- **Personal use.** Local files only — streaming APIs legally forbid
  third-party full-track playback. Never distribute separated stems.

## Stack

| Layer | Tech |
|---|---|
| Analysis core (Python) | librosa, madmom, Essentia, pyloudnorm, numpy |
| Mix renderer (Python) | numpy + Signalsmith Stretch (via binding or port) |
| Stems | Demucs / HTDemucs (batch, cached as FLAC/Opus) |
| App (later) | Next.js + Web Audio API |
| Storage | SQLite (index) + JSON sidecars (curves, grids) |

## Status

**All seven phases complete (2026-11-08) — every acceptance gate passed.**
See [PLAN.md](PLAN.md) for phase details and [RESUME.md](RESUME.md) for live
project state.

## Quick start

```bash
# one-time setup
python -m venv .venv
.venv/Scripts/python -m pip install -r core/requirements.lock
.venv/Scripts/python -m pip install -e core          # phaselock CLI
.venv/Scripts/python -m pip install demucs           # optional: stems

# analyze a music folder (mp3/flac/wav; m4a/ogg via ffmpeg)
.venv/Scripts/phaselock analyze "C:\path\to\music"

# plan + render a set
.venv/Scripts/phaselock plan --duration 60 --vibe "dark bpm 124-128 end 126"
.venv/Scripts/phaselock render data/sets/set_<stamp>.json --mp3 out.mp3

# optional: stem-separate tracks for true bass-swap transitions
.venv/Scripts/phaselock stems "C:\path\to\music\track.mp3"

# the booth (web app)
cd app && npm install && npm run build && npx next start -p 3100
# → http://localhost:3100
```

## Repo layout

```
phaselock/
├── README.md      # this file — vision + principles
├── PLAN.md        # overview plan + multiphase plan (all phases ✅)
├── RESUME.md      # continuously updated: state, next steps, decisions
├── core/          # Python package: analyze / plan / render / stems + CLI
│   └── tests/     # acceptance gate suites (test_*.py) + track generator
├── app/           # Next.js booth — library browser, plan UI, render jobs
├── data/          # phaselock.db + sidecar cache + sets  (gitignored)
├── library/       # YOUR music                        (gitignored)
├── stems_cache/   # separated stems                   (gitignored)
└── renders/       # rendered mixes                    (gitignored)
```
