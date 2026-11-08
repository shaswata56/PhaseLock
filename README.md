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

**Phase 0 — scaffold (this repo).** See [PLAN.md](PLAN.md) for the full
multiphase plan and [RESUME.md](RESUME.md) for live project state.

## Repo layout

```
phaselock/
├── README.md      # this file — vision + principles
├── PLAN.md        # overview plan + multiphase plan
├── RESUME.md      # continuously updated: state, next steps, decisions
├── core/          # Python analysis + rendering engine (Phase 1+)
├── app/           # Next.js web app (Phase 5+)
├── library/       # YOUR music — gitignored
└── stems_cache/   # separated stems — gitignored (large!)
```
