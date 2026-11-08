# PhaseLock — RESUME

*Live project state. Update at the end of every session. A new session starts
by reading this file.*

## Current state (2026-11-08, night — v2 COMPLETE)

**Status: v1 (all 7 phases) + v2 complete. Showpiece mix rendered.**

**v2 delivered:**
- Full real library: 110 tracks (98 of 99 Liked Songs; 1 unicode filename
  lost in copy; 15 duplicate real_sample rows removed).
- New transitions: drum_drop (drum-stem build → hard drop on the downbeat),
  acapella_overlay (vocals stem over a blend); graceful degradation; planner
  assigns by stems availability. Booth v2: detail view (waveform + energy +
  downbeat ticks), Range-streaming preview, transition editing.
- **Showcase mix rendered:** `renders/showcase_60min.wav` (+ 70 MB MP3) —
  55.5 min, 20 real tracks, Rasputin → Forbidden Voices, Camelot
  11A→8B→3A, energy 0.23→0.99, 19 transitions (3 drum_drop, 6
  acapella_overlay), **11 real stem swaps**, −1.00 dBFS, −11.2 LUFS,
  finite, deterministic. 12 tracks stem-separated (~36 min CPU).

**Booth:** rebuilt + live at http://localhost:3100 (all v2 APIs verified:
curves, preview 200/206, plan/edit).

## Next steps (v3 ideas in PLAN.md — nothing pending)

- **Listen to `renders/showcase_60min.mp3`** — the blind-listen gate is the
  one human check left
- Live dual-deck Web Audio booth; MusicGen textures; Opus stems; GPU batch

**What exists:**
- **Analysis core** (`core/phaselock/`): BPM (sub-frame via beat-index
  regression + onset-ACF cross-check with grid rebuild), key→Camelot
  (Krumhansl–Schmuckler + human override), beatgrid/downbeats, energy curve
  + LUFS + novelty, waveform peaks. SQLite + JSON sidecars, change detection.
- **Planner**: Camelot compat graph, deterministic beam search over energy
  arcs, phrase-aligned in/out points, transition typing, vibe DSL
  (presets + `bpm A-B`/`end NNN`/modifiers), optional LLM ordering hook
  (`PHASELOCK_LLM_CMD`) with compatibility repair.
- **Renderer**: single-pass variable-rate WSOLA (sample-exact beat mapping,
  delta-shared stem decks), phase-locked downbeat-on-downbeat timeline,
  blend/bass_swap/filter_sweep transitions (LR4 crossover), stem bass swaps
  from Demucs htdemucs cache, LUFS gain riding, look-ahead limiter,
  byte-identical determinism.
- **Stems**: `phaselock stems` CLI, FLAC cache + manifest staleness.
- **Booth** (`app/`): Next.js 16 — library table (sort/filter), plan panel
  (duration/vibe/DSL), setlist with transition arrows, background render
  jobs with live log, in-browser player + WAV download. All API routes
  shell to the CLI.

**Verified:** analysis gates (BPM 10/10 ±0.5%, key 9/10), planning gates,
render gates (no-clip −0.98 dBTP, −11.4 LUFS, byte-identical, phase-locked
overlaps), stems gates (fresh/stale, real stem swap + fallback), vibe gates
(DSL, steering, LLM hook + repair + fallback); real-mp3 end-to-end render;
full web loop (plan → job 5.9s → 57 MB WAV streamed).

**Data:** `data/phaselock.db` — 27 tracks (10 synthetic ground truth, 2 stem
tests, 15 real Liked-Songs sample). User's full library: 99 mp3s at
`~/Downloads/Liked Songs` — analyze wholesale any time.

**The booth is (probably) still running:** `http://localhost:3100`
(started with `npx next start -p 3100` from `app/`).

## Next steps (v2 ideas — nothing is pending)

- Analyze the full 99-track Liked Songs library; render a real 60-min set
- Booth polish: waveform/energy detail view, per-track preview, transition
  editing, live dual-deck Web Audio mode
- Drum-only drops + acapella overlays (stems cached, transitions unbuilt)
- Optional: MusicGen textures, Opus stems, Cloud Run GPU batch

## Decision log

| Date | Decision | Why |
|---|---|---|
| 2026-11-08 | Name: PhaseLock | PLL metaphor; beat-matching is phase-locking. Alternates: Dropwise, Selector, Downbeat. |
| 2026-11-08 | Location: `C:\workstation\phaselock` | Workstation = product workspace (chonky.cam precedent). |
| 2026-11-08 | Repo-local git identity: `shaswata56 <shaswata56@users.noreply.github.com>` | Global identity is the forbidden orbitax one; set before first commit. |
| 2026-11-08 | MVP = Phases 1–3; Python core before web app | Render quality is the credibility gate; UI wraps a proven core. |
| 2026-11-08 | Remote: `git@github.com:shaswata56/PhaseLock.git` (SSH) | User-created; push after every change. |
| 2026-11-08 | librosa backend, not madmom | madmom won't build on py3.13. Compensated: beat regression + ACF rebuild. |
| 2026-11-08 | Own KS key detector, no Essentia | Essentia has no Windows wheels; override + top-3 candidates cover misses. |
| 2026-11-08 | Own WSOLA instead of Signalsmith/binding | No pip wheel for Python; ≤4% ratios are WSOLA-transparent; single-pass variable-rate gives sample-exact grids. |
| 2026-11-08 | Test material = seeded synthetic generator | Exact ground truth for precision gates; real mp3s are the sanity layer. |
| 2026-11-08 | Booth shells out to the CLI | One source of truth; no analysis logic duplicated in JS. |
| 2026-11-08 | LLM = optional external hook, not a dependency | $0/offline by default; the LLM refines order only, compat repair keeps it safe. |
| 2026-11-08 | data/ fully gitignored | Derived artifacts (db, sidecars, sets); accidentally committed once, untracked. |

## Conventions

- Audio/stems/renders/data never committed.
- Determinism gate: same set JSON → byte-identical WAV.
- Repo-local git identity only.
- Run gates: `.venv/Scripts/python.exe core/tests/test_{analysis,planning,render,stems,vibe}.py`
  (regenerate `library/test/` via `core/tests/gen_test_tracks.py` if missing).
