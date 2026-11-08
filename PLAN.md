# PhaseLock — Plan

## Overview plan

**Product:** PhaseLock is a personal AI DJ + mixing engine for EDM. It turns a
folder of local music files into (a) an analyzed library, (b) ordered DJ sets,
and (c) rendered, fully-mixed sessions — with optional stem-level control.

**MVP definition (Phases 1–3):** point it at a music folder, get back a
rendered 60-minute beat-matched mix with phrase-aware transitions. Everything
after that (stems, web app, LLM taste) deepens an already-working core.

**Architecture in one line:** Python core (analysis → set planning → offline
render) + SQLite/JSON cache; Next.js + Web Audio front-end bolted on after the
core proves itself. No servers, no accounts, no cloud dependency.

**Why this is tractable:**
- All ML components are pre-trained and open source — zero training.
- EDM is the easiest genre to auto-mix (4/4, steady BPM, 16/32-bar phrases,
  obvious energy structure).
- The genuinely novel value (transition craft) is deterministic code, not ML.

**Standing constraints:**
- Local audio files only. No streaming-API playback (legally forbidden).
- Personal use only — no redistribution of stems or mixes of commercial tracks.
- Big media (audio, stems) never enters git.
- Every session updates RESUME.md before it ends.

---

## Multiphase plan

Each phase ends at a usable milestone. Acceptance criteria are hard gates —
we don't advance until they pass on real tracks.

### Phase 0 — Scaffold ✅
**Goal:** repo, plans, identity, structure.
- [x] Name + directory (`phaselock`, "beat-matching = phase-locking two signals")
- [x] README (vision/principles), PLAN (this file), RESUME, .gitignore
- [x] git init with repo-local identity (shaswata56 — global is the forbidden
      orbitax identity)
- [ ] Python venv + pinned deps (deferred to Phase 1 kickoff)

### Phase 1 — The Ears (analysis core) ✅ (2026-11-08)
**Goal:** `phaselock analyze <folder>` builds the library index.
**Built:**
- [x] Ingest: mp3/flac/wav native; m4a/aac/ogg/opus via ffmpeg pipe
- [x] Per track: BPM + beatgrid + downbeats, key → Camelot (KS profiles),
      energy curve (RMS dB/0.5 s + novelty), waveform peaks, LUFS
- [x] Storage: SQLite + JSON sidecars, change detection (size+mtime),
      resumable, human key overrides survive re-analysis
- [x] CLI: analyze / list / stats / override
**Acceptance — PASSED:**
- 10 synthetic ground-truth tracks: BPM 10/10 within ±0.5% (worst 0.16%),
  key 9/10 (one relative major/minor confusion — known KS behavior,
  override covers it), beatgrids 10/10 monotonic + tempo-consistent
- Real-track sanity: 15 commercial mp3s, 0 errors, BPMs plausible vs known
  values, LUFS/keys sensible
**Est:** 2–4 sessions → took 1

Implementation notes: madmom does not build on Python 3.13 (Cython) →
librosa backend with two additions that met the gate: integer-index beat
regression (kills ~±1% frame-snap quantization) and onset-envelope ACF
cross-check with uniform-grid rebuild (fixes 4/3 metrical-level locks).
Sub-frame BPM precision comes from the regression, not librosa's estimate.

### Phase 2 — The Brain (set generation) ✅ (2026-11-08)
**Goal:** `phaselock plan --vibe "<desc>" --duration 60` → ordered set with
transition points.
**Built:**
- [x] Camelot compatibility graph (same/relative 1.0, adjacent 0.9,
      same-tonic M↔m 0.55 "energy boost", else clash)
- [x] Energy-arc profiler (library percentile + per-vibe arc shapes)
- [x] BPM chaining via deterministic beam search: no clashes, drift ≤3%/step,
      stretch ≤4% from native
- [x] Phrase-aligned in/out points (16-beat grid anchored to downbeat phase)
- [x] Transition typing (blend / bass_swap / filter_sweep) + set JSON +
      readable setlist; vibe presets: auto/peak/warmup/melodic/dark/closing
**Acceptance — PASSED:** camelot unit 10/10; zero clashes + bounded drift/
stretch; phrase-aligned points; peak vs warmup produce different sets;
deterministic. Verified on mixed real+synthetic library (99-track-ready).
**Est:** 1–2 sessions → took 1

### Phase 3 — The Hands (mix renderer) — MVP COMPLETE ✅ (2026-11-08)
**Goal:** `phaselock render <set.json>` → one continuous WAV/MP3 mix.
**Built:**
- [x] WSOLA time-stretch, single pass with piecewise-constant rate: fixed
      synthesis hop + continuous analysis cursor → sample-exact beat mapping
      across rate changes; correlation offsets from mono reference, applied
      to all channels
- [x] Phase-locked timeline: deck k's final transition beats run at deck
      k+1's mix BPM; the incoming downbeat lands sample-exact on the rate
      change (downbeat on downbeat)
- [x] Transitions: blend (low band swaps by midpoint), bass_swap (LR4
      crossover, hard low swap on the beat), filter_sweep (swept HP exit)
- [x] LUFS gain riding per deck, look-ahead limiter at −1 dBFS, 24-bit WAV
      + optional MP3 encode
**Acceptance — PASSED:** peak −1.00 dBFS / true-peak −0.98 dBTP, −11.4 LUFS;
duration within 0.5 s of plan; byte-identical re-render; onset period in
every overlap = overlap tempo within 1.5%; solo regions at planned tempo.
End-to-end verified on real stereo mp3s + synthetic WAVs.
**Est:** 3–5 sessions → took 1

### Phase 4 — The Anatomy (stems) ✅ (2026-11-08)
**Goal:** stem-aware library and transitions.
**Built:**
- [x] Demucs/htdemucs separation (CPU on this machine; CUDA auto if
      present), stems cached as FLAC + manifest with size/mtime staleness
- [x] `phaselock stems PATH...` CLI; stale caches reported, never reused
- [x] Stem bass_swap: bass-stem decks warped with the SAME WSOLA deltas as
      the full decks (identical time-warp) replace the low band; LR4 keeps
      spectral consistency; automatic crossover fallback without stems
**Acceptance — PASSED:** manifest+4 stems cached & fresh-check; simulated
source change → stale; bass_swap between different-key tracks rendered from
real stems (engine reports the swap, −1.00 dBFS, finite); fallback path
verified without stems.
**Est:** 1–2 sessions → took 1

### Phase 5 — The Booth (web app) ✅ (2026-11-08)
**Goal:** Next.js app: browse library, see waveforms/curves, queue sets,
play + export.
**Built:**
- [x] `app/` Next.js 16 + Tailwind: dark booth UI — sortable/filterable
      library table, plan panel (duration + vibe presets + DSL input),
      setlist with transition typing, render button → live job log →
      in-browser player + WAV download
- [x] API routes shell out to the `phaselock` CLI (one source of truth):
      GET /api/tracks (60 s cache), POST /api/plan, POST /api/render
      (background job), GET /api/jobs (status + mix streaming; paths
      validated against data/sets and renders/)
- [x] Full loop verified live: plan → job → 57 MB WAV streamed over HTTP
**Deferred:** waveform/energy-curve detail view and per-track preview
streaming (v2 booth polish); live dual-deck mode remains the stretch goal.
**Est:** ongoing → v1 took 1 session

### Phase 6 — The Taste (vibe DSL + LLM hook) ✅ (2026-11-08)
**Goal:** natural-language curation without a hard external dependency.
**Built:**
- [x] Vibe DSL: presets (auto/peak/warmup/melodic/dark/closing) + directives
      `bpm A-B`, `end NNN`, `start NNN`, modifiers hype/chill/soft/minor/
      major — unknown words ignored, so free-text prompts work
- [x] Optional LLM hook: `PHASELOCK_LLM_CMD` receives {library, prompt} on
      stdin, returns a path order; planner honors it after compatibility
      repair (clashing/incompatible entries dropped, never mixed through);
      any hook failure falls back silently to the built-in beam search
- [x] `phaselock plan --llm`
**Acceptance — PASSED:** DSL parsing gates; disjoint DSL windows steer
selection; fake LLM hook (reverse order) honored end-to-end with zero
clashes after repair; absent hook falls back.
**Est:** 1 session → took <1

---

## Non-goals (explicit)

- ❌ Streaming-service playback (Spotify/Apple forbid it; dead end)
- ❌ Real-time stem separation in the browser (precompute instead)
- ❌ Turntable/scratch emulation
- ❌ Multi-user / accounts / cloud sync
- ❌ Training or fine-tuning any model
- ❌ Publishing or distributing mixes/stems of commercial tracks

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| madmom deps painful on Windows | Essentia/librosa fallbacks; pin versions early (Phase 1) |
| Time-stretch artifacts on EDM pads/pianos | Signalsmith Stretch (best OSS quality); A/B gate in Phase 3 acceptance |
| Key detection accuracy (~85%) | Human override field in schema from day 1 |
| Stem storage growth (~50 MB/track) | Opus stems option; only separate the working subset |
| Transition quality plateau ("podcast crossfade") | Phrase rules + energy-curve conditioning; blind-listen gate each iteration |

## Milestones at a glance

| # | Milestone | Phase |
|---|---|---|
| M0 | Scaffold | 0 ✅ |
| M1 | It knows your library | 1 ✅ |
| M2 | It plans the journey | 2 ✅ |
| M3 | **It plays the set (MVP)** | 3 ✅ |
| M4 | It takes the track apart | 4 ✅ |
| M5 | It lives in the browser | 5 ✅ |
| M6 | It has taste | 6 ✅ |

**All seven phases complete 2026-11-08.** Every phase passed its acceptance
gates; full regression (`core/tests/test_*.py`) is green.

## v2 (2026-11-08, same day)

- [x] Full 99-track real library analyzed (98 ok, 1 unicode filename lost in copy)
- [x] **drum_drop** transition: 16-beat drum-stem build over the outgoing,
      hard drop to incoming full on the downbeat (needs stems; degrades to
      filter_sweep)
- [x] **acapella_overlay** transition: vocals stem rides a half-sine window
      over a level blend (needs stems; degrades to blend)
- [x] Planner assigns stem moves when the incoming track has stems
- [x] Booth v2: track detail view (canvas waveform + energy curve + downbeat
      ticks), per-track preview streaming (Range/206, DB-validated paths),
      per-transition type editing with save
- [x] v2 transition gates + full regression green
- [ ] Showcase 60-min all-real-music set rendered with stems (running)

## v3 ideas (unplanned)

- Booth: live dual-deck Web Audio mode with stem toggles
- MusicGen/Riffusion transition textures
- Opus stem cache option; Cloud Run GPU batch separation
- Library dedupe by audio fingerprint, not path
