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

### Phase 2 — The Brain (set generation)
**Goal:** `phaselock plan --vibe "<desc>" --duration 60` → ordered set with
transition points.
**Build:**
- Camelot harmonic-compatibility graph (±1 key steps, relative major/minor,
  energy-jump rules)
- Energy-arc profiler (opening → build → peak → landing) over the library
- BPM chaining (monotone-ish drift, max ±3–4% stretch per track)
- Transition point selection: last phrase boundary before a sensible mix-out
  (32/16 bars), informed by energy curve
- Output: set JSON (order, in/out points, transition types) + human-readable
  setlist
**Acceptance:**
- 60-min set from a 50+ track library: zero key clashes (Camelot-compatible
  throughout), BPM drift smooth (no jump > 4% at any transition)
- Two different vibes from the same library produce meaningfully different sets
**Est:** 1–2 sessions

### Phase 3 — The Hands (mix renderer) — MVP complete
**Goal:** `phaselock render <set.json>` → one continuous WAV/MP3 mix.
**Build:**
- Time-alignment + stretch to match tempos (Signalsmith Stretch; fallback
  librosa/soxr) with phase-lock to beatgrid
- Transition types v1: full mix-blend, bass swap on downbeat, EQ automation
  (low-cut sweep), filter sweep out of drops
- Phrase-aware placement: transitions start/land on 16/32-bar boundaries
- Gain riding: EBU R128 loudness matching between tracks
- Export: WAV master → optional MP3/Opus encode (ffmpeg)
**Acceptance:**
- Rendered 30-min set passes the blind-listen test: someone who DJs would not
  immediately call the transitions machine-made
- Zero audible stretch artifacts on sustained notes; no clipping
  (true peak < −1 dBTP)
- Deterministic: same set JSON → byte-identical WAV
**Est:** 3–5 sessions (the craft lives here; expect iteration)

### Phase 4 — The Anatomy (stems)
**Goal:** stem-aware library and transitions.
**Build:**
- Demucs/HTDemucs batch separation (local GPU if available, else CPU or
  Cloud Run spot job — < $1 per full library)
- Stem cache: 4 stems/track as FLAC (or Opus to halve storage), manifest with
  source hash → invalidate on re-analyze
- Stem transitions v2: acapella overlay, drum-only drop, bass-swap across
  differing keys
**Acceptance:**
- One bass-swap transition between two different-key tracks rendered with
  stems, no vocal/bleed artifacts
- Cache invalidation works (re-analyze → stems marked stale, not silently
  reused)
**Est:** 1–2 sessions

### Phase 5 — The Booth (web app)
**Goal:** Next.js app: browse library, see waveforms/curves, queue sets,
play + export.
**Build:**
- Library browser (filters: key, BPM, energy), track detail with waveform +
  energy overlay
- Set builder: drag/edit a generated set, transition markers
- Playback of rendered mixes; live A/B of transition variants
- Stretch (only if core is solid): dual-deck live mode with stem toggles
  (Web Audio scheduling, Chris Wilson lookahead pattern)
**Acceptance:**
- Full loop in browser: analyze → plan → render → listen without touching a CLI
**Est:** ongoing

### Phase 6 — The Taste (LLM + generative layer)
**Goal:** natural-language curation; optional generative textures.
**Build:**
- LLM set curation over the analyzed library metadata ("dark melodic techno,
  90 min, end at 126 BPM") → set JSON in Phase 2's format
- Optional: MusicGen/Riffusion-generated transition textures, one-shots
**Acceptance:**
- Vibe prompt → set that a human agrees matches the prompt
**Est:** 1 session

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
| M1 | It knows your library | 1 |
| M2 | It plans the journey | 2 |
| M3 | **It plays the set (MVP)** | 3 |
| M4 | It takes the track apart | 4 |
| M5 | It lives in the browser | 5 |
| M6 | It has taste | 6 |
