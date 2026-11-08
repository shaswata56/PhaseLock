# PhaseLock — RESUME

*Live project state. Update at the end of every session. A new session starts
by reading this file.*

## Current state (2026-11-09, early hours — v2.8)

**Status: v1 + v2 + musically-verified mashup (v7) complete. Key detector fixed.**

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

**v2.1 (listening-feedback pass):** user's verdict "some good, some wasn't" →
diagnosis: 17/19 joins at drift/phase risk (live Voice recording drifts 1.4%,
δ up to ±209 ms). Engine now: local-tempo overlap regions, measured
kick-comb phase alignment per join (residual-verified, demote-to-sweep when
unsalvageable), 16-beat blends, drum_drop outgoing HP-sweep, tamed acapella.
**Showcase v2 rendered:** `renders/showcase_60min_v2.mp3` — 18/19 joins
corrected (up to ±209 ms), 9 stem swaps, −1.00 dBFS. Alignment gates
(test_alignment.py: 100 ms grid-error → 0.3 ms residual) + full regression
green. The v1 mix stays at `renders/showcase_60min.mp3` for A/B.

**v2.2 (manual mix studio, on user request "mix any with any for anytime"):**
booth MANUAL SET panel — `+` adds any library track, reorder, per-track
minutes 0.5–15, transition type + beats, compat badges (clash/bpm-gap warn,
never block) → build → per-join type editing → render. Backed by
`phaselock build` CLI + `planner.build_manual_set` (manual tempo limits ±8%
stretch / ±6% drift, clamps + warns on extreme gaps). Gates
`test_build.py` pass; live-verified hand-picked pair → render → stream.

**v2.3 (mashup, on user request "ultimate test"):** `phaselock mashup` —
instrumental bed from one base track's stems + guest vocals from donors,
energy-ranked phrase-anchored windows, anchor-phrase warp, uniform
mix-tempo section grid, per-section slots. Gates `test_mashup.py`
(G1 finite/duration, G2 placements, G3 determinism). Artifact:
`renders/mashup_zedd_bounce_days.mp3`.

**v2.4 (VAULT):** booth renders panel lists `renders/` with inline audio
players (auto-refresh on job completion); Range-streaming API with strict
filename validation.

**v2.5 (listening rig, on user challenge "you need a way to listen"):**
`phaselock listen AUDIO --start S --dur D -o png` renders a 4-panel view
(mel spectrogram + downbeat ticks, waveform + onsets, sub/vocal-band dB,
highs/low-mid dB) + machine notes (clicks, kick-band flams, band balance)
— `rendering/listen.py`. **First honest listen of the mashup exposed two
real flaws:** per-stem section gains defined but never applied (one summed
envelope → drop1−build1 sub = −0.7 dB, inverted), and the bed read the
source linearly so mashup drops sat on the source's breakdowns. **v3
fixes:** per-stem per-section mixing (drops bass ×1.7, builds duck+ramp,
melody stem HP @90 Hz), phrase-sampled bed (`_plan_bed_windows` — source
phrases ranked by sub energy; drops get the source's biggest moments).
Verified by listening: drop1−build1 **+16.7 dB**, drop1−break +15.3 dB,
drops consistent ≤2.2 dB, 0 clicks; vision-model spectrogram read
independently confirms (on-grid single kicks, drop energy jump, break
sub dropout, vocal formants, no clipping).

**v2.6 (reference-driven rebuild, after user's "not a good mix"):** user
supplied the reference — Million Voices x Calling (Tiesto mashup). Torn
down with the listening rig + yt-dlp + vision reads: two-act arc, rising
builds, ONE breakdown with the vocal holding, final = single biggest
moment (+9.4 dB over drop 1), continuous phrase-run topline, hard cuts.
v4: `renders/mashup_bounce_iwytk_v4.mp3` (Bounce bed x I Want You To Know
topline, 3:11). Renderer: run/solo/callback vocal roles, noise risers,
enforced section ladder (post-limiter calibration — the limiter eats a
preset ladder unevenly), strong-open window filter. Ladder verified = ref
(drop1 −9.4, break −19.2, final 0), 0 clicks, LUFS −10. Tools:
`core/tools/band_scan.py` + `band_diff.py` (any-track structure diffs
vs a reference). Known residuals: internal window dips (source
arrangement shows through), bass sits higher vs the ref's sub-focused
master.

**v2.7 (v6, after "not even close"):** vision-diffed my drop vs the ref
TEXTURE — the real gap was groove/integration: smeared kicks, un-pumped
sub wall, vocal pasted on top, stitching seams. v6 = `mashup_ref.py`:
assemble the reference arc FROM THE SOURCE'S OWN PARTS (contiguous high
runs; final = its biggest run, build = its own riser, break = the final
beat-stripped), continuous native-rate slices (zero WSOLA on the bed),
rebuilt beat-locked sidechain pump, glued topline (EQ + plate reverb,
L2-normalized IR — RMS-norm IR = 235x convolution bomb), reference
ladder down-only, pair picked by full-library template match (Find You x
We'll Be Coming Back, 6A/6A, same scene). `renders/mashup_findyou_wbbc_
v6.mp3`. Vision re-pass: crisp on-grid kicks, pumping sub, integrated
vocal, clean downbeat slam, no seams. Debug trail banked: limiter was
innocent (spy instrumentation found +43dBFS reverb spikes); dB sentinels
must be None not >0; fixed-window part pickers straddle gaps — use
contiguous runs.

**v2.8 (after "you're still not getting what you're creating"):** first
genuinely musical question — what NOTES does the topline sing over the
bed's CHORDS — exposed that v6 ground a semitone (WBBC's dominant G
natural over a G#-centered bed) because the Phase-1 key detector was
wrong for 25/79 tracks (cosine on uncentered chroma + middle-only
window). key.py fixed (Pearson + two-window chroma, verified vs known
keys), library re-keyed, and pair selection now runs a bar-level
consonance check (donor melody PCs vs the bed's measured chord loop)
BEFORE rendering. v7: Find You (Eb maj) × This Is What It Feels Like
(Ab maj, 0.73 consonance) — `renders/mashup_findyou_tiwifl_v7.mp3`,
0 clicks, LUFS −14.6, vision clean w/ distinct register lanes.

**Booth:** rebuilt + live at http://localhost:3100 (all v2 APIs verified:
curves, preview 200/206, plan/edit, build).

## Next steps (v3 ideas in PLAN.md — nothing pending)

- **A/B listen: `showcase_60min_v2.mp3` vs the v1** — the corrected joins
  are #2, #5, #10, #12, #15, #17, #18 (largest corrections)
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
