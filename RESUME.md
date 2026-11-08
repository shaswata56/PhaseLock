# PhaseLock — RESUME

*Live project state. Update at the end of every session. A new session starts
by reading this file.*

## Current state (2026-11-08)

**Phase:** 1 complete (The Ears — analysis core, all gates passed). Next:
**Phase 2 kickoff — The Brain (set generation)**.

**Done this session:**
- venv at `.venv` (Python 3.13.4), pinned deps in `core/requirements.lock`,
  package installed editable (`pip install -e core` → `phaselock` CLI).
- Full analysis core under `core/phaselock/`: decode (mp3/flac/wav native,
  ffmpeg pipe for the rest), tempo (librosa + regression + ACF rebuild),
  key (Krumhansl–Schmuckler → Camelot), energy (RMS curve, LUFS, novelty),
  waveform peaks; SQLite + JSON sidecars with change detection.
- Synthetic test rig: `core/tests/gen_test_tracks.py` (10 exact-ground-truth
  EDM tracks) + `core/tests/test_analysis.py` (the acceptance gates).
- **Gates: BPM 10/10 within ±0.5% (worst 0.16%), key 9/10 (gate ≥8),
  beatgrids 10/10.** Real sanity: 15 Liked-Songs mp3s, 0 errors, plausible
  values throughout.

**Next steps (Phase 2 kickoff):**
1. Camelot compatibility graph (±1 steps, relative major/minor, energy-jump
   rules) — pure function + unit tests.
2. Energy-arc profiler over the library (read sidecar curves; intro→build→
   peak→land).
3. BPM chaining (monotone-ish drift, ≤ ±4% stretch), transition point
   selection at phrase boundaries (32/16 bars from downbeat grid).
4. `phaselock plan --duration 60` → set JSON + readable setlist; acceptance:
   zero key clashes, no BPM jump > 4% at transitions, two vibes → two
   meaningfully different sets.

**Data on disk:** `data/phaselock.db` (25 tracks: 10 synthetic + 15 real
sample), sidecars in `data/cache/`. `library/test/` regenerable via the
generator; `library/real_sample/` holds 15 of the user's 99 Liked Songs
(full library at `~/Downloads/Liked Songs` — analyze wholesale when useful).

**Blockers/questions for user:** none.

## Decision log

| Date | Decision | Why |
|---|---|---|
| 2026-11-08 | Name: PhaseLock | PLL metaphor; beat-matching is phase-locking. Alternates considered: Dropwise, Selector, Downbeat. |
| 2026-11-08 | Location: `C:\workstation\phaselock` | Workstation = product workspace (chonky.cam precedent); `~/work` reserved for hunts. |
| 2026-11-08 | Repo-local git identity: `shaswata56 <shaswata56@users.noreply.github.com>` | Global identity is the forbidden orbitax one; noreply address is tied to the personal account and exposes nothing. Set BEFORE first commit (per sites-repo lesson). |
| 2026-11-08 | MVP = Phases 1–3 (analyze → plan → render) | Stems/app/taste all build on a working render core; render is the credibility gate. |
| 2026-11-08 | Python core before web app | Render quality is the hard part; UI last so it wraps a proven core. |
| 2026-11-08 | Remote: `git@github.com:shaswata56/PhaseLock.git` (SSH) | User-created repo; SSH = personal key per convention. Push after every change. |
| 2026-11-08 | librosa backend, not madmom | madmom won't build on Python 3.13 (Cython). Compensated: beat-index regression (sub-frame BPM) + ACF cross-check/rebuild (metrical-level fixes). madmom stays an optional backend if the Python version ever allows. |
| 2026-11-08 | Own KS key detector, no Essentia | Essentia has no Windows wheels; KS profiles are textbook and passed the gate. Known limitation: relative major/minor confusion (1/10 synthetic misses) → `key_override` + sidecar top-3 candidates cover it. |
| 2026-11-08 | Test material = synthetic generator with exact ground truth | BPM/key precision gates need exact truth; real tracks (no ground truth) are the sanity layer. Regenerable, deterministic, seeded. |

## Conventions

- Every session: end by updating this file (state, next steps, decisions).
- Audio/stems never committed; `library/` and `stems_cache/` are gitignored.
- Determinism gate from Phase 3 on: same input → byte-identical render.
- Repo-local git identity only — never rely on global config here.
- Run tests: `.venv/Scripts/python.exe core/tests/test_analysis.py`
  (regenerate material first if `library/test/` is missing).
