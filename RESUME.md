# PhaseLock — RESUME

*Live project state. Update at the end of every session. A new session starts
by reading this file.*

## Current state (2026-11-08)

**Phase:** 0 complete (scaffold). Next: Phase 1 kickoff — The Ears.

**Done:**
- Project named **PhaseLock** (beat-matching = phase-locking two signals; PLL).
- Repo at `C:\workstation\phaselock`, git initialized, initial commit made with
  repo-local identity.
- README (vision/principles), PLAN (overview + 7-phase plan with acceptance
  gates), RESUME (this file), .gitignore (media + stems excluded).

**Next steps (Phase 1 kickoff checklist):**
1. Python 3.11+ venv at `core/.venv` (or repo root `.venv`), pin deps:
   librosa, madmom, essentia (conda-forge if pip fails on Windows), pyloudnorm,
   soundfile, numpy, click.
2. Assemble 10 test tracks with known BPM + key (from Beatport or user's own
   tagged library) into `library/test/` — gitignored.
3. Schema first: SQLite tables + JSON sidecar format, THEN analyzers.
4. CLI skeleton: `phaselock analyze/list/stats` (click).

**Blockers/questions for user:** none. (If you have a preferred test folder of
EDM tracks on this machine, point me at it — otherwise I'll work with
whatever's findable/generatable.)

## Decision log

| Date | Decision | Why |
|---|---|---|
| 2026-11-08 | Name: PhaseLock | PLL metaphor; beat-matching is phase-locking. Alternates considered: Dropwise, Selector, Downbeat. |
| 2026-11-08 | Location: `C:\workstation\phaselock` | Workstation = product workspace (chonky.cam precedent); `~/work` reserved for hunts. |
| 2026-11-08 | Repo-local git identity: `shaswata56 <shaswata56@users.noreply.github.com>` | Global identity is the forbidden orbitax one; noreply address is tied to the personal account and exposes nothing. Set BEFORE first commit (per sites-repo lesson). |
| 2026-11-08 | MVP = Phases 1–3 (analyze → plan → render) | Stems/app/taste all build on a working render core; render is the credibility gate. |
| 2026-11-08 | Python core before web app | Render quality is the hard part; UI last so it wraps a proven core. |

## Conventions

- Every session: end by updating this file (state, next steps, decisions).
- Audio/stems never committed; `library/` and `stems_cache/` are gitignored.
- Determinism gate from Phase 3 on: same input → byte-identical render.
- Repo-local git identity only — never rely on global config here.
