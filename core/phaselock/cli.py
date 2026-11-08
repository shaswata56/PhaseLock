"""PhaseLock CLI: analyze / list / stats / override."""

from __future__ import annotations

import statistics
from pathlib import Path

import click

from . import analyze as analyze_mod
from . import db as db_mod
from . import decode

# repo_root/data/ — core/phaselock/cli.py → up two levels
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = _REPO_ROOT / "data" / "phaselock.db"
DEFAULT_CACHE = _REPO_ROOT / "data" / "cache"


def _db(db_path: str | None):
    return db_mod.connect(db_path or DEFAULT_DB)


@click.group()
@click.version_option()
def main() -> None:
    """PhaseLock — personal AI DJ for EDM. Library analysis tools."""


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--db", "db_path", default=None, help="Library DB path.")
@click.option("--cache", "cache_dir", default=None, help="Sidecar cache dir.")
@click.option("--recursive/--no-recursive", default=True)
@click.option("--force", is_flag=True, help="Re-analyze unchanged files too.")
def analyze(path: str, db_path: str | None, cache_dir: str | None,
            recursive: bool, force: bool) -> None:
    """Analyze every audio file under PATH into the library."""
    conn = _db(db_path)
    cache = Path(cache_dir) if cache_dir else DEFAULT_CACHE
    files = decode.scan_audio_files(path, recursive=recursive)
    if not files:
        click.echo(f"no audio files found under {path}")
        return
    ok = err = skip = 0
    with click.progressbar(files, label="analyzing") as bar:
        for f in bar:
            r = analyze_mod.analyze_file(conn, cache, f, force=force)
            if r["analysis_status"] == "ok":
                ok += 1 if r["analyzer_version"] else 0
            else:
                err += 1
    # Count skips separately (analyze_file returns existing row on skip).
    click.echo(f"done: {len(files)} file(s), {err} error(s)")
    if err:
        for row in conn.execute(
            "SELECT path, error FROM tracks WHERE analysis_status = 'error'"
        ):
            click.echo(f"  ERROR {row['path']}: {row['error']}")


@main.command("list")
@click.option("--db", "db_path", default=None)
@click.option("--limit", default=50, show_default=True)
def list_tracks(db_path: str | None, limit: int) -> None:
    """Show analyzed tracks: BPM, key, Camelot, loudness."""
    conn = _db(db_path)
    rows = conn.execute(
        "SELECT path, bpm, key_name, camelot, loudness_lufs, duration_s, "
        "analysis_status FROM tracks ORDER BY path LIMIT ?", (limit,)
    ).fetchall()
    if not rows:
        click.echo("library empty — run `phaselock analyze <folder>`")
        return
    w = max(len(click.format_filename(r["path"])) for r in rows)
    w = min(w, 60)
    click.echo(f"{'file':<{w}}  {'bpm':>6}  {'key':<10} {'cam':<4} {'lufs':>6}  status")
    for r in rows:
        bpm = f"{r['bpm']:.1f}" if r["bpm"] else "-"
        key = r["key_name"] or "-"
        cam = r["camelot"] or "-"
        lufs = f"{r['loudness_lufs']:.1f}" if r["loudness_lufs"] is not None else "-"
        click.echo(f"{click.format_filename(r['path']):<{w}}  {bpm:>6}  {key:<10} "
                   f"{cam:<4} {lufs:>6}  {r['analysis_status']}")


@main.command()
@click.option("--db", "db_path", default=None)
def stats(db_path: str | None) -> None:
    """Library summary: counts, BPM spread, key distribution."""
    conn = _db(db_path)
    n = conn.execute("SELECT COUNT(*) c FROM tracks WHERE analysis_status='ok'").fetchone()["c"]
    if not n:
        click.echo("library empty — run `phaselock analyze <folder>`")
        return
    bpms = [r["bpm"] for r in conn.execute(
        "SELECT bpm FROM tracks WHERE analysis_status='ok' AND bpm IS NOT NULL")]
    lufs = [r["loudness_lufs"] for r in conn.execute(
        "SELECT loudness_lufs FROM tracks WHERE analysis_status='ok' "
        "AND loudness_lufs IS NOT NULL")]
    click.echo(f"tracks: {n}")
    if bpms:
        click.echo(f"bpm: min {min(bpms):.1f} / median {statistics.median(bpms):.1f} "
                   f"/ max {max(bpms):.1f}")
    if lufs:
        click.echo(f"loudness: mean {statistics.mean(lufs):.1f} LUFS")
    click.echo("keys:")
    for r in conn.execute(
        "SELECT camelot, key_name, COUNT(*) c FROM tracks "
        "WHERE analysis_status='ok' GROUP BY camelot, key_name "
        "ORDER BY c DESC LIMIT 12"
    ):
        click.echo(f"  {r['camelot'] or '-':<4} {r['key_name'] or '-':<10} ×{r['c']}")


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.argument("key_name")
@click.option("--db", "db_path", default=None)
def override(path: str, key_name: str, db_path: str | None) -> None:
    """Set a human key override: PATH 'A minor'."""
    conn = _db(db_path)
    if not analyze_mod.set_key_override(conn, path, key_name):
        click.echo(f"could not parse key {key_name!r} — use e.g. 'A minor', 'F# major', 'Bb major'")
        raise SystemExit(1)
    click.echo(f"override set: {Path(path).name} → {key_name}")


@main.command()
@click.option("--duration", default=60.0, show_default=True, help="Target minutes.")
@click.option("--vibe", default=None,
              help="peak / warmup / melodic / dark / closing / auto")
@click.option("--db", "db_path", default=None)
@click.option("--cache", "cache_dir", default=None)
@click.option("--out", "out_path", default=None, help="Set JSON path.")
@click.option("--json", "as_json", is_flag=True, help="Print set JSON only.")
def plan(duration: float, vibe: str | None, db_path: str | None,
         cache_dir: str | None, out_path: str | None, as_json: bool) -> None:
    """Generate a DJ set from the analyzed library."""
    import json as _json

    from .planning import planner

    conn = _db(db_path)
    cache = Path(cache_dir) if cache_dir else DEFAULT_CACHE
    cards = planner.load_library(conn, cache)
    if len(cards) < 2:
        click.echo("need ≥2 analyzed tracks — run `phaselock analyze <folder>`")
        raise SystemExit(1)
    s = planner.plan_set(cards, duration, vibe)

    if out_path is None:
        sets = _REPO_ROOT / "data" / "sets"
        sets.mkdir(parents=True, exist_ok=True)
        stamp = _dt_now()
        out_path = sets / f"set_{stamp}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(_json.dumps(s, indent=2))

    if as_json:
        click.echo(_json.dumps(s))
    else:
        click.echo(f"set → {out_path}")
        click.echo(f"vibe={s['vibe']}  tracks={len(s['tracks'])}  "
                   f"target={s['duration_target_min']:.0f} min")
        for k, t in enumerate(s["tracks"], 1):
            tr = t.get("transition_out", {})
            tr_s = f"→ {tr['type']}/{tr['beats']}b" if tr else ""
            click.echo(f"  {k:>2}. {Path(t['path']).stem:<28} {t['bpm']:>6.1f}bpm "
                       f"{t['camelot']:<3} E{t['energy']:.2f} "
                       f"play {t['play_beats']}b @ {t['mix_bpm']:.2f} {tr_s}")


def _dt_now() -> str:
    import datetime

    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


@main.command()
@click.argument("set_json", type=click.Path(exists=True))
@click.option("-o", "out_wav", default=None, help="Output WAV path.")
@click.option("--mp3", "mp3_out", default=None, help="Also encode MP3 to path.")
@click.option("--db", "db_path", default=None)
@click.option("--cache", "cache_dir", default=None)
def render(set_json: str, out_wav: str | None, mp3_out: str | None,
           db_path: str | None, cache_dir: str | None) -> None:
    """Render a set JSON to a continuous mix."""
    import json as _json
    import subprocess
    from .rendering import engine

    s = _json.loads(Path(set_json).read_text())
    if out_wav is None:
        renders = _REPO_ROOT / "renders"
        renders.mkdir(parents=True, exist_ok=True)
        stem = Path(set_json).stem
        out_wav = renders / f"{stem}.wav"
    conn = _db(db_path) if db_path or Path(_REPO_ROOT / "data" / "phaselock.db").exists() else None
    cache = Path(cache_dir) if cache_dir else DEFAULT_CACHE

    info = engine.render_set(
        s, cache, out_wav, conn=conn,
        progress=lambda m: click.echo(f"  {m}"),
    )
    click.echo(f"mix → {info['out']} ({info['duration_s']}s, "
               f"{len(info['transitions'])} transitions)")
    if mp3_out:
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", info["out"],
                        "-codec:a", "libmp3lame", "-qscale:a", "2", mp3_out],
                       check=True)
        click.echo(f"mp3 → {mp3_out}")


if __name__ == "__main__":
    main()
