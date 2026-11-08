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


if __name__ == "__main__":
    main()
