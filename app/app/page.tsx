"use client";

import { useEffect, useMemo, useState } from "react";

type Track = {
  path: string;
  bpm: number | null;
  key_name: string | null;
  camelot: string | null;
  loudness_lufs: number | null;
  duration_s: number | null;
  energy_mean: number | null;
  analysis_status: string;
};

type SetTrack = {
  path: string;
  bpm: number;
  key: string;
  camelot: string;
  energy: number;
  mix_bpm: number;
  play_beats: number;
  transition_out?: { type: string; beats: number };
};

type Set = { vibe: string; duration_target_min: number; tracks: SetTrack[] };

type Job = {
  id: string;
  status: string;
  log: string[];
  mixUrl?: string;
};

const VIBES = ["auto", "peak", "warmup", "melodic", "dark", "closing"];
const name = (p: string) => p.split(/[\\/]/).pop()?.replace(/\.[^.]+$/, "") ?? p;
const fmtDur = (s: number | null) =>
  s == null ? "—" : `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, "0")}`;

export default function Booth() {
  const [tracks, setTracks] = useState<Track[]>([]);
  const [filter, setFilter] = useState("");
  const [sort, setSort] = useState<{ k: keyof Track; dir: 1 | -1 }>({
    k: "bpm",
    dir: 1,
  });
  const [duration, setDuration] = useState(30);
  const [vibe, setVibe] = useState("auto");
  const [dsl, setDsl] = useState("");
  const [planning, setPlanning] = useState(false);
  const [set, setSet] = useState<Set | null>(null);
  const [setPath, setSetPath] = useState<string | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = () =>
    fetch("/api/tracks")
      .then((r) => (r.ok ? r.json() : Promise.reject(r.text())))
      .then((t: Track[]) => setTracks(t.filter((x) => x.analysis_status === "ok")))
      .catch(() => setError("could not load library — is `phaselock analyze` done?"));

  useEffect(() => {
    load();
  }, []);

  useEffect(() => {
    if (!job || job.status !== "running") return;
    const t = setInterval(() => {
      fetch(`/api/jobs?id=${job.id}`)
        .then((r) => r.json())
        .then((j: Job) => setJob(j));
    }, 2000);
    return () => clearInterval(t);
  }, [job]);

  const shown = useMemo(() => {
    const f = filter.toLowerCase();
    const rows = tracks.filter(
      (t) =>
        !f ||
        name(t.path).toLowerCase().includes(f) ||
        (t.camelot ?? "").toLowerCase().includes(f) ||
        (t.key_name ?? "").toLowerCase().includes(f),
    );
    const { k, dir } = sort;
    return rows.sort((a, b) => {
      const x = a[k] ?? -1e9;
      const y = b[k] ?? -1e9;
      if (typeof x === "string" || typeof y === "string")
        return String(x).localeCompare(String(y)) * dir;
      return ((x as number) - (y as number)) * dir;
    });
  }, [tracks, filter, sort]);

  const bpms = tracks.map((t) => t.bpm).filter((b): b is number => b != null);
  const stats = tracks.length
    ? `${tracks.length} tracks · ${Math.min(...bpms).toFixed(0)}–${Math.max(
        ...bpms,
      ).toFixed(0)} bpm`
    : "library empty";

  const plan = async () => {
    setPlanning(true);
    setError(null);
    setJob(null);
    try {
      const r = await fetch("/api/plan", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          duration,
          vibe: dsl.trim() ? `${vibe} ${dsl.trim()}` : vibe,
        }),
      });
      const j = await r.json();
      if (!r.ok) throw new Error(j.error || "plan failed");
      setSet(j.set);
      setSetPath(j.setPath);
    } catch (e: any) {
      setError(String(e));
    } finally {
      setPlanning(false);
    }
  };

  const render = async () => {
    if (!setPath) return;
    setError(null);
    setJob({ id: "starting", status: "running", log: [] });
    try {
      const r = await fetch("/api/render", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ setPath }),
      });
      const j = await r.json();
      if (!r.ok) throw new Error(j.error || "render failed");
      setJob({ id: j.jobId, status: "running", log: [] });
    } catch (e: any) {
      setError(String(e));
      setJob(null);
    }
  };

  const th = (label: string, k: keyof Track) => (
    <th
      className="cursor-pointer px-2 py-1.5 text-left font-medium text-zinc-400 select-none hover:text-emerald-300"
      onClick={() =>
        setSort((s) => ({ k, dir: s.k === k && s.dir === 1 ? -1 : 1 }))
      }
    >
      {label}
      {sort.k === k ? (sort.dir === 1 ? " ↑" : " ↓") : ""}
    </th>
  );

  const TR_COLORS: Record<string, string> = {
    blend: "text-sky-300",
    bass_swap: "text-fuchsia-300",
    filter_sweep: "text-amber-300",
  };

  return (
    <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-6">
      <header className="mb-6 flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="font-mono text-2xl font-bold tracking-tight text-zinc-100">
            PHASE<span className="text-emerald-400">LOCK</span>
            <span className="ml-3 align-middle text-xs font-normal tracking-widest text-zinc-500">
              THE BOOTH
            </span>
          </h1>
          <p className="mt-1 text-sm text-zinc-500">
            {stats} — analyze · plan · render, all local
          </p>
        </div>
        <button
          onClick={load}
          className="rounded border border-zinc-700 px-3 py-1 text-xs text-zinc-400 hover:border-emerald-500 hover:text-emerald-300"
        >
          refresh
        </button>
      </header>

      {error && (
        <div className="mb-4 rounded border border-red-800 bg-red-950/50 px-3 py-2 text-sm text-red-300">
          {error}
        </div>
      )}

      <div className="grid gap-6 lg:grid-cols-[1.2fr_1fr]">
        {/* Library */}
        <section className="rounded-lg border border-zinc-800 bg-zinc-900/60">
          <div className="flex items-center justify-between border-b border-zinc-800 px-3 py-2">
            <h2 className="text-sm font-semibold tracking-wide text-zinc-300">
              LIBRARY
            </h2>
            <input
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              placeholder="filter…"
              className="w-40 rounded bg-zinc-950 px-2 py-1 text-xs text-zinc-300 outline-none ring-1 ring-zinc-800 focus:ring-emerald-600"
            />
          </div>
          <div className="max-h-[60vh] overflow-auto">
            <table className="w-full text-xs">
              <thead className="sticky top-0 bg-zinc-900">
                <tr>
                  {th("track", "path")}
                  {th("bpm", "bpm")}
                  {th("key", "key_name")}
                  {th("cam", "camelot")}
                  {th("lufs", "loudness_lufs")}
                  {th("len", "duration_s")}
                </tr>
              </thead>
              <tbody className="font-mono">
                {shown.map((t) => (
                  <tr
                    key={t.path}
                    className="border-t border-zinc-800/60 hover:bg-zinc-800/40"
                  >
                    <td className="max-w-[220px] truncate px-2 py-1 text-zinc-300">
                      {name(t.path)}
                    </td>
                    <td className="px-2 py-1 tabular-nums text-zinc-400">
                      {t.bpm?.toFixed(1) ?? "—"}
                    </td>
                    <td className="px-2 py-1 text-zinc-400">{t.key_name ?? "—"}</td>
                    <td className="px-2 py-1 text-emerald-400/80">
                      {t.camelot ?? "—"}
                    </td>
                    <td className="px-2 py-1 tabular-nums text-zinc-500">
                      {t.loudness_lufs?.toFixed(1) ?? "—"}
                    </td>
                    <td className="px-2 py-1 tabular-nums text-zinc-500">
                      {fmtDur(t.duration_s)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        {/* Planner + setlist */}
        <section className="space-y-4">
          <div className="rounded-lg border border-zinc-800 bg-zinc-900/60 p-4">
            <h2 className="mb-3 text-sm font-semibold tracking-wide text-zinc-300">
              PLAN A SET
            </h2>
            <div className="flex flex-wrap items-center gap-2">
              <input
                type="number"
                min={5}
                max={240}
                value={duration}
                onChange={(e) => setDuration(Number(e.target.value) || 30)}
                className="w-16 rounded bg-zinc-950 px-2 py-1 text-xs text-zinc-200 ring-1 ring-zinc-800 focus:ring-emerald-600"
              />
              <span className="text-xs text-zinc-500">min</span>
              <select
                value={vibe}
                onChange={(e) => setVibe(e.target.value)}
                className="rounded bg-zinc-950 px-2 py-1 text-xs text-zinc-200 ring-1 ring-zinc-800 focus:ring-emerald-600"
              >
                {VIBES.map((v) => (
                  <option key={v}>{v}</option>
                ))}
              </select>
              <input
                value={dsl}
                onChange={(e) => setDsl(e.target.value)}
                placeholder="DSL: bpm 122-128, end 126, minor"
                className="min-w-40 flex-1 rounded bg-zinc-950 px-2 py-1 text-xs text-zinc-200 ring-1 ring-zinc-800 placeholder:text-zinc-600 focus:ring-emerald-600"
              />
              <button
                onClick={plan}
                disabled={planning}
                className="rounded bg-emerald-600 px-3 py-1.5 text-xs font-semibold text-zinc-950 hover:bg-emerald-500 disabled:opacity-40"
              >
                {planning ? "planning…" : "generate"}
              </button>
            </div>
          </div>

          {set && (
            <div className="rounded-lg border border-zinc-800 bg-zinc-900/60">
              <div className="flex items-center justify-between border-b border-zinc-800 px-3 py-2">
                <h2 className="text-sm font-semibold tracking-wide text-zinc-300">
                  SET · {set.vibe} · {set.tracks.length} tracks
                </h2>
                <button
                  onClick={render}
                  disabled={job?.status === "running"}
                  className="rounded bg-fuchsia-600 px-3 py-1 text-xs font-semibold text-zinc-950 hover:bg-fuchsia-500 disabled:opacity-40"
                >
                  {job?.status === "running" ? "rendering…" : "render mix"}
                </button>
              </div>
              <ol className="max-h-72 overflow-auto p-2 font-mono text-xs">
                {set.tracks.map((t, i) => (
                  <li key={t.path} className="flex items-center gap-2 px-1 py-1">
                    <span className="w-5 text-right text-zinc-600">{i + 1}</span>
                    <span className="flex-1 truncate text-zinc-300">
                      {name(t.path)}
                    </span>
                    <span className="tabular-nums text-zinc-400">
                      {t.mix_bpm.toFixed(1)}
                    </span>
                    <span className="text-emerald-400/80">{t.camelot}</span>
                    {t.transition_out && (
                      <span
                        className={`text-[10px] ${TR_COLORS[t.transition_out.type] ?? "text-zinc-500"}`}
                      >
                        ↦ {t.transition_out.type}
                      </span>
                    )}
                  </li>
                ))}
              </ol>
            </div>
          )}

          {job && (
            <div className="rounded-lg border border-zinc-800 bg-zinc-900/60 p-3">
              <h2 className="mb-2 text-sm font-semibold tracking-wide text-zinc-300">
                RENDER ·{" "}
                <span
                  className={
                    job.status === "done"
                      ? "text-emerald-400"
                      : job.status === "error"
                        ? "text-red-400"
                        : "text-amber-300"
                  }
                >
                  {job.status}
                </span>
              </h2>
              {job.status === "running" && (
                <pre className="max-h-24 overflow-auto rounded bg-zinc-950 p-2 font-mono text-[10px] leading-4 text-zinc-500">
                  {job.log.slice(-6).join("\n")}
                </pre>
              )}
              {job.status === "done" && job.mixUrl && (
                <div className="space-y-2">
                  <audio controls src={job.mixUrl} className="w-full" />
                  <a
                    href={job.mixUrl}
                    download="phaselock-mix.wav"
                    className="inline-block rounded border border-emerald-700 px-3 py-1 text-xs text-emerald-300 hover:bg-emerald-950"
                  >
                    download wav
                  </a>
                </div>
              )}
              {job.status === "error" && (
                <pre className="max-h-24 overflow-auto rounded bg-zinc-950 p-2 font-mono text-[10px] text-red-400">
                  {job.log.join("\n")}
                </pre>
              )}
            </div>
          )}
        </section>
      </div>
    </main>
  );
}
