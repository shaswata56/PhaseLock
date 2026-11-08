"use client";

import { useEffect, useMemo, useRef, useState } from "react";

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
const TRANSITIONS = [
  "blend",
  "bass_swap",
  "filter_sweep",
  "drum_drop",
  "acapella_overlay",
];
const BEAT_CHOICES = [8, 16, 32, 64];
const name = (p: string) => p.split(/[\\/]/).pop()?.replace(/\.[^.]+$/, "") ?? p;
const fmtDur = (s: number | null) =>
  s == null ? "—" : `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, "0")}`;

// Camelot compatibility, same rules as the planner
function compat(a?: string | null, b?: string | null): number | null {
  if (!a || !b) return null;
  const pa = /^(\d+)([AB])$/.exec(a.toUpperCase());
  const pb = /^(\d+)([AB])$/.exec(b.toUpperCase());
  if (!pa || !pb) return null;
  const na = +pa[1], nb = +pb[1];
  const d = Math.min(Math.abs(na - nb), 12 - Math.abs(na - nb));
  if (d === 0) return 1;
  if (d === 1) return 0.9;
  if (d === 3 && pa[2] !== pb[2]) return 0.55;
  return 0;
}

function compatBadge(v: number | null) {
  if (v == null) return null;
  if (v >= 1) return <span className="text-emerald-400">harmonic</span>;
  if (v >= 0.9) return <span className="text-sky-300">compatible</span>;
  if (v > 0) return <span className="text-amber-300">bold</span>;
  return <span className="text-red-400 font-semibold">clash</span>;
}

/* ---------- waveform + energy detail view ---------- */

function Curves({ path }: { path: string }) {
  const [data, setData] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    setData(null);
    setErr(null);
    fetch(`/api/curves?path=${encodeURIComponent(path)}`)
      .then((r) => r.json())
      .then((d) => (d.error ? setErr(d.error) : setData(d)))
      .catch(() => setErr("curves unavailable"));
  }, [path]);

  useEffect(() => {
    const cv = ref.current;
    if (!cv || !data?.waveform) return;
    const dpr = window.devicePixelRatio || 1;
    const w = cv.clientWidth * dpr;
    const h = 180 * dpr;
    cv.width = w;
    cv.height = h;
    const g = cv.getContext("2d")!;
    g.clearRect(0, 0, w, h);

    const { min, max } = data.waveform;
    const n = min.length;
    const mid = h / 2;
    g.fillStyle = "rgba(52,211,153,0.55)";
    for (let i = 0; i < n; i++) {
      const x = (i / n) * w;
      const y0 = mid - max[i] * mid * 0.95;
      const y1 = mid - min[i] * mid * 0.95;
      g.fillRect(x, Math.min(y0, y1), Math.max(1, w / n), Math.max(1, y1 - y0));
    }

    // energy curve (dB → height)
    const ec: number[] = data.energy_curve_db ?? [];
    if (ec.length) {
      g.strokeStyle = "rgba(56,189,248,0.9)";
      g.lineWidth = 1.5 * dpr;
      g.beginPath();
      for (let i = 0; i < ec.length; i++) {
        const x = (i / ec.length) * w;
        const y = h - ((ec[i] + 60) / 60) * h * 0.92;
        i ? g.lineTo(x, y) : g.moveTo(x, y);
      }
      g.stroke();
    }

    // downbeat ticks
    const beats: number[] = data.beats ?? [];
    const phase = data.downbeat_phase;
    const total = data.energy_curve_db
      ? (ec.length * 0.5)
      : beats[beats.length - 1] ?? 1;
    if (beats.length && phase != null) {
      g.strokeStyle = "rgba(244,114,182,0.5)";
      g.lineWidth = dpr;
      for (let i = phase; i < beats.length; i += 16) {
        const x = (beats[i] / total) * w;
        g.beginPath();
        g.moveTo(x, 0);
        g.lineTo(x, 8 * dpr);
        g.stroke();
      }
    }
  }, [data]);

  if (err) return <p className="text-xs text-zinc-500">{err}</p>;
  if (!data) return <p className="text-xs text-zinc-500">loading curves…</p>;
  return <canvas ref={ref} className="w-full rounded bg-zinc-950" style={{ height: 180 }} />;
}

function TrackDetail({ track, onClose }: { track: Track; onClose: () => void }) {
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/60 p-3">
      <div className="mb-2 flex items-start justify-between gap-2">
        <div className="min-w-0">
          <p className="truncate text-sm font-semibold text-zinc-200">
            {name(track.path)}
          </p>
          <p className="font-mono text-xs text-zinc-500">
            {track.bpm?.toFixed(1)} bpm · {track.key_name} · {track.camelot} ·{" "}
            {track.loudness_lufs?.toFixed(1)} LUFS · {fmtDur(track.duration_s)}
          </p>
        </div>
        <button
          onClick={onClose}
          className="rounded border border-zinc-700 px-2 text-xs text-zinc-400 hover:border-emerald-500 hover:text-emerald-300"
        >
          close
        </button>
      </div>
      <Curves path={track.path} />
      <audio
        controls
        src={`/api/preview?path=${encodeURIComponent(track.path)}`}
        className="mt-2 w-full"
      />
    </div>
  );
}

/* ---------- main booth ---------- */

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
  const [selected, setSelected] = useState<Track | null>(null);
  const [edits, setEdits] = useState<Record<number, string>>({});
  const [manual, setManual] = useState<{ path: string; mins: number }[]>([]);
  const [mType, setMType] = useState("blend");
  const [mBeats, setMBeats] = useState(16);
  const [building, setBuilding] = useState(false);
  const [buildWarn, setBuildWarn] = useState<string[]>([]);

  const manualTrack = (p: string) => tracks.find((t) => t.path === p);
  const addToManual = (t: Track) => {
    setManual((m) =>
      m.some((x) => x.path === t.path) ? m : [...m, { path: t.path, mins: 3 }],
    );
  };
  const moveManual = (i: number, dir: -1 | 1) =>
    setManual((m) => {
      const j = i + dir;
      if (j < 0 || j >= m.length) return m;
      const n = [...m];
      [n[i], n[j]] = [n[j], n[i]];
      return n;
    });

  const buildManual = async () => {
    if (manual.length < 2) return;
    setBuilding(true);
    setError(null);
    setBuildWarn([]);
    try {
      const r = await fetch("/api/build", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          paths: manual.map((m) => m.path),
          mins: manual.map((m) => m.mins),
          type: mType,
          beats: mBeats,
        }),
      });
      const j = await r.json();
      if (!r.ok) throw new Error(j.error || "build failed");
      setSet(j.set);
      setSetPath(j.setPath);
      setEdits({});
      setJob(null);
      setBuildWarn(j.set.warnings ?? []);
    } catch (e: any) {
      setError(String(e));
    } finally {
      setBuilding(false);
    }
  };

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
    setEdits({});
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

  const saveEdits = async () => {
    if (!setPath || !set) return;
    const payload = Object.entries(edits).map(([index, type]) => ({
      index: Number(index),
      type,
    }));
    const r = await fetch("/api/plan/edit", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ setPath, edits: payload }),
    });
    if (r.ok) {
      const { applied } = await r.json();
      setSet({
        ...set,
        tracks: set.tracks.map((t, i) =>
          edits[i] && t.transition_out
            ? { ...t, transition_out: { ...t.transition_out, type: edits[i] } }
            : t,
        ),
      });
      setEdits({});
      if (applied !== payload.length)
        setError("some edits were rejected (bad index/type)");
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
    drum_drop: "text-red-300",
    acapella_overlay: "text-violet-300",
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
          <div className="max-h-[62vh] overflow-auto">
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
                    onClick={() => setSelected(t)}
                    className={`cursor-pointer border-t border-zinc-800/60 hover:bg-zinc-800/40 ${
                      selected?.path === t.path ? "bg-zinc-800/60" : ""
                    }`}
                  >
                    <td className="w-6 px-1 py-1">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          addToManual(t);
                        }}
                        title="add to manual set"
                        className="rounded px-1 text-zinc-500 hover:bg-emerald-900/60 hover:text-emerald-300"
                      >
                        +
                      </button>
                    </td>
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
          {selected && (
            <TrackDetail track={selected} onClose={() => setSelected(null)} />
          )}

          <div className="rounded-lg border border-zinc-800 bg-zinc-900/60 p-4">
            <h2 className="mb-3 text-sm font-semibold tracking-wide text-zinc-300">
              MANUAL SET — any track, any order, any length
            </h2>
            {manual.length === 0 && (
              <p className="text-xs text-zinc-500">
                add tracks from the library with <span className="text-emerald-400">+</span>
                , then build &amp; render.
              </p>
            )}
            <ol className="space-y-1">
              {manual.map((m, i) => {
                const t = manualTrack(m.path);
                const prev = i > 0 ? manualTrack(manual[i - 1].path) : null;
                const c = prev ? compat(prev.camelot, t?.camelot) : null;
                const gap =
                  prev && t && prev.bpm && t.bpm
                    ? Math.abs(t.bpm - prev.bpm) / prev.bpm
                    : 0;
                return (
                  <li key={m.path}>
                    {prev && (
                      <div className="flex items-center gap-2 py-0.5 pl-3 font-mono text-[10px] text-zinc-500">
                        <span>{compatBadge(c)}</span>
                        {gap > 0.06 && (
                          <span className="text-amber-300">
                            bpm gap {(gap * 100).toFixed(0)}%
                          </span>
                        )}
                      </div>
                    )}
                    <div className="flex items-center gap-1.5 font-mono text-xs">
                      <span className="w-4 text-right text-zinc-600">{i + 1}</span>
                      <span className="flex-1 truncate text-zinc-300">
                        {name(m.path)}
                      </span>
                      {t?.bpm != null && (
                        <span className="tabular-nums text-zinc-500">
                          {t.bpm.toFixed(1)}
                        </span>
                      )}
                      {t?.camelot && (
                        <span className="text-emerald-400/80">{t.camelot}</span>
                      )}
                      <input
                        type="number"
                        min={0.5}
                        max={15}
                        step={0.5}
                        value={m.mins}
                        onChange={(e) =>
                          setManual((ms) =>
                            ms.map((x, j) =>
                              j === i
                                ? { ...x, mins: Number(e.target.value) || 3 }
                                : x,
                            ),
                          )
                        }
                        className="w-14 rounded bg-zinc-950 px-1 py-0.5 text-right tabular-nums text-zinc-300 ring-1 ring-zinc-800 focus:ring-emerald-600"
                      />
                      <span className="text-zinc-600">min</span>
                      <button
                        onClick={() => moveManual(i, -1)}
                        className="px-1 text-zinc-500 hover:text-emerald-300"
                        aria-label="up"
                      >
                        ↑
                      </button>
                      <button
                        onClick={() => moveManual(i, 1)}
                        className="px-1 text-zinc-500 hover:text-emerald-300"
                        aria-label="down"
                      >
                        ↓
                      </button>
                      <button
                        onClick={() =>
                          setManual((ms) => ms.filter((_, j) => j !== i))
                        }
                        className="px-1 text-zinc-500 hover:text-red-400"
                        aria-label="remove"
                      >
                        ✕
                      </button>
                    </div>
                  </li>
                );
              })}
            </ol>
            {manual.length >= 2 && (
              <div className="mt-3 flex flex-wrap items-center gap-2">
                <select
                  value={mType}
                  onChange={(e) => setMType(e.target.value)}
                  className="rounded bg-zinc-950 px-2 py-1 text-xs text-zinc-200 ring-1 ring-zinc-800"
                >
                  {TRANSITIONS.map((v) => (
                    <option key={v}>{v}</option>
                  ))}
                </select>
                <select
                  value={mBeats}
                  onChange={(e) => setMBeats(Number(e.target.value))}
                  className="rounded bg-zinc-950 px-2 py-1 text-xs text-zinc-200 ring-1 ring-zinc-800"
                >
                  {BEAT_CHOICES.map((v) => (
                    <option key={v} value={v}>
                      {v} beats
                    </option>
                  ))}
                </select>
                <span className="font-mono text-xs text-zinc-500">
                  ~{manual.reduce((a, m) => a + (m.mins || 0), 0).toFixed(1)} min
                </span>
                <button
                  onClick={buildManual}
                  disabled={building}
                  className="ml-auto rounded bg-emerald-600 px-3 py-1.5 text-xs font-semibold text-zinc-950 hover:bg-emerald-500 disabled:opacity-40"
                >
                  {building ? "building…" : "build set"}
                </button>
              </div>
            )}
            {buildWarn.length > 0 && (
              <ul className="mt-2 space-y-0.5 font-mono text-[10px] text-amber-300">
                {buildWarn.map((w) => (
                  <li key={w}>⚠ {w}</li>
                ))}
              </ul>
            )}
          </div>

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
                <div className="flex items-center gap-2">
                  {Object.keys(edits).length > 0 && (
                    <button
                      onClick={saveEdits}
                      className="rounded bg-sky-600 px-2 py-1 text-xs font-semibold text-zinc-950 hover:bg-sky-500"
                    >
                      save edits ({Object.keys(edits).length})
                    </button>
                  )}
                  <button
                    onClick={render}
                    disabled={job?.status === "running"}
                    className="rounded bg-fuchsia-600 px-3 py-1 text-xs font-semibold text-zinc-950 hover:bg-fuchsia-500 disabled:opacity-40"
                  >
                    {job?.status === "running" ? "rendering…" : "render mix"}
                  </button>
                </div>
              </div>
              <ol className="max-h-72 overflow-auto p-2 font-mono text-xs">
                {set.tracks.map((t, i) => {
                  const cur = t.transition_out;
                  const edited = edits[i];
                  return (
                    <li
                      key={t.path}
                      className="flex items-center gap-2 px-1 py-1"
                    >
                      <span className="w-5 text-right text-zinc-600">{i + 1}</span>
                      <span className="flex-1 truncate text-zinc-300">
                        {name(t.path)}
                      </span>
                      <span className="tabular-nums text-zinc-400">
                        {t.mix_bpm.toFixed(1)}
                      </span>
                      <span className="text-emerald-400/80">{t.camelot}</span>
                      {cur && (
                        <select
                          value={edited ?? cur.type}
                          onChange={(e) =>
                            setEdits((s) => ({ ...s, [i]: e.target.value }))
                          }
                          className={`rounded bg-zinc-950 px-1 py-0.5 text-[10px] ring-1 ring-zinc-800 ${
                            TR_COLORS[edited ?? cur.type] ?? "text-zinc-500"
                          }`}
                        >
                          {TRANSITIONS.map((v) => (
                            <option key={v} value={v}>
                              ↦ {v}
                            </option>
                          ))}
                        </select>
                      )}
                    </li>
                  );
                })}
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
