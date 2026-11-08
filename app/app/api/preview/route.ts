import { NextRequest, NextResponse, connection } from "next/server";
import fs from "fs";
import path from "path";
import { Readable } from "stream";
import { repoRoot, runCli } from "../../../lib/phaselock";

// Source-file preview streaming. Paths are validated against the library DB
// (via the CLI) before any byte is read; Range requests supported for seeking.

const MIME: Record<string, string> = {
  ".mp3": "audio/mpeg",
  ".flac": "audio/flac",
  ".wav": "audio/wav",
  ".m4a": "audio/mp4",
  ".ogg": "audio/ogg",
  ".opus": "audio/ogg",
};

let known: { at: number; paths: Set<string> } | null = null;

async function knownPaths(): Promise<Set<string>> {
  if (!known || Date.now() - known.at > 60_000) {
    const rows = JSON.parse(await runCli(["tracks"]));
    known = { at: Date.now(), paths: new Set(rows.map((r: any) => r.path)) };
  }
  return known.paths;
}

export async function GET(req: NextRequest) {
  await connection();
  const p = req.nextUrl.searchParams.get("path");
  if (!p) return NextResponse.json({ error: "path required" }, { status: 400 });
  const abs = path.resolve(p);
  let allowed: Set<string>;
  try {
    allowed = await knownPaths();
  } catch {
    return NextResponse.json({ error: "library unavailable" }, { status: 500 });
  }
  if (!allowed.has(abs)) {
    return NextResponse.json({ error: "not a library track" }, { status: 403 });
  }
  if (!fs.existsSync(abs)) {
    return NextResponse.json({ error: "file missing" }, { status: 404 });
  }

  const type = MIME[path.extname(abs).toLowerCase()] ?? "application/octet-stream";
  const size = fs.statSync(abs).size;
  const range = req.headers.get("range");

  const streamRange = (start: number, end: number, status: number) => {
    const stream = Readable.toWeb(
      fs.createReadStream(abs, { start, end }),
    ) as unknown as ReadableStream;
    const headers: Record<string, string> = {
      "content-type": type,
      "content-length": String(end - start + 1),
      "accept-ranges": "bytes",
    };
    if (status === 206) headers["content-range"] = `bytes ${start}-${end}/${size}`;
    return new NextResponse(stream, { status, headers });
  };

  if (range) {
    const m = /bytes=(\d*)-(\d*)/.exec(range);
    if (m) {
      const start = m[1] ? parseInt(m[1], 10) : 0;
      const end = m[2] ? Math.min(parseInt(m[2], 10), size - 1) : size - 1;
      if (start <= end) return streamRange(start, end, 206);
    }
    return new NextResponse(null, {
      status: 416,
      headers: { "content-range": `bytes */${size}` },
    });
  }
  return streamRange(0, size - 1, 200);
}
