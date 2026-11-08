import { NextRequest, NextResponse, connection } from "next/server";
import fs from "fs";
import path from "path";
import { repoRoot, runCli } from "../../../lib/phaselock";
import { sendAudio } from "../../../lib/sendfile";

// Source-file preview streaming. Paths are validated against the library DB
// (via the CLI) before any byte is read; Range requests supported for seeking.

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
  return sendAudio(req, abs);
}
