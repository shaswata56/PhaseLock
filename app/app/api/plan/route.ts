import { NextRequest, NextResponse } from "next/server";
import path from "path";
import { repoRoot, runCli } from "../../../lib/phaselock";



export async function POST(req: NextRequest) {
  try {
    const { duration, vibe, llm } = await req.json();
    const stamp = new Date().toISOString().replace(/[-:T.]/g, "").slice(0, 14);
    const setPath = path.join(repoRoot(), "data", "sets", `web_${stamp}.json`);
    const args = [
      "plan",
      "--duration",
      String(Number(duration) || 30),
      "--out",
      setPath,
    ];
    if (vibe) args.push("--vibe", String(vibe));
    if (llm) args.push("--llm");
    const text = await runCli(args, 180_000);
    const fs = await import("fs");
    const set = JSON.parse(fs.readFileSync(setPath, "utf8"));
    return NextResponse.json({ set, setPath, log: text.trim() });
  } catch (e: any) {
    return NextResponse.json({ error: String(e) }, { status: 500 });
  }
}
