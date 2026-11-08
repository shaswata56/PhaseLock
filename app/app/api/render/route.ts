import { NextRequest, NextResponse } from "next/server";
import path from "path";
import { repoRoot, startJob } from "../../../lib/phaselock";



export async function POST(req: NextRequest) {
  try {
    const { setPath } = await req.json();
    const root = repoRoot();
    const setAbs = path.resolve(String(setPath));
    const setsDir = path.resolve(root, "data", "sets");
    if (!setAbs.startsWith(setsDir)) {
      return NextResponse.json(
        { error: "setPath must live under data/sets" },
        { status: 400 },
      );
    }
    const stamp = path.basename(setAbs).replace(/\.json$/, "");
    const outWav = path.join(root, "renders", `webmix_${stamp}.wav`);
    const id = `r${Date.now().toString(36)}`;
    startJob(id, ["render", setAbs, "-o", outWav], outWav);
    return NextResponse.json({ jobId: id });
  } catch (e: any) {
    return NextResponse.json({ error: String(e) }, { status: 500 });
  }
}
