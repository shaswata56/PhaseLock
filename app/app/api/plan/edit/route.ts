import { NextRequest, NextResponse, connection } from "next/server";
import fs from "fs";
import path from "path";
import { repoRoot } from "../../../../lib/phaselock";

const TYPES = ["blend", "bass_swap", "filter_sweep", "drum_drop", "acapella_overlay"];

export async function POST(req: NextRequest) {
  await connection();
  try {
    const { setPath, edits } = await req.json();
    const setAbs = path.resolve(String(setPath));
    const setsDir = path.resolve(repoRoot(), "data", "sets");
    if (!setAbs.startsWith(setsDir) || !setAbs.endsWith(".json")) {
      return NextResponse.json(
        { error: "setPath must be a json under data/sets" },
        { status: 400 },
      );
    }
    const set = JSON.parse(fs.readFileSync(setAbs, "utf8"));
    let applied = 0;
    for (const e of edits ?? []) {
      const i = Number(e?.index);
      const t = String(e?.type);
      if (
        Number.isInteger(i) && i >= 0 && i < set.tracks.length - 1
        && TYPES.includes(t)
      ) {
        set.tracks[i].transition_out = {
          ...set.tracks[i].transition_out,
          type: t,
        };
        applied++;
      }
    }
    fs.writeFileSync(setAbs, JSON.stringify(set, null, 2));
    return NextResponse.json({ ok: true, applied });
  } catch (e: any) {
    return NextResponse.json({ error: String(e) }, { status: 500 });
  }
}
