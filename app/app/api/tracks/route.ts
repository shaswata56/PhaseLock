import { NextResponse, connection } from "next/server";
import { runCli } from "../../../lib/phaselock";

let cache: { at: number; body: string } | null = null;

export async function GET() {
  await connection(); // always fresh — this reads the live library DB
  try {
    if (!cache || Date.now() - cache.at > 60_000) {
      const body = await runCli(["tracks"]);
      cache = { at: Date.now(), body };
    }
    return new NextResponse(cache.body, {
      headers: { "content-type": "application/json" },
    });
  } catch (e: any) {
    return NextResponse.json({ error: String(e) }, { status: 500 });
  }
}
