import { NextRequest, NextResponse, connection } from "next/server";
import path from "path";
import { runCli } from "../../../lib/phaselock";

export async function GET(req: NextRequest) {
  await connection();
  const p = req.nextUrl.searchParams.get("path");
  if (!p) return NextResponse.json({ error: "path required" }, { status: 400 });
  try {
    // the CLI validates the path against the library DB (no traversal)
    const body = await runCli(["curves", path.resolve(p)], 30_000);
    return new NextResponse(body, {
      headers: { "content-type": "application/json" },
    });
  } catch (e: any) {
    return NextResponse.json({ error: String(e) }, { status: 500 });
  }
}
