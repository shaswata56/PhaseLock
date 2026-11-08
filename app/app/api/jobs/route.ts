import { NextRequest, NextResponse, connection } from "next/server";
import fs from "fs";
import { Readable } from "stream";
import { getJob } from "../../../lib/phaselock";

export async function GET(req: NextRequest) {
  await connection(); // job state is per-request live
  const id = req.nextUrl.searchParams.get("id");
  const wantMix = req.nextUrl.searchParams.get("mix");
  if (!id) return NextResponse.json({ error: "id required" }, { status: 400 });
  const job = getJob(id);
  if (!job) return NextResponse.json({ error: "unknown job" }, { status: 404 });

  if (wantMix) {
    if (job.status !== "done" || !job.out || !fs.existsSync(job.out)) {
      return NextResponse.json({ error: "mix not ready" }, { status: 404 });
    }
    const stream = Readable.toWeb(
      fs.createReadStream(job.out),
    ) as unknown as ReadableStream;
    return new NextResponse(stream, {
      headers: {
        "content-type": "audio/wav",
        "content-length": String(fs.statSync(job.out).size),
      },
    });
  }

  return NextResponse.json({
    id: job.id,
    status: job.status,
    log: job.log.slice(-30),
    mixUrl:
      job.status === "done" && job.out
        ? `/api/jobs?id=${job.id}&mix=1`
        : undefined,
  });
}
