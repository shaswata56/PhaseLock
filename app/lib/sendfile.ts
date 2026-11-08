import fs from "fs";
import path from "path";
import { NextRequest, NextResponse } from "next/server";
import { Readable } from "stream";

/** Range-aware audio file streaming shared by preview and the vault. */

const MIME: Record<string, string> = {
  ".mp3": "audio/mpeg",
  ".flac": "audio/flac",
  ".wav": "audio/wav",
  ".m4a": "audio/mp4",
  ".ogg": "audio/ogg",
  ".opus": "audio/ogg",
};

export function sendAudio(req: NextRequest, abs: string): NextResponse {
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
