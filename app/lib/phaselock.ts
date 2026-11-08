import { spawn } from "child_process";
import path from "path";
import fs from "fs";

/** The web app shells out to the phaselock CLI — one source of truth. */

export function repoRoot(): string {
  // Next runs from app/ (dev and start); repo root is one level up.
  let root = path.resolve(process.cwd(), "..");
  if (!fs.existsSync(path.join(root, ".venv"))) root = process.cwd();
  return root;
}

export function cliExe(): string {
  const exe = path.join(repoRoot(), ".venv", "Scripts", "phaselock.exe");
  if (!fs.existsSync(exe)) {
    throw new Error(`phaselock CLI not found at ${exe} — run pip install -e core`);
  }
  return exe;
}

export function runCli(args: string[], timeoutMs = 120_000): Promise<string> {
  return new Promise((resolve, reject) => {
    const child = spawn(cliExe(), args, { cwd: repoRoot() });
    let out = "";
    let err = "";
    const timer = setTimeout(() => {
      child.kill();
      reject(new Error(`phaselock ${args[0]} timed out`));
    }, timeoutMs);
    child.stdout.on("data", (d) => (out += d));
    child.stderr.on("data", (d) => (err += d));
    child.on("error", (e) => {
      clearTimeout(timer);
      reject(e);
    });
    child.on("close", (code) => {
      clearTimeout(timer);
      if (code === 0) resolve(out);
      else reject(new Error(err || `phaselock ${args[0]} exited ${code}`));
    });
  });
}

/* ----- render jobs (in-memory; personal app, one user) ----- */

export type Job = {
  id: string;
  status: "running" | "done" | "error";
  log: string[];
  out?: string; // absolute wav path
  startedAt: number;
};

const jobs = new Map<string, Job>();

export function startJob(id: string, args: string[], outPath?: string): Job {
  const job: Job = { id, status: "running", log: [], startedAt: Date.now() };
  jobs.set(id, job);
  const child = spawn(cliExe(), args, { cwd: repoRoot() });
  const push = (d: any) =>
    job.log.push(...String(d).split(/\r?\n/).filter(Boolean));
  child.stdout.on("data", push);
  child.stderr.on("data", push);
  child.on("error", (e) => {
    job.status = "error";
    job.log.push(String(e));
  });
  child.on("close", (code) => {
    if (job.status === "running") {
      job.status = code === 0 ? "done" : "error";
      if (code === 0 && outPath) job.out = outPath;
      if (code !== 0) job.log.push(`exited ${code}`);
    }
    job.log.push(
      `— ${job.status} in ${((Date.now() - job.startedAt) / 1000).toFixed(1)}s`,
    );
  });
  return job;
}

export function getJob(id: string): Job | undefined {
  return jobs.get(id);
}
