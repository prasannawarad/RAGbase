import "server-only";
import type { NextResponse } from "next/server";
import { apiError } from "@/lib/api/http";

/**
 * Lightweight abuse guard for public API routes. Two independent layers:
 *
 *  1. Same-origin check — rejects browser requests coming from a page NOT served
 *     by this deployment (classic CSRF-style Origin/Referer allowlist). This stops
 *     someone calling your endpoints from another site's page or console.
 *  2. Per-IP rate limit — a best-effort in-memory fixed window. It caps how fast a
 *     single IP can hit an endpoint, blunting scripted floods that would burn LLM /
 *     embedding quota or spam the database.
 *
 * LIMITATIONS (be honest in an interview):
 *  - The rate limiter is in-memory, so it is PER serverless instance and resets on
 *    cold start. It slows abuse; it is not a hard global quota. For strict limits,
 *    back it with Upstash Redis (`@upstash/ratelimit`) and swap `hit()` for a Redis
 *    call — the call sites below don't change.
 *  - The origin check only constrains browsers (curl/servers send no Origin). That's
 *    fine: cross-site browser abuse is what it targets; floods are the rate limiter's job.
 *
 * Allowed origins:
 *  - localhost / 127.0.0.1 when NODE_ENV !== "production"
 *  - the deployment's own host (Vercel `VERCEL_URL`, `VERCEL_PROJECT_PRODUCTION_URL`)
 *  - any host in the optional `ALLOWED_ORIGINS` env (comma-separated, e.g.
 *    "https://ragbase.example.com,https://www.ragbase.example.com")
 */

function allowedHosts(): Set<string> {
  const hosts = new Set<string>();
  if (process.env.NODE_ENV !== "production") {
    hosts.add("localhost");
    hosts.add("127.0.0.1");
  }
  // Vercel system env vars are hostnames (no protocol).
  for (const v of [process.env.VERCEL_URL, process.env.VERCEL_PROJECT_PRODUCTION_URL]) {
    if (v?.trim()) hosts.add(v.trim().toLowerCase());
  }
  for (const raw of (process.env.ALLOWED_ORIGINS ?? "").split(",")) {
    const entry = raw.trim();
    if (!entry) continue;
    hosts.add(hostOf(entry) ?? entry.toLowerCase());
  }
  return hosts;
}

/** Extract a lowercase hostname from a URL/origin string, or null if unparseable. */
function hostOf(value: string): string | null {
  try {
    return new URL(value).hostname.toLowerCase();
  } catch {
    return null;
  }
}

/**
 * Returns a 403 response if the request comes from a browser page on a host that is
 * not in the allowlist. Requests with no Origin AND no Referer (curl, server-to-server)
 * are allowed through here — the rate limiter is their backstop.
 */
function checkOrigin(req: Request): NextResponse | null {
  const source = req.headers.get("origin") ?? req.headers.get("referer");
  if (!source) return null; // non-browser client; let the rate limiter handle it

  const host = hostOf(source);
  if (host && allowedHosts().has(host)) return null;

  return apiError(403, "forbidden", "Forbidden: cross-origin request");
}

// --- Best-effort in-memory rate limiter (per warm instance) ---------------------

type Bucket = { count: number; resetAt: number };
const buckets = new Map<string, Bucket>();

function clientIp(req: Request): string {
  const fwd = req.headers.get("x-forwarded-for");
  if (fwd) return fwd.split(",")[0]!.trim();
  return req.headers.get("x-real-ip")?.trim() || "unknown";
}

/** Records a hit; returns { ok, retryAfter } where retryAfter is seconds until reset. */
function hit(bucketKey: string, limit: number, windowMs: number): { ok: boolean; retryAfter: number } {
  const now = Date.now();
  const existing = buckets.get(bucketKey);
  if (!existing || existing.resetAt <= now) {
    buckets.set(bucketKey, { count: 1, resetAt: now + windowMs });
    return { ok: true, retryAfter: 0 };
  }
  existing.count += 1;
  if (existing.count > limit) {
    return { ok: false, retryAfter: Math.ceil((existing.resetAt - now) / 1000) };
  }
  return { ok: true, retryAfter: 0 };
}

// Opportunistic cleanup so the Map can't grow unbounded on a long-lived instance.
function sweep(now: number): void {
  if (buckets.size < 5000) return;
  for (const [k, v] of buckets) if (v.resetAt <= now) buckets.delete(k);
}

export type GuardOptions = {
  /** Max requests per IP per window. Default 30. */
  limit?: number;
  /** Window length in ms. Default 60_000 (1 minute). */
  windowMs?: number;
  /** Stable name for this route, so limits are tracked per-route. */
  name: string;
};

/**
 * Run both guards. Returns a blocking NextResponse (403 or 429) to return from the
 * route handler, or null when the request may proceed.
 *
 *   const blocked = guardRequest(req, { name: "chat", limit: 20 });
 *   if (blocked) return blocked;
 */
export function guardRequest(req: Request, opts: GuardOptions): NextResponse | null {
  const origin = checkOrigin(req);
  if (origin) return origin;

  const limit = opts.limit ?? 30;
  const windowMs = opts.windowMs ?? 60_000;
  const now = Date.now();
  sweep(now);

  const { ok, retryAfter } = hit(`${opts.name}:${clientIp(req)}`, limit, windowMs);
  if (!ok) {
    const res = apiError(429, "rate_limited", "Too many requests. Please slow down.", { retryAfter });
    res.headers.set("Retry-After", String(retryAfter));
    return res;
  }
  return null;
}

/** Test hook: clears all rate-limit buckets. */
export function __resetGuardForTests(): void {
  buckets.clear();
}
