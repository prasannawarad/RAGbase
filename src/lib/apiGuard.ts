import "server-only";
import type { NextResponse } from "next/server";
import { apiError } from "@/lib/api/http";
import { __resetRateLimitForTests, rateLimit } from "@/lib/rateLimit";

/**
 * Abuse guard for the API routes, in two stages around authentication:
 *
 *  guardRequest (before auth)
 *    1. Same-origin check — rejects browser requests coming from a page NOT served by this
 *       deployment (Origin/Referer allowlist), so another site can't drive your endpoints.
 *    2. Per-IP rate limit.
 *  guardUser / guardDailyBudget (after auth)
 *    3. Per-user rate limit and daily budget, so rotating IPs doesn't evade the limits.
 *
 * Limiter backend (src/lib/rateLimit.ts): Upstash Redis sliding window when
 * UPSTASH_REDIS_REST_URL/TOKEN are set (shared across all serverless instances — a real limit);
 * otherwise an in-memory window that is per warm instance and resets on cold start.
 *
 * LIMITATIONS (be honest in an interview):
 *  - Without Redis configured the limit is only best-effort per instance.
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

// --- Rate limiting is delegated to src/lib/rateLimit.ts (Redis when configured, else memory) ---

function clientIp(req: Request): string {
  const fwd = req.headers.get("x-forwarded-for");
  if (fwd) return fwd.split(",")[0]!.trim();
  return req.headers.get("x-real-ip")?.trim() || "unknown";
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
 * Layer 1 (before authentication): same-origin check + per-IP rate limit. Returns a blocking
 * NextResponse (403 or 429) to return from the route handler, or null when the request may proceed.
 *
 *   const blocked = await guardRequest(req, { name: "chat", limit: 20 });
 *   if (blocked) return blocked;
 */
export async function guardRequest(req: Request, opts: GuardOptions): Promise<NextResponse | null> {
  const origin = checkOrigin(req);
  if (origin) return origin;

  const { ok, retryAfter } = await rateLimit(
    `ip:${opts.name}:${clientIp(req)}`,
    opts.limit ?? 30,
    opts.windowMs ?? 60_000
  );
  return ok ? null : tooMany(retryAfter);
}

/**
 * Layer 2 (after authentication): per-USER limit. The IP limit alone is not enough — one user can
 * rotate IPs, and one IP (an office, a university) can hold many legitimate users.
 */
export async function guardUser(userId: string, opts: GuardOptions): Promise<NextResponse | null> {
  const { ok, retryAfter } = await rateLimit(
    `user:${opts.name}:${userId}`,
    opts.limit ?? 30,
    opts.windowMs ?? 60_000
  );
  return ok ? null : tooMany(retryAfter);
}

const DAY_MS = 24 * 3600 * 1000;

/** Per-user daily budget for an expensive action (protects shared free-tier LLM quota). */
export function guardDailyBudget(userId: string, name: string, perDay: number): Promise<NextResponse | null> {
  return guardUser(userId, { name: `${name}-daily`, limit: perDay, windowMs: DAY_MS });
}

function tooMany(retryAfter: number): NextResponse {
  const res = apiError(429, "rate_limited", "Too many requests. Please slow down.", { retryAfter });
  res.headers.set("Retry-After", String(retryAfter));
  return res;
}

/** Test hook: clears all rate-limit state. */
export function __resetGuardForTests(): void {
  __resetRateLimitForTests();
}
