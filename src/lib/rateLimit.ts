import "server-only";

import { Ratelimit } from "@upstash/ratelimit";
import { getRedis, withTimeout } from "@/lib/redis";

/**
 * Rate limiting with two backends behind one function:
 *
 *  - Redis (Upstash) sliding window — shared across every serverless instance, so the limit
 *    is real. Used whenever UPSTASH_REDIS_REST_* is configured.
 *  - In-memory fixed window — per warm instance, resets on cold start. Used when Redis is not
 *    configured (local dev) or when Redis errors. It degrades to "weaker limit", never to
 *    "no limit" and never to "request fails".
 */

export type LimitResult = { ok: boolean; retryAfter: number };

// --- in-memory backend ---------------------------------------------------------

type Bucket = { count: number; resetAt: number };
const buckets = new Map<string, Bucket>();

function memoryHit(key: string, limit: number, windowMs: number): LimitResult {
  const now = Date.now();
  if (buckets.size >= 5000) {
    for (const [k, v] of buckets) if (v.resetAt <= now) buckets.delete(k);
  }
  const existing = buckets.get(key);
  if (!existing || existing.resetAt <= now) {
    buckets.set(key, { count: 1, resetAt: now + windowMs });
    return { ok: true, retryAfter: 0 };
  }
  existing.count += 1;
  if (existing.count > limit) {
    return { ok: false, retryAfter: Math.ceil((existing.resetAt - now) / 1000) };
  }
  return { ok: true, retryAfter: 0 };
}

// --- Redis backend -------------------------------------------------------------

const limiters = new Map<string, Ratelimit>();

function redisLimiter(limit: number, windowMs: number): Ratelimit | null {
  const redis = getRedis();
  if (!redis) return null;
  const id = `${limit}:${windowMs}`;
  let rl = limiters.get(id);
  if (!rl) {
    rl = new Ratelimit({
      redis,
      limiter: Ratelimit.slidingWindow(limit, `${windowMs} ms`),
      prefix: "rl",
      // The library's own timeout resolves { success: true } (allow) when Redis is slow, which
      // would silently disable limiting. Disable it; rateLimit() enforces a timeout that falls
      // back to the in-memory limiter instead.
      timeout: 0,
    });
    limiters.set(id, rl);
  }
  return rl;
}

/**
 * Count one hit against `key`. Returns { ok: false, retryAfter } (seconds) once `limit` hits
 * have been made within `windowMs`.
 */
export async function rateLimit(key: string, limit: number, windowMs: number): Promise<LimitResult> {
  const rl = redisLimiter(limit, windowMs);
  if (rl) {
    try {
      const r = await withTimeout(rl.limit(key));
      return { ok: r.success, retryAfter: r.success ? 0 : Math.max(1, Math.ceil((r.reset - Date.now()) / 1000)) };
    } catch (e) {
      console.warn("[rateLimit] Redis failed or timed out, using in-memory limiter:", e instanceof Error ? e.message : e);
    }
  }
  return memoryHit(key, limit, windowMs);
}

/** Test hook. */
export function __resetRateLimitForTests(): void {
  buckets.clear();
  limiters.clear();
}
