import "server-only";

import { Redis } from "@upstash/redis";

/**
 * Optional Redis (Upstash REST) client shared by the rate limiter and the cache.
 * Returns null when UPSTASH_REDIS_REST_URL / UPSTASH_REDIS_REST_TOKEN are unset, in which
 * case callers degrade: rate limiting falls back to in-memory, caching becomes a no-op.
 * (Upstash's REST transport is stateless HTTP, so it works in serverless with no connection pool.)
 */
let client: Redis | null | undefined;

export function getRedis(): Redis | null {
  if (client !== undefined) return client;
  const url = process.env.UPSTASH_REDIS_REST_URL?.trim();
  const token = process.env.UPSTASH_REDIS_REST_TOKEN?.trim();
  // retries: 1 — the default (5 with backoff) can stall a request for seconds when Redis is degraded.
  client = url && token ? new Redis({ url, token, retry: { retries: 1 } }) : null;
  return client;
}

/** Test hook: forget the memoised client so env changes take effect. */
export function __resetRedisForTests(): void {
  client = undefined;
}

/** Max time any single Redis call may add to a request before we give up and degrade. */
export const REDIS_TIMEOUT_MS = 1500;

/** Reject if `p` has not settled within `ms`. Callers decide how to degrade (fail open / memory). */
export function withTimeout<T>(p: Promise<T>, ms: number = REDIS_TIMEOUT_MS): Promise<T> {
  let timer: ReturnType<typeof setTimeout>;
  const timeout = new Promise<never>((_, reject) => {
    timer = setTimeout(() => reject(new Error(`Redis call timed out after ${ms}ms`)), ms);
  });
  return Promise.race([p, timeout]).finally(() => clearTimeout(timer));
}
