import "server-only";

import { getRedis, withTimeout } from "@/lib/redis";

/**
 * Thin JSON cache over Redis.
 *
 * Design rules:
 *  - NO in-memory fallback. With several serverless instances an in-memory cache can't be
 *    invalidated consistently, so it would serve stale data after an upload. Without Redis the
 *    cache is a no-op and every request computes fresh results.
 *  - Fail open. Any Redis error or timeout (1.5s) is treated as a miss / skipped write; the cache can slow things
 *    down by one failed call but can never break a request.
 *  - Per-user invalidation by version: every user has a counter embedded in their cache keys.
 *    Writes (ingest, delete) bump it, which orphans all of that user's old entries at once
 *    (they expire by TTL) — no key scanning, no stale reads.
 */

export async function cacheGet<T>(key: string): Promise<T | null> {
  const redis = getRedis();
  if (!redis) return null;
  try {
    return (await withTimeout(redis.get<T>(key))) ?? null;
  } catch (e) {
    console.warn("[cache] get failed:", e instanceof Error ? e.message : e);
    return null;
  }
}

export async function cacheSet(key: string, value: unknown, ttlSeconds: number): Promise<void> {
  const redis = getRedis();
  if (!redis) return;
  try {
    await withTimeout(redis.set(key, value, { ex: ttlSeconds }));
  } catch (e) {
    console.warn("[cache] set failed:", e instanceof Error ? e.message : e);
  }
}

const versionKey = (userId: string) => `ver:${userId}`;

/** Current cache generation for a user (0 if never bumped or Redis is unavailable). */
export async function getUserCacheVersion(userId: string): Promise<number> {
  const redis = getRedis();
  if (!redis) return 0;
  try {
    return Number((await withTimeout(redis.get<number>(versionKey(userId)))) ?? 0);
  } catch {
    return 0;
  }
}

/** Invalidate everything cached for this user. Call after any write to their documents/chunks. */
export async function bumpUserCacheVersion(userId: string): Promise<void> {
  const redis = getRedis();
  if (!redis) return;
  try {
    await withTimeout(redis.incr(versionKey(userId)));
  } catch (e) {
    console.warn("[cache] version bump failed:", e instanceof Error ? e.message : e);
  }
}
