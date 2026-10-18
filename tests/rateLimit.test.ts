import { beforeEach, describe, expect, it, vi } from "vitest";

const limitFn = vi.fn();
vi.mock("@/lib/redis", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/redis")>()),
  getRedis: vi.fn(),
}));
vi.mock("@upstash/ratelimit", () => {
  class Ratelimit {
    static slidingWindow = vi.fn(() => ({}));
    limit = limitFn;
  }
  return { Ratelimit };
});

import { getRedis } from "@/lib/redis";
import { __resetRateLimitForTests, rateLimit } from "@/lib/rateLimit";

beforeEach(() => {
  vi.clearAllMocks();
  __resetRateLimitForTests();
});

describe("rateLimit: in-memory backend (no Redis)", () => {
  beforeEach(() => vi.mocked(getRedis).mockReturnValue(null));

  it("allows up to the limit, then blocks with a positive retryAfter", async () => {
    expect((await rateLimit("k", 2, 60_000)).ok).toBe(true);
    expect((await rateLimit("k", 2, 60_000)).ok).toBe(true);
    const r = await rateLimit("k", 2, 60_000);
    expect(r.ok).toBe(false);
    expect(r.retryAfter).toBeGreaterThan(0);
  });

  it("keeps keys independent", async () => {
    await rateLimit("a", 1, 60_000);
    expect((await rateLimit("a", 1, 60_000)).ok).toBe(false);
    expect((await rateLimit("b", 1, 60_000)).ok).toBe(true);
  });

  it("starts a fresh window after it expires", async () => {
    vi.useFakeTimers();
    await rateLimit("k", 1, 1000);
    expect((await rateLimit("k", 1, 1000)).ok).toBe(false);
    vi.advanceTimersByTime(1001);
    expect((await rateLimit("k", 1, 1000)).ok).toBe(true);
    vi.useRealTimers();
  });
});

describe("rateLimit: Redis backend", () => {
  beforeEach(() => vi.mocked(getRedis).mockReturnValue({} as never));

  it("uses the Redis limiter's verdict", async () => {
    limitFn.mockResolvedValueOnce({ success: false, reset: Date.now() + 5000 });
    const r = await rateLimit("k", 5, 60_000);
    expect(r.ok).toBe(false);
    expect(r.retryAfter).toBeGreaterThanOrEqual(4);
    limitFn.mockResolvedValueOnce({ success: true, reset: Date.now() + 5000 });
    expect(await rateLimit("k", 5, 60_000)).toEqual({ ok: true, retryAfter: 0 });
  });

  it("falls back to the in-memory limiter (still enforcing) when Redis errors", async () => {
    limitFn.mockRejectedValue(new Error("redis down"));
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    expect((await rateLimit("k", 1, 60_000)).ok).toBe(true);
    expect((await rateLimit("k", 1, 60_000)).ok).toBe(false); // limit still applies
    expect(warn).toHaveBeenCalled();
    warn.mockRestore();
  });

  it("treats a hung Redis as a failure (not as 'allowed') and uses the in-memory limiter", async () => {
    vi.useFakeTimers();
    limitFn.mockImplementation(() => new Promise(() => {})); // never settles
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const first = rateLimit("slow", 1, 60_000);
    await vi.advanceTimersByTimeAsync(2000);
    expect((await first).ok).toBe(true);
    const second = rateLimit("slow", 1, 60_000);
    await vi.advanceTimersByTimeAsync(2000);
    expect((await second).ok).toBe(false); // limit enforced despite Redis being down
    warn.mockRestore();
    vi.useRealTimers();
  });
});
