import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/redis", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/redis")>()),
  getRedis: vi.fn(),
}));

import { getRedis } from "@/lib/redis";
import { bumpUserCacheVersion, cacheGet, cacheSet, getUserCacheVersion } from "@/lib/cache";
import { retrievalCacheKey, wantsFreshResults } from "@/lib/retrievalCache";

/** Tiny in-memory stand-in for the subset of the Upstash client we use. */
function fakeRedis() {
  const store = new Map<string, unknown>();
  const sets: Array<{ key: string; ex?: number }> = [];
  return {
    store,
    sets,
    get: vi.fn(async (k: string) => (store.has(k) ? store.get(k) : null)),
    set: vi.fn(async (k: string, v: unknown, opts?: { ex?: number }) => {
      store.set(k, v);
      sets.push({ key: k, ex: opts?.ex });
    }),
    incr: vi.fn(async (k: string) => {
      const n = Number(store.get(k) ?? 0) + 1;
      store.set(k, n);
      return n;
    }),
  };
}

beforeEach(() => vi.clearAllMocks());

describe("cache without Redis configured", () => {
  beforeEach(() => vi.mocked(getRedis).mockReturnValue(null));

  it("is a harmless no-op (miss, silent write, version 0)", async () => {
    expect(await cacheGet("k")).toBeNull();
    await expect(cacheSet("k", { a: 1 }, 60)).resolves.toBeUndefined();
    await expect(bumpUserCacheVersion("u")).resolves.toBeUndefined();
    expect(await getUserCacheVersion("u")).toBe(0);
  });
});

describe("cache with Redis", () => {
  it("round-trips a value and passes the TTL", async () => {
    const r = fakeRedis();
    vi.mocked(getRedis).mockReturnValue(r as never);
    await cacheSet("k", { hello: "world" }, 300);
    expect(r.sets[0]).toEqual({ key: "k", ex: 300 });
    expect(await cacheGet("k")).toEqual({ hello: "world" });
  });

  it("fails open: Redis errors become a miss / skipped write, never an exception", async () => {
    const r = fakeRedis();
    r.get.mockRejectedValue(new Error("boom"));
    r.set.mockRejectedValue(new Error("boom"));
    r.incr.mockRejectedValue(new Error("boom"));
    vi.mocked(getRedis).mockReturnValue(r as never);
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    expect(await cacheGet("k")).toBeNull();
    await expect(cacheSet("k", 1, 10)).resolves.toBeUndefined();
    await expect(bumpUserCacheVersion("u")).resolves.toBeUndefined();
    expect(await getUserCacheVersion("u")).toBe(0);
    warn.mockRestore();
  });

  it("a hung Redis call times out into a miss instead of stalling the request", async () => {
    vi.useFakeTimers();
    const r = fakeRedis();
    r.get.mockImplementation(() => new Promise(() => {}));
    vi.mocked(getRedis).mockReturnValue(r as never);
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const pending = cacheGet("k");
    await vi.advanceTimersByTimeAsync(2000);
    expect(await pending).toBeNull();
    warn.mockRestore();
    vi.useRealTimers();
  });

  it("bumping a user's version makes their old retrieval entries unreachable", async () => {
    const r = fakeRedis();
    vi.mocked(getRedis).mockReturnValue(r as never);
    const shape = { query: "q", topK: 5, useHybrid: false, rerank: false, rewrite: false, filterDocumentId: null };

    const keyBefore = retrievalCacheKey("u1", await getUserCacheVersion("u1"), shape);
    await cacheSet(keyBefore, { results: ["old"] }, 300);
    expect(await cacheGet(keyBefore)).toEqual({ results: ["old"] });

    await bumpUserCacheVersion("u1"); // e.g. a document was just uploaded
    const keyAfter = retrievalCacheKey("u1", await getUserCacheVersion("u1"), shape);
    expect(keyAfter).not.toBe(keyBefore);
    expect(await cacheGet(keyAfter)).toBeNull();
  });
});

describe("retrievalCacheKey", () => {
  const shape = { query: "q", topK: 5, useHybrid: false, rerank: false, rewrite: false, filterDocumentId: null };

  it("is deterministic and scoped to user and generation", () => {
    expect(retrievalCacheKey("u1", 0, shape)).toBe(retrievalCacheKey("u1", 0, { ...shape }));
    expect(retrievalCacheKey("u1", 0, shape)).not.toBe(retrievalCacheKey("u2", 0, shape));
    expect(retrievalCacheKey("u1", 0, shape)).not.toBe(retrievalCacheKey("u1", 1, shape));
  });

  it.each([
    ["query", { query: "other" }],
    ["topK", { topK: 6 }],
    ["hybrid", { useHybrid: true }],
    ["rerank", { rerank: true }],
    ["rewrite", { rewrite: true }],
    ["document filter", { filterDocumentId: "123e4567-e89b-42d3-a456-426614174000" }],
  ])("changes when %s changes", (_n, change) => {
    expect(retrievalCacheKey("u1", 0, { ...shape, ...change })).not.toBe(retrievalCacheKey("u1", 0, shape));
  });
});

describe("wantsFreshResults", () => {
  const r = (h: Record<string, string>) => new Request("http://x", { headers: h });
  it("honours Cache-Control no-cache / no-store", () => {
    expect(wantsFreshResults(r({ "cache-control": "no-cache" }))).toBe(true);
    expect(wantsFreshResults(r({ "cache-control": "No-Store" }))).toBe(true);
    expect(wantsFreshResults(r({}))).toBe(false);
    expect(wantsFreshResults(r({ "cache-control": "max-age=60" }))).toBe(false);
  });
});
