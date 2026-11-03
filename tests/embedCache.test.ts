import { beforeEach, describe, expect, it, vi } from "vitest";

const cacheGet = vi.fn();
const cacheSet = vi.fn();
const embedTexts = vi.fn();

vi.mock("@/lib/cache", () => ({ cacheGet: (...a: unknown[]) => cacheGet(...a), cacheSet: (...a: unknown[]) => cacheSet(...a) }));
vi.mock("@/lib/gemini", () => ({
  EMBEDDING_DIMS: 768,
  EMBEDDING_MODEL: "gemini-embedding-001",
  embedTexts: (...a: unknown[]) => embedTexts(...a),
}));

import { embeddingCacheKey, embedQueryCached } from "@/lib/embedCache";

const vec = (n: number) => new Array(768).fill(n);

beforeEach(() => vi.clearAllMocks());

describe("embeddingCacheKey", () => {
  it("includes model, dimensions and a text hash", () => {
    const k = embeddingCacheKey("hello");
    expect(k).toMatch(/^emb:v1:gemini-embedding-001:768:[0-9a-f]{64}$/);
    expect(embeddingCacheKey("hello")).toBe(k);
    expect(embeddingCacheKey("hello!")).not.toBe(k);
  });
});

describe("embedQueryCached", () => {
  it("returns a cached vector without calling Gemini", async () => {
    cacheGet.mockResolvedValue(vec(0.5));
    const r = await embedQueryCached("q");
    expect(r.cached).toBe(true);
    expect(embedTexts).not.toHaveBeenCalled();
  });

  it("on a miss, calls Gemini once and stores the vector with a TTL", async () => {
    cacheGet.mockResolvedValue(null);
    embedTexts.mockResolvedValue([vec(0.25)]);
    const r = await embedQueryCached("q");
    expect(r.cached).toBe(false);
    expect(embedTexts).toHaveBeenCalledTimes(1);
    expect(cacheSet).toHaveBeenCalledWith(embeddingCacheKey("q"), vec(0.25), expect.any(Number));
  });

  it("ignores a malformed cached value (wrong dimensions) and recomputes", async () => {
    cacheGet.mockResolvedValue([1, 2, 3]);
    embedTexts.mockResolvedValue([vec(0.1)]);
    expect((await embedQueryCached("q")).cached).toBe(false);
  });

  it("never caches when Gemini fails — so the mock fallback can't poison the cache", async () => {
    cacheGet.mockResolvedValue(null);
    embedTexts.mockRejectedValue(new Error("quota"));
    await expect(embedQueryCached("q")).rejects.toThrow("quota");
    expect(cacheSet).not.toHaveBeenCalled();
  });

  it("rejects (and does not cache) a wrong-shaped embedding from Gemini", async () => {
    cacheGet.mockResolvedValue(null);
    embedTexts.mockResolvedValue([[1, 2, 3]]);
    await expect(embedQueryCached("q")).rejects.toThrow();
    expect(cacheSet).not.toHaveBeenCalled();
  });
});
