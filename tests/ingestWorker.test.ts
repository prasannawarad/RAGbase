import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  processDocument,
  type ChunkWrite,
  type DocRow,
  type IngestStore,
  type StagedChunk,
} from "@/lib/ingestWorker";
import { contentHash } from "@/lib/ingestStore";

/** In-memory IngestStore that mimics the database semantics the worker relies on. */
function fakeStore(initial: Partial<DocRow> & { chunks?: string[] }) {
  const texts = initial.chunks ?? ["a", "b", "c", "d", "e"];
  const doc: DocRow = {
    id: "doc1",
    name: "doc.txt",
    status: "queued",
    chunk_count: texts.length,
    chunks_done: 0,
    error: null,
    ...initial,
  };
  const staged: (StagedChunk & { claimed: boolean })[] = texts.map((content, i) => ({
    id: `s${i}`,
    chunk_index: i,
    content,
    attempts: 0,
    claimed: false,
  }));
  const written = new Map<number, ChunkWrite>();
  const calls = { claim: 0, release: 0, reset: 0 };

  const store: IngestStore = {
    getDocument: async (id) => (id === doc.id ? { ...doc } : null),
    claimBatch: async (_id, limit) => {
      calls.claim++;
      const rows = staged.filter((r) => !r.claimed).slice(0, limit);
      rows.forEach((r) => {
        r.claimed = true;
        r.attempts += 1;
      });
      return rows.map(({ claimed: _c, ...r }) => ({ ...r }));
    },
    writeChunks: async (_d, rows) => rows.forEach((r) => written.set(r.chunk_index, r)),
    deleteStaged: async (ids) => {
      for (const id of ids) staged.splice(staged.findIndex((r) => r.id === id), 1);
    },
    releaseClaims: async (ids) => {
      calls.release++;
      staged.filter((r) => ids.includes(r.id)).forEach((r) => (r.claimed = false));
    },
    resetStaged: async () => {
      calls.reset++;
      staged.forEach((r) => {
        r.attempts = 0;
        r.claimed = false;
      });
    },
    countStaged: async () => staged.length,
    countChunks: async () => written.size,
    updateDocument: async (_id, patch) => {
      Object.assign(doc, patch);
    },
  };
  return { store, doc, staged, written, calls };
}

const okEmbed = vi.fn(async (texts: string[]) => texts.map(() => [0.1, 0.2]));

beforeEach(() => {
  okEmbed.mockClear();
});

describe("processDocument", () => {
  it("returns null for an unknown / invisible document", async () => {
    const { store } = fakeStore({});
    expect(await processDocument({ ...store, getDocument: async () => null }, "nope", { embed: okEmbed })).toBeNull();
  });

  it("embeds every chunk in small batches and finishes as indexed", async () => {
    const { store, doc, written, staged } = fakeStore({ chunks: Array.from({ length: 20 }, (_, i) => `t${i}`) });
    const r = await processDocument(store, "doc1", { embed: okEmbed, batchSize: 8 });
    expect(r).toMatchObject({ status: "indexed", chunksDone: 20, chunkCount: 20 });
    expect(written.size).toBe(20);
    expect(staged).toHaveLength(0);
    expect(doc.status).toBe("indexed");
    expect(okEmbed).toHaveBeenCalledTimes(3); // 8 + 8 + 4
    expect(Math.max(...okEmbed.mock.calls.map((c) => c[0].length))).toBeLessThanOrEqual(8);
  });

  it("is a no-op for an already indexed document", async () => {
    const { store } = fakeStore({ status: "indexed", chunks_done: 5 });
    const r = await processDocument(store, "doc1", { embed: okEmbed });
    expect(r).toMatchObject({ status: "indexed", chunksDone: 5 });
    expect(okEmbed).not.toHaveBeenCalled();
  });

  it("a transient embedding failure releases the claim and reports a retry, not a failure", async () => {
    const { store, doc, staged, calls } = fakeStore({});
    const embed = vi.fn().mockRejectedValueOnce(new Error("429 quota"));
    const r = await processDocument(store, "doc1", { embed });
    expect(r).toMatchObject({ status: "processing", retrying: true, error: "429 quota" });
    expect(doc.status).toBe("processing");
    expect(calls.release).toBe(1);
    expect(staged.every((s) => !s.claimed)).toBe(true); // claimable again
    expect(staged).toHaveLength(5); // nothing lost

    // next call succeeds from where it left off
    const r2 = await processDocument(store, "doc1", { embed: okEmbed });
    expect(r2).toMatchObject({ status: "indexed", chunksDone: 5 });
  });

  it("marks the document failed after the attempt limit and keeps the staged text", async () => {
    const { store, doc, staged } = fakeStore({ chunks: ["only"] });
    const embed = vi.fn().mockRejectedValue(new Error("gemini down"));
    let r;
    for (let i = 0; i < 3; i++) r = await processDocument(store, "doc1", { embed, maxAttempts: 3 });
    expect(r).toMatchObject({ status: "failed", error: "gemini down" });
    expect(doc.status).toBe("failed");
    expect(doc.error).toBe("gemini down");
    expect(staged).toHaveLength(1); // retained for a manual retry
  });

  it("a failed document stays failed unless retry is requested", async () => {
    const { store, calls } = fakeStore({ status: "failed", error: "gemini down" });
    const r = await processDocument(store, "doc1", { embed: okEmbed });
    expect(r).toMatchObject({ status: "failed", error: "gemini down" });
    expect(calls.claim).toBe(0);
    expect(okEmbed).not.toHaveBeenCalled();
  });

  it("retry resets attempts and finishes the document", async () => {
    const { store, doc, calls } = fakeStore({ status: "failed", error: "gemini down" });
    const r = await processDocument(store, "doc1", { embed: okEmbed, retry: true });
    expect(calls.reset).toBe(1);
    expect(r).toMatchObject({ status: "indexed", chunksDone: 5 });
    expect(doc.error).toBeNull();
  });

  it("reports 'processing' (not indexed) when another worker holds the remaining rows", async () => {
    const { store, staged } = fakeStore({ status: "processing" });
    staged.forEach((s) => (s.claimed = true)); // someone else's live lease
    const r = await processDocument(store, "doc1", { embed: okEmbed });
    expect(r).toMatchObject({ status: "processing", chunksDone: 0 });
    expect(okEmbed).not.toHaveBeenCalled();
  });

  it("stops at the time budget and leaves the rest for the next call", async () => {
    const { store } = fakeStore({ chunks: Array.from({ length: 30 }, (_, i) => `t${i}`) });
    let t = 0;
    const r = await processDocument(store, "doc1", {
      embed: okEmbed,
      batchSize: 8,
      budgetMs: 1000,
      now: () => (t += 600), // each loop check advances the clock
    });
    expect(r?.status).toBe("processing");
    expect(r!.chunksDone).toBeGreaterThan(0);
    expect(r!.chunksDone).toBeLessThan(30);
  });

  it("is idempotent when re-run after a crash between write and delete (no duplicate chunks)", async () => {
    const { store, written, staged } = fakeStore({ chunks: ["a", "b"] });
    // Simulate: chunks were written but the worker died before deleting staging rows + lease expired.
    written.set(0, { chunk_index: 0, content: "a", embedding: [1] });
    written.set(1, { chunk_index: 1, content: "b", embedding: [1] });
    const r = await processDocument(store, "doc1", { embed: okEmbed });
    expect(r).toMatchObject({ status: "indexed", chunksDone: 2 });
    expect(written.size).toBe(2); // upsert by chunk_index, not appended
    expect(staged).toHaveLength(0);
  });

  it("rejects an embedding-count mismatch instead of writing misaligned vectors", async () => {
    const { store, written } = fakeStore({ chunks: ["a", "b"] });
    const bad = vi.fn(async () => [[1]]); // 1 vector for 2 texts
    const r = await processDocument(store, "doc1", { embed: bad });
    expect(r).toMatchObject({ status: "processing", retrying: true });
    expect(written.size).toBe(0);
  });
});

describe("contentHash", () => {
  it("is stable for the same chunks and differs when text or order changes", () => {
    expect(contentHash(["a", "b"])).toBe(contentHash(["a", "b"]));
    expect(contentHash(["a", "b"])).not.toBe(contentHash(["b", "a"]));
    expect(contentHash(["a", "b"])).not.toBe(contentHash(["a", "c"]));
    expect(contentHash(["a"])).toMatch(/^[0-9a-f]{64}$/);
  });
});
