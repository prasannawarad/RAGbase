/**
 * Resumable ingestion worker (Stage 4). Pure orchestration over an `IngestStore`, so the
 * retry / lease / failure behaviour is unit-testable without a database (see tests/ingestWorker.test.ts)
 * and the same code can later be driven by a queue instead of the browser.
 *
 * Flow per call:  claim a small batch -> embed -> write real chunk rows -> delete staging rows,
 * repeating until the staging table is empty or the time budget is spent.
 *
 * Failure rules (the part that matters):
 *  - Embedding fails  -> release the claims so they can be retried; after MAX_ATTEMPTS the
 *    document is marked 'failed' with the error (staged text is kept so it can be retried).
 *  - Worker dies mid-batch -> the lease expires and another call re-claims the rows.
 *  - Writes are upserts keyed (document_id, chunk_index), so re-processing never duplicates.
 *  - No claimable rows but rows still staged -> another worker holds them; report 'processing'.
 */

export type DocStatus = "queued" | "processing" | "indexed" | "failed";

export type DocRow = {
  id: string;
  name: string;
  status: DocStatus;
  chunk_count: number;
  chunks_done: number;
  error: string | null;
};

export type StagedChunk = { id: string; chunk_index: number; content: string; attempts: number };
export type ChunkWrite = { chunk_index: number; content: string; embedding: number[] };

export interface IngestStore {
  getDocument(id: string): Promise<DocRow | null>;
  claimBatch(documentId: string, limit: number): Promise<StagedChunk[]>;
  /** Upsert real chunk rows (idempotent on document_id + chunk_index). */
  writeChunks(doc: DocRow, rows: ChunkWrite[]): Promise<void>;
  deleteStaged(ids: string[]): Promise<void>;
  /** Give claimed rows back (claimed_at = null) so the next call can retry them. */
  releaseClaims(ids: string[]): Promise<void>;
  /** Reset attempts/claims for a failed document so it can run again. */
  resetStaged(documentId: string): Promise<void>;
  countStaged(documentId: string): Promise<number>;
  countChunks(documentId: string): Promise<number>;
  updateDocument(id: string, patch: Partial<Pick<DocRow, "status" | "chunks_done" | "error">>): Promise<void>;
}

export type ProcessResult = {
  status: DocStatus;
  chunksDone: number;
  chunkCount: number;
  error?: string;
  /** True when this call hit an embedding error but will be retried (not yet failed). */
  retrying?: boolean;
};

export type WorkerOptions = {
  embed: (texts: string[]) => Promise<number[][]>;
  batchSize?: number;
  maxAttempts?: number;
  budgetMs?: number;
  retry?: boolean;
  now?: () => number;
};

export const INGEST_BATCH_SIZE = 8;
export const INGEST_MAX_ATTEMPTS = 3;
export const INGEST_BUDGET_MS = 20_000;

const result = (doc: Pick<DocRow, "chunk_count">, status: DocStatus, done: number, extra: Partial<ProcessResult> = {}): ProcessResult => ({
  status,
  chunksDone: done,
  chunkCount: doc.chunk_count,
  ...extra,
});

/** Returns null when the document does not exist (or is not visible to this user). */
export async function processDocument(
  store: IngestStore,
  documentId: string,
  opts: WorkerOptions
): Promise<ProcessResult | null> {
  const batchSize = opts.batchSize ?? INGEST_BATCH_SIZE;
  const maxAttempts = opts.maxAttempts ?? INGEST_MAX_ATTEMPTS;
  const budgetMs = opts.budgetMs ?? INGEST_BUDGET_MS;
  const now = opts.now ?? Date.now;
  const started = now();

  let doc = await store.getDocument(documentId);
  if (!doc) return null;

  if (doc.status === "indexed") return result(doc, "indexed", doc.chunk_count);

  if (doc.status === "failed") {
    if (!opts.retry) return result(doc, "failed", doc.chunks_done, { error: doc.error ?? "Ingestion failed" });
    await store.resetStaged(doc.id);
    await store.updateDocument(doc.id, { status: "processing", error: null });
    doc = { ...doc, status: "processing", error: null };
  } else if (doc.status === "queued") {
    await store.updateDocument(doc.id, { status: "processing" });
    doc = { ...doc, status: "processing" };
  }

  while (now() - started < budgetMs) {
    const batch = await store.claimBatch(doc.id, batchSize);

    if (batch.length === 0) {
      // Nothing claimable. Either we are finished, or another worker holds the remaining rows.
      const staged = await store.countStaged(doc.id);
      const done = await store.countChunks(doc.id);
      if (staged === 0) {
        await store.updateDocument(doc.id, { status: "indexed", chunks_done: done, error: null });
        return result(doc, "indexed", done);
      }
      return result(doc, "processing", done);
    }

    const ids = batch.map((b) => b.id);
    try {
      const embeddings = await opts.embed(batch.map((b) => b.content));
      if (embeddings.length !== batch.length) throw new Error("Embedding count mismatch");
      await store.writeChunks(
        doc,
        batch.map((b, i) => ({ chunk_index: b.chunk_index, content: b.content, embedding: embeddings[i]! }))
      );
      await store.deleteStaged(ids);
      await store.updateDocument(doc.id, { chunks_done: await store.countChunks(doc.id) });
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      const done = await store.countChunks(doc.id);
      if (Math.max(...batch.map((b) => b.attempts)) >= maxAttempts) {
        await store.updateDocument(doc.id, { status: "failed", error: message, chunks_done: done });
        return result(doc, "failed", done, { error: message });
      }
      await store.releaseClaims(ids);
      return result(doc, "processing", done, { retrying: true, error: message });
    }
  }

  // Time budget spent with work remaining: the caller simply calls again.
  return result(doc, "processing", await store.countChunks(doc.id));
}
