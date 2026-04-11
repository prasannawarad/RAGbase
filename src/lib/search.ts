import "server-only";

/**
 * Supabase pgvector search + hybrid BM25 (RRF), server-side only.
 * Uses L2 distance (<->); scores are mapped to (0,1] for UI thresholds.
 */

import { BM25 } from "@/lib/bm25";
import type { DocMetadata, VectorDoc } from "@/lib/vectorStore";
import { tokenize } from "@/lib/tokenizer";
import { getSupabaseAdmin } from "@/lib/supabase/server";

const EMBEDDING_DIM = 768;
const RRF_K = 60;

/** pgvector / PostgREST expects a string literal for vector columns. */
export function toVectorParam(embedding: number[]): string {
  if (embedding.length !== EMBEDDING_DIM) {
    throw new Error(`Expected ${EMBEDDING_DIM}-dim embedding, got ${embedding.length}`);
  }
  return `[${embedding.join(",")}]`;
}

/** Map L2 distance to a 0–1 style score (higher = closer), for thresholding. */
export function distanceToVectorScore(distance: number): number {
  return 1 / (1 + distance);
}

export type ChunkRow = {
  id: string;
  content: string;
  document_id: string;
  chunk_index: number;
  word_count: number | null;
  metadata: Record<string, unknown> | null;
};

function rowToVectorDoc(
  row: ChunkRow,
  extras: Partial<VectorDoc> = {}
): VectorDoc {
  const meta = (row.metadata || {}) as Record<string, unknown>;
  const metadata: DocMetadata = {
    docName: typeof meta.docName === "string" ? meta.docName : undefined,
    docId: row.document_id,
    chunkIndex: row.chunk_index,
    wordCount: row.word_count ?? undefined,
  };
  return {
    id: row.id,
    text: row.content,
    embedding: [],
    metadata,
    tokens: tokenize(row.content),
    ...extras,
  };
}

/**
 * K-nearest chunks by L2 distance (matches user SQL intent).
 */
export async function vectorSearch(
  queryEmbedding: number[],
  limit = 5
): Promise<VectorDoc[]> {
  const supabase = getSupabaseAdmin();
  const { data, error } = await supabase.rpc("match_chunks", {
    query_embedding: toVectorParam(queryEmbedding),
    match_count: limit,
  });
  if (error) throw new Error(`match_chunks: ${error.message}`);

  const rows = (data || []) as Array<{
    id: string;
    content: string;
    document_id: string;
    chunk_index: number;
    distance: number;
    word_count: number | null;
    metadata: Record<string, unknown> | null;
  }>;

  return rows.map((r) =>
    rowToVectorDoc(
      {
        id: r.id,
        content: r.content,
        document_id: r.document_id,
        chunk_index: r.chunk_index,
        word_count: r.word_count,
        metadata: r.metadata,
      },
      {
        score: distanceToVectorScore(r.distance),
        vectorScore: distanceToVectorScore(r.distance),
      }
    )
  );
}

async function fetchChunkRows(filterDocumentId?: string | null): Promise<ChunkRow[]> {
  const supabase = getSupabaseAdmin();
  let q = supabase
    .from("chunks")
    .select("id, content, document_id, chunk_index, word_count, metadata");
  if (filterDocumentId) {
    q = q.eq("document_id", filterDocumentId);
  }
  const { data, error } = await q;
  if (error) throw new Error(`chunks select: ${error.message}`);
  return (data || []) as ChunkRow[];
}

/**
 * Hybrid: BM25 + vector RRF (same spirit as in-memory VectorStore), using DB distances.
 */
export async function hybridSearchDb(
  query: string,
  queryEmbedding: number[],
  topK: number,
  filterDocumentId?: string | null
): Promise<VectorDoc[]> {
  const [rows, distRows] = await Promise.all([
    fetchChunkRows(filterDocumentId ?? undefined),
    getSupabaseAdmin().rpc("chunk_vector_distances", {
      query_embedding: toVectorParam(queryEmbedding),
      filter_document_id: filterDocumentId ?? null,
    }),
  ]);

  if (distRows.error) {
    throw new Error(`chunk_vector_distances: ${distRows.error.message}`);
  }

  if (!rows.length) return [];

  const distanceById = new Map(
    (distRows.data || []).map((r: { id: string; distance: number }) => [r.id, r.distance])
  );

  const bm25 = new BM25();
  bm25.build(rows.map((r) => ({ tokens: tokenize(r.content) })));
  const qt = tokenize(query);

  const bm25Ranked = rows
    .map((r, i) => ({ id: r.id, bm25Score: bm25.scoreDoc(qt, i) }))
    .sort((a, b) => b.bm25Score - a.bm25Score);

  const vectorRankOrder: string[] = (distRows.data || []).map((r: { id: string }) => r.id);

  const rrfScores = new Map<string, number>();
  vectorRankOrder.forEach((id, rank) => {
    rrfScores.set(id, (rrfScores.get(id) || 0) + 1 / (RRF_K + rank + 1));
  });
  bm25Ranked.forEach(({ id }, rank) => {
    rrfScores.set(id, (rrfScores.get(id) || 0) + 1 / (RRF_K + rank + 1));
  });

  const bm25ById = new Map(bm25Ranked.map((b) => [b.id, b.bm25Score]));

  const merged = rows
    .map((row) => {
      const dist = distanceById.get(row.id);
      const vectorScore = typeof dist === "number" ? distanceToVectorScore(dist) : 0;
      const bm25Score = bm25ById.get(row.id) ?? 0;
      return rowToVectorDoc(row, {
        score: rrfScores.get(row.id) || 0,
        vectorScore,
        bm25Score,
      });
    })
    .sort((a, b) => (b.score ?? 0) - (a.score ?? 0))
    .slice(0, topK);

  return merged;
}

/** Vector-only search with optional document filter (chunk inspector). */
export async function vectorSearchFiltered(
  queryEmbedding: number[],
  topK: number,
  filterDocumentId?: string | null
): Promise<VectorDoc[]> {
  const { data, error } = await getSupabaseAdmin().rpc("chunk_vector_distances", {
    query_embedding: toVectorParam(queryEmbedding),
    filter_document_id: filterDocumentId ?? null,
  });
  if (error) throw new Error(`chunk_vector_distances: ${error.message}`);

  const ordered = (data || []) as { id: string; distance: number }[];
  const takeIds = ordered.slice(0, topK).map((r) => r.id);
  if (!takeIds.length) return [];

  const distMap = new Map(ordered.map((r) => [r.id, r.distance]));

  const { data: rows, error: e2 } = await getSupabaseAdmin()
    .from("chunks")
    .select("id, content, document_id, chunk_index, word_count, metadata")
    .in("id", takeIds);
  if (e2) throw new Error(`chunks select: ${e2.message}`);

  const rowMap = new Map((rows as ChunkRow[]).map((r) => [r.id, r]));

  return takeIds
    .map((id) => {
      const row = rowMap.get(id);
      const dist = distMap.get(id);
      if (!row || dist === undefined) return null;
      const score = distanceToVectorScore(dist);
      return rowToVectorDoc(row, { score, vectorScore: score });
    })
    .filter((x): x is VectorDoc => x != null);
}
