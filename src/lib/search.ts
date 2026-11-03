import "server-only";

/**
 * Supabase pgvector search, server-side only.
 *
 * Both modes run entirely inside Postgres on indexes (see migration 006) and return only the
 * top candidates:
 *   vector  -> `match_chunks`          HNSW nearest neighbours (L2 distance)
 *   hybrid  -> `hybrid_search_chunks`  HNSW + full-text candidates fused with RRF (k = 60)
 *
 * Every function takes the caller's user-scoped Supabase client, so Row Level Security
 * restricts results to that user's own chunks. Never pass the service-role client here.
 */

import type { SupabaseClient } from "@supabase/supabase-js";
import type { DocMetadata, VectorDoc } from "@/lib/vectorStore";
import { tokenize } from "@/lib/tokenizer";

const EMBEDDING_DIM = 768;

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

/** Row shape shared by `match_chunks` and `hybrid_search_chunks`. */
export type SearchRow = {
  id: string;
  content: string;
  document_id: string;
  chunk_index: number;
  distance: number;
  word_count: number | null;
  metadata: Record<string, unknown> | null;
  /** hybrid only */
  keyword_score?: number;
  /** hybrid only */
  rrf_score?: number;
};

function rowToVectorDoc(row: SearchRow, extras: Partial<VectorDoc>): VectorDoc {
  const meta = row.metadata || {};
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

export function mapVectorRow(row: SearchRow): VectorDoc {
  const score = distanceToVectorScore(row.distance);
  return rowToVectorDoc(row, { score, vectorScore: score });
}

/**
 * `bm25Score` carries the Postgres full-text rank (ts_rank_cd). The field name is kept so the
 * UI and retrieval contract don't change; it is no longer literal BM25.
 */
export function mapHybridRow(row: SearchRow): VectorDoc {
  return rowToVectorDoc(row, {
    score: row.rrf_score ?? 0,
    vectorScore: distanceToVectorScore(row.distance),
    bm25Score: row.keyword_score ?? 0,
  });
}

/** K-nearest chunks by L2 distance, optionally scoped to one document. */
export async function vectorSearch(
  supabase: SupabaseClient,
  queryEmbedding: number[],
  limit = 5,
  filterDocumentId?: string | null
): Promise<VectorDoc[]> {
  const { data, error } = await supabase.rpc("match_chunks", {
    query_embedding: toVectorParam(queryEmbedding),
    match_count: limit,
    filter_document_id: filterDocumentId ?? null,
  });
  if (error) throw new Error(`match_chunks: ${error.message}`);
  return ((data || []) as SearchRow[]).map(mapVectorRow);
}

/** Hybrid: index-backed vector + keyword candidates, fused with Reciprocal Rank Fusion in SQL. */
export async function hybridSearchDb(
  supabase: SupabaseClient,
  query: string,
  queryEmbedding: number[],
  topK: number,
  filterDocumentId?: string | null
): Promise<VectorDoc[]> {
  const { data, error } = await supabase.rpc("hybrid_search_chunks", {
    query_text: query,
    query_embedding: toVectorParam(queryEmbedding),
    match_count: topK,
    filter_document_id: filterDocumentId ?? null,
  });
  if (error) throw new Error(`hybrid_search_chunks: ${error.message}`);
  return ((data || []) as SearchRow[]).map(mapHybridRow);
}
