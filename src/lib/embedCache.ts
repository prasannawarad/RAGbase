import "server-only";

import { createHash } from "node:crypto";
import { cacheGet, cacheSet } from "@/lib/cache";
import { EMBEDDING_DIMS, EMBEDDING_MODEL, embedTexts } from "@/lib/gemini";

/**
 * Query-embedding cache. The Gemini free tier allows 100 embed requests/min, and the same
 * question is asked repeatedly (retries, evals, popular questions), so embedding is the
 * scarcest shared resource.
 *
 * Safety rules:
 *  - Only REAL Gemini embeddings are cached. embedTexts() throws on failure and the caller
 *    swaps in the mock fallback; since that happens outside this function, a mock vector can
 *    never be written to the cache (it would otherwise poison retrieval for the TTL).
 *  - The key includes the model name and dimensions, so changing the embedding model
 *    automatically stops reusing old vectors.
 *  - Keyed by text hash only (not user): an embedding is a pure function of the text and
 *    contains no tenant data.
 */

const TTL_SECONDS = 7 * 24 * 3600;

export function embeddingCacheKey(text: string): string {
  const hash = createHash("sha256").update(text).digest("hex");
  return `emb:v1:${EMBEDDING_MODEL}:${EMBEDDING_DIMS}:${hash}`;
}

export type EmbedQueryResult = { embedding: number[]; cached: boolean };

/** Embed one query string, using the cache. Throws if Gemini fails (caller handles fallback). */
export async function embedQueryCached(text: string): Promise<EmbedQueryResult> {
  const key = embeddingCacheKey(text);
  const hit = await cacheGet<number[]>(key);
  if (hit && hit.length === EMBEDDING_DIMS) return { embedding: hit, cached: true };

  const [embedding] = await embedTexts([text]);
  if (!embedding || embedding.length !== EMBEDDING_DIMS) {
    throw new Error("embedTexts returned an unexpected shape");
  }
  await cacheSet(key, embedding, TTL_SECONDS);
  return { embedding, cached: false };
}
