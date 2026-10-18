import "server-only";

import { createHash } from "node:crypto";
import type { RetrievalResult } from "@/lib/retrievalResult";

/**
 * Cache for whole /api/retrieve responses (including query rewrite + LLM rerank, so a hit saves
 * embedding quota AND LLM tokens). Scoped per user and per cache generation:
 *
 *   ret:v1:<userId>:<generation>:<hash of the request shape>
 *
 * The generation comes from getUserCacheVersion(); ingest/delete bump it, which makes every
 * earlier entry unreachable immediately — a freshly uploaded document is always searchable.
 */

export const RETRIEVAL_TTL_SECONDS = 300;

export type RetrievalCacheShape = {
  query: string;
  topK: number;
  useHybrid: boolean;
  rerank: boolean;
  rewrite: boolean;
  filterDocumentId: string | null;
};

export type CachedRetrieval = { results: RetrievalResult[]; rewrittenQuery?: string };

export function retrievalCacheKey(userId: string, generation: number, shape: RetrievalCacheShape): string {
  // Explicit field order => the hash never depends on JSON key order.
  const canonical = JSON.stringify([
    shape.query,
    shape.topK,
    shape.useHybrid,
    shape.rerank,
    shape.rewrite,
    shape.filterDocumentId,
  ]);
  const hash = createHash("sha256").update(canonical).digest("hex");
  return `ret:v1:${userId}:${generation}:${hash}`;
}

/** Honour the standard `Cache-Control: no-cache` request header (used by the eval harness). */
export function wantsFreshResults(req: Request): boolean {
  const cc = req.headers.get("cache-control")?.toLowerCase() ?? "";
  return cc.includes("no-cache") || cc.includes("no-store");
}
