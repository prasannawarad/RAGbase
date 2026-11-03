import { NextResponse } from "next/server";
import { hybridSearchDb, vectorSearch } from "@/lib/search";
import { vectorDocsToRetrievalResults } from "@/lib/vectorDocsToRetrieval";
import { embedQueryCached } from "@/lib/embedCache";
import { cacheGet, cacheSet, getUserCacheVersion } from "@/lib/cache";
import {
  RETRIEVAL_TTL_SECONDS,
  retrievalCacheKey,
  wantsFreshResults,
  type CachedRetrieval,
} from "@/lib/retrievalCache";
import { mockEmbeddingsForTexts } from "@/lib/embedFallback";
import { requireUser } from "@/lib/auth";
import { startTrace, flushTraces } from "@/lib/observability";
import { rewriteQuery } from "@/lib/queryRewrite";
import { rerankDocs, rerankCandidateCount } from "@/lib/rerank";
import { guardRequest, guardUser } from "@/lib/apiGuard";
import { apiError, parseBody } from "@/lib/api/http";
import { retrieveSchema } from "@/lib/api/schemas";

export async function POST(req: Request) {
  const blocked = await guardRequest(req, { name: "retrieve", limit: 30 });
  if (blocked) return blocked;
  const auth = await requireUser(req);
  if (!auth.ok) return auth.response;
  const { supabase, user } = auth;
  const perUser = await guardUser(user.id, { name: "retrieve", limit: 30 });
  if (perUser) return perUser;

  const parsed = await parseBody(req, retrieveSchema);
  if (!parsed.ok) return parsed.response;
  const { query } = parsed.data;
  const topK = parsed.data.topK ?? 5;
  const useHybrid = parsed.data.useHybrid ?? false;
  const useRerank = parsed.data.rerank ?? false;
  const useRewrite = parsed.data.rewrite ?? false;
  const filterDocumentId = parsed.data.filterDocumentId ?? null;

  // Cache lookup comes before tracing/embedding/LLM work: a hit costs two Redis reads.
  const fresh = wantsFreshResults(req);
  const cacheKey = retrievalCacheKey(user.id, await getUserCacheVersion(user.id), {
    query,
    topK,
    useHybrid,
    rerank: useRerank,
    rewrite: useRewrite,
    filterDocumentId,
  });
  if (!fresh) {
    const hit = await cacheGet<CachedRetrieval>(cacheKey);
    if (hit) return NextResponse.json(hit, { headers: { "X-Cache": "HIT" } });
  }

  const trace = startTrace("retrieve", {
    input: { query, topK, useHybrid, rerank: useRerank, rewrite: useRewrite, filterDocumentId },
    tags: [
      useHybrid ? "hybrid" : "vector",
      ...(useRerank ? ["rerank"] : []),
      ...(useRewrite ? ["rewrite"] : []),
    ],
  });

  try {
    let searchQuery = query;
    if (useRewrite) {
      const rewriteSpan = trace?.span({ name: "rewrite-query", input: { query } });
      searchQuery = await rewriteQuery(query);
      rewriteSpan?.end({ output: { searchQuery, changed: searchQuery !== query } });
    }

    const embedSpan = trace?.span({ name: "embed-query", input: { query: searchQuery } });
    let queryEmbedding: number[] | undefined;
    let embeddingSource = "gemini/gemini-embedding-001";
    let embeddingCached = false;
    try {
      const r = await embedQueryCached(searchQuery);
      queryEmbedding = r.embedding;
      embeddingCached = r.cached;
    } catch {
      [queryEmbedding] = mockEmbeddingsForTexts([searchQuery]);
      embeddingSource = "mock-fallback";
    }
    embedSpan?.end({ output: { embeddingSource, embeddingCached, dims: queryEmbedding?.length ?? 0 } });

    if (!queryEmbedding?.length) {
      trace?.update({ output: { error: "Failed to build query embedding" } });
      await flushTraces();
      return apiError(500, "internal_error", "Failed to build query embedding");
    }

    const fetchK = useRerank ? rerankCandidateCount(topK) : topK;
    const searchSpan = trace?.span({
      name: useHybrid ? "hybrid-search" : "vector-search",
      input: { topK, fetchK, filterDocumentId },
    });
    let raw;
    if (useHybrid) {
      raw = await hybridSearchDb(supabase, searchQuery, queryEmbedding, fetchK, filterDocumentId);
    } else {
      raw = await vectorSearch(supabase, queryEmbedding, fetchK, filterDocumentId);
    }
    searchSpan?.end({ output: { candidates: raw.length } });

    if (useRerank) {
      const rerankSpan = trace?.span({
        name: "llm-rerank",
        input: { candidates: raw.length, topK },
      });
      // Rerank against the original question — user intent, not the rewrite.
      raw = await rerankDocs(query, raw, topK);
      rerankSpan?.end({ output: raw.map((d) => ({ id: d.id, chunkIndex: d.metadata.chunkIndex })) });
    }

    const results = await vectorDocsToRetrievalResults(supabase, raw);
    trace?.update({
      output: {
        resultCount: results.length,
        results: results.map((r) => ({
          documentName: r.documentName,
          chunkIndex: r.chunkIndex,
          score: r.score,
          vectorScore: r.vectorScore,
          bm25Score: r.bm25Score,
        })),
      },
    });
    await flushTraces();
    const payload: CachedRetrieval = {
      results,
      ...(useRewrite && searchQuery !== query ? { rewrittenQuery: searchQuery } : {}),
    };
    // Never cache a response built from the mock-embedding fallback: it is near-random and
    // would stay wrong for the whole TTL after Gemini recovers.
    if (embeddingSource !== "mock-fallback") {
      await cacheSet(cacheKey, payload, RETRIEVAL_TTL_SECONDS);
    }
    return NextResponse.json(payload, { headers: { "X-Cache": fresh ? "BYPASS" : "MISS" } });
  } catch (e) {
    console.error("[api/retrieve]", e);
    trace?.update({ output: { error: e instanceof Error ? e.message : String(e) } });
    await flushTraces();
    return apiError(500, "internal_error", "Retrieval failed");
  }
}
