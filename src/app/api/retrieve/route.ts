import { NextResponse } from "next/server";
import { hybridSearchDb, vectorSearch } from "@/lib/search";
import { vectorDocsToRetrievalResults } from "@/lib/vectorDocsToRetrieval";
import { embedTexts } from "@/lib/gemini";
import { mockEmbeddingsForTexts } from "@/lib/embedFallback";
import { requireUser } from "@/lib/auth";
import { startTrace, flushTraces } from "@/lib/observability";
import { rewriteQuery } from "@/lib/queryRewrite";
import { rerankDocs, rerankCandidateCount } from "@/lib/rerank";
import { guardRequest } from "@/lib/apiGuard";
import { apiError, parseBody } from "@/lib/api/http";
import { retrieveSchema } from "@/lib/api/schemas";

export async function POST(req: Request) {
  const blocked = guardRequest(req, { name: "retrieve", limit: 30 });
  if (blocked) return blocked;
  const auth = await requireUser(req);
  if (!auth.ok) return auth.response;
  const { supabase } = auth;

  const parsed = await parseBody(req, retrieveSchema);
  if (!parsed.ok) return parsed.response;
  const { query } = parsed.data;
  const topK = parsed.data.topK ?? 5;
  const useHybrid = parsed.data.useHybrid ?? false;
  const useRerank = parsed.data.rerank ?? false;
  const useRewrite = parsed.data.rewrite ?? false;
  const filterDocumentId = parsed.data.filterDocumentId ?? null;

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
    let embeddings: number[][];
    let embeddingSource = "gemini/gemini-embedding-001";
    try {
      embeddings = await embedTexts([searchQuery]);
    } catch {
      embeddings = mockEmbeddingsForTexts([searchQuery]);
      embeddingSource = "mock-fallback";
    }
    embedSpan?.end({ output: { embeddingSource, dims: embeddings[0]?.length ?? 0 } });

    const [queryEmbedding] = embeddings;
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
    return NextResponse.json({
      results,
      ...(useRewrite && searchQuery !== query ? { rewrittenQuery: searchQuery } : {}),
    });
  } catch (e) {
    console.error("[api/retrieve]", e);
    trace?.update({ output: { error: e instanceof Error ? e.message : String(e) } });
    await flushTraces();
    return apiError(500, "internal_error", "Retrieval failed");
  }
}
