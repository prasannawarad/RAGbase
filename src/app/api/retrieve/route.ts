import { NextResponse } from "next/server";
import {
  hybridSearchDb,
  vectorSearch,
  vectorSearchFiltered,
} from "@/lib/search";
import { vectorDocsToRetrievalResults } from "@/lib/vectorDocsToRetrieval";
import { embedTexts } from "@/lib/gemini";
import { mockEmbeddingsForTexts } from "@/lib/embedFallback";
import { isSupabaseConfigured } from "@/lib/supabase/server";
import { startTrace, flushTraces } from "@/lib/observability";
import { rewriteQuery } from "@/lib/queryRewrite";
import { rerankDocs, rerankCandidateCount } from "@/lib/rerank";

export async function POST(req: Request) {
  if (!isSupabaseConfigured()) {
    return NextResponse.json(
      {
        error:
          "Supabase is not configured (set SUPABASE_SERVICE_ROLE_KEY and SUPABASE_URL or NEXT_PUBLIC_SUPABASE_URL)",
      },
      { status: 503 }
    );
  }

  let body: Record<string, unknown>;
  try {
    body = (await req.json()) as Record<string, unknown>;
  } catch {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
  }

  const query = typeof body.query === "string" ? body.query.trim() : "";
  const topK = typeof body.topK === "number" && body.topK > 0 ? body.topK : 5;
  const useHybrid = Boolean(body.useHybrid);
  const useRerank = Boolean(body.rerank);
  const useRewrite = Boolean(body.rewrite);
  const filterDocumentId =
    typeof body.filterDocumentId === "string" && body.filterDocumentId.length > 0
      ? body.filterDocumentId
      : null;

  if (!query) {
    return NextResponse.json(
      { error: "Expected body: { query: string, topK?: number, useHybrid?: boolean, rerank?: boolean, rewrite?: boolean, filterDocumentId?: string }" },
      { status: 400 }
    );
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
      return NextResponse.json({ error: "Failed to build query embedding" }, { status: 500 });
    }

    const fetchK = useRerank ? rerankCandidateCount(topK) : topK;
    const searchSpan = trace?.span({
      name: useHybrid ? "hybrid-search" : "vector-search",
      input: { topK, fetchK, filterDocumentId },
    });
    let raw;
    if (useHybrid) {
      raw = await hybridSearchDb(searchQuery, queryEmbedding, fetchK, filterDocumentId);
    } else if (filterDocumentId) {
      raw = await vectorSearchFiltered(queryEmbedding, fetchK, filterDocumentId);
    } else {
      raw = await vectorSearch(queryEmbedding, fetchK);
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

    const results = await vectorDocsToRetrievalResults(raw);
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
    return NextResponse.json(
      { error: e instanceof Error ? e.message : String(e) },
      { status: 500 }
    );
  }
}
