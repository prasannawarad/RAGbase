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

export async function POST(req: Request) {
  if (!isSupabaseConfigured()) {
    return NextResponse.json(
      { error: "Supabase is not configured (SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY)" },
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
  const filterDocumentId =
    typeof body.filterDocumentId === "string" && body.filterDocumentId.length > 0
      ? body.filterDocumentId
      : null;

  if (!query) {
    return NextResponse.json(
      { error: "Expected body: { query: string, topK?: number, useHybrid?: boolean, filterDocumentId?: string }" },
      { status: 400 }
    );
  }

  try {
    let embeddings: number[][];
    try {
      embeddings = await embedTexts([query]);
    } catch {
      embeddings = mockEmbeddingsForTexts([query]);
    }
    const [queryEmbedding] = embeddings;
    if (!queryEmbedding?.length) {
      return NextResponse.json({ error: "Failed to build query embedding" }, { status: 500 });
    }

    let raw;
    if (useHybrid) {
      raw = await hybridSearchDb(query, queryEmbedding, topK, filterDocumentId);
    } else if (filterDocumentId) {
      raw = await vectorSearchFiltered(queryEmbedding, topK, filterDocumentId);
    } else {
      raw = await vectorSearch(queryEmbedding, topK);
    }
    const results = await vectorDocsToRetrievalResults(raw);
    return NextResponse.json({ results });
  } catch (e) {
    console.error("[api/retrieve]", e);
    return NextResponse.json(
      { error: e instanceof Error ? e.message : String(e) },
      { status: 500 }
    );
  }
}
