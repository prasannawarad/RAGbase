import "server-only";

import type { VectorDoc } from "@/lib/vectorStore";
import { getSupabaseAdmin } from "@/lib/supabase/server";
import { makeContentSnippet, type RetrievalResult } from "@/lib/retrievalResult";

/** Server-only: map vector search rows to attributed retrieval results; fills document names from DB when missing in chunk metadata. */
export async function vectorDocsToRetrievalResults(docs: VectorDoc[]): Promise<RetrievalResult[]> {
  const docIds = [
    ...new Set(
      docs.map((d) => d.metadata.docId).filter((id): id is string => typeof id === "string" && id.length > 0)
    ),
  ];

  const nameById = new Map<string, string>();
  if (docIds.length) {
    try {
      const supabase = getSupabaseAdmin();
      const { data } = await supabase.from("documents").select("id, name").in("id", docIds);
      for (const row of data || []) {
        const id = row.id as string;
        const name = row.name as string;
        if (id && name) nameById.set(id, name);
      }
    } catch {
      /* names stay empty → fall back to metadata / Unknown */
    }
  }

  return docs.map((d) => {
    const docId = d.metadata.docId ?? "";
    const fromMeta = d.metadata.docName?.trim();
    const documentName = fromMeta || nameById.get(docId) || "Unknown document";
    const chunkIndex = d.metadata.chunkIndex ?? 0;
    const content = d.text;
    return {
      id: d.id,
      documentId: docId,
      documentName,
      chunkIndex,
      content,
      contentSnippet: makeContentSnippet(content),
      score: d.score,
      vectorScore: d.vectorScore,
      bm25Score: d.bm25Score,
    };
  });
}
