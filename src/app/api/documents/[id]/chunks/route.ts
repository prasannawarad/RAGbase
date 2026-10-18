import { NextResponse } from "next/server";
import { requireUser } from "@/lib/auth";
import type { VectorDoc } from "@/lib/vectorStore";
import { tokenize } from "@/lib/tokenizer";
import { apiError, validate } from "@/lib/api/http";
import { documentIdParamSchema } from "@/lib/api/schemas";

export async function GET(
  req: Request,
  context: { params: Promise<{ id: string }> }
) {
  const auth = await requireUser(req);
  if (!auth.ok) return auth.response;
  const { supabase } = auth;

  const params = validate(documentIdParamSchema, await context.params);
  if (!params.ok) return params.response;
  const documentId = params.data.id;

  try {
    const { data, error } = await supabase
      .from("chunks")
      .select("id, content, document_id, chunk_index, word_count, metadata")
      .eq("document_id", documentId)
      .order("chunk_index", { ascending: true });

    if (error) throw new Error(error.message);

    const chunks: VectorDoc[] = (data || []).map((row) => {
      const meta = (row.metadata || {}) as Record<string, unknown>;
      return {
        id: row.id as string,
        text: row.content as string,
        embedding: [],
        metadata: {
          docName: typeof meta.docName === "string" ? meta.docName : undefined,
          docId: row.document_id as string,
          chunkIndex: row.chunk_index as number,
          wordCount: (row.word_count as number) ?? undefined,
        },
        tokens: tokenize(row.content as string),
      };
    });

    return NextResponse.json({ chunks });
  } catch (e) {
    console.error("[api/documents/[id]/chunks]", e);
    return apiError(500, "internal_error", "Failed to load chunks", { chunks: [] });
  }
}
