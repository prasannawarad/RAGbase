import { NextResponse } from "next/server";
import { getSupabaseAdmin, isSupabaseConfigured } from "@/lib/supabase/server";
import type { VectorDoc } from "@/lib/vectorStore";
import { tokenize } from "@/lib/tokenizer";

export async function GET(
  _req: Request,
  context: { params: Promise<{ id: string }> }
) {
  if (!isSupabaseConfigured()) {
    return NextResponse.json({ error: "Supabase is not configured", chunks: [] }, { status: 503 });
  }

  const { id: documentId } = await context.params;
  if (!documentId) {
    return NextResponse.json({ error: "Missing id" }, { status: 400 });
  }

  try {
    const supabase = getSupabaseAdmin();
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
    return NextResponse.json(
      { error: e instanceof Error ? e.message : String(e), chunks: [] },
      { status: 500 }
    );
  }
}
