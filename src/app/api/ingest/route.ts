import { NextResponse } from "next/server";
import { getSupabaseAdmin, isSupabaseConfigured } from "@/lib/supabase/server";
import { toVectorParam } from "@/lib/search";
import { embedTexts } from "@/lib/gemini";
import { mockEmbeddingsForTexts } from "@/lib/embedFallback";

/**
 * Body (minimum):
 *   { documentName: string, chunks: string[] }
 *
 * Optional (preserves upload metadata / UI):
 *   fileType?: string
 *   pages?: number | null
 */
type IngestBody = {
  documentName?: unknown;
  chunks?: unknown;
  fileType?: unknown;
  pages?: unknown;
};

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

  let body: IngestBody;
  try {
    body = (await req.json()) as IngestBody;
  } catch {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
  }

  const documentName = typeof body.documentName === "string" ? body.documentName.trim() : "";
  if (!documentName) {
    return NextResponse.json(
      {
        error:
          "Expected body: { documentName: string, chunks: string[], fileType?: string, pages?: number | null }",
      },
      { status: 400 }
    );
  }

  const rawChunks = body.chunks;
  if (!Array.isArray(rawChunks) || rawChunks.length === 0) {
    return NextResponse.json({ error: "No chunks provided" }, { status: 400 });
  }

  const chunks = rawChunks.map((c) => String(c));

  if (chunks.some((t) => !t.trim())) {
    return NextResponse.json({ error: "chunks must be non-empty strings" }, { status: 400 });
  }

  const fileType = typeof body.fileType === "string" && body.fileType.trim() ? body.fileType.trim() : "text";
  const pages =
    typeof body.pages === "number" && Number.isFinite(body.pages)
      ? body.pages
      : body.pages === null
        ? null
        : null;

  let embeddings: number[][];
  try {
    embeddings = await embedTexts(chunks);
  } catch {
    embeddings = mockEmbeddingsForTexts(chunks);
  }

  if (!embeddings || embeddings.length !== chunks.length) {
    return NextResponse.json({ error: "Embedding mismatch" }, { status: 500 });
  }

  if (embeddings[0]?.length !== 768) {
    return NextResponse.json({ error: "Invalid embedding dimension" }, { status: 500 });
  }

  const rawText = chunks.join("\n\n");
  const charCount = rawText.length;
  const wordCount = rawText.split(/\s+/).filter(Boolean).length || 0;
  const avgChunkSize =
    chunks.length > 0
      ? Math.round(chunks.reduce((acc, c) => acc + c.length, 0) / chunks.length)
      : 0;
  const uploadedAt = new Date().toISOString();

  const supabase = getSupabaseAdmin();

  const { data: insertedDocs, error } = await supabase
    .from("documents")
    .insert({
      name: documentName,
      raw_text: rawText,
      char_count: charCount,
      word_count: wordCount,
      chunk_count: chunks.length,
      avg_chunk_size: avgChunkSize,
      pages,
      file_type: fileType,
      status: "indexed",
      uploaded_at: uploadedAt,
    })
    .select("id");

  const docRow = insertedDocs?.[0];

  if (error) {
    console.error("[INGEST ERROR - DOCUMENT INSERT]:", {
      message: error.message,
      code: error.code,
      details: error.details,
      hint: error.hint,
    });

    return NextResponse.json(
      {
        error: "Failed to insert document",
        details: error.message,
        code: error.code,
        hint: error.hint,
        supabaseDetails: error.details,
      },
      { status: 500 }
    );
  }

  if (!docRow) {
    console.error("[INGEST ERROR - DOCUMENT INSERT]:", "No row returned");
    return NextResponse.json(
      {
        error: "Failed to insert document",
        details: "No row returned",
      },
      { status: 500 }
    );
  }

  const docId = docRow.id as string;

  const rows = chunks.map((text, i) => {
    const wordCountChunk = text.split(/\s+/).filter(Boolean).length || 0;
    return {
      document_id: docId,
      chunk_index: i,
      content: text,
      word_count: wordCountChunk,
      embedding: toVectorParam(embeddings[i]!),
      metadata: {
        docName: documentName,
        docId,
      },
    };
  });

  const { error: chunkError } = await supabase.from("chunks").insert(rows);

  if (chunkError) {
    console.error("[INGEST ERROR - CHUNKS INSERT]:", {
      message: chunkError.message,
      code: chunkError.code,
      details: chunkError.details,
      hint: chunkError.hint,
    });

    await supabase.from("documents").delete().eq("id", docId);

    return NextResponse.json(
      {
        error: "Failed to insert chunks",
        details: chunkError.message,
        code: chunkError.code,
        hint: chunkError.hint,
        supabaseDetails: chunkError.details,
      },
      { status: 500 }
    );
  }

  return NextResponse.json({
    document: {
      id: docId,
      name: documentName,
      text: rawText,
      chunks: chunks.length,
      charCount: charCount,
      wordCount: wordCount,
      avgChunkSize,
      pages,
      fileType,
      status: "indexed" as const,
      uploadedAt: new Date(uploadedAt).toLocaleString(),
    },
  });
}
