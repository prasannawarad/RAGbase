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
      { error: "Supabase is not configured (SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY)" },
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
  const rawChunks = body.chunks;
  if (!documentName || !Array.isArray(rawChunks) || rawChunks.length === 0) {
    return NextResponse.json(
      {
        error:
          "Expected body: { documentName: string, chunks: string[], fileType?: string, pages?: number | null }",
      },
      { status: 400 }
    );
  }

  const chunkTexts = rawChunks.map((c) => String(c));
  if (chunkTexts.some((t) => !t.trim())) {
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
    embeddings = await embedTexts(chunkTexts);
  } catch {
    embeddings = mockEmbeddingsForTexts(chunkTexts);
  }

  if (embeddings.length !== chunkTexts.length) {
    return NextResponse.json({ error: "Embedding count mismatch" }, { status: 500 });
  }
  for (const emb of embeddings) {
    if (!emb || emb.length !== 768) {
      return NextResponse.json({ error: "Invalid embedding dimensions" }, { status: 500 });
    }
  }

  const rawText = chunkTexts.join("\n\n");
  const charCount = rawText.length;
  const wordCount = rawText.split(/\s+/).filter(Boolean).length || 0;
  const avgChunkSize = Math.round(
    chunkTexts.reduce((s, t) => s + t.length, 0) / chunkTexts.length
  );
  const uploadedAt = new Date().toISOString();

  const supabase = getSupabaseAdmin();

  const { data: docRow, error: docErr } = await supabase
    .from("documents")
    .insert({
      name: documentName,
      raw_text: rawText,
      char_count: charCount,
      word_count: wordCount,
      chunk_count: chunkTexts.length,
      avg_chunk_size: avgChunkSize,
      pages,
      file_type: fileType,
      status: "indexed",
      uploaded_at: uploadedAt,
    })
    .select("id")
    .single();

  if (docErr || !docRow) {
    console.error("[api/ingest] document insert", docErr);
    return NextResponse.json({ error: docErr?.message ?? "document insert failed" }, { status: 500 });
  }

  const docId = docRow.id as string;

  const rows = chunkTexts.map((text, i) => {
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

  const { error: chunkErr } = await supabase.from("chunks").insert(rows);

  if (chunkErr) {
    console.error("[api/ingest] chunks insert", chunkErr);
    await supabase.from("documents").delete().eq("id", docId);
    return NextResponse.json({ error: chunkErr.message }, { status: 500 });
  }

  return NextResponse.json({
    document: {
      id: docId,
      name: documentName,
      text: rawText,
      chunks: chunkTexts.length,
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
