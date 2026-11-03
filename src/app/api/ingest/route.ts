import { NextResponse } from "next/server";
import { apiError, devDetails, parseBody } from "@/lib/api/http";
import { ingestSchema, LIMITS } from "@/lib/api/schemas";
import { requireUser } from "@/lib/auth";
import { guardDailyBudget, guardRequest, guardUser } from "@/lib/apiGuard";
import { contentHash } from "@/lib/ingestStore";

const STAGE_INSERT_BATCH = 500;

type DocRow = {
  id: string;
  name: string;
  raw_text: string | null;
  chunk_count: number;
  chunks_done: number;
  char_count: number | null;
  word_count: number | null;
  avg_chunk_size: number | null;
  pages: number | null;
  file_type: string | null;
  status: string | null;
  uploaded_at: string;
};

const DOC_COLUMNS =
  "id, name, raw_text, chunk_count, chunks_done, char_count, word_count, avg_chunk_size, pages, file_type, status, uploaded_at";

function toClientDocument(row: DocRow) {
  return {
    id: row.id,
    name: row.name,
    text: row.raw_text ?? "",
    chunks: row.chunk_count,
    chunksDone: row.chunks_done,
    charCount: row.char_count ?? 0,
    wordCount: row.word_count ?? 0,
    avgChunkSize: row.avg_chunk_size ?? 0,
    pages: row.pages,
    fileType: row.file_type ?? "text",
    status: row.status ?? "queued",
    uploadedAt: new Date(row.uploaded_at).toLocaleString(),
  };
}

/**
 * Stage 4: ingestion is a two-step, resumable pipeline.
 *   POST /api/ingest              validate + dedup + store the chunk TEXT (fast, no embedding)  -> 202
 *   POST /api/ingest/:id/process  embed in small batches until the document is 'indexed'
 *
 * Re-uploading identical text is a no-op: the existing document is returned (200, duplicate: true),
 * so no embedding quota is spent twice.
 */
export async function POST(req: Request) {
  // Writes rows — keep it low.
  const blocked = await guardRequest(req, { name: "ingest", limit: 10 });
  if (blocked) return blocked;
  const auth = await requireUser(req);
  if (!auth.ok) return auth.response;
  const { supabase, user } = auth;
  const perUser =
    (await guardUser(user.id, { name: "ingest", limit: 10 })) ??
    (await guardDailyBudget(user.id, "ingest", 100));
  if (perUser) return perUser;

  const parsed = await parseBody(req, ingestSchema, LIMITS.bodyBytesIngest);
  if (!parsed.ok) return parsed.response;
  const { documentName, chunks, pages = null } = parsed.data;
  const fileType = parsed.data.fileType ?? "text";

  const hash = contentHash(chunks);

  const findExisting = async () => {
    const { data } = await supabase
      .from("documents")
      .select(DOC_COLUMNS)
      .eq("user_id", user.id)
      .eq("content_hash", hash)
      .maybeSingle();
    return (data as DocRow | null) ?? null;
  };

  const existing = await findExisting();
  if (existing) {
    return NextResponse.json({ document: toClientDocument(existing), duplicate: true });
  }

  const rawText = chunks.join("\n\n");
  const avgChunkSize = Math.round(chunks.reduce((acc, c) => acc + c.length, 0) / chunks.length);

  const { data: inserted, error } = await supabase
    .from("documents")
    .insert({
      user_id: user.id,
      name: documentName,
      raw_text: rawText,
      char_count: rawText.length,
      word_count: rawText.split(/\s+/).filter(Boolean).length,
      chunk_count: chunks.length,
      chunks_done: 0,
      avg_chunk_size: avgChunkSize,
      pages,
      file_type: fileType,
      status: "queued",
      content_hash: hash,
    })
    .select(DOC_COLUMNS)
    .single();

  if (error || !inserted) {
    // Two identical uploads racing past the check above: the unique index decides, return the winner.
    if (error?.code === "23505") {
      const winner = await findExisting();
      if (winner) return NextResponse.json({ document: toClientDocument(winner), duplicate: true });
    }
    console.error("[INGEST ERROR - DOCUMENT INSERT]:", error);
    return apiError(500, "internal_error", "Failed to create document", devDetails(error?.message));
  }

  const docId = (inserted as DocRow).id;

  for (let i = 0; i < chunks.length; i += STAGE_INSERT_BATCH) {
    const rows = chunks.slice(i, i + STAGE_INSERT_BATCH).map((content, j) => ({
      document_id: docId,
      user_id: user.id,
      chunk_index: i + j,
      content,
    }));
    const { error: stageError } = await supabase.from("ingest_chunks").insert(rows);
    if (stageError) {
      console.error("[INGEST ERROR - STAGING]:", stageError);
      // Cascade removes any staging rows already written.
      await supabase.from("documents").delete().eq("id", docId);
      return apiError(500, "internal_error", "Failed to queue chunks", devDetails(stageError.message));
    }
  }

  return NextResponse.json({ document: toClientDocument(inserted as DocRow), duplicate: false }, { status: 202 });
}
