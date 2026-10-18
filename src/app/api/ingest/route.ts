import { NextResponse } from "next/server";
import { apiError, devDetails, parseBody } from "@/lib/api/http";
import { ingestSchema, LIMITS } from "@/lib/api/schemas";
import { requireUser } from "@/lib/auth";
import { toVectorParam } from "@/lib/search";
import { embedTexts } from "@/lib/gemini";
import { mockEmbeddingsForTexts } from "@/lib/embedFallback";
import { guardDailyBudget, guardRequest, guardUser } from "@/lib/apiGuard";
import { bumpUserCacheVersion } from "@/lib/cache";

export async function POST(req: Request) {
  // Writes rows + burns embedding quota — keep it low.
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

  let embeddings: number[][];
  try {
    embeddings = await embedTexts(chunks);
  } catch {
    embeddings = mockEmbeddingsForTexts(chunks);
  }

  if (!embeddings || embeddings.length !== chunks.length) {
    return apiError(500, "internal_error", "Embedding mismatch");
  }

  if (embeddings[0]?.length !== 768) {
    return apiError(500, "internal_error", "Invalid embedding dimension");
  }

  const rawText = chunks.join("\n\n");
  const charCount = rawText.length;
  const wordCount = rawText.split(/\s+/).filter(Boolean).length || 0;
  const avgChunkSize =
    chunks.length > 0
      ? Math.round(chunks.reduce((acc, c) => acc + c.length, 0) / chunks.length)
      : 0;
  const uploadedAt = new Date().toISOString();

  const { data: insertedDocs, error } = await supabase
    .from("documents")
    .insert({
      user_id: user.id,
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

    return apiError(500, "internal_error", "Failed to insert document", devDetails(error.message));
  }

  if (!docRow) {
    console.error("[INGEST ERROR - DOCUMENT INSERT]:", "No row returned");
    return apiError(500, "internal_error", "Failed to insert document");
  }

  const docId = docRow.id as string;

  const rows = chunks.map((text, i) => {
    const wordCountChunk = text.split(/\s+/).filter(Boolean).length || 0;
    return {
      user_id: user.id,
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

    return apiError(500, "internal_error", "Failed to insert chunks", devDetails(chunkError.message));
  }

  // New chunks exist: orphan this user's cached retrieval results.
  await bumpUserCacheVersion(user.id);

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
