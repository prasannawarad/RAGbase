import { NextResponse } from "next/server";
import { requireUser } from "@/lib/auth";
import { guardRequest } from "@/lib/apiGuard";
import { bumpUserCacheVersion } from "@/lib/cache";
import { apiError } from "@/lib/api/http";

function mapRow(row: Record<string, unknown>) {
  const uploaded = row.uploaded_at as string | null;
  return {
    id: row.id as string,
    name: row.name as string,
    text: (row.raw_text as string) ?? "",
    chunks: (row.chunk_count as number) ?? 0,
    charCount: (row.char_count as number) ?? 0,
    wordCount: (row.word_count as number) ?? 0,
    avgChunkSize: (row.avg_chunk_size as number) ?? 0,
    pages: (row.pages as number | null) ?? null,
    fileType: (row.file_type as string) ?? "text",
    status: (row.status as string) ?? "indexed",
    chunksDone: (row.chunks_done as number) ?? 0,
    error: (row.error as string | null) ?? null,
    uploadedAt: uploaded ? new Date(uploaded).toLocaleString() : undefined,
  };
}

export async function GET(req: Request) {
  const auth = await requireUser(req);
  if (!auth.ok) return auth.response;
  const { supabase } = auth;
  try {
    const { data: docs, error } = await supabase
      .from("documents")
      .select("*")
      .order("uploaded_at", { ascending: false });

    if (error) {
      console.error("[Supabase error]:", error);
      throw new Error(error.message);
    }

    const { count, error: chunksError } = await supabase.from("chunks").select("*", { count: "exact", head: true });

    if (chunksError) {
      console.error("[Supabase error]:", chunksError);
      throw new Error(chunksError.message);
    }

    return NextResponse.json({
      documents: (docs || []).map((r) => mapRow(r as Record<string, unknown>)),
      totalChunks: count ?? 0,
    });
  } catch (error: unknown) {
    console.error("[API /documents ERROR]:", error);
    return apiError(500, "internal_error", "Request failed");
  }
}

/** Delete ALL of the caller's documents (RLS scopes it to their own; chunks cascade via FK). */
export async function DELETE(req: Request) {
  // Wipes everything — the most destructive call in the API.
  const blocked = await guardRequest(req, { name: "documents-delete-all", limit: 5 });
  if (blocked) return blocked;
  const auth = await requireUser(req);
  if (!auth.ok) return auth.response;
  const { supabase, user } = auth;
  try {
    const { error } = await supabase
      .from("documents")
      .delete()
      .neq("id", "00000000-0000-0000-0000-000000000000");

    if (error) {
      console.error("[Supabase error]:", error);
      throw new Error(error.message);
    }

    await bumpUserCacheVersion(user.id);
    return NextResponse.json({ ok: true });
  } catch (error: unknown) {
    console.error("[API /documents ERROR]:", error);
    return apiError(500, "internal_error", "Request failed");
  }
}
