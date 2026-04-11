import { NextResponse } from "next/server";
import { getSupabaseAdmin, isSupabaseConfigured } from "@/lib/supabase/server";

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
    uploadedAt: uploaded ? new Date(uploaded).toLocaleString() : undefined,
  };
}

export async function GET() {
  try {
    if (!isSupabaseConfigured()) {
      return NextResponse.json(
        { error: "Supabase is not configured", documents: [], totalChunks: 0 },
        { status: 503 }
      );
    }

    const supabase = getSupabaseAdmin();
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
    return NextResponse.json(
      { error: error instanceof Error ? error.message || "Unknown error" : "Unknown error" },
      { status: 500 }
    );
  }
}

/** Delete all documents (chunks cascade via FK). */
export async function DELETE() {
  try {
    if (!isSupabaseConfigured()) {
      return NextResponse.json({ error: "Supabase is not configured" }, { status: 503 });
    }

    const supabase = getSupabaseAdmin();
    const { error } = await supabase
      .from("documents")
      .delete()
      .neq("id", "00000000-0000-0000-0000-000000000000");

    if (error) {
      console.error("[Supabase error]:", error);
      throw new Error(error.message);
    }

    return NextResponse.json({ ok: true });
  } catch (error: unknown) {
    console.error("[API /documents ERROR]:", error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message || "Unknown error" : "Unknown error" },
      { status: 500 }
    );
  }
}
