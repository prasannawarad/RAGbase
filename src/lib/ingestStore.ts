import "server-only";

import type { SupabaseClient } from "@supabase/supabase-js";
import { createHash } from "node:crypto";
import { toVectorParam } from "@/lib/search";
import type { ChunkWrite, DocRow, IngestStore, StagedChunk } from "@/lib/ingestWorker";

/** Stable identity of a document's text, used to make re-uploads idempotent per user. */
export function contentHash(chunks: string[]): string {
  return createHash("sha256").update(chunks.join("\n\n")).digest("hex");
}

/** Supabase-backed store. `supabase` must be the user-scoped client so RLS applies. */
export function supabaseIngestStore(supabase: SupabaseClient, userId: string): IngestStore {
  const fail = (what: string, error: { message: string } | null) => {
    if (error) throw new Error(`${what}: ${error.message}`);
  };

  return {
    async getDocument(id) {
      const { data, error } = await supabase
        .from("documents")
        .select("id, name, status, chunk_count, chunks_done, error")
        .eq("id", id)
        .maybeSingle();
      fail("get document", error);
      return (data as DocRow | null) ?? null;
    },

    async claimBatch(documentId, limit) {
      const { data, error } = await supabase.rpc("claim_ingest_batch", {
        p_document_id: documentId,
        p_limit: limit,
      });
      fail("claim_ingest_batch", error);
      return ((data ?? []) as StagedChunk[]).sort((a, b) => a.chunk_index - b.chunk_index);
    },

    async writeChunks(doc: DocRow, rows: ChunkWrite[]) {
      const { error } = await supabase.from("chunks").upsert(
        rows.map((r) => ({
          user_id: userId,
          document_id: doc.id,
          chunk_index: r.chunk_index,
          content: r.content,
          embedding: toVectorParam(r.embedding),
          word_count: r.content.split(/\s+/).filter(Boolean).length,
          metadata: { docName: doc.name, docId: doc.id },
        })),
        { onConflict: "document_id,chunk_index" }
      );
      fail("write chunks", error);
    },

    async deleteStaged(ids) {
      const { error } = await supabase.from("ingest_chunks").delete().in("id", ids);
      fail("delete staged", error);
    },

    async releaseClaims(ids) {
      const { error } = await supabase.from("ingest_chunks").update({ claimed_at: null }).in("id", ids);
      fail("release claims", error);
    },

    async resetStaged(documentId) {
      const { error } = await supabase
        .from("ingest_chunks")
        .update({ attempts: 0, claimed_at: null })
        .eq("document_id", documentId);
      fail("reset staged", error);
    },

    async countStaged(documentId) {
      const { count, error } = await supabase
        .from("ingest_chunks")
        .select("id", { count: "exact", head: true })
        .eq("document_id", documentId);
      fail("count staged", error);
      return count ?? 0;
    },

    async countChunks(documentId) {
      const { count, error } = await supabase
        .from("chunks")
        .select("id", { count: "exact", head: true })
        .eq("document_id", documentId);
      fail("count chunks", error);
      return count ?? 0;
    },

    async updateDocument(id, patch) {
      const { error } = await supabase
        .from("documents")
        .update({ ...patch, updated_at: new Date().toISOString() })
        .eq("id", id);
      fail("update document", error);
    },
  };
}
