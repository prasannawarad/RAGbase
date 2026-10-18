import { NextResponse } from "next/server";
import { guardRequest, guardUser } from "@/lib/apiGuard";
import { apiError, validate } from "@/lib/api/http";
import { documentIdParamSchema } from "@/lib/api/schemas";
import { requireUser } from "@/lib/auth";
import { bumpUserCacheVersion } from "@/lib/cache";
import { embedTexts } from "@/lib/gemini";
import { mockEmbeddingsForTexts } from "@/lib/embedFallback";
import { processDocument } from "@/lib/ingestWorker";
import { supabaseIngestStore } from "@/lib/ingestStore";

// Embedding retries can wait on Gemini's rate limit; leave headroom beyond the worker's 20s budget.
export const maxDuration = 60;

/**
 * Real embeddings only, by default. The old ingest path silently stored mock vectors when Gemini
 * failed, which corrupts retrieval permanently (they survive until a full reembed). The worker
 * retries and then marks the document failed instead. Set ALLOW_MOCK_EMBEDDINGS=true to opt back
 * into the mock fallback for offline demos.
 */
async function embedForIngest(texts: string[]): Promise<number[][]> {
  try {
    return await embedTexts(texts);
  } catch (e) {
    if (process.env.ALLOW_MOCK_EMBEDDINGS === "true") return mockEmbeddingsForTexts(texts);
    throw e;
  }
}

/**
 * POST /api/ingest/:id/process[?retry=1]
 * Advances ingestion of one document by up to ~20s of work and reports progress. Idempotent and
 * safe to call concurrently or repeatedly: the client calls it until status is indexed | failed.
 */
export async function POST(req: Request, context: { params: Promise<{ id: string }> }) {
  const blocked = await guardRequest(req, { name: "ingest-process", limit: 120 });
  if (blocked) return blocked;
  const auth = await requireUser(req);
  if (!auth.ok) return auth.response;
  const { supabase, user } = auth;
  const perUser = await guardUser(user.id, { name: "ingest-process", limit: 60 });
  if (perUser) return perUser;

  const params = validate(documentIdParamSchema, await context.params);
  if (!params.ok) return params.response;

  try {
    const retry = new URL(req.url).searchParams.get("retry") === "1";
    const result = await processDocument(supabaseIngestStore(supabase, user.id), params.data.id, {
      embed: embedForIngest,
      retry,
    });
    if (!result) return apiError(404, "not_found", "Document not found");
    if (result.status === "indexed") await bumpUserCacheVersion(user.id);
    return NextResponse.json(result);
  } catch (e) {
    console.error("[api/ingest/process]", e);
    return apiError(500, "internal_error", "Ingestion step failed");
  }
}
