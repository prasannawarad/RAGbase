import { NextResponse } from "next/server";
import { requireUser } from "@/lib/auth";
import { guardRequest } from "@/lib/apiGuard";
import { bumpUserCacheVersion } from "@/lib/cache";
import { apiError, validate } from "@/lib/api/http";
import { documentIdParamSchema } from "@/lib/api/schemas";

export async function DELETE(
  req: Request,
  context: { params: Promise<{ id: string }> }
) {
  // Destructive — same-origin only, and rate-limited against delete storms.
  const blocked = await guardRequest(req, { name: "documents-delete", limit: 20 });
  if (blocked) return blocked;
  const auth = await requireUser(req);
  if (!auth.ok) return auth.response;
  const { supabase, user } = auth;

  const params = validate(documentIdParamSchema, await context.params);
  if (!params.ok) return params.response;

  try {
    const { error } = await supabase.from("documents").delete().eq("id", params.data.id);
    if (error) throw new Error(error.message);
    await bumpUserCacheVersion(user.id);
    return NextResponse.json({ ok: true });
  } catch (e) {
    console.error("[api/documents/[id] DELETE]", e);
    return apiError(500, "internal_error", "Failed to delete document");
  }
}
