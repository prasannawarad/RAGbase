import { NextResponse } from "next/server";
import { embedTexts } from "@/lib/gemini";
import { mockEmbeddingsForTexts } from "@/lib/embedFallback";
import { guardRequest } from "@/lib/apiGuard";
import { requireUser } from "@/lib/auth";
import { parseBody } from "@/lib/api/http";
import { embedSchema } from "@/lib/api/schemas";

export async function POST(req: Request) {
  const blocked = guardRequest(req, { name: "embed", limit: 20 });
  if (blocked) return blocked;
  // Burns LLM / embedding quota — signed-in users only.
  const auth = await requireUser(req);
  if (!auth.ok) return auth.response;

  const parsed = await parseBody(req, embedSchema);
  if (!parsed.ok) return parsed.response;
  const { texts } = parsed.data;

  try {
    const embeddings = await embedTexts(texts);
    return NextResponse.json({ embeddings });
  } catch {
    const embeddings = mockEmbeddingsForTexts(texts);
    return NextResponse.json({ embeddings });
  }
}
