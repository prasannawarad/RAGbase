import { NextResponse } from "next/server";
import { embedTexts } from "@/lib/gemini";
import { mockEmbeddingsForTexts } from "@/lib/embedFallback";

export async function POST(req: Request) {
  try {
    const body = (await req.json()) as { texts?: unknown };
    const texts = body.texts;
    if (!Array.isArray(texts)) {
      return NextResponse.json({ error: "Expected body: { texts: string[] }" }, { status: 400 });
    }
    const strTexts = texts.map((t) => String(t));

    try {
      const embeddings = await embedTexts(strTexts);
      return NextResponse.json({ embeddings });
    } catch {
      const embeddings = mockEmbeddingsForTexts(strTexts);
      return NextResponse.json({ embeddings });
    }
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }
}
