import { NextResponse } from "next/server";
import { generateText, openGeminiChatTextStream } from "@/lib/gemini";

function parseSummaryFromModel(raw: string): {
  summary: string;
  topics: string[];
  key_entities: string[];
  sentiment: string;
  complexity: string;
} | null {
  const cleaned = raw.replace(/```json|```/g, "").trim();
  try {
    const o = JSON.parse(cleaned) as Record<string, unknown>;
    if (typeof o.summary !== "string") return null;
    return {
      summary: o.summary,
      topics: Array.isArray(o.topics) ? o.topics.map(String) : [],
      key_entities: Array.isArray(o.key_entities) ? o.key_entities.map(String) : [],
      sentiment: typeof o.sentiment === "string" ? o.sentiment : "neutral",
      complexity: typeof o.complexity === "string" ? o.complexity : "intermediate",
    };
  } catch {
    return null;
  }
}

const FALLBACK_TEXT = `The answer could not be generated right now. Please try again or check server configuration.

<metadata>{"confidence":"low","sources_used":[],"key_entities":[],"follow_up_questions":[]}</metadata>`;

const FALLBACK_SUMMARY = {
  mode: "summary" as const,
  summary: "Summary unavailable. Please try again later.",
  topics: ["N/A"],
  key_entities: [],
  sentiment: "neutral",
  complexity: "intermediate",
};

/**
 * Body: { prompt: string }
 * - RAG chat → streaming text/plain (UTF-8 deltas) with X-Chat-Stream: 1, or JSON fallback { mode: "text", text }
 * - Document summary → { mode: "summary", ... } (JSON, non-streaming)
 */
export async function POST(req: Request) {
  try {
    const body = (await req.json()) as { prompt?: unknown };
    const prompt = typeof body.prompt === "string" ? body.prompt : "";
    if (!prompt) {
      return NextResponse.json({ error: "Expected body: { prompt: string }" }, { status: 400 });
    }

    const isSummary =
      prompt.includes("Summarize this document") && prompt.includes("Respond ONLY in JSON");

    if (isSummary) {
      try {
        const raw = await generateText(prompt, { temperature: 0.4, maxOutputTokens: 3000 });
        const parsed = parseSummaryFromModel(raw);
        if (parsed) {
          return NextResponse.json({
            mode: "summary",
            ...parsed,
          });
        }
        return NextResponse.json(FALLBACK_SUMMARY);
      } catch {
        return NextResponse.json(FALLBACK_SUMMARY);
      }
    }

    try {
      const stream = await openGeminiChatTextStream(prompt, { temperature: 0.3, maxOutputTokens: 3000 });
      return new Response(stream, {
        headers: {
          "Content-Type": "text/plain; charset=utf-8",
          "Cache-Control": "no-store",
          "X-Chat-Stream": "1",
        },
      });
    } catch (streamErr) {
      console.warn("[api/chat] stream failed, falling back to generateContent:", streamErr);
      try {
        const text = await generateText(prompt, { temperature: 0.3, maxOutputTokens: 3000 });
        const trimmed = text.trim();
        if (!trimmed) {
          return NextResponse.json({ mode: "text", text: FALLBACK_TEXT });
        }
        let out = trimmed;
        if (!out.includes("<metadata>")) {
          out += `\n\n<metadata>{"confidence":"medium","sources_used":[],"key_entities":[],"follow_up_questions":[]}</metadata>`;
        }
        return NextResponse.json({ mode: "text", text: out });
      } catch {
        return NextResponse.json({ mode: "text", text: FALLBACK_TEXT });
      }
    }
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }
}
