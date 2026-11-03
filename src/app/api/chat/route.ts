import { NextResponse } from "next/server";
import { generateText as generateGroq, openGroqChatTextStream, groqModelName } from "@/lib/groq";
import { generateText as generateGemini, openGeminiChatTextStream, GEMINI_TEXT_MODEL } from "@/lib/gemini";
import { startTrace, flushTraces, tapTextStream } from "@/lib/observability";
import { guardDailyBudget, guardRequest, guardUser } from "@/lib/apiGuard";
import { requireUser } from "@/lib/auth";
import { apiError, parseBody } from "@/lib/api/http";
import { chatSchema } from "@/lib/api/schemas";

const GROQ_MODEL = groqModelName();

/** Per-user daily cap on LLM generations; env-tunable. */
function dailyLimit(envName: string, fallback: number): number {
  const n = Number(process.env[envName]);
  return Number.isFinite(n) && n > 0 ? n : fallback;
}
const GEMINI_MODEL = GEMINI_TEXT_MODEL;

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

function jsonTextFromCompletion(text: string) {
  const trimmed = text.trim();
  if (!trimmed) {
    return NextResponse.json({ mode: "text", text: FALLBACK_TEXT });
  }
  let out = trimmed;
  if (!out.includes("<metadata>")) {
    out += `\n\n<metadata>{"confidence":"medium","sources_used":[],"key_entities":[],"follow_up_questions":[]}</metadata>`;
  }
  return NextResponse.json({ mode: "text", text: out });
}

const streamHeaders = {
  "Content-Type": "text/plain; charset=utf-8",
  "Cache-Control": "no-store",
  "X-Chat-Stream": "1",
} as const;

/**
 * Body: { prompt: string }
 * - RAG chat → streaming text/plain (UTF-8 deltas) with X-Chat-Stream: 1, or JSON fallback { mode: "text", text }
 * - Document summary → { mode: "summary", ... } (JSON, non-streaming)
 */
export async function POST(req: Request) {
  // LLM generation is the most expensive call — cap it tightest.
  const blocked = await guardRequest(req, { name: "chat", limit: 20 });
  if (blocked) return blocked;
  // Burns LLM / embedding quota — signed-in users only.
  const auth = await requireUser(req);
  if (!auth.ok) return auth.response;
  const perUser =
    (await guardUser(auth.user.id, { name: "chat", limit: 20 })) ??
    (await guardDailyBudget(auth.user.id, "chat", dailyLimit("DAILY_CHAT_LIMIT", 300)));
  if (perUser) return perUser;
  const parsed = await parseBody(req, chatSchema);
  if (!parsed.ok) return parsed.response;
  const { prompt } = parsed.data;

  try {

    const isSummary =
      prompt.includes("Summarize this document") && prompt.includes("Respond ONLY in JSON");

    const trace = startTrace(isSummary ? "chat-summary" : "chat-rag", {
      input: { prompt },
      tags: [isSummary ? "summary" : "rag"],
    });

    if (isSummary) {
      let raw: string;
      let model = GROQ_MODEL;
      const gen = trace?.generation({
        name: "summary",
        input: prompt,
        model,
        modelParameters: { temperature: 0.4, max_tokens: 3000 },
      });
      try {
        raw = await generateGroq(prompt, { temperature: 0.4, maxOutputTokens: 3000 });
      } catch {
        try {
          model = GEMINI_MODEL;
          raw = await generateGemini(prompt, { temperature: 0.4, maxOutputTokens: 3000 });
        } catch {
          gen?.end({ output: null, level: "ERROR", statusMessage: "all providers failed" });
          await flushTraces();
          return NextResponse.json(FALLBACK_SUMMARY);
        }
      }
      gen?.end({ output: raw, model });
      const parsed = parseSummaryFromModel(raw);
      trace?.update({ output: parsed ?? { error: "unparseable summary" } });
      await flushTraces();
      if (parsed) {
        return NextResponse.json({
          mode: "summary",
          ...parsed,
        });
      }
      return NextResponse.json(FALLBACK_SUMMARY);
    }

    const streamResponse = (stream: ReadableStream<Uint8Array>, model: string) => {
      if (!trace) return new Response(stream, { headers: streamHeaders });
      const gen = trace.generation({
        name: "rag-answer",
        input: prompt,
        model,
        modelParameters: { temperature: 0.3, max_tokens: 3000 },
      });
      const tapped = tapTextStream(stream, async (fullText) => {
        gen.end({ output: fullText });
        trace.update({ output: { text: fullText, streamed: true, model } });
        await flushTraces();
      });
      return new Response(tapped, { headers: streamHeaders });
    };

    const traceCompletion = async (text: string, model: string) => {
      if (!trace) return;
      trace.generation({
        name: "rag-answer",
        input: prompt,
        model,
        modelParameters: { temperature: 0.3, max_tokens: 3000 },
        output: text,
        endTime: new Date(),
      });
      trace.update({ output: { text, streamed: false, model } });
      await flushTraces();
    };

    try {
      const stream = await openGroqChatTextStream(prompt, { temperature: 0.3, maxOutputTokens: 3000 });
      return streamResponse(stream, GROQ_MODEL);
    } catch (groqStreamErr) {
      console.warn("[api/chat] Groq stream failed:", groqStreamErr);
      try {
        const stream = await openGeminiChatTextStream(prompt, { temperature: 0.3, maxOutputTokens: 3000 });
        return streamResponse(stream, GEMINI_MODEL);
      } catch (geminiStreamErr) {
        console.warn("[api/chat] Gemini stream failed:", geminiStreamErr);
        try {
          const text = await generateGroq(prompt, { temperature: 0.3, maxOutputTokens: 3000 });
          await traceCompletion(text, GROQ_MODEL);
          return jsonTextFromCompletion(text);
        } catch (groqTextErr) {
          console.warn("[api/chat] Groq generateText failed:", groqTextErr);
          try {
            const text = await generateGemini(prompt, { temperature: 0.3, maxOutputTokens: 3000 });
            await traceCompletion(text, GEMINI_MODEL);
            return jsonTextFromCompletion(text);
          } catch (geminiTextErr) {
            console.warn("[api/chat] Gemini generateText failed:", geminiTextErr);
            trace?.update({ output: { error: "all providers failed" } });
            await flushTraces();
            return NextResponse.json({ mode: "text", text: FALLBACK_TEXT });
          }
        }
      }
    }
  } catch (e) {
    console.error("[api/chat] unexpected", e);
    return apiError(500, "internal_error", "Chat request failed");
  }
}
