import { generateText as generateGroq } from "@/lib/groq";
import { generateText as generateGemini } from "@/lib/gemini";

/**
 * Rewrite a user question into a self-contained search query optimized for
 * embedding + BM25 retrieval (expanded synonyms, preserved entities).
 * Falls back to the original query on any provider or parse failure, so
 * enabling rewrite can never break retrieval.
 */
export async function rewriteQuery(query: string): Promise<string> {
  const prompt = `You optimize search queries for a document retrieval system that combines semantic embeddings with BM25 keyword matching.

Rewrite the user's question into ONE self-contained search query:
- Keep every named entity, number, and technical term exactly as written
- Expand abbreviations and add 2-4 close synonyms for the key concepts
- Drop filler words ("please", "can you tell me")
- Do NOT answer the question or add facts not implied by it

User question: "${query}"

Respond ONLY with JSON: {"query": "rewritten search query"}`;

  let raw: string;
  try {
    raw = await generateGroq(prompt, { temperature: 0, maxOutputTokens: 200 });
  } catch {
    try {
      raw = await generateGemini(prompt, { temperature: 0, maxOutputTokens: 200 });
    } catch {
      return query;
    }
  }

  try {
    const cleaned = raw.replace(/```json|```/g, "").trim();
    const parsed = JSON.parse(cleaned) as { query?: unknown };
    if (typeof parsed.query === "string" && parsed.query.trim().length > 0) {
      return parsed.query.trim();
    }
  } catch {
    // fall through to original
  }
  return query;
}
