import { generateText as generateGroq } from "@/lib/groq";
import { generateText as generateGemini } from "@/lib/gemini";
import type { VectorDoc } from "@/lib/vectorStore";

/** How many candidates to pull from first-stage retrieval when reranking. */
export function rerankCandidateCount(topK: number): number {
  return Math.max(topK * 4, 20);
}

const MAX_PASSAGE_CHARS = 600;

/**
 * Listwise LLM rerank: first-stage retrieval optimizes recall (many
 * candidates), the LLM re-orders for precision against the original question.
 * Falls back to the incoming order on any provider or parse failure.
 */
export async function rerankDocs(
  query: string,
  docs: VectorDoc[],
  topK: number
): Promise<VectorDoc[]> {
  if (docs.length <= topK) return docs.slice(0, topK);

  const passages = docs
    .map((d, i) => `[${i}] ${d.text.slice(0, MAX_PASSAGE_CHARS)}`)
    .join("\n\n");

  const prompt = `You are a retrieval reranker. Rank the passages below by how useful each is for answering the question. Judge only relevance to THIS question; ignore writing quality.

Question: "${query}"

Passages:
${passages}

Respond ONLY with JSON listing ALL passage indices from most to least relevant:
{"ranking": [3, 0, 7, ...]}`;

  let raw: string;
  try {
    raw = await generateGroq(prompt, { temperature: 0, maxOutputTokens: 500 });
  } catch {
    try {
      raw = await generateGemini(prompt, { temperature: 0, maxOutputTokens: 500 });
    } catch {
      return docs.slice(0, topK);
    }
  }

  try {
    const cleaned = raw.replace(/```json|```/g, "").trim();
    const parsed = JSON.parse(cleaned) as { ranking?: unknown };
    if (!Array.isArray(parsed.ranking)) return docs.slice(0, topK);

    const seen = new Set<number>();
    const ordered: VectorDoc[] = [];
    for (const idx of parsed.ranking) {
      if (typeof idx === "number" && Number.isInteger(idx) && idx >= 0 && idx < docs.length && !seen.has(idx)) {
        seen.add(idx);
        ordered.push(docs[idx]);
      }
    }
    // Any indices the model dropped keep their first-stage order at the tail.
    docs.forEach((d, i) => {
      if (!seen.has(i)) ordered.push(d);
    });
    return ordered.slice(0, topK);
  } catch {
    return docs.slice(0, topK);
  }
}
