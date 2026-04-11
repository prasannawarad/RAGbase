/**
 * Normalized retrieval row returned by /api/retrieve (client + server).
 */
export type RetrievalResult = {
  id: string;
  documentId: string;
  documentName: string;
  chunkIndex: number;
  /** Full chunk text (for model context and inspectors). */
  content: string;
  /** Short preview for UI lists. */
  contentSnippet: string;
  score?: number;
  vectorScore?: number;
  bm25Score?: number;
};

export function makeContentSnippet(text: string, maxChars = 200): string {
  const t = text.trim();
  if (t.length <= maxChars) return t;
  return t.slice(0, Math.max(0, maxChars - 1)) + "…";
}
