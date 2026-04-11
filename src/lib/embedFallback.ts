/** Deterministic L2-normalized pseudo-embedding when Gemini is unavailable (same dim as text-embedding-004). */
const EMBED_DIM = 768;

export function mockEmbedding(text: string): number[] {
  let h = 2166136261;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  const out: number[] = [];
  let seed = h >>> 0;
  for (let i = 0; i < EMBED_DIM; i++) {
    seed = (Math.imul(seed, 1103515245) + 12345 + i) >>> 0;
    out.push((seed / 0xffffffff) * 2 - 1);
  }
  let norm = 0;
  for (const x of out) norm += x * x;
  norm = Math.sqrt(norm) + 1e-8;
  return out.map((x) => x / norm);
}

export function mockEmbeddingsForTexts(texts: string[]): number[][] {
  return texts.map((t) => mockEmbedding(String(t)));
}
