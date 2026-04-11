import { tokenize } from "./tokenizer";

// Okapi BM25: state-of-the-art keyword ranking used in Elasticsearch,
// Solr, and most production search engines. Combined with vector search
// via Reciprocal Rank Fusion for hybrid retrieval.
export class BM25 {
  k1: number;
  b: number;
  docs: { tokens?: string[] }[];
  avgDL: number;
  idf: Record<string, number>;
  tf: Record<string, number>[];

  constructor(k1 = 1.5, b = 0.75) {
    this.k1 = k1; // term frequency saturation (1.2–2.0 typical)
    this.b = b; // document length normalization (0.75 typical)
    this.docs = [];
    this.avgDL = 0;
    this.idf = {};
    this.tf = [];
  }

  build(docs: { tokens?: string[] }[]) {
    this.docs = docs;
    if (!docs.length) return;
    this.avgDL = docs.reduce((s, d) => s + (d.tokens?.length || 0), 0) / docs.length;
    const N = docs.length;
    const df: Record<string, number> = {};
    docs.forEach((doc) => {
      const seen = new Set<string>();
      (doc.tokens || []).forEach((t) => {
        if (!seen.has(t)) {
          df[t] = (df[t] || 0) + 1;
          seen.add(t);
        }
      });
    });
    this.idf = Object.fromEntries(
      Object.entries(df).map(([t, n]) => [t, Math.log((N - n + 0.5) / (n + 0.5) + 1)])
    );
    this.tf = docs.map((doc) => {
      const freq: Record<string, number> = {};
      (doc.tokens || []).forEach((t) => {
        freq[t] = (freq[t] || 0) + 1;
      });
      return freq;
    });
  }

  scoreDoc(queryTokens: string[], docIdx: number) {
    const tf = this.tf[docIdx];
    if (!tf) return 0;
    const dl = this.docs[docIdx]?.tokens?.length || 0;
    return queryTokens.reduce((s, t) => {
      if (!tf[t]) return s;
      const idf = this.idf[t] || 0;
      const tfNorm =
        (tf[t] * (this.k1 + 1)) /
        (tf[t] + this.k1 * (1 - this.b + (this.b * dl) / (this.avgDL || 1)));
      return s + idf * tfNorm;
    }, 0);
  }

  search(query: string, topK = 10) {
    const qt = tokenize(query);
    return this.docs
      .map((_, i) => ({ idx: i, score: this.scoreDoc(qt, i) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);
  }
}
