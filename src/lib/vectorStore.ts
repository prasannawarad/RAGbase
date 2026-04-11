import { BM25 } from "./bm25";
import { tokenize } from "./tokenizer";

export type DocMetadata = {
  docName?: string;
  docId?: string;
  chunkIndex?: number;
  wordCount?: number;
};

export type VectorDoc = {
  id: string;
  text: string;
  embedding: number[];
  metadata: DocMetadata;
  tokens: string[];
  score?: number;
  vectorScore?: number;
  bm25Score?: number;
};

function cosineSimilarity(a: number[], b: number[]) {
  let dot = 0,
    normA = 0,
    normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB) + 1e-8);
}

export class VectorStore {
  documents: VectorDoc[];
  bm25: BM25;

  constructor() {
    this.documents = [];
    this.bm25 = new BM25();
  }

  add(id: string, text: string, embedding: number[], metadata: DocMetadata = {}) {
    this.documents.push({ id, text, embedding, metadata, tokens: tokenize(text) });
    this._rebuildBM25();
  }

  addBatch(entries: { id: string; text: string; embedding: number[]; metadata: DocMetadata }[]) {
    entries.forEach(({ id, text, embedding, metadata }) => {
      this.documents.push({ id, text, embedding, metadata, tokens: tokenize(text) });
    });
    this._rebuildBM25();
  }

  _rebuildBM25() {
    this.bm25.build(this.documents);
  }

  search(queryEmbedding: number[], topK = 5, filter: ((d: VectorDoc) => boolean) | null = null) {
    let docs = filter ? this.documents.filter(filter) : this.documents;
    return docs
      .map((doc) => ({ ...doc, score: cosineSimilarity(queryEmbedding, doc.embedding) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);
  }

  // Hybrid search: BM25 + Vector fused via Reciprocal Rank Fusion
  // RRF(d) = Σ 1/(k + rank_i(d))  — k=60 is standard default
  hybridSearch(
    query: string,
    queryEmbedding: number[],
    topK = 5,
    filter: ((d: VectorDoc) => boolean) | null = null,
    k = 60
  ) {
    let docs = filter ? this.documents.filter(filter) : this.documents;
    if (!docs.length) return [];

    const vectorRanked = docs
      .map((doc, i) => ({ i, vectorScore: cosineSimilarity(queryEmbedding, doc.embedding) }))
      .sort((a, b) => b.vectorScore - a.vectorScore);

    const qt = tokenize(query);
    const bm25Ranked = docs
      .map((doc, i) => ({
        i,
        bm25Score: this.bm25.scoreDoc(qt, this.documents.indexOf(doc)),
      }))
      .sort((a, b) => b.bm25Score - a.bm25Score);

    const rrfScores: Record<number, number> = {};
    vectorRanked.forEach(({ i }, rank) => {
      rrfScores[i] = (rrfScores[i] || 0) + 1 / (k + rank + 1);
    });
    bm25Ranked.forEach(({ i }, rank) => {
      rrfScores[i] = (rrfScores[i] || 0) + 1 / (k + rank + 1);
    });

    const vectorScoreMap = Object.fromEntries(vectorRanked.map(({ i, vectorScore }) => [i, vectorScore]));
    const bm25ScoreMap = Object.fromEntries(bm25Ranked.map(({ i, bm25Score }) => [i, bm25Score]));

    return docs
      .map((doc, i) => ({
        ...doc,
        score: rrfScores[i] || 0,
        vectorScore: vectorScoreMap[i] || 0,
        bm25Score: bm25ScoreMap[i] || 0,
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);
  }

  getByDocId(docId: string) {
    return this.documents.filter((d) => d.metadata.docId === docId);
  }
  clear() {
    this.documents = [];
    this._rebuildBM25();
  }
  removeByDocId(docId: string) {
    this.documents = this.documents.filter((d) => d.metadata.docId !== docId);
    this._rebuildBM25();
  }
  get size() {
    return this.documents.length;
  }
  getStats() {
    const byDoc: Record<string, { count: number; totalLen: number }> = {};
    this.documents.forEach((d) => {
      const name = d.metadata.docName || "Unknown";
      if (!byDoc[name]) byDoc[name] = { count: 0, totalLen: 0 };
      byDoc[name].count++;
      byDoc[name].totalLen += d.text.length;
    });
    return byDoc;
  }
}
