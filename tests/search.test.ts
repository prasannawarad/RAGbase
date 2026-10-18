import { describe, expect, it } from "vitest";
import {
  distanceToVectorScore,
  mapHybridRow,
  mapVectorRow,
  toVectorParam,
  type SearchRow,
} from "@/lib/search";

const row: SearchRow = {
  id: "c1",
  content: "Backups run nightly at 02:00 UTC.",
  document_id: "d1",
  chunk_index: 3,
  distance: 1,
  word_count: 6,
  metadata: { docName: "ops.md" },
  keyword_score: 0.42,
  rrf_score: 0.031,
};

describe("toVectorParam", () => {
  it("formats a 768-dim embedding as a pgvector literal", () => {
    const lit = toVectorParam(new Array(768).fill(0.5));
    expect(lit.startsWith("[0.5,0.5")).toBe(true);
    expect(lit.endsWith("]")).toBe(true);
  });

  it("rejects wrong dimensions instead of sending them to the database", () => {
    expect(() => toVectorParam([1, 2, 3])).toThrow(/768/);
  });
});

describe("distanceToVectorScore", () => {
  it("maps distance into (0, 1], higher = closer", () => {
    expect(distanceToVectorScore(0)).toBe(1);
    expect(distanceToVectorScore(1)).toBe(0.5);
    expect(distanceToVectorScore(0.2)).toBeGreaterThan(distanceToVectorScore(0.9));
  });
});

describe("row mappers", () => {
  it("mapVectorRow scores by distance and carries metadata", () => {
    const doc = mapVectorRow(row);
    expect(doc.score).toBe(0.5);
    expect(doc.vectorScore).toBe(0.5);
    expect(doc.metadata).toMatchObject({ docId: "d1", chunkIndex: 3, docName: "ops.md", wordCount: 6 });
    expect(doc.text).toBe(row.content);
  });

  it("mapHybridRow ranks by the RRF score and exposes the keyword score as bm25Score", () => {
    const doc = mapHybridRow(row);
    expect(doc.score).toBe(0.031);
    expect(doc.bm25Score).toBe(0.42);
    expect(doc.vectorScore).toBe(0.5);
  });

  it("tolerates missing metadata and keyword fields", () => {
    const doc = mapHybridRow({ ...row, metadata: null, keyword_score: undefined, rrf_score: undefined });
    expect(doc.metadata.docName).toBeUndefined();
    expect(doc.score).toBe(0);
    expect(doc.bm25Score).toBe(0);
  });
});
