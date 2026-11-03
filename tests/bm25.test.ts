import { describe, expect, it } from "vitest";
import { BM25 } from "@/lib/bm25";
import { tokenize } from "@/lib/tokenizer";

function build(texts: string[]) {
  const bm25 = new BM25();
  bm25.build(texts.map((t) => ({ tokens: tokenize(t) })));
  return bm25;
}

describe("tokenize", () => {
  it("lowercases and splits on non-word characters", () => {
    expect(tokenize("Hello, World!")).toEqual(expect.arrayContaining(["hello", "world"]));
  });
});

describe("BM25", () => {
  const corpus = [
    "postgres supports vector similarity search with pgvector",
    "the weather today is sunny and warm",
    "bm25 is a keyword ranking function used in search engines",
  ];

  it("ranks the document containing the query terms highest", () => {
    const bm25 = build(corpus);
    const q = tokenize("pgvector similarity");
    const scores = corpus.map((_, i) => bm25.scoreDoc(q, i));
    expect(scores[0]!).toBeGreaterThan(scores[1]!);
    expect(scores[0]!).toBeGreaterThan(scores[2]!);
  });

  it("scores zero when no query term appears", () => {
    const bm25 = build(corpus);
    expect(bm25.scoreDoc(tokenize("zebra"), 1)).toBe(0);
  });

  it("gives rarer terms more weight than common ones", () => {
    const bm25 = build(["search search engine", "search tools", "rare unicorn search"]);
    const rare = bm25.scoreDoc(tokenize("unicorn"), 2);
    const common = bm25.scoreDoc(tokenize("search"), 2);
    expect(rare).toBeGreaterThan(common);
  });
});
