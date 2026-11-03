import { describe, expect, it } from "vitest";
import { chunkText } from "@/lib/chunker";

const sentence = (n: number) => `This is sentence number ${n} in the test document.`;

describe("chunkText", () => {
  it("returns a single chunk for short text", () => {
    const chunks = chunkText("Hello world. Short text.");
    expect(chunks).toHaveLength(1);
    expect(chunks[0]!.index).toBe(0);
  });

  it("splits long text into multiple chunks near the target size", () => {
    const text = Array.from({ length: 60 }, (_, i) => sentence(i)).join(" ");
    const chunks = chunkText(text, { chunkSize: 200, overlap: 100 });
    expect(chunks.length).toBeGreaterThan(5);
    for (const c of chunks) expect(c.text.length).toBeLessThanOrEqual(200 + 100 + 60);
  });

  it("numbers chunks sequentially with stable ids", () => {
    const text = Array.from({ length: 30 }, (_, i) => sentence(i)).join(" ");
    const chunks = chunkText(text, { chunkSize: 150 });
    chunks.forEach((c, i) => {
      expect(c.index).toBe(i);
      expect(c.id).toBe(`chunk_${i}`);
    });
  });

  it("overlaps adjacent chunks by repeating trailing sentences", () => {
    const text = Array.from({ length: 30 }, (_, i) => sentence(i)).join(" ");
    const chunks = chunkText(text, { chunkSize: 200, overlap: 120 });
    const overlapped = chunks.slice(1).some((c, i) => {
      const lastSentenceOfPrev = chunks[i]!.text.split(/(?<=\.)\s/).pop()!;
      return c.text.includes(lastSentenceOfPrev);
    });
    expect(overlapped).toBe(true);
  });

  it("does not lose content: every sentence appears in some chunk", () => {
    const sentences = Array.from({ length: 25 }, (_, i) => sentence(i));
    const chunks = chunkText(sentences.join(" "), { chunkSize: 180 });
    const joined = chunks.map((c) => c.text).join(" ");
    for (const s of sentences) expect(joined).toContain(s);
  });
});
