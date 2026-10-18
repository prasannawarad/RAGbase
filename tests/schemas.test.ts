import { describe, expect, it } from "vitest";
import {
  chatSchema,
  documentIdParamSchema,
  embedSchema,
  ingestSchema,
  LIMITS,
  retrieveSchema,
} from "@/lib/api/schemas";

const uuid = "123e4567-e89b-42d3-a456-426614174000";

describe("ingestSchema", () => {
  it("accepts the minimal payload and trims strings", () => {
    const r = ingestSchema.parse({ documentName: "  notes.txt ", chunks: [" hello "] });
    expect(r.documentName).toBe("notes.txt");
    expect(r.chunks).toEqual(["hello"]);
  });

  it("accepts null pages (non-PDF uploads)", () => {
    expect(ingestSchema.safeParse({ documentName: "a", chunks: ["x"], pages: null }).success).toBe(true);
  });

  it.each([
    ["missing name", { chunks: ["x"] }],
    ["blank name", { documentName: "  ", chunks: ["x"] }],
    ["no chunks", { documentName: "a", chunks: [] }],
    ["blank chunk", { documentName: "a", chunks: ["ok", "  "] }],
    ["non-string chunk", { documentName: "a", chunks: [42] }],
    ["negative pages", { documentName: "a", chunks: ["x"], pages: -1 }],
  ])("rejects %s", (_label, body) => {
    expect(ingestSchema.safeParse(body).success).toBe(false);
  });

  it("enforces size caps", () => {
    expect(
      ingestSchema.safeParse({ documentName: "a", chunks: ["x".repeat(LIMITS.chunkChars + 1)] }).success
    ).toBe(false);
    expect(
      ingestSchema.safeParse({ documentName: "a", chunks: Array(LIMITS.chunksPerIngest + 1).fill("x") }).success
    ).toBe(false);
  });
});

describe("retrieveSchema", () => {
  it("accepts a full valid payload", () => {
    const r = retrieveSchema.parse({
      query: "what is rag",
      topK: 5,
      useHybrid: true,
      rerank: true,
      rewrite: false,
      filterDocumentId: uuid,
    });
    expect(r.topK).toBe(5);
  });

  it("allows a null filterDocumentId", () => {
    expect(retrieveSchema.safeParse({ query: "q", filterDocumentId: null }).success).toBe(true);
  });

  it.each([
    ["empty query", { query: "" }],
    ["topK 0", { query: "q", topK: 0 }],
    ["topK too large", { query: "q", topK: LIMITS.topKMax + 1 }],
    ["fractional topK", { query: "q", topK: 2.5 }],
    ["bad document id", { query: "q", filterDocumentId: "not-a-uuid" }],
    ["string boolean", { query: "q", useHybrid: "true" }],
  ])("rejects %s", (_label, body) => {
    expect(retrieveSchema.safeParse(body).success).toBe(false);
  });
});

describe("chatSchema / embedSchema / params", () => {
  it("chat requires a non-empty prompt under the cap", () => {
    expect(chatSchema.safeParse({ prompt: "hi" }).success).toBe(true);
    expect(chatSchema.safeParse({ prompt: "" }).success).toBe(false);
    expect(chatSchema.safeParse({ prompt: "x".repeat(LIMITS.promptChars + 1) }).success).toBe(false);
  });

  it("embed caps batch size", () => {
    expect(embedSchema.safeParse({ texts: ["a"] }).success).toBe(true);
    expect(embedSchema.safeParse({ texts: [] }).success).toBe(false);
    expect(embedSchema.safeParse({ texts: Array(LIMITS.embedTexts + 1).fill("a") }).success).toBe(false);
  });

  it("document id param must be a uuid", () => {
    expect(documentIdParamSchema.safeParse({ id: uuid }).success).toBe(true);
    expect(documentIdParamSchema.safeParse({ id: "1; drop table documents" }).success).toBe(false);
  });
});
