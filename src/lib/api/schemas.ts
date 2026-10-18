import { z } from "zod";

/**
 * API request contracts — the single source of truth for what each route accepts.
 * Kept free of server-only imports so the schemas are unit-testable and could be
 * shared with the client later.
 */

export const LIMITS = {
  /** Max raw request body, in bytes, by route class. */
  bodyBytesDefault: 256 * 1024,
  bodyBytesIngest: 6 * 1024 * 1024,
  documentName: 255,
  fileType: 50,
  chunksPerIngest: 3000,
  chunkChars: 4000,
  query: 2000,
  topKMax: 50,
  promptChars: 32_000,
  embedTexts: 100,
  embedTextChars: 8000,
} as const;

const uuid = z.string().uuid();

export const ingestSchema = z.object({
  documentName: z.string().trim().min(1).max(LIMITS.documentName),
  chunks: z
    .array(z.string().trim().min(1).max(LIMITS.chunkChars))
    .min(1)
    .max(LIMITS.chunksPerIngest),
  fileType: z.string().trim().min(1).max(LIMITS.fileType).optional(),
  pages: z.number().int().nonnegative().nullable().optional(),
});
export type IngestInput = z.infer<typeof ingestSchema>;

export const retrieveSchema = z.object({
  query: z.string().trim().min(1).max(LIMITS.query),
  topK: z.number().int().min(1).max(LIMITS.topKMax).optional(),
  useHybrid: z.boolean().optional(),
  rerank: z.boolean().optional(),
  rewrite: z.boolean().optional(),
  filterDocumentId: uuid.nullish(),
});
export type RetrieveInput = z.infer<typeof retrieveSchema>;

export const chatSchema = z.object({
  prompt: z.string().min(1).max(LIMITS.promptChars),
});
export type ChatInput = z.infer<typeof chatSchema>;

export const embedSchema = z.object({
  texts: z.array(z.string().max(LIMITS.embedTextChars)).min(1).max(LIMITS.embedTexts),
});
export type EmbedInput = z.infer<typeof embedSchema>;

export const documentIdParamSchema = z.object({ id: uuid });
