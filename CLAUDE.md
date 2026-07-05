# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
npm run dev          # start dev server (localhost:3000)
npm run dev:turbo    # dev with Turbopack (faster HMR)
npm run dev:clean    # wipe .next cache then start dev
npm run build        # production build
npm run lint         # ESLint
npm run eval         # LLM-as-judge eval harness (see evals/README.md; dev server must be running)
npm run reembed      # re-embed all chunks in Supabase (after embedding model changes)
```

No unit test suite exists yet. TypeScript checking runs as part of `next build`. Retrieval/answer quality is measured with `npm run eval` against a golden dataset (`evals/dataset.json`).

## Environment Variables

Copy `.env.example` to `.env.local`. Required keys:

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | Embeddings (`gemini-embedding-001`) + Gemini LLM fallback |
| `LANGFUSE_SECRET_KEY` / `LANGFUSE_PUBLIC_KEY` | Optional — enables tracing; all tracing is a no-op when unset |
| `GROQ_API_KEY` | Primary LLM (Llama 3.3 70B by default) |
| `GROQ_MODEL` | Optional override of the Groq model — free-tier token budgets are per-model, so switching models recovers capacity after a daily quota is exhausted |
| `SUPABASE_URL` | Supabase project URL (server-preferred; falls back to `NEXT_PUBLIC_SUPABASE_URL`) |
| `SUPABASE_SERVICE_ROLE_KEY` | Bypasses RLS — server/API routes only, never client |
| `NEXT_PUBLIC_SUPABASE_URL` | Browser-safe project URL |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | Browser-safe anon key (reserved for future client-side use) |

Server-only keys (`GEMINI_API_KEY`, `GROQ_API_KEY`, `SUPABASE_SERVICE_ROLE_KEY`) must never appear in `NEXT_PUBLIC_*` variables or be imported from `"use client"` modules.

## Architecture

RAGBase is a Next.js 15 App Router application with a single-page UI backed by API routes. The frontend is one large client component (`src/components/RAGBase.tsx`) rendered by `src/app/page.tsx`. All business logic lives in `src/lib/`.

### Data flow

**Ingest**: Client sends text chunks → `POST /api/ingest` → embeds via Gemini `gemini-embedding-001` truncated to 768 dims and L2-normalized in `src/lib/gemini.ts` (the model's non-3072 outputs are not unit length; `text-embedding-004` was retired by Google) → inserts into Supabase `documents` + `chunks` tables (pgvector `vector(768)` column). If the embedding model ever changes, run `npm run reembed` — query and chunk vectors must come from the same model.

**Retrieve**: Client query → `POST /api/retrieve` → embed query → Supabase RPC:
- Vector-only: `match_chunks` (ANN, L2 distance)
- Hybrid: `chunk_vector_distances` (full L2 ordering) + in-server BM25 → Reciprocal Rank Fusion (k=60)
- Optional `filterDocumentId` scopes both modes to a single document
- Optional `rewrite: true` — LLM rewrites the query for retrieval (`src/lib/queryRewrite.ts`); response includes `rewrittenQuery`; the original question is still used for generation and reranking
- Optional `rerank: true` — first stage fetches `max(topK*4, 20)` candidates, then listwise LLM rerank (`src/lib/rerank.ts`) selects `topK`; both options fail open (fall back to unmodified behavior on any LLM error)

**Generate**: Retrieved chunks + query → `POST /api/chat` → Groq streaming (`llama-3.3-70b-versatile`), falls back to Gemini streaming, then to non-streaming variants of each. The response is `text/plain` UTF-8 deltas with `X-Chat-Stream: 1`, or a JSON fallback `{ mode: "text", text }`. Document summarization (detected by prompt content) returns `{ mode: "summary", ... }` JSON.

### LLM provider fallback chain

`groq stream → gemini stream → groq non-stream → gemini non-stream → FALLBACK_TEXT`

Provider quota reality (free tiers, July 2026): Groq is ~100–200k tokens/day *per model* (use `GROQ_MODEL` to switch to a fresh budget); Gemini `gemini-2.5-flash-lite` (the `GEMINI_TEXT_MODEL`) allows only **20 generate requests/day**, so the Gemini leg of the chain is a thin emergency fallback, not real capacity. `gemini-2.0-flash` has zero free-tier quota — do not revert to it.

Embedding also falls back: Gemini → `embedFallback.ts` (deterministic mock, same 768 dimensions, L2-normalized). WARNING: this fallback is silent — if Gemini embedding breaks, retrieval degrades to near-random while appearing to work. Check the `embed-query` span in Langfuse (embeddingSource: `mock-fallback`) or compare `/api/embed` output against `mockEmbeddingsForTexts`.

### Observability & evals

- `src/lib/observability.ts` — optional Langfuse tracing (no-op without keys). `/api/retrieve` traces embed + search spans; `/api/chat` traces generations, including streamed output via `tapTextStream`. Serverless requires `flushTraces()` before the response finishes.
- `scripts/eval.ts` (`npm run eval`) — LLM-as-judge harness (Groq Llama 3.3 judge) over the live API. Golden dataset in `evals/dataset.json` (gitignored; example checked in). Supports `--mode vector|hybrid|both` for A/B. Results written to `evals/results/`.
- `scripts/reembed.ts` (`npm run reembed`) — re-embeds every chunk row; `--dry` to count only. Gemini free tier is 100 embed requests/min; `embedOne` retries on 429.

### Database schema

Two tables, applied via migrations in `supabase/migrations/` (run in Supabase SQL editor in order):

- `documents` — metadata: name, raw_text, char_count, word_count, chunk_count, avg_chunk_size, pages, file_type, status, uploaded_at
- `chunks` — document_id (FK + cascade delete), chunk_index, content, embedding `vector(768)`, word_count, metadata jsonb

Two Supabase RPC functions:
- `match_chunks(query_embedding, match_count)` — ANN nearest-neighbor, returns ordered rows with L2 distance
- `chunk_vector_distances(query_embedding, filter_document_id)` — full distance ordering for all chunks, used by hybrid search

### Scoring

L2 distances from pgvector are mapped to `(0, 1]` via `1 / (1 + distance)`. The `VectorStore` class (`src/lib/vectorStore.ts`) is an in-memory fallback that uses cosine similarity — it is not used in the production DB path but exists for offline testing.

### Chunking

`src/lib/chunker.ts` splits on sentence boundaries (regex), targeting 500 chars with ~100-char overlap using the last 2 sentences of the previous chunk.

### PDF support

pdf.js is loaded dynamically from CDN at runtime in the client component (no npm install). Extracted text is passed to the chunker before ingest.

### Supabase client split

| File | Use |
|---|---|
| `src/lib/supabase/server.ts` | API routes only — singleton service-role client, guarded by `assertNotBrowser()` |
| `src/lib/supabase/client.ts` | Browser (anon key) — reserved for future client-side Supabase calls |
| `src/lib/supabase/env.ts` | URL resolution logic shared by both |

### `next.config.js` notes

`recharts` is listed in `transpilePackages` to fix a webpack crash with React 19. Polling watch is enabled (`poll: 1000`) for Docker/VM environments.
