# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

RAGBase — a RAG pipeline app: upload documents (text/PDF), chunk and embed them into Supabase pgvector, retrieve via vector or hybrid search (with optional query rewrite and LLM rerank), and chat over the results through a streaming LLM fallback chain. Retrieval quality is measured with an LLM-as-judge eval harness.

## Commands

```bash
npm run dev          # start dev server (localhost:3000)
npm run dev:turbo    # dev with Turbopack (faster HMR)
npm run dev:clean    # wipe .next cache then start dev
npm run build        # production build
npm run lint         # ESLint
npm run typecheck    # tsc --noEmit
npm test             # Vitest unit tests (tests/*.test.ts); no network or DB needed
npm run eval         # LLM-as-judge eval harness (see evals/README.md; dev server must be running)
npm run reembed      # re-embed all chunks in Supabase (after embedding model changes)
```

Unit tests (Vitest) cover pure logic and the security seams: schemas, error/body parsing, rate limiter, auth gate, search mappers, chunker, BM25. They do **not** call an LLM or a database. CI (`.github/workflows/ci.yml`) runs lint → typecheck → test → build. Retrieval/answer quality is measured separately with `npm run eval` against a golden dataset (`evals/dataset.json`) — manual, not a CI gate (it spends free-tier tokens). SQL/RLS is tested on a scratch Postgres via `supabase/tests/` (see `docs/ARCHITECTURE.md` §7).

Design decisions and their reasoning live in `docs/ARCHITECTURE.md` — read it before changing auth, RLS, search, or the API contract.

## Environment Variables

Copy `.env.example` to `.env.local`. Required keys:

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | Embeddings (`gemini-embedding-001`) + Gemini LLM fallback |
| `LANGFUSE_SECRET_KEY` / `LANGFUSE_PUBLIC_KEY` | Optional — enables tracing; all tracing is a no-op when unset |
| `GROQ_API_KEY` | Primary LLM (Llama 3.3 70B by default) |
| `GROQ_MODEL` | Optional override of the Groq model — free-tier token budgets are per-model, so switching models recovers capacity after a daily quota is exhausted |
| `SUPABASE_URL` | Supabase project URL (server-preferred; falls back to `NEXT_PUBLIC_SUPABASE_URL`) |
| `SUPABASE_SERVICE_ROLE_KEY` | Bypasses RLS — admin scripts only (`npm run reembed`). The web app never uses it; there is no service-role client in `src/` |
| `NEXT_PUBLIC_SUPABASE_URL` | Browser-safe project URL |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | **Required.** Browser-safe anon key; all API data access runs as the logged-in user under RLS |
| `UPSTASH_REDIS_REST_URL` / `UPSTASH_REDIS_REST_TOKEN` | Optional — Upstash Redis for shared rate limits + caches. Unset = in-memory limiter, caching off |
| `ALLOW_MOCK_EMBEDDINGS` | Optional — `true` re-enables mock vectors at *ingest* (offline demos). Default off on purpose |
| `DAILY_CHAT_LIMIT` | Optional — per-user daily chat cap (default 300) |
| `EVAL_EMAIL` / `EVAL_PASSWORD` | Account `npm run eval` signs in as (must own the eval documents) |
| `ALLOWED_ORIGINS` | Optional — comma-separated extra origins allowed to call API routes (see API abuse guard). Deployment host + localhost-in-dev are always allowed |

Server-only keys (`GEMINI_API_KEY`, `GROQ_API_KEY`, `SUPABASE_SERVICE_ROLE_KEY`, `EVAL_PASSWORD`) must never appear in `NEXT_PUBLIC_*` variables or be imported from `"use client"` modules.

## Architecture

RAGBase is a Next.js 15 App Router application with a single-page UI backed by API routes. The frontend is one large client component (`src/components/RAGBase.tsx`) rendered by `src/app/page.tsx`. All business logic lives in `src/lib/`.

### Data flow

**Ingest** (two steps, resumable): Client sends text chunks → `POST /api/ingest` validates, dedups by content hash and stores the chunk *text* in `ingest_chunks` (document `status: queued`, 202) → the client calls `POST /api/ingest/:id/process` repeatedly; each call claims batches (`claim_ingest_batch`, leased, `SKIP LOCKED`), embeds via Gemini `gemini-embedding-001` truncated to 768 dims and L2-normalized in `src/lib/gemini.ts` (the model's non-3072 outputs are not unit length; `text-embedding-004` was retired by Google) → upserts real rows into `chunks` as the logged-in user (`user_id` set; pgvector `vector(768)` column) until the document is `indexed` (or `failed` after 3 attempts per chunk; the UI offers Retry). `processDocument()` in `src/lib/ingestWorker.ts` holds the logic behind an `IngestStore` interface (unit-tested with a fake). **Ingest does NOT fall back to mock embeddings** (they would be stored permanently) unless `ALLOW_MOCK_EMBEDDINGS=true`; query-time embedding still falls back. Unfinished documents auto-resume when the app loads. If the embedding model ever changes, run `npm run reembed` — query and chunk vectors must come from the same model.

**Retrieve**: Client query → `POST /api/retrieve` → embed query → Supabase RPC (both run in Postgres on indexes and are scoped to the caller by RLS):
- Vector-only: `match_chunks` (HNSW ANN, L2 distance)
- Hybrid: `hybrid_search_chunks` — HNSW top-100 ∪ Postgres full-text (GIN) top-100, fused with Reciprocal Rank Fusion (k=60) in SQL. The keyword score is `ts_rank_cd`, returned in the legacy `bm25Score` field (UI contract unchanged)
- Optional `filterDocumentId` scopes both modes to a single document
- Optional `rewrite: true` — LLM rewrites the query for retrieval (`src/lib/queryRewrite.ts`); response includes `rewrittenQuery`; the original question is still used for generation and reranking
- Optional `rerank: true` — first stage fetches `max(topK*4, 20)` candidates, then listwise LLM rerank (`src/lib/rerank.ts`) selects `topK`; both options fail open (fall back to unmodified behavior on any LLM error)

**Generate**: Retrieved chunks + query → `POST /api/chat` → Groq streaming (`llama-3.3-70b-versatile`), falls back to Gemini streaming, then to non-streaming variants of each. The response is `text/plain` UTF-8 deltas with `X-Chat-Stream: 1`, or a JSON fallback `{ mode: "text", text }`. Document summarization (detected by prompt content) returns `{ mode: "summary", ... }` JSON.

### LLM provider fallback chain

`groq stream → gemini stream → groq non-stream → gemini non-stream → FALLBACK_TEXT`

Provider quota reality (free tiers, July 2026): Groq is ~100–200k tokens/day *per model* (use `GROQ_MODEL` to switch to a fresh budget); Gemini `gemini-2.5-flash-lite` (the `GEMINI_TEXT_MODEL`) allows only **20 generate requests/day**, so the Gemini leg of the chain is a thin emergency fallback, not real capacity. `gemini-2.0-flash` has zero free-tier quota — do not revert to it.

Embedding also falls back: Gemini → `embedFallback.ts` (deterministic mock, same 768 dimensions, L2-normalized). WARNING: this fallback is silent — if Gemini embedding breaks, retrieval degrades to near-random while appearing to work. Check the `embed-query` span in Langfuse (embeddingSource: `mock-fallback`) or compare `/api/embed` output against `mockEmbeddingsForTexts`.

### Request pipeline (every data route)

Order matters — each layer rejects early: `await guardRequest` (origin + per-IP limit) → `requireUser(req)` (`src/lib/auth.ts`) → `guardUser` / `guardDailyBudget` (per-user limits) → `parseBody(req, schema)` (`src/lib/api/http.ts`, schemas in `src/lib/api/schemas.ts`) → query via the **user-scoped** client, so Row Level Security applies.

- **Auth**: Supabase Auth. Browser uses cookie sessions (`@supabase/ssr`, refreshed by `src/middleware.ts`, which redirects signed-out page loads to `/login`); CLI tools send `Authorization: Bearer <access_token>`. `/api/*` is never redirected — routes return 401 JSON. Login page has email/password and "Continue as guest" (Supabase anonymous sign-ins).
- **Tenancy**: `documents.user_id` and `chunks.user_id` + RLS policies (migration 005). Never add a service-role client to request handling — it would bypass isolation.
- **Contracts**: all errors are `{ error: string, code: string, ... }` via `apiError()`; DB details are only returned outside production.
- **Abuse guard** (`src/lib/apiGuard.ts` + `src/lib/rateLimit.ts`): same-origin check (403 for browser requests from foreign origins; no-Origin curl passes), per-IP and per-user limits (`chat` 20/min, `retrieve` 30/min, `embed` 20/min, `ingest` 10/min, `documents` DELETE 20/min, delete-all 5/min) and daily budgets (`chat` 300, `ingest` 100). Backend is Upstash sliding window when configured, else per-instance in-memory (resets on cold start). Redis errors/slowness (>1.5 s) fall back to memory — never to "allow all". `guardRequest` is **async**: always `await` it (an un-awaited call returns a truthy Promise). The GET routes (`documents`, `documents/[id]/chunks`) are not rate-limited, only auth-gated.

### Caching (`src/lib/cache.ts`, needs Redis; no-op without it)

- Query embeddings (`embedCache.ts`, 7 d) and whole retrieve responses (`retrievalCache.ts`, 5 min, per user). Responses carry `X-Cache: HIT|MISS|BYPASS`; send `Cache-Control: no-cache` to bypass (the eval harness does).
- Invalidation = per-user generation counter in the key; ingest and both DELETE routes call `bumpUserCacheVersion(user.id)`. **Any new route that writes documents/chunks must bump it**, or users get stale search results.
- Never cache anything derived from `embedFallback.ts` mock vectors. No in-memory cache fallback by design (can't be invalidated across instances). Chat answers and document lists are intentionally not cached — see `docs/ARCHITECTURE.md` D3.4.

### Observability & evals

- `src/lib/observability.ts` — optional Langfuse tracing (no-op without keys). `/api/retrieve` traces embed + search spans; `/api/chat` traces generations, including streamed output via `tapTextStream`. Serverless requires `flushTraces()` before the response finishes.
- `scripts/eval.ts` (`npm run eval`) — LLM-as-judge harness (Groq Llama 3.3 judge) over the live API; signs in with `EVAL_EMAIL`/`EVAL_PASSWORD` and sends a Bearer token. Golden dataset in `evals/dataset.json` (gitignored; example checked in). Supports `--mode vector|hybrid|both` for A/B. Results written to `evals/results/`.
- `scripts/reembed.ts` (`npm run reembed`) — re-embeds every chunk row; `--dry` to count only. Gemini free tier is 100 embed requests/min; `embedOne` retries on 429.

### Database schema

Two tables, applied via numbered, idempotent migrations in `supabase/migrations/` (run in the Supabase SQL editor in order; 005 has a manual backfill step — see `docs/ARCHITECTURE.md` §7):

- `documents` — user_id (owner), name, raw_text, char_count, word_count, chunk_count, avg_chunk_size, pages, file_type, status (`queued|processing|indexed|failed`), chunks_done, error, content_hash (unique per user), uploaded_at
- `ingest_chunks` — staging for text awaiting embedding (document_id, chunk_index, content, attempts, claimed_at); rows are deleted as they become `chunks`
- `chunks` — user_id, document_id (FK + cascade delete), chunk_index, content, embedding `vector(768)`, word_count, metadata jsonb, `fts` (generated tsvector)

Indexes (006): HNSW on `embedding` (`vector_l2_ops`), GIN on `fts`, btree on `user_id`/`document_id`. RPCs (`match_chunks`, `hybrid_search_chunks`) are `SECURITY INVOKER` so RLS applies; `anon` has no access to tables or functions. Known limit: HNSW applies the tenant filter after the index scan, so mid-sized tenants can under-return at scale (see ARCHITECTURE §6).

### Scoring

L2 distances from pgvector are mapped to `(0, 1]` via `1 / (1 + distance)`. The `VectorStore` class (`src/lib/vectorStore.ts`) is an in-memory fallback that uses cosine similarity — it is not used in the production DB path but exists for offline testing.

### Chunking

`src/lib/chunker.ts` splits on sentence boundaries (regex), targeting 500 chars with ~100-char overlap using the last 2 sentences of the previous chunk.

### PDF support

pdf.js is loaded dynamically from CDN at runtime in the client component (no npm install). Extracted text is passed to the chunker before ingest.

### Supabase client split

| File | Use |
|---|---|
| `src/lib/auth.ts` | `requireUser(req)` — the gate for API routes; returns a user-scoped client (cookie or Bearer) |
| `src/lib/supabase/request.ts` | Per-request server client built from cookies (server-only) |
| `src/lib/supabase/client.ts` | Browser client (anon key, cookie session) — used by the login page |
| `src/lib/supabase/env.ts` | URL / anon-key resolution |

`scripts/reembed.ts` builds its own service-role client (admin job, intentionally bypasses RLS).

### `next.config.js` notes

`recharts` is listed in `transpilePackages` to fix a webpack crash with React 19. Polling watch is enabled (`poll: 1000`) for Docker/VM environments.
