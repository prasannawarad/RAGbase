# RAGBase

> A production-grade RAG system with hybrid search (BM25 + vector), streaming AI responses, and Supabase-backed persistence.

Upload documents, embed them into Postgres + pgvector, retrieve with hybrid search, and chat over your knowledge base with streaming answers and source attribution.

---

## Demo

RAGBase is a full-stack reference implementation: **chunk → embed → persist → retrieve → generate**. The UI covers upload, document management, hybrid Q&A with citations, a chunk inspector, and analytics — suitable as a portfolio piece or starting point for a document intelligence product.

---

## Why this project

Most RAG demos rely purely on vector search and thin client-side glue.

RAGBase implements a **production-style architecture**:

- Server-owned embedding pipelines
- Hybrid retrieval (BM25 + vector + RRF)
- Streaming LLM responses with fallback handling
- Persistent vector storage with pgvector
- Source attribution and inspectable chunks

That puts it closer to real-world systems like Perplexity, Glean, or internal enterprise search — not a notebook demo.

---

## Features

- **Multi-format upload** — PDF (client-side extraction), TXT, MD, CSV; batch-friendly
- **Sentence-aware chunking** — configurable size and overlap on the client before ingest
- **Server-side pipeline** — `POST /api/ingest` embeds (Gemini) and persists to Supabase
- **Hybrid retrieval** — BM25 + vector similarity fused with **Reciprocal Rank Fusion (RRF)**
- **Streaming chat** — SSE-style streaming from `/api/chat` into the UI
- **Source attribution** — answers cite document and chunk; jump-to-chunk from chat
- **Chunk inspector** — browse chunks per document; hybrid search within the inspector
- **Analytics** — query log, confidence, and document-level summaries (JSON mode)
- **Durable storage** — documents and vectors in Supabase; cascade delete for cleanup
- **Resilient API layer** — embedding/chat fallbacks and defensive client handling for failed routes

---

## Tech Stack

| Layer | Technology |
|--------|------------|
| Framework | **Next.js 15** (App Router), **React 19**, **TypeScript** |
| Database | **Supabase** (PostgreSQL + **pgvector**) |
| Embeddings & LLM | **Google Gemini** (e.g. text-embedding-004, streaming generation) |
| Search | Custom **BM25** + pgvector distance + **RRF** |
| Styling | **Tailwind CSS** (project configured); main app UI uses component-scoped styling |
| Charts | **Recharts** |
| Client resilience | **Error boundaries**, guarded `fetch` + JSON parsing |

---

## Architecture

### Ingest

```text
┌────────┐   chunk (client)    ┌─────────────┐   embed + persist   ┌──────────┐
│ Client │ ──────────────────► │ /api/ingest │ ───────────────────► │ Supabase │
└────────┘                    └─────────────┘                     │ documents│
                                                                  │ + chunks │
                                                                  └──────────┘
```

1. User selects files → text extracted (PDF via pdf.js CDN).
2. Text is split with the shared chunker (`lib/chunker`).
3. Chunks are sent to **`/api/ingest`**, which **embeds internally using Gemini** (`lib/gemini`), with a deterministic fallback if embedding fails, then inserts rows into **`documents`** and **`chunks`** (768-dim vectors). A separate **`/api/embed`** route exists for standalone embedding calls.

### Query (retrieval)

```text
┌────────┐   POST JSON          ┌──────────────┐   embed query    ┌─────────────┐
│ Client │ ───────────────────► │ /api/retrieve│ ───────────────► │ Hybrid rank │
└────────┘                      └──────────────┘                  │ BM25 + vec  │
        ◄──────────────────────  RetrievalResult[]  ◄─────────────│ + RRF       │
                                                                   └─────────────┘
```

1. Query string → query embedding from the server.
2. **BM25** over chunk text + **vector** ordering from Supabase RPCs → **RRF** to merge ranks.
3. Optional filter by `document_id` for scoped search.

### Chat

```text
┌────────┐   POST /api/chat     ┌─────────────┐   stream          ┌─────┐
│ Client │ ──────────────────► │ Gemini SSE  │ ─────────────────► │ UI  │
└────────┘                     └─────────────┘                    └─────┘
```

1. Client builds a prompt from retrieved chunks + optional conversation tail.
2. **`/api/chat`** streams tokens; metadata block parsed for confidence and follow-ups.
3. UI updates incrementally; sources link back to the chunk inspector.

---

## Key Design Decisions

- **Server-side embeddings** — consistent retrieval vectors, API keys stay on the server, and clients never touch provider credentials
- **Hybrid search (BM25 + vector)** — better recall than pure dense retrieval on keyword-heavy queries
- **RRF fusion** — balances lexical and semantic rankings without brittle score calibration
- **Streaming responses** — lower perceived latency; metadata appended after the visible answer
- **Service-role Supabase access from API routes** — ingestion and RPCs run with a controlled server identity (complement with RLS if you expose the browser client)

---

## Project Structure

```text
src/
├── app/
│   ├── api/
│   │   ├── chat/          # Streaming LLM + JSON fallbacks
│   │   ├── embed/         # Gemini embeddings (768-dim)
│   │   ├── ingest/        # Chunk persistence + document row
│   │   ├── retrieve/      # Hybrid search orchestration
│   │   └── documents/     # List / delete + per-doc chunks
│   ├── layout.tsx
│   ├── page.tsx
│   └── globals.css
├── components/
│   ├── RAGBase.tsx        # Main application shell + views
│   └── RAGBaseErrorBoundary.tsx
└── lib/
    ├── bm25.ts            # Lexical scoring
    ├── chunker.ts         # Text splitting
    ├── gemini.ts          # Embeddings + generate + stream helpers
    ├── search.ts          # RRF + hybrid assembly
    ├── supabase/          # Server + optional browser client
    ├── embedFallback.ts
    └── …                  # tokenizer, vector types, retrieval DTOs

supabase/migrations/       # SQL for pgvector + RPCs
```

---

## Setup Instructions

1. **Clone and install**

   ```bash
   git clone <your-repo-url> ragbase
   cd ragbase
   npm install
   ```

2. **Create a Supabase project**  
   Note the project URL and keys from **Project Settings → API**.

3. **Run the database migration** (see [Supabase Setup](#supabase-setup)).

4. **Configure environment variables** — copy `.env.example` to `.env.local` and fill values (see below).

5. **Start the dev server**

   ```bash
   npm run dev
   ```

6. Open [http://localhost:3000](http://localhost:3000).

---

## Environment Variables

Create **`.env.local`** in the project root (never commit secrets):

```bash
# Server-only — used by /api/embed and /api/chat (do not prefix with NEXT_PUBLIC_)
GEMINI_API_KEY=

# Supabase — server / API routes only (service role for ingestion & RPC)
SUPABASE_URL=
SUPABASE_SERVICE_ROLE_KEY=

# Optional — browser-safe keys if you add direct Supabase client usage with RLS
NEXT_PUBLIC_SUPABASE_URL=
NEXT_PUBLIC_SUPABASE_ANON_KEY=
```

- **`GEMINI_API_KEY`** — required for embeddings and chat.
- **`SUPABASE_*`** — required for listing, ingesting, and searching documents/chunks.

---

## Supabase Setup

1. In the Supabase dashboard, open **SQL Editor**.
2. Run the migration in **`supabase/migrations/001_ragbase_pgvector.sql`** (or use Supabase CLI: `supabase db push` if you link the project).

This script:

- Enables the **`vector`** extension
- Creates **`documents`** and **`chunks`** (768-dimensional embeddings)
- Adds **`match_chunks`** and **`chunk_vector_distances`** RPCs for vector search
- Grants rights used by the **service role** from API routes

After migration, confirm tables appear under **Table Editor** and test a simple query in the SQL editor if needed.

---

## Running Locally

| Command | Description |
|---------|-------------|
| `npm run dev` | Next.js dev server (Turbopack optional: `npm run dev:turbo`) |
| `npm run build` | Production build |
| `npm run start` | Serve production build |
| `npm run lint` | ESLint |

For a clean dev cache: `npm run dev:clean`.

---

## Deployment

- **Recommended:** [Vercel](https://vercel.com) for Next.js (App Router + API routes on the same deployment)
- **Database:** Supabase (hosted Postgres + pgvector)
- **Secrets:** configure `GEMINI_API_KEY`, `SUPABASE_URL`, and `SUPABASE_SERVICE_ROLE_KEY` in the host’s environment (e.g. Vercel Project Settings → Environment Variables)

Production build locally:

```bash
npm run build
npm start
```

---

## Dev Notes

- Next.js dev server may need a cache reset in rare cases: `rm -rf .next` (this repo includes `npm run dev:clean`)
- Supabase must have the **pgvector** extension enabled (see migration)
- Chat may fall back to non-streaming JSON if streaming/SSE cannot be established
- Keep **service role** keys server-only; never expose them with `NEXT_PUBLIC_`

---

## Future Improvements

- **IndexedDB** (or similar) for client-side embedding or chunk cache and faster re-opens
- **IVFFLAT / HNSW** index tuning on `chunks.embedding` at scale (commented hints exist in migration)
- **Auth** (e.g. Clerk / Supabase Auth) and per-tenant namespaces
- **Evaluation suite** — retrieval hit rate, answer faithfulness
- **Observability** — structured logs, tracing, and cost dashboards for Gemini usage

---

## Author

Built by **Prasanna Warad** as a production-style RAG system.

---

<p align="center">
  <sub>Next.js · Supabase · Gemini · Hybrid search</sub>
</p>
