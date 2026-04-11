# 🚀 RAGBase

> A production-grade Retrieval-Augmented Generation (RAG) platform with hybrid search (BM25 + vector), streaming AI responses, and Supabase-backed persistence.

---

## 🔗 Live Demo

**Production:** [https://ragbase-gamma.vercel.app](https://ragbase-gamma.vercel.app)

---

## 🧠 Overview

RAGBase is a full-stack document intelligence system that lets you:

* Upload documents (PDF, TXT, MD, CSV)
* Convert them into vector embeddings
* Store them in a persistent vector database
* Retrieve relevant context using hybrid search
* Generate streaming AI responses with source attribution

This project is designed to reflect **real-world RAG system architecture**, not just a demo.

---

## 🎯 Why this project

Most RAG examples rely only on vector search and client-side logic.

RAGBase implements a **production-style pipeline**:

* Server-side embedding generation (Gemini)
* Hybrid retrieval (BM25 + vector + RRF)
* Streaming LLM responses via Groq with automatic Gemini fallback
* Persistent vector storage (pgvector)
* Source attribution with chunk-level inspection

This makes it closer to systems like **Perplexity, Glean, or enterprise knowledge assistants**.

---

## ✨ Features

* 📄 Multi-format document upload (PDF, TXT, MD, CSV)
* ✂️ Sentence-aware chunking with overlap
* 🧠 Server-side embeddings (Gemini `text-embedding-004`, 768-dim)
* 🔍 Hybrid search (BM25 + vector similarity + RRF)
* ⚡ Streaming AI responses via Groq (SSE) with Gemini fallback
* 📚 Source attribution (document + chunk-level)
* 🗂 Chunk inspector (browse + semantic search)
* 📊 Analytics dashboard (chunks per document, confidence distribution)
* 💾 Supabase persistence (Postgres + pgvector)
* 🛡 Multi-level fallback for both embeddings and chat generation

---

## 🧱 Tech Stack

| Layer | Technology |
| ----- | ---------- |
| Frontend | Next.js 15 (App Router), React 19, TypeScript |
| Backend | Next.js API routes (Node runtime) |
| Database | Supabase (PostgreSQL + pgvector) |
| Embeddings | Google Gemini (`text-embedding-004`) |
| LLM (primary) | Groq (`llama-3.3-70b-versatile`) |
| LLM (fallback) | Google Gemini (`gemini-2.0-flash`) |
| Search | BM25 + vector similarity + RRF |
| Styling | Tailwind CSS |
| Charts | Recharts |

---

## 🧠 Architecture

### Ingest Pipeline

```text
Client → chunk → /api/ingest → embed (Gemini, server-side) → Supabase (documents + chunks)
```

* Documents are chunked client-side (sentence-aware, configurable size and overlap)
* `/api/ingest` generates 768-dim embeddings via Gemini server-side
* Data is stored in Supabase with pgvector

### Retrieval Pipeline

```text
Client → /api/retrieve → embed query (Gemini) → BM25 + vector → RRF → results
```

* Query is embedded on the server
* BM25 keyword scores and vector similarity scores are combined via RRF
* Optional per-document filtering for chunk inspector

### Chat Pipeline

```text
/api/chat: Groq stream → Gemini stream → Groq text → Gemini text → fallback
```

* Context is built from retrieved chunks and injected into the prompt
* Responses stream token-by-token when streaming succeeds
* Automatic fallback chain if a provider is down or rate-limited

---

## 🧩 Key Design Decisions

* **Groq as primary LLM** — fast inference; failover to Gemini when needed
* **Gemini as LLM fallback** — automatic backup if Groq quota or availability fails
* **Gemini for embeddings** — `text-embedding-004` produces 768-dim vectors matching the pgvector schema
* **Server-side embeddings** — consistency, security, centralized control
* **Hybrid search (BM25 + vector)** — better recall than pure vector search
* **Reciprocal Rank Fusion (RRF)** — balances lexical and semantic ranking
* **Supabase service role (server-only)** — secure ingestion, RPC-based retrieval

---

## 📁 Project Structure

```text
src/
├── app/
│   ├── api/
│   │   ├── chat/
│   │   ├── ingest/
│   │   ├── retrieve/
│   │   ├── documents/
│   │   └── embed/
│   ├── layout.tsx
│   └── page.tsx
├── components/
│   └── RAGBase.tsx
└── lib/
    ├── gemini.ts       ← embeddings + Gemini text/stream fallback
    ├── groq.ts         ← primary LLM + streaming
    ├── search.ts       ← hybrid search (BM25 + vector + RRF)
    ├── chunker.ts
    ├── bm25.ts
    ├── tokenizer.ts
    ├── embedFallback.ts
    └── supabase/
        ├── client.ts
        ├── server.ts
        └── env.ts
supabase/
└── migrations/
    ├── 001_ragbase_pgvector.sql
    ├── 002_fix_documents_schema.sql
    ├── 003_documents_full_insert_contract.sql
    └── 004_grants_postgrest_reload.sql
```

---

## ⚙️ Setup

### 1. Clone

```bash
git clone https://github.com/prasannawarad/RAGbase.git
cd RAGbase
```

### 2. Install dependencies

```bash
npm install
```

### 3. Environment variables

Create `.env.local` with all four keys:

```env
GEMINI_API_KEY=your_gemini_key
GROQ_API_KEY=your_groq_key
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
```

Get your keys:

* **Gemini:** [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
* **Groq:** [console.groq.com/keys](https://console.groq.com/keys)
* **Supabase:** Project Settings → API

### 4. Supabase setup

In the Supabase **SQL Editor**, run the migrations **in order**:

1. `supabase/migrations/001_ragbase_pgvector.sql` — core schema, pgvector, RPCs (`match_chunks`, `chunk_vector_distances`)
2. `supabase/migrations/002_fix_documents_schema.sql`
3. `supabase/migrations/003_documents_full_insert_contract.sql`
4. `supabase/migrations/004_grants_postgrest_reload.sql` — run last

Migration 001 enables `vector`, creates `documents` and `chunks` (768-dim), and adds retrieval RPCs. Run all four in sequence.

### 5. Run locally

```bash
npm run dev
```

Open: [http://localhost:3000](http://localhost:3000)

---

## 🚀 Deployment (Vercel)

1. Push to GitHub
2. Import the repo at [vercel.com](https://vercel.com)
3. Add the same four environment variables in project settings
4. Deploy

```bash
npm run build
npm start
```

**Live:** [https://ragbase-gamma.vercel.app](https://ragbase-gamma.vercel.app)

---

## 🧪 Dev Notes

* If the dev server misbehaves, clear the Next.js cache:

  ```bash
  rm -rf .next
  # or
  npm run dev:clean
  ```

* Supabase must have the **pgvector** extension enabled (migration 001 handles this)
* Chat streaming falls back to JSON if streaming fails
* Never put `SUPABASE_SERVICE_ROLE_KEY` in `NEXT_PUBLIC_*` variables

---

## 🔮 Future Improvements

* Authentication (multi-user support)
* LLM reranking for retrieval quality
* Background ingestion jobs (large document support)
* Vector index optimization (IVFFLAT / HNSW)
* Observability and query analytics

---

## 📄 Resume Summary

Built a production-grade RAG system with hybrid retrieval (BM25 + vector + RRF), server-side Gemini embedding pipelines, and real-time streaming LLM responses using Groq and Next.js, backed by Supabase pgvector.

---

## 👤 Author

**Prasanna Warad**

---

<p align="center">
  <sub>Next.js · Supabase · Groq · Gemini · Hybrid Search</sub>
</p>
