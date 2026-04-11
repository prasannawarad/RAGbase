# 🚀 RAGBase

> A production-grade Retrieval-Augmented Generation (RAG) platform with hybrid search (BM25 + vector), streaming AI responses, and Supabase-backed persistence.

![RAGBase UI](./public/screenshot.png)

---

## 🧠 Overview

RAGBase is a full-stack document intelligence system that lets you:

* Upload documents
* Convert them into embeddings
* Store them in a vector database
* Retrieve relevant context using hybrid search
* Generate streaming AI responses with source attribution

This project is designed to reflect **real-world RAG system architecture**, not just a demo.

---

## 🎯 Why this project

Most RAG examples rely only on vector search and client-side logic.

RAGBase implements a **production-style pipeline**:

* Server-side embedding generation
* Hybrid retrieval (BM25 + vector + RRF)
* Streaming LLM responses with fallback handling
* Persistent vector storage (pgvector)
* Source attribution with chunk-level inspection

This makes it closer to systems like **Perplexity, Glean, or enterprise knowledge assistants**.

---

## ✨ Features

* 📄 Multi-format document upload (PDF, TXT, MD, CSV)
* ✂️ Sentence-aware chunking with overlap
* 🧠 Server-side embeddings (Gemini)
* 🔍 Hybrid search (BM25 + vector similarity + RRF)
* ⚡ Streaming AI responses (SSE)
* 📚 Source attribution (document + chunk-level)
* 🗂 Chunk inspector (browse + jump to source)
* 📊 Analytics dashboard
* 💾 Supabase persistence (Postgres + pgvector)
* 🛡 Fallback mechanisms for embeddings and chat

---

## 🧱 Tech Stack

| Layer    | Technology                                    |
| -------- | --------------------------------------------- |
| Frontend | Next.js 15 (App Router), React 19, TypeScript |
| Backend  | Next.js API routes                            |
| Database | Supabase (PostgreSQL + pgvector)              |
| AI       | Google Gemini (embeddings + streaming chat)   |
| Search   | BM25 + vector similarity + RRF                |
| Styling  | Tailwind CSS                                  |
| Charts   | Recharts                                      |

---

## 🧠 Architecture

### Ingest Pipeline

```text
Client → chunk → /api/ingest → embed (Gemini, server-side) → Supabase (documents + chunks)
```

* Documents are chunked on the client
* `/api/ingest` generates embeddings server-side (`lib/gemini`; deterministic fallback if embedding fails)
* Data is stored in Supabase with pgvector

---

### Retrieval Pipeline

```text
Client → /api/retrieve → embed query → BM25 + vector → RRF → results
```

* Query is embedded on the server
* BM25 + vector similarity are combined
* RRF merges rankings for better recall

---

### Chat Pipeline

```text
Client → /api/chat → Gemini (streaming) → UI
```

* Context is built from retrieved chunks
* Responses stream token-by-token
* Metadata is appended after completion

---

## 🧩 Key Design Decisions

* **Server-side embeddings**
  Ensures consistency, security, and centralized control

* **Hybrid search (BM25 + vector)**
  Improves retrieval quality over pure vector search

* **Reciprocal Rank Fusion (RRF)**
  Balances lexical and semantic ranking

* **Streaming responses**
  Reduces perceived latency and improves UX

* **Supabase service role (server-only)**
  Secure ingestion and RPC-based retrieval

---

## 📁 Project Structure

```text
src/
├── app/
│   ├── api/
│   │   ├── chat/
│   │   ├── embed/
│   │   ├── ingest/
│   │   ├── retrieve/
│   │   └── documents/
│   ├── layout.tsx
│   └── page.tsx
├── components/
│   └── RAGBase.tsx
└── lib/
    ├── gemini.ts
    ├── search.ts
    ├── chunker.ts
    ├── bm25.ts
    ├── supabase/
```

---

## ⚙️ Setup

### 1. Clone

```bash
git clone https://github.com/prasannawarad/RAGbase.git
cd RAGbase
```

---

### 2. Install dependencies

```bash
npm install
```

---

### 3. Environment variables

Create `.env.local` (see also `.env.example`):

```env
GEMINI_API_KEY=your_key_here

SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
```

Optional (browser client + RLS): `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY`.

---

### 4. Supabase setup

In the Supabase **SQL Editor**, run the full migration:

**[`supabase/migrations/001_ragbase_pgvector.sql`](./supabase/migrations/001_ragbase_pgvector.sql)**

It enables **`vector`**, creates **`documents`** and **`chunks`** (768-dim embeddings), and adds RPCs (`match_chunks`, `chunk_vector_distances`) used by retrieval. Do not use a minimal stub schema — the app expects this shape.

---

### 5. Run locally

```bash
npm run dev
```

Open: [http://localhost:3000](http://localhost:3000)

---

## 🚀 Deployment

* Recommended: **Vercel**
* Database: **Supabase**
* Add environment variables in deployment settings

```bash
npm run build
npm start
```

---

## 🧪 Dev Notes

* If dev server breaks:

  ```bash
  rm -rf .next
  ```

  Or: `npm run dev:clean`

* Supabase must have **pgvector enabled**
* Streaming falls back to JSON if SSE fails
* Keep service role keys server-only

---

## 🔮 Future Improvements

* Authentication (multi-user support)
* Reranking with LLM
* Background ingestion jobs
* Vector indexing optimization (IVFFLAT / HNSW)
* Observability & analytics

---

## 📄 Resume Summary

* Built a production-grade RAG system with hybrid retrieval (BM25 + vector), server-side embedding pipelines, and real-time streaming LLM responses using Next.js and Supabase.

---

## 👤 Author

**Prasanna Warad**

---

<p align="center">
  <sub>Next.js · Supabase · Gemini · Hybrid Search</sub>
</p>
