# RAGBase Evals

LLM-as-judge evaluation harness for the retrieve → generate pipeline. Run it
before and after any retrieval change (reranking, query rewriting, chunking
tweaks) to know whether the change actually helped.

## Setup

1. Start the app: `npm run dev` (the harness calls the live API routes).
2. Ingest your documents through the UI.
3. Copy `dataset.example.json` to `dataset.json` and write 10–20 real
   question/answer pairs about those documents. Include at least one
   unanswerable question to test refusal behavior.
4. `GROQ_API_KEY` must be set in `.env.local` (the judge runs on Groq).

## Run

```bash
npm run eval                 # vector search only
npm run eval -- --mode hybrid
npm run eval -- --mode both  # A/B: vector vs hybrid side by side
npm run eval -- --configs vector,vector+rerank,vector+rewrite,hybrid+rerank+rewrite
```

A config is a base mode (`vector` | `hybrid`) plus optional `+rerank` and/or
`+rewrite` stages — these map to the corresponding `/api/retrieve` flags.

Options: `--base http://localhost:3000`, `--dataset evals/dataset.json`,
`--topk 5`.

## Metrics (0–1, judged by Llama 3.3 70B)

- **faithfulness** — is every claim in the answer supported by the retrieved context?
- **answer_relevancy** — does the answer actually address the question?
- **context_relevance** — how much of the retrieved context was useful for the question?
- **correctness** — agreement with `expectedAnswer` (skipped when not provided).

Results are printed as a table and written to `evals/results/*.json` so runs
can be diffed over time. If Langfuse keys are configured, each eval item is
also logged as a trace with scores attached.

## Free-tier quota budget (matters a lot)

- **Groq** `llama-3.3-70b`: 100k tokens/day. One ~28-run eval (chat + judge)
  burns most of it. The judge falls back to Gemini automatically.
- **Gemini** `gemini-2.5-flash-lite`: ~20 requests/min. The harness paces 15s
  between items and honors "retry in Ns" hints, but a 6-config × 14-item run
  will still crawl. Prefer 2–3 configs per run and compare across result files.
- **Gemini embeddings**: 100 requests/min (handled by retry in `embedOne`).

Rule of thumb: ~5k LLM tokens per eval item. Plan runs so
`items × configs × 5k` stays inside the daily budget.
