/**
 * RAGBase eval harness — LLM-as-judge over the live API routes.
 *
 * Usage (app must be running, docs ingested):
 *   npm run eval                       # vector mode
 *   npm run eval -- --mode hybrid
 *   npm run eval -- --mode both       # A/B vector vs hybrid
 *   npm run eval -- --configs vector,vector+rerank,vector+rewrite,hybrid+rerank+rewrite
 *   npm run eval -- --base http://localhost:3000 --dataset evals/dataset.json --topk 5
 *
 * Config syntax: base mode ("vector" | "hybrid") plus optional "+rerank" /
 * "+rewrite" stages, matching the /api/retrieve flags.
 *
 * See evals/README.md for the full workflow.
 */

import fs from "node:fs";
import path from "node:path";
import Groq from "groq-sdk";
import { generateText as generateGeminiText } from "../src/lib/gemini";

// ---------- env (.env.local) ----------

function loadEnvLocal() {
  const p = path.join(process.cwd(), ".env.local");
  if (!fs.existsSync(p)) return;
  for (const line of fs.readFileSync(p, "utf8").split("\n")) {
    const m = line.match(/^\s*([A-Z0-9_]+)\s*=\s*(.*)\s*$/);
    if (m && !(m[1] in process.env)) {
      process.env[m[1]] = m[2].replace(/^["']|["']$/g, "");
    }
  }
}
loadEnvLocal();

// ---------- args ----------

function arg(name: string, fallback: string): string {
  const i = process.argv.indexOf(`--${name}`);
  return i !== -1 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}

const BASE = arg("base", "http://localhost:3000");
const DATASET = arg("dataset", "evals/dataset.json");
const TOP_K = Number(arg("topk", "5"));
const JUDGE_MODEL = process.env.EVAL_JUDGE_MODEL?.trim() || "llama-3.3-70b-versatile";

type EvalConfig = { label: string; useHybrid: boolean; rerank: boolean; rewrite: boolean };

function parseConfigs(): EvalConfig[] {
  const mode = arg("mode", "");
  const configsArg = arg("configs", "");
  const labels = configsArg
    ? configsArg.split(",").map((s) => s.trim()).filter(Boolean)
    : mode === "both"
      ? ["vector", "hybrid"]
      : [mode || "vector"];
  return labels.map((label) => {
    const parts = label.split("+");
    if (parts[0] !== "vector" && parts[0] !== "hybrid") {
      throw new Error(`Config "${label}" must start with "vector" or "hybrid"`);
    }
    return {
      label,
      useHybrid: parts[0] === "hybrid",
      rerank: parts.includes("rerank"),
      rewrite: parts.includes("rewrite"),
    };
  });
}

// ---------- types ----------

type DatasetItem = {
  id: string;
  question: string;
  expectedAnswer?: string;
  filterDocumentId?: string;
};

type RetrievalResult = {
  documentName: string;
  chunkIndex: number;
  content: string;
  score?: number;
  vectorScore?: number;
  bm25Score?: number;
};

type Judgment = {
  faithfulness: number;
  answer_relevancy: number;
  context_relevance: number;
  correctness: number | null;
  reasoning: string;
};

type EvalRow = {
  id: string;
  mode: string;
  question: string;
  answer: string;
  retrievedCount: number;
  latencyMs: { retrieve: number; chat: number };
  judgment: Judgment;
};

// ---------- pipeline calls ----------

async function retrieve(
  item: DatasetItem,
  config: EvalConfig
): Promise<{ results: RetrievalResult[]; rewrittenQuery?: string; ms: number }> {
  const t0 = Date.now();
  const res = await fetch(`${BASE}/api/retrieve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query: item.question,
      topK: TOP_K,
      useHybrid: config.useHybrid,
      rerank: config.rerank,
      rewrite: config.rewrite,
      filterDocumentId: item.filterDocumentId,
    }),
  });
  if (!res.ok) throw new Error(`retrieve ${res.status}: ${await res.text()}`);
  const json = (await res.json()) as { results: RetrievalResult[]; rewrittenQuery?: string };
  return { results: json.results, rewrittenQuery: json.rewrittenQuery, ms: Date.now() - t0 };
}

/** Mirrors the prompt built by src/components/RAGBase.tsx so evals measure the real pipeline. */
function buildRagPrompt(question: string, results: RetrievalResult[], useHybrid: boolean): string {
  const contextBlock = results
    .map(
      (c, i) =>
        `[Source ${i + 1}] (from: ${c.documentName}, chunk ${c.chunkIndex}; ` +
        (useHybrid
          ? `vector: ${((c.vectorScore ?? 0) * 100).toFixed(1)}%, bm25: ${c.bm25Score?.toFixed(2) || "N/A"}`
          : `relevance: ${((c.score ?? 0) * 100).toFixed(1)}%`) +
        `)\n${c.content}`
    )
    .join("\n\n");

  return `You are RAGBase, a document intelligence assistant. Answer using ONLY the retrieved context.

Retrieved context:
---
${contextBlock}
---

Question: "${question}"

Rules:
1. Use ONLY information from the context above
2. Cite sources using [Source N] inline throughout your answer
3. If context is insufficient, say so clearly
4. Be specific and thorough
5. If sources conflict, mention the discrepancy

Write your full answer first (natural language, with [Source N] citations inline).
Then end your response with EXACTLY this block (replace values):

<metadata>{"confidence": "high|medium|low", "sources_used": [1, 2], "key_entities": ["entity1", "entity2"], "follow_up_questions": ["q1", "q2", "q3"]}</metadata>`;
}

async function chat(prompt: string): Promise<{ answer: string; ms: number }> {
  const t0 = Date.now();
  const res = await fetch(`${BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt }),
  });
  if (!res.ok) throw new Error(`chat ${res.status}: ${await res.text()}`);
  let text: string;
  if (res.headers.get("x-chat-stream") === "1") {
    text = await res.text();
  } else {
    const json = (await res.json()) as { text?: string };
    text = json.text ?? "";
  }
  const answer = text.replace(/<metadata>[\s\S]*?<\/metadata>/, "").trim();
  return { answer, ms: Date.now() - t0 };
}

// ---------- judge ----------

const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });

function suggestedRetryMs(err: unknown): number {
  const msg = err instanceof Error ? err.message : String(err);
  const m = msg.match(/retry in ([\d.]+)\s*s/i);
  return m ? Math.ceil(parseFloat(m[1]) * 1000) + 2_000 : 30_000;
}

/** Groq judge with Gemini fallback — Groq's free tier is 100k tokens/day, which a full A/B run can exhaust. */
async function judgeCompletion(judgePrompt: string): Promise<string> {
  // Some models reject response_format; some windows are rate-limited. Try
  // strict JSON mode, then plain mode, then Gemini, honoring retry hints.
  for (let attempt = 0; ; attempt++) {
    try {
      const completion = await groq.chat.completions.create({
        model: JUDGE_MODEL,
        messages: [{ role: "user", content: judgePrompt }],
        temperature: 0,
        ...(attempt === 0 ? { response_format: { type: "json_object" as const } } : {}),
      });
      return completion.choices[0]?.message?.content ?? "{}";
    } catch (e) {
      if (attempt >= 3) break;
      if (attempt > 0) await new Promise((r) => setTimeout(r, suggestedRetryMs(e)));
    }
  }
  for (let attempt = 0; ; attempt++) {
    try {
      return await generateGeminiText(judgePrompt, { temperature: 0, maxOutputTokens: 500 });
    } catch (e) {
      if (attempt >= 2) throw e;
      await new Promise((r) => setTimeout(r, suggestedRetryMs(e)));
    }
  }
}

async function judge(item: DatasetItem, context: string, answer: string): Promise<Judgment> {
  const judgePrompt = `You are a strict RAG evaluation judge. Score the assistant's answer.

QUESTION:
${item.question}

RETRIEVED CONTEXT:
${context}

ASSISTANT ANSWER:
${answer}
${item.expectedAnswer ? `\nREFERENCE ANSWER (ground truth):\n${item.expectedAnswer}\n` : ""}
Score each metric from 0.0 to 1.0:
- faithfulness: every claim in the answer is supported by the retrieved context (1.0 = fully grounded, 0.0 = hallucinated). If the answer correctly states the context is insufficient, score 1.0.
- answer_relevancy: the answer directly addresses the question.
- context_relevance: fraction of the retrieved context that is actually useful for this question.
- correctness: agreement with the reference answer.${item.expectedAnswer ? "" : " No reference provided — return null."}

Respond ONLY with JSON:
{"faithfulness": 0.0, "answer_relevancy": 0.0, "context_relevance": 0.0, "correctness": ${item.expectedAnswer ? "0.0" : "null"}, "reasoning": "one or two sentences"}`;

  const raw = await judgeCompletion(judgePrompt);
  const o = JSON.parse(raw.replace(/```json|```/g, "").trim()) as Partial<Judgment>;
  const clamp = (v: unknown) => (typeof v === "number" ? Math.min(1, Math.max(0, v)) : 0);
  return {
    faithfulness: clamp(o.faithfulness),
    answer_relevancy: clamp(o.answer_relevancy),
    context_relevance: clamp(o.context_relevance),
    correctness: item.expectedAnswer != null && typeof o.correctness === "number" ? clamp(o.correctness) : null,
    reasoning: typeof o.reasoning === "string" ? o.reasoning : "",
  };
}

// ---------- optional Langfuse logging ----------

async function logToLangfuse(rows: EvalRow[]) {
  if (!process.env.LANGFUSE_SECRET_KEY || !process.env.LANGFUSE_PUBLIC_KEY) return;
  const { Langfuse } = await import("langfuse");
  const lf = new Langfuse({
    secretKey: process.env.LANGFUSE_SECRET_KEY,
    publicKey: process.env.LANGFUSE_PUBLIC_KEY,
    baseUrl: process.env.LANGFUSE_BASEURL || "https://cloud.langfuse.com",
  });
  for (const row of rows) {
    const trace = lf.trace({
      name: "eval-run",
      input: { question: row.question },
      output: { answer: row.answer },
      tags: ["eval", row.mode],
      metadata: { itemId: row.id, latencyMs: row.latencyMs },
    });
    trace.score({ name: "faithfulness", value: row.judgment.faithfulness });
    trace.score({ name: "answer_relevancy", value: row.judgment.answer_relevancy });
    trace.score({ name: "context_relevance", value: row.judgment.context_relevance });
    if (row.judgment.correctness != null) {
      trace.score({ name: "correctness", value: row.judgment.correctness });
    }
  }
  await lf.flushAsync();
  console.log("Scores logged to Langfuse.");
}

// ---------- run ----------

function avg(nums: number[]): number {
  return nums.length ? nums.reduce((a, b) => a + b, 0) / nums.length : 0;
}

function fmt(n: number): string {
  return n.toFixed(2);
}

async function runMode(items: DatasetItem[], config: EvalConfig): Promise<EvalRow[]> {
  const mode = config.label;
  const rows: EvalRow[] = [];
  for (const item of items) {
    process.stdout.write(`  [${mode}] ${item.id}: retrieving... `);
    try {
      const { results, rewrittenQuery, ms: retrieveMs } = await retrieve(item, config);
      if (rewrittenQuery) process.stdout.write(`(rewrote: "${rewrittenQuery.slice(0, 60)}") `);
      const prompt = buildRagPrompt(item.question, results, config.useHybrid);
      process.stdout.write("generating... ");
      let { answer, ms: chatMs } = await chat(prompt);
      if (answer.includes("could not be generated")) {
        // Every provider was rate-limited; wait out the window and retry once.
        process.stdout.write("(providers exhausted, retrying in 45s) ");
        await new Promise((r) => setTimeout(r, 45_000));
        ({ answer, ms: chatMs } = await chat(prompt));
      }
      process.stdout.write("judging... ");
      const context = results.map((r, i) => `[Source ${i + 1}]\n${r.content}`).join("\n\n");
      const judgment = await judge(item, context, answer);
      rows.push({
        id: item.id,
        mode,
        question: item.question,
        answer,
        retrievedCount: results.length,
        latencyMs: { retrieve: retrieveMs, chat: chatMs },
        judgment,
      });
      console.log(
        `faith=${fmt(judgment.faithfulness)} rel=${fmt(judgment.answer_relevancy)} ctx=${fmt(judgment.context_relevance)}` +
          (judgment.correctness != null ? ` corr=${fmt(judgment.correctness)}` : "")
      );
    } catch (e) {
      console.log(`FAILED: ${e instanceof Error ? e.message : e}`);
    }
    // Pace for free-tier RPM limits; raise this if providers start 429ing.
    await new Promise((r) => setTimeout(r, 5_000));
  }
  return rows;
}

function printSummary(rows: EvalRow[]) {
  const byMode = new Map<string, EvalRow[]>();
  for (const r of rows) {
    byMode.set(r.mode, [...(byMode.get(r.mode) ?? []), r]);
  }
  console.log("\n=== Summary ===");
  console.log(
    "config".padEnd(24) +
      "n".padEnd(4) +
      "faithful".padEnd(10) +
      "relevancy".padEnd(11) +
      "ctx-rel".padEnd(9) +
      "correct".padEnd(9) +
      "p50 retrieve".padEnd(14) +
      "p50 chat"
  );
  for (const [mode, rs] of byMode) {
    const correct = rs.map((r) => r.judgment.correctness).filter((c): c is number => c != null);
    const p50 = (nums: number[]) => nums.sort((a, b) => a - b)[Math.floor(nums.length / 2)] ?? 0;
    console.log(
      mode.padEnd(24) +
        String(rs.length).padEnd(4) +
        fmt(avg(rs.map((r) => r.judgment.faithfulness))).padEnd(10) +
        fmt(avg(rs.map((r) => r.judgment.answer_relevancy))).padEnd(11) +
        fmt(avg(rs.map((r) => r.judgment.context_relevance))).padEnd(9) +
        (correct.length ? fmt(avg(correct)) : "n/a").padEnd(9) +
        `${p50(rs.map((r) => r.latencyMs.retrieve))}ms`.padEnd(14) +
        `${p50(rs.map((r) => r.latencyMs.chat))}ms`
    );
  }
}

async function main() {
  if (!process.env.GROQ_API_KEY) {
    console.error("GROQ_API_KEY is required (judge model). Set it in .env.local.");
    process.exit(1);
  }

  let datasetPath = DATASET;
  if (!fs.existsSync(datasetPath)) {
    const example = "evals/dataset.example.json";
    console.warn(`${datasetPath} not found — falling back to ${example}. Copy it to ${DATASET} and add real questions.`);
    datasetPath = example;
  }
  const dataset = JSON.parse(fs.readFileSync(datasetPath, "utf8")) as { items: DatasetItem[] };
  const configs = parseConfigs();
  console.log(
    `Evaluating ${dataset.items.length} items against ${BASE} (configs: ${configs.map((c) => c.label).join(", ")}, topK: ${TOP_K})\n`
  );

  const rows: EvalRow[] = [];
  for (const config of configs) {
    rows.push(...(await runMode(dataset.items, config)));
  }

  if (!rows.length) {
    console.error("\nNo items completed. Is the dev server running and are documents ingested?");
    process.exit(1);
  }

  printSummary(rows);

  const outDir = path.join("evals", "results");
  fs.mkdirSync(outDir, { recursive: true });
  const runLabel = configs.map((c) => c.label).join("_");
  const outPath = path.join(outDir, `${new Date().toISOString().replace(/[:.]/g, "-")}-${runLabel}.json`);
  fs.writeFileSync(outPath, JSON.stringify({ base: BASE, configs: configs.map((c) => c.label), topK: TOP_K, rows }, null, 2));
  console.log(`\nFull results written to ${outPath}`);

  await logToLangfuse(rows);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
