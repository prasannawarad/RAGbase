"use client";

import { useState, useRef, useEffect, useMemo, useCallback } from "react";
import type { ChangeEvent, CSSProperties, DragEvent } from "react";

// ══════════════════════════════════════════════════════════════
// RAGBase v2 — Document Intelligence & RAG Platform
// ── New in v2 ──────────────────────────────────────────────
//   ✦ PDF support via pdf.js (dynamic CDN load)
//   ✦ Hybrid search: BM25 + Vector via Reciprocal Rank Fusion
//   ✦ Supabase + pgvector persistence
//   ✦ Streaming generation (answer text streams live)
// ══════════════════════════════════════════════════════════════

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line,
  CartesianGrid,
} from "recharts";

import { chunkText } from "@/lib/chunker";
import type { RetrievalResult } from "@/lib/retrievalResult";
import type { VectorDoc } from "@/lib/vectorStore";

declare global {
  interface Window {
    pdfjsLib: {
      GlobalWorkerOptions: { workerSrc: string };
      getDocument: (opts: { data: ArrayBuffer }) => { promise: Promise<PDFDocumentProxy> };
    };
  }
}

interface PDFDocumentProxy {
  numPages: number;
  getPage: (n: number) => Promise<PDFPageProxy>;
}

interface PDFPageProxy {
  getTextContent: () => Promise<{ items: { str: string }[] }>;
}

type StoredDocument = {
  id: string;
  name: string;
  text: string;
  chunks: number;
  charCount: number;
  wordCount: number;
  avgChunkSize: number;
  pages?: number | null;
  fileType?: string;
  status?: string;
  uploadedAt?: string;
};

type ProcessLogEntry = { time: string; msg: string };

type ChatMessage =
  | {
      role: "user";
      text: string;
      timestamp: string;
    }
  | {
      id: string;
      role: "assistant";
      streaming: true;
      streamText: string;
      searchMode: "hybrid" | "vector";
      timestamp: string;
    }
  | {
      id: string;
      role: "assistant";
      streaming: false;
      text: string;
      confidence: string;
      sourcesUsed: number[];
      keyEntities: string[];
      followUps: string[];
      retrievedChunks: RetrievalResult[];
      searchMode: "hybrid" | "vector";
      timestamp: string;
      streamText?: string;
    };

type QueryLogEntry = {
  question: string;
  timestamp: string;
  confidence: string;
  topScore: number;
  sourcesUsed: number;
  chunksSearched: number;
  searchMode: "hybrid" | "vector";
};

type DocSummary = {
  summary: string;
  topics?: string[];
  key_entities?: string[];
  sentiment?: string;
  complexity?: string;
};

// ─── PDF Extraction ───────────────────────────────────────────
// Dynamically loads pdf.js from CDN so we don't need a build step.
// Caches the loaded library in window.pdfjsLib after first load.
async function loadPDFJS(): Promise<Window["pdfjsLib"]> {
  if (window.pdfjsLib) return window.pdfjsLib;
  return new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js";
    script.onload = () => {
      window.pdfjsLib.GlobalWorkerOptions.workerSrc =
        "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";
      resolve(window.pdfjsLib as Window["pdfjsLib"]);
    };
    script.onerror = () => reject(new Error("Failed to load PDF.js from CDN"));
    document.head.appendChild(script);
  });
}

async function extractPDFText(file: File) {
  const pdfjsLib = await loadPDFJS();
  const arrayBuffer = await file.arrayBuffer();
  const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
  const pageTexts: string[] = [];
  for (let i = 1; i <= pdf.numPages; i++) {
    const page = await pdf.getPage(i);
    const content = await page.getTextContent();
    // Join items; insert space between text runs, newline between pages
    pageTexts.push(
      content.items.map((item) => item.str).join(" ").replace(/\s{2,}/g, " ")
    );
  }
  return { text: pageTexts.join("\n\n"), pages: pdf.numPages };
}

// ─── Backend API (no API keys in the browser) ─────────────────
type ChatApiResponse =
  | { mode: "text"; text: string }
  | ({ mode: "summary" } & DocSummary);

/** Single read of JSON body; avoids crashes on HTML/plain error pages. */
async function readJsonOnce<T>(res: Response): Promise<{ parsed: T | null; parseError: boolean }> {
  try {
    const parsed = (await res.json()) as T;
    return { parsed, parseError: false };
  } catch {
    return { parsed: null, parseError: true };
  }
}

type FetchRetrieveResult = { results: RetrievalResult[]; error: string | null };

/** JSON responses only (e.g. document summary). RAG chat uses streaming text from /api/chat. */
async function fetchChatJson(prompt: string): Promise<ChatApiResponse> {
  let res: Response;
  try {
    res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt }),
    });
  } catch (e) {
    console.error("[API /api/chat] (JSON) network:", e);
    throw new Error(e instanceof Error ? e.message : "Network error");
  }
  const { parsed: data, parseError } = await readJsonOnce<{ error?: string } & ChatApiResponse>(res);
  if (parseError || data === null) {
    console.error("[API /api/chat] (JSON) invalid JSON", res.status);
    throw new Error(`Chat request failed (${res.status})`);
  }
  if (!res.ok) {
    const msg = data.error || `Chat API error (${res.status})`;
    console.error("[API /api/chat] (JSON)", res.status, msg);
    throw new Error(msg);
  }
  return data as ChatApiResponse;
}

/**
 * RAG chat: reads Gemini stream (text/plain) or JSON fallback ({ mode: "text", text }).
 * Calls onDelta with visible text (metadata block hidden while streaming).
 */
async function consumeRagChatResponse(
  prompt: string,
  onDelta: (displayText: string) => void
): Promise<string> {
  let res: Response;
  try {
    res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt }),
    });
  } catch (e) {
    console.error("[API /api/chat] (stream) network:", e);
    throw new Error(e instanceof Error ? e.message : "Network error");
  }

  const ct = res.headers.get("Content-Type") ?? "";
  const isTextStream =
    res.ok &&
    res.body &&
    (res.headers.get("X-Chat-Stream") === "1" || ct.includes("text/plain"));

  if (!res.ok) {
    const { parsed: errBody } = await readJsonOnce<{ error?: string }>(res);
    const msg = errBody?.error || `Chat request failed (${res.status})`;
    console.error("[API /api/chat]", res.status, msg);
    throw new Error(msg);
  }

  if (isTextStream) {
    const reader = res.body!.getReader();
    const decoder = new TextDecoder();
    let raw = "";
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        raw += decoder.decode(value, { stream: true });
        const displayText = raw.replace(/<metadata>[\s\S]*$/, "").trim();
        onDelta(displayText);
      }
    } catch (e) {
      console.error("[API /api/chat] stream read failed:", e);
      throw new Error(e instanceof Error ? e.message : "Chat stream interrupted");
    } finally {
      reader.releaseLock();
    }
    let out = raw.trim();
    if (!out.includes("<metadata>")) {
      out += `\n\n<metadata>{"confidence":"medium","sources_used":[],"key_entities":[],"follow_up_questions":[]}</metadata>`;
    }
    return out;
  }

  const { parsed: data, parseError } = await readJsonOnce<{ error?: string; mode?: string; text?: string }>(res);
  if (parseError || data === null) {
    console.error("[API /api/chat] invalid JSON (non-stream)", res.status);
    throw new Error(`Invalid chat response (${res.status})`);
  }
  if (data.mode === "text" && typeof data.text === "string") {
    const full = data.text;
    onDelta(full.replace(/<metadata>[\s\S]*$/, "").trim());
    return full;
  }
  console.error("[API /api/chat] unexpected shape", data);
  throw new Error("Unexpected chat response");
}

async function fetchRetrieve(params: {
  query: string;
  topK: number;
  useHybrid: boolean;
  filterDocumentId?: string | null;
}): Promise<FetchRetrieveResult> {
  try {
    const res = await fetch("/api/retrieve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: params.query,
        topK: params.topK,
        useHybrid: params.useHybrid,
        filterDocumentId: params.filterDocumentId ?? null,
      }),
    });
    const { parsed: data, parseError } = await readJsonOnce<{ error?: string; results?: RetrievalResult[] }>(res);
    if (parseError || data === null) {
      const msg = res.ok ? "Invalid response from retrieve API" : `Retrieve failed (${res.status})`;
      console.error("[API /api/retrieve] invalid JSON", res.status, res.ok);
      return { results: [], error: msg };
    }
    if (!res.ok) {
      const msg = data.error || `Retrieve failed (${res.status})`;
      console.error("[API /api/retrieve]", res.status, msg);
      return { results: [], error: msg };
    }
    return { results: data.results ?? [], error: null };
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    console.error("[API /api/retrieve] network:", e);
    return { results: [], error: msg };
  }
}

// Parse streamed response into answer + metadata
type StreamMetadata = {
  confidence?: string;
  sources_used?: number[];
  key_entities?: string[];
  follow_up_questions?: string[];
};

function parseStreamedResponse(fullText: string) {
  const metaMatch = fullText.match(/<metadata>([\s\S]*?)<\/metadata>/);
  const answer = fullText.replace(/<metadata>[\s\S]*?<\/metadata>/, "").trim();
  let metadata: StreamMetadata = {};
  if (metaMatch) {
    try {
      metadata = JSON.parse(metaMatch[1].trim()) as StreamMetadata;
    } catch {
      /* ignore */
    }
  }
  return { answer, metadata };
}

// ─── Constants ───────────────────────────────────────────────
const VIEWS = {
  UPLOAD: "upload", DOCUMENTS: "documents", CHAT: "chat",
  CHUNKS: "chunks", ANALYTICS: "analytics", PIPELINE: "pipeline",
};

/** Blank-screen bisect: set each to `false` to re-enable (one at a time). */
const DEBUG_DISABLE_RECHARTS = true;
const DEBUG_DISABLE_PDF = true;

// ══════════════════════════════════════════════════════════════
export default function RAGBase() {
  console.log("[RAGBase] render start");

  // ─── Core State ─────────────────────────────────────────────
  const [view, setView] = useState<(typeof VIEWS)[keyof typeof VIEWS]>(VIEWS.UPLOAD);
  const [documents, setDocuments] = useState<StoredDocument[]>([]);
  const [totalChunks, setTotalChunks] = useState(0);
  const [chunksByDocId, setChunksByDocId] = useState<Record<string, VectorDoc[]>>({});

  // ─── Persistence State ──────────────────────────────────────
  const [dbLoading, setDbLoading] = useState(true); // loading from Supabase on mount
  const [docsLoading, setDocsLoading] = useState(true);
  const [dbSynced, setDbSynced] = useState(false); // documents exist in Supabase
  const [docsError, setDocsError] = useState<string | null>(null);

  // ─── Processing ─────────────────────────────────────────────
  const [processing, setProcessing] = useState(false);
  const [processStep, setProcessStep] = useState("");
  const [processProgress, setProcessProgress] = useState(0);
  const [processLog, setProcessLog] = useState<ProcessLogEntry[]>([]);

  // ─── Chat ───────────────────────────────────────────────────
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const [lastRetrieved, setLastRetrieved] = useState<RetrievalResult[]>([]);
  const [showRetrieval, setShowRetrieval] = useState(false);

  // ─── Chunk Inspector ────────────────────────────────────────
  const [selectedDocForChunks, setSelectedDocForChunks] = useState<string | null>(null);
  const [chunkSearchQuery, setChunkSearchQuery] = useState("");
  const [chunkSearchResults, setChunkSearchResults] = useState<RetrievalResult[] | null>(null);
  /** Expand rows in "All Chunks by Document" (controlled <details>). */
  const [chunksDocExpanded, setChunksDocExpanded] = useState<Record<string, boolean>>({});
  /** After navigating from chat Sources, scroll to this chunk in Chunk view. */
  const [chunkJumpTarget, setChunkJumpTarget] = useState<{ docId: string; chunkId: string } | null>(null);
  const [chunkSearchLoading, setChunkSearchLoading] = useState(false);

  // ─── Analytics / Summaries ──────────────────────────────────
  const [docSummaries, setDocSummaries] = useState<Record<string, DocSummary>>({});
  const [analyticsLoading, setAnalyticsLoading] = useState(false);
  const [queryLog, setQueryLog] = useState<QueryLogEntry[]>([]);

  // ─── Settings ───────────────────────────────────────────────
  const [chunkSize, setChunkSize] = useState(500);
  const [chunkOverlap, setChunkOverlap] = useState(100);
  const [topK, setTopK] = useState(5);
  const [similarityThreshold, setSimilarityThreshold] = useState(0.3);
  const [useHybridSearch, setUseHybridSearch] = useState(true);  // NEW: BM25 + Vector

  const [error, setError] = useState("");
  const chatEndRef = useRef<HTMLDivElement | null>(null);
  const chatInputRef = useRef<HTMLInputElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const docFileInputRef = useRef<HTMLInputElement | null>(null);

  console.log("[RAGBase] before effect", "mount-log");
  useEffect(() => {
    console.log("[RAGBase] mounted");
    return () => console.log("[RAGBase] unmounted");
  }, []);

  console.log("[RAGBase] before effect", "chat-scroll");
  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: "smooth" }); }, [chatMessages]);

  console.log("[RAGBase] before effect", "chat-focus");
  useEffect(() => {
    if (!chatLoading && view === VIEWS.CHAT) chatInputRef.current?.focus();
  }, [chatLoading, view]);

  // ─── Load documents from Supabase (mount + Retry) ────────────
  const refetchDocuments = useCallback(async () => {
    try {
      setDbLoading(true);
      setDocsLoading(true);
      const res = await fetch("/api/documents");
      if (!res.ok) {
        const { parsed: errBody } = await readJsonOnce<{ error?: string }>(res);
        const msg = errBody?.error || `Failed to load documents (${res.status})`;
        console.error("[API /api/documents]", res.status, msg);
        setDocuments([]);
        setTotalChunks(0);
        setDocsError(msg);
        setDbSynced(false);
        return;
      }
      const { parsed: data, parseError } = await readJsonOnce<{
        error?: string;
        documents?: StoredDocument[];
        totalChunks?: number;
      }>(res);
      if (parseError || data === null) {
        console.error("[API /api/documents] invalid JSON", res.status);
        setDocuments([]);
        setTotalChunks(0);
        setDocsError("Invalid response from documents API");
        setDbSynced(false);
        return;
      }
      const docs = data.documents ?? [];
      setDocsError(null);
      setDocuments(docs);
      setTotalChunks(data.totalChunks ?? 0);
      if (docs.length) {
        setDbSynced(true);
        setView(VIEWS.CHAT);
      }
    } catch (error: unknown) {
      console.error("Documents API failed:", error);
      setDocuments([]);
      setTotalChunks(0);
      setDocsError(error instanceof Error ? error.message : "Failed to load documents");
      setDbSynced(false);
    } finally {
      setDbLoading(false);
      setDocsLoading(false);
    }
  }, []);

  console.log("[RAGBase] before effect", "supabase-load-documents");
  useEffect(() => {
    void refetchDocuments();
  }, [refetchDocuments]);

  console.log("[RAGBase] before effect", "chunks-by-doc");
  useEffect(() => {
    if (view !== VIEWS.CHUNKS || !documents.length) {
      setChunksByDocId({});
      return;
    }
    let cancelled = false;
    (async () => {
      const next: Record<string, VectorDoc[]> = {};
      await Promise.all(
        documents.map(async (doc) => {
          try {
            const res = await fetch(`/api/documents/${doc.id}/chunks`);
            if (!res.ok) {
              console.error(`[API /api/documents/${doc.id}/chunks]`, res.status);
              return;
            }
            const { parsed: data, parseError } = await readJsonOnce<{ chunks?: VectorDoc[] }>(res);
            if (parseError || !data) {
              console.error(`[API /api/documents/${doc.id}/chunks] invalid JSON`);
              return;
            }
            if (data.chunks) next[doc.id] = data.chunks;
          } catch (e) {
            console.error(`[API /api/documents/${doc.id}/chunks]`, e);
          }
        })
      );
      if (!cancelled) setChunksByDocId(next);
    })();
    return () => {
      cancelled = true;
    };
  }, [view, documents]);

  useEffect(() => {
    if (view !== VIEWS.CHUNKS || !chunkJumpTarget) return;
    const chunks = chunksByDocId[chunkJumpTarget.docId];
    if (!chunks?.some((c) => c.id === chunkJumpTarget.chunkId)) return;
    const t = window.setTimeout(() => {
      const el = document.querySelector(`[data-chunk-jump="${chunkJumpTarget.chunkId}"]`);
      el?.scrollIntoView({ behavior: "smooth", block: "center" });
      setChunkJumpTarget(null);
    }, 120);
    return () => clearTimeout(t);
  }, [view, chunkJumpTarget, chunksByDocId]);

  const addLog = (msg: string) =>
    setProcessLog((prev) => [...prev, { time: new Date().toLocaleTimeString(), msg }]);

  // ─── Document Processing Pipeline ───────────────────────────
  const processDocuments = async (
    texts: { name: string; text: string; pages?: number; fileType?: string }[]
  ) => {
    setProcessing(true); setError(""); setProcessLog([]);

    try {
      addLog("Starting document processing pipeline...");
      setProcessStep("Parsing documents..."); setProcessProgress(5);

      const createdDocs: StoredDocument[] = [];
      const totalChunkEstimate = texts.reduce((acc, doc) => {
        const parts = chunkText(doc.text, { chunkSize, overlap: chunkOverlap });
        return acc + parts.length;
      }, 0);

      let processedChunks = 0;

      for (let di = 0; di < texts.length; di++) {
        const doc = texts[di];
        const chunks = chunkText(doc.text, { chunkSize, overlap: chunkOverlap });
        const avgChunkSize = chunks.length
          ? Math.round(chunks.reduce((s, c) => s + c.length, 0) / chunks.length)
          : 0;

        addLog(
          `Chunked "${doc.name}": ${chunks.length} chunks (avg ${avgChunkSize} chars)` +
            (doc.pages ? ` · ${doc.pages} pages` : "")
        );

        setProcessStep(`Ingesting "${doc.name}" (${chunks.length} chunks, server-side embeddings)...`);
        const pct = 5 + Math.round((processedChunks / Math.max(totalChunkEstimate, 1)) * 75);
        setProcessProgress(pct);

        const chunkTexts = chunks.map((c) => c.text);
        addLog(`POST /api/ingest — ${chunkTexts.length} chunks (embed + persist on server)`);

        const ingestPayload: {
          documentName: string;
          chunks: string[];
          fileType?: string;
          pages?: number | null;
        } = {
          documentName: doc.name,
          chunks: chunkTexts,
        };
        if (doc.fileType) ingestPayload.fileType = doc.fileType;
        if (doc.pages != null) ingestPayload.pages = doc.pages;

        let res: Response;
        try {
          res = await fetch("/api/ingest", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(ingestPayload),
          });
        } catch (e) {
          console.error("[API /api/ingest] network:", e);
          throw new Error(e instanceof Error ? e.message : "Ingest network error");
        }
        const { parsed: payload, parseError } = await readJsonOnce<{ error?: string; document?: StoredDocument }>(res);
        if (!res.ok) {
          const msg = (!parseError && payload?.error) || `Ingest failed (${res.status})`;
          console.error("[API /api/ingest]", res.status, msg);
          throw new Error(msg);
        }
        if (parseError || !payload) {
          console.error("[API /api/ingest] invalid JSON", res.status);
          throw new Error("Invalid response from ingest API");
        }
        if (!payload.document) throw new Error("Ingest returned no document");
        createdDocs.push(payload.document);
        processedChunks += chunks.length;
      }

      setProcessProgress(92);
      setProcessStep("Finalizing…");
      const updatedDocs = [...documents, ...createdDocs];
      setDocuments(updatedDocs);
      setDocsError(null);
      setTotalChunks((t) => t + createdDocs.reduce((s, d) => s + d.chunks, 0));
      setDbSynced(true);
      addLog(`Saved to Supabase — ${createdDocs.length} document(s), ${processedChunks} chunks`);

      setProcessProgress(100);
      setProcessStep("Pipeline complete!");
      addLog(`Knowledge base ready: ${updatedDocs.reduce((s, d) => s + d.chunks, 0)} chunks in database`);

      setTimeout(() => {
        setView(VIEWS.CHAT);
        setProcessing(false);
        setProcessStep("");
        setProcessProgress(0);
      }, 1000);
    } catch (err: unknown) {
      console.error(err);
      const message = err instanceof Error ? err.message : String(err);
      addLog(`ERROR: ${message}`);
      setError("Processing failed: " + message);
      setProcessing(false);
    }
  };

  // ─── File Handlers ───────────────────────────────────────────
  const handleFileUpload = async (e: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    if (!files.length) return;
    const texts = [];

    for (const file of files) {
      const ext = (file.name.split(".").pop() ?? "").toLowerCase();
      if (ext === "pdf") {
        if (DEBUG_DISABLE_PDF) {
          setError(`PDF temporarily disabled (debug). Use .txt / .md / .csv for "${file.name}".`);
          continue;
        }
        addLog(`Extracting text from PDF: ${file.name}...`);
        try {
          const { text, pages } = await extractPDFText(file);
          if (!text.trim()) throw new Error("No extractable text found in PDF");
          texts.push({ name: file.name, text, pages, fileType: "pdf" });
        } catch (err: unknown) {
          const message = err instanceof Error ? err.message : String(err);
          setError(`PDF extraction failed for "${file.name}": ${message}`);
        }
      } else {
        texts.push({ name: file.name, text: await file.text(), fileType: ext });
      }
    }
    if (texts.length) processDocuments(texts);
  };

  const handlePasteText = () => {
    const text = prompt("Paste your document text:");
    if (text?.trim())
      processDocuments([{ name: `Pasted Doc ${documents.length + 1}`, text: text.trim(), fileType: "text" }]);
  };

  const handleDrop = (e: DragEvent) => {
    e.preventDefault();
    const files = Array.from(e.dataTransfer.files).filter((f) =>
      /\.(txt|md|csv|pdf)$/i.test(f.name)
    );
    if (files.length) {
      const dt = new DataTransfer();
      files.forEach((f) => dt.items.add(f));
      handleFileUpload({ target: { files: dt.files } } as ChangeEvent<HTMLInputElement>);
    }
  };

  const removeDocument = async (docId: string) => {
    const removed = documents.find((d) => d.id === docId);
    try {
      const res = await fetch(`/api/documents/${docId}`, { method: "DELETE" });
      const { parsed: data, parseError } = await readJsonOnce<{ error?: string }>(res);
      if (!res.ok) {
        const msg = (!parseError && data?.error) || `Failed to delete document (${res.status})`;
        console.error("[API DELETE /api/documents/:id]", res.status, msg);
        setError(msg);
        return;
      }
    } catch (e) {
      console.error("[API DELETE /api/documents/:id]", e);
      setError(e instanceof Error ? e.message : "Failed to delete document");
      return;
    }
    const updated = documents.filter((d) => d.id !== docId);
    setDocuments(updated);
    setChunksByDocId((prev) => {
      const next = { ...prev };
      delete next[docId];
      return next;
    });
    setTotalChunks((t) => Math.max(0, t - (removed?.chunks ?? 0)));
    if (!updated.length) setDbSynced(false);
  };

  const goToRetrievedSource = (s: RetrievalResult) => {
    setChunksDocExpanded((prev) => ({ ...prev, [s.documentId]: true }));
    setSelectedDocForChunks(s.documentId);
    setChunkJumpTarget({ docId: s.documentId, chunkId: s.id });
    setView(VIEWS.CHUNKS);
  };

  // ─── Chat / Q&A (Streaming) ───────────────────────────────
  const submitQuestion = async () => {
    if (!chatInput.trim() || chatLoading) return;
    const question = chatInput.trim();
    setChatInput(""); setChatLoading(true); setError("");
    setChatMessages((prev) => [
      ...prev,
      { role: "user", text: question, timestamp: new Date().toLocaleTimeString() },
    ]);

    try {
      const { results: retrieved, error: retrieveErr } = await fetchRetrieve({
        query: question,
        topK,
        useHybrid: useHybridSearch,
      });
      if (retrieveErr) {
        console.warn("[chat] retrieval failed; continuing with empty context:", retrieveErr);
      }
      let results = retrieved;
      if (useHybridSearch) {
        results = results.filter((r) => (r.vectorScore ?? 0) >= similarityThreshold);
      } else {
        results = results.filter((r) => (r.score ?? 0) >= similarityThreshold);
      }
      setLastRetrieved(results);

      const contextBlock = results
        .map(
          (c, i) =>
            `[Source ${i + 1}] (from: ${c.documentName}, chunk ${c.chunkIndex}; ` +
            (useHybridSearch
              ? `vector: ${((c.vectorScore ?? 0) * 100).toFixed(1)}%, bm25: ${c.bm25Score?.toFixed(2) || "N/A"}`
              : `relevance: ${((c.score ?? 0) * 100).toFixed(1)}%`) +
            `)\n${c.content}`
        )
        .join("\n\n");

      const history = chatMessages.slice(-6);
      const historyBlock = history.length
        ? "\nRecent conversation:\n" +
          history
            .map((m) => {
              if (m.role === "user") return `${m.role}: ${m.text}`;
              if (m.streaming) return `${m.role}: ${m.streamText || ""}`;
              return `${m.role}: ${m.text || ""}`;
            })
            .join("\n") + "\n"
        : "";

      // Streaming-friendly prompt: answer first (streams live), metadata block at end (hidden during stream)
      const prompt = `You are RAGBase, a document intelligence assistant. Answer using ONLY the retrieved context.
${historyBlock}
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

      // Add a streaming placeholder message
      const streamId = `stream_${Date.now()}`;
      setChatMessages((prev) => [
        ...prev,
        {
          id: streamId,
          role: "assistant",
          streaming: true,
          streamText: "",
          searchMode: useHybridSearch ? "hybrid" : "vector",
          timestamp: new Date().toLocaleTimeString(),
        },
      ]);

      const fullRaw = await consumeRagChatResponse(prompt, (displayText) => {
        setChatMessages((prev) =>
          prev.map((m) =>
            m.role === "assistant" && "id" in m && m.id === streamId
              ? { ...m, streamText: displayText }
              : m
          )
        );
      });

      const { answer, metadata } = parseStreamedResponse(fullRaw);

      const finalEntry: Extract<ChatMessage, { role: "assistant"; streaming: false }> = {
        id: streamId,
        role: "assistant",
        streaming: false,
        text: answer,
        confidence: metadata.confidence || "medium",
        sourcesUsed: metadata.sources_used || [],
        keyEntities: metadata.key_entities || [],
        followUps: metadata.follow_up_questions || [],
        retrievedChunks: results,
        searchMode: useHybridSearch ? "hybrid" : "vector",
        timestamp: new Date().toLocaleTimeString(),
      };

      setChatMessages((prev) =>
        prev.map((m) =>
          m.role === "assistant" && "id" in m && m.id === streamId ? finalEntry : m
        )
      );
      setQueryLog((prev) => [
        ...prev,
        {
          question,
          timestamp: new Date().toLocaleString(),
          confidence: metadata.confidence || "medium",
          topScore: results[0]?.vectorScore || results[0]?.score || 0,
          sourcesUsed: metadata.sources_used?.length || 0,
          chunksSearched: results.length,
          searchMode: useHybridSearch ? "hybrid" : "vector",
        },
      ]);
    } catch (err: unknown) {
      console.error(err);
      setChatMessages((prev) =>
        prev.map((m) =>
          m.role === "assistant" && "streaming" in m && m.streaming
            ? ({
                id: m.id,
                role: "assistant",
                streaming: false,
                text: "Error processing your question. Please try again.",
                confidence: "low",
                sourcesUsed: [],
                keyEntities: [],
                followUps: [],
                retrievedChunks: [],
                searchMode: m.searchMode,
                timestamp: m.timestamp,
              } satisfies Extract<ChatMessage, { role: "assistant"; streaming: false }>)
            : m
        )
      );
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setChatLoading(false);
    }
  };

  // ─── Chunk Search (Hybrid) ────────────────────────────────
  const searchChunks = async () => {
    if (!chunkSearchQuery.trim()) return;
    setChunkSearchLoading(true);
    try {
      const { results, error: retrieveErr } = await fetchRetrieve({
        query: chunkSearchQuery,
        topK: 20,
        useHybrid: useHybridSearch,
        filterDocumentId: selectedDocForChunks,
      });
      if (retrieveErr) {
        console.error("[chunk search]", retrieveErr);
        setError(retrieveErr);
        setChunkSearchResults([]);
        return;
      }
      setChunkSearchResults(results);
    } catch (err: unknown) {
      console.error("[chunk search] unexpected", err);
      setError(err instanceof Error ? err.message : String(err));
      setChunkSearchResults([]);
    } finally {
      setChunkSearchLoading(false);
    }
  };

  // ─── Doc Summary ─────────────────────────────────────────
  const generateDocSummary = async (docId: string) => {
    const doc = documents.find((d) => d.id === docId);
    if (!doc || docSummaries[docId]) return;
    setAnalyticsLoading(true);
    try {
      const summaryRes = await fetchChatJson(
        `Summarize this document concisely. Respond ONLY in JSON:
{"summary": "2-3 sentence summary", "topics": ["topic1","topic2","topic3"], "key_entities": ["entity1","entity2"], "sentiment": "positive|neutral|negative", "complexity": "basic|intermediate|advanced"}

Document (first 3000 chars):
${doc.text.substring(0, 3000)}`
      );
      if (summaryRes.mode !== "summary") throw new Error("Unexpected summary response");
      const { mode: _m, ...parsed } = summaryRes;
      setDocSummaries((prev) => ({ ...prev, [docId]: parsed }));
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setAnalyticsLoading(false);
    }
  };

  // ─── Analytics ───────────────────────────────────────────
  const storeStats = useMemo(() => {
    const byDoc: Record<string, { count: number; totalLen: number }> = {};
    documents.forEach((d) => {
      const name = d.name || "Unknown";
      byDoc[name] = {
        count: d.chunks,
        totalLen: d.text.length,
      };
    });
    return byDoc;
  }, [documents]);
  const chunkDistribution = useMemo(
    () =>
      Object.entries(storeStats).map(([name, data]) => ({
        name: name.length > 15 ? name.substring(0, 15) + "…" : name,
        chunks: data.count,
        avgSize: Math.round(data.totalLen / data.count),
      })),
    [storeStats]
  );
  const queryStats = useMemo(() => {
    if (!queryLog.length) return null;
    const avgScore = queryLog.reduce((s, q) => s + q.topScore, 0) / queryLog.length;
    const confCounts = { high: 0, medium: 0, low: 0 };
    queryLog.forEach((q) => {
      const k = q.confidence as keyof typeof confCounts;
      if (k in confCounts) confCounts[k]++;
    });
    return { avgScore, confCounts, total: queryLog.length };
  }, [queryLog]);

  // ─── Clear All ───────────────────────────────────────────
  const clearAll = async () => {
    if (!confirm("Clear all documents and chunks from Supabase?")) return;
    try {
      const res = await fetch("/api/documents", { method: "DELETE" });
      const { parsed: data, parseError } = await readJsonOnce<{ error?: string }>(res);
      if (!res.ok) {
        const msg = (!parseError && data?.error) || `Failed to clear database (${res.status})`;
        console.error("[API DELETE /api/documents]", res.status, msg);
        setError(msg);
        return;
      }
    } catch (e) {
      console.error("[API DELETE /api/documents]", e);
      setError(e instanceof Error ? e.message : "Failed to clear database");
      return;
    }
    setDocuments([]); setTotalChunks(0); setChunksByDocId({});
    setDocsError(null);
    setChatMessages([]); setLastRetrieved([]); setQueryLog([]);
    setDocSummaries({}); setChunkSearchResults(null);
    setView(VIEWS.UPLOAD);
    setDbSynced(false);
  };

  // ─── Sub-Components ──────────────────────────────────────
  const ConfBadge = ({ level }: { level: string }) => {
    const c: Record<string, string> = { high: "#06D6A0", medium: "#FFD166", low: "#EF476F" };
    return (
      <span style={{
        fontSize: 10, fontWeight: 800, padding: "2px 8px", borderRadius: 4,
        textTransform: "uppercase", letterSpacing: "0.05em",
        background: (c[level] || c.low) + "18", color: c[level] || c.low,
        fontFamily: "'IBM Plex Mono', monospace",
      }}>{level}</span>
    );
  };

  const SearchModeBadge = ({ mode }: { mode: "hybrid" | "vector" }) => (
    <span style={{
      fontSize: 9, fontWeight: 700, padding: "1px 6px", borderRadius: 4,
      background: mode === "hybrid" ? "var(--accent2)" + "18" : "var(--primary)" + "18",
      color: mode === "hybrid" ? "var(--accent2)" : "var(--primary)",
      fontFamily: "'IBM Plex Mono', monospace", textTransform: "uppercase",
    }}>{mode === "hybrid" ? "⚡ Hybrid" : "◈ Vector"}</span>
  );

  const ScoreBar = ({ score, max = 1 }: { score: number; max?: number }) => (
    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
      <div style={{ flex: 1, height: 4, background: "#1A1D2E", borderRadius: 2, overflow: "hidden" }}>
        <div style={{
          width: `${(score / max) * 100}%`, height: "100%", borderRadius: 2,
          background: score >= 0.7 ? "#06D6A0" : score >= 0.4 ? "#FFD166" : "#EF476F",
          transition: "width 0.6s ease",
        }} />
      </div>
      <span style={{
        fontSize: 11, fontWeight: 700, color: "#E4E7F0",
        fontFamily: "'IBM Plex Mono', monospace", minWidth: 40,
      }}>{(score * 100).toFixed(1)}%</span>
    </div>
  );

  const fileTypeIcon = (type: string | undefined) => {
    if (type === "pdf") return "📕";
    if (type === "md") return "📝";
    if (type === "csv") return "📊";
    return "📄";
  };

  const documentsLoadErrorCard = docsError ? (
    <div
      style={{
        ...S.card,
        marginBottom: 16,
        border: "1px solid rgba(239, 71, 111, 0.35)",
        background: "rgba(239, 71, 111, 0.06)",
      }}
    >
      <h3 style={{ fontSize: 15, fontWeight: 700, color: "var(--danger)", marginBottom: 10 }}>
        ⚠️ Failed to load documents
      </h3>
      <p style={{ fontSize: 13, color: "var(--text-bright)", lineHeight: 1.55, marginBottom: 14 }}>{docsError}</p>
      <button
        type="button"
        onClick={() => void refetchDocuments()}
        disabled={docsLoading}
        className="hov"
        style={{
          ...S.btnOut,
          borderColor: "var(--danger)",
          color: "var(--danger)",
          opacity: docsLoading ? 0.65 : 1,
          cursor: docsLoading ? "wait" : "pointer",
        }}
      >
        {docsLoading ? "Loading…" : "Retry"}
      </button>
    </div>
  ) : null;

  // ══════════════════════════════════════════════════════════
  // RENDER
  // ══════════════════════════════════════════════════════════
  return (
    <div style={S.wrapper}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
        :root {
          --bg: #08090E; --surface: #0F1118; --surface2: #171A26; --surface3: #1F2333;
          --border: #252940; --border2: #343850; --text: #8B92AB; --text-bright: #E4E7F0;
          --primary: #7C5CFC; --primary2: #5B3FD9; --accent: #06D6A0; --accent2: #3EDBF0;
          --danger: #EF476F; --warn: #FFD166;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::selection { background: var(--primary); color: white; }
        @keyframes fadeUp { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        @keyframes pulse { 0%,100% { opacity: 0.3; } 50% { opacity: 1; } }
        @keyframes slideIn { from { opacity: 0; transform: translateX(-8px); } to { opacity: 1; transform: translateX(0); } }
        @keyframes glow { 0%,100% { box-shadow: 0 0 4px var(--primary); } 50% { box-shadow: 0 0 20px var(--primary); } }
        @keyframes blink { 0%,100% { opacity: 1; } 50% { opacity: 0; } }
        @keyframes progressPulse { 0%,100% { opacity: 0.8; } 50% { opacity: 1; } }
        @keyframes shimmer {
          0% { background-position: -200% 0; }
          100% { background-position: 200% 0; }
        }
        .hov { transition: all 0.15s ease; cursor: pointer; }
        .hov:hover { transform: translateY(-1px); }
        input:focus, textarea:focus { border-color: var(--primary) !important; outline: none; }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }
        .drop-t { transition: all 0.3s ease; }
        .drop-t:hover { border-color: var(--primary) !important; background: rgba(124,92,252,0.04) !important; }
        details > summary { list-style: none; }
        details > summary::-webkit-details-marker { display: none; }
        .streaming-cursor { display: inline-block; width: 2px; height: 14px; background: var(--primary); margin-left: 2px; animation: blink 0.8s step-end infinite; vertical-align: text-bottom; }
      `}</style>

      {/* ─── Sidebar ─── */}
      <div style={S.sidebar}>
        <div style={S.sidebarTop}>
          <div style={S.logoRow}>
            <div style={S.logoGem}>R</div>
            <div>
              <div style={S.logoTitle}>RAGBase</div>
              <div style={S.logoSub}>Document Intelligence</div>
            </div>
          </div>
          {/* Persistence indicator */}
          <div style={{ marginTop: 8, display: "flex", alignItems: "center", gap: 5 }}>
            <div style={{
              width: 6, height: 6, borderRadius: "50%",
              background: dbLoading ? "var(--warn)" : dbSynced ? "var(--accent)" : "var(--border2)",
              flexShrink: 0,
            }} />
            <span style={{ fontSize: 9, color: "var(--text)", fontFamily: "'IBM Plex Mono', monospace" }}>
              {dbLoading ? "loading…" : dbSynced ? "Supabase synced" : "not persisted"}
            </span>
          </div>
        </div>

        <div style={S.sidebarNav}>
          {[
            { id: VIEWS.UPLOAD, icon: "⬆", label: "Upload" },
            { id: VIEWS.DOCUMENTS, icon: "📚", label: "Documents", count: documents.length },
            { id: VIEWS.CHAT, icon: "💬", label: "Chat", disabled: !documents.length },
            { id: VIEWS.CHUNKS, icon: "🔬", label: "Chunk Inspector", disabled: !documents.length },
            { id: VIEWS.ANALYTICS, icon: "📊", label: "Analytics", disabled: !documents.length },
            { id: VIEWS.PIPELINE, icon: "⚙️", label: "Pipeline" },
          ].map((item) => (
            <button
              key={item.id}
              className="hov"
              onClick={() => !item.disabled && setView(item.id)}
              style={{
                ...S.navItem,
                background: view === item.id ? "var(--surface3)" : "transparent",
                borderLeft: view === item.id ? "2px solid var(--primary)" : "2px solid transparent",
                opacity: item.disabled ? 0.35 : 1,
                cursor: item.disabled ? "not-allowed" : "pointer",
              }}
            >
              <span style={{ fontSize: 14 }}>{item.icon}</span>
              <span style={{ flex: 1, fontSize: 12 }}>{item.label}</span>
              {(item.count ?? 0) > 0 && <span style={S.badge}>{item.count}</span>}
            </button>
          ))}
        </div>

        <div style={S.sidebarStats}>
          {[
            { label: "Docs", value: documents.length, color: "var(--primary)" },
            { label: "Chunks", value: totalChunks, color: "var(--accent)" },
            { label: "Queries", value: queryLog.length, color: "var(--accent2)" },
          ].map((s) => (
            <div key={s.label} style={S.statItem}>
              <span style={{ fontSize: 10, color: "var(--text)" }}>{s.label}</span>
              <span style={{
                fontSize: 14, fontWeight: 800, color: s.color,
                fontFamily: "'IBM Plex Mono', monospace",
              }}>{s.value}</span>
            </div>
          ))}
        </div>

        <div style={S.sidebarFoot}>
          {/* Search mode toggle */}
          <div style={{ marginBottom: 8, padding: "6px 10px", background: "var(--surface2)", borderRadius: 5 }}>
            <div style={{ fontSize: 9, color: "var(--text)", marginBottom: 4, fontFamily: "'IBM Plex Mono', monospace", textTransform: "uppercase" }}>
              Search Mode
            </div>
            <div style={{ display: "flex", gap: 4 }}>
              {[
                { label: "Hybrid", value: true, color: "var(--accent2)" },
                { label: "Vector", value: false, color: "var(--primary)" },
              ].map((opt) => (
                <button
                  key={opt.label}
                  onClick={() => setUseHybridSearch(opt.value)}
                  style={{
                    flex: 1, padding: "3px 0", fontSize: 10, fontWeight: 700,
                    border: "none", borderRadius: 4, cursor: "pointer", fontFamily: "'IBM Plex Mono', monospace",
                    background: useHybridSearch === opt.value ? opt.color + "22" : "transparent",
                    color: useHybridSearch === opt.value ? opt.color : "var(--text)",
                    borderBottom: useHybridSearch === opt.value ? `1px solid ${opt.color}` : "1px solid transparent",
                  }}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* ─── Main ─── */}
      <div style={S.main}>
        {error && (
          <div style={S.errorBar}>
            <span>⚠️</span>
            <span style={{ flex: 1 }}>{error}</span>
            <button onClick={() => setError("")} style={{ background: "none", border: "none", color: "var(--danger)", cursor: "pointer" }}>✕</button>
          </div>
        )}

        {/* ─── Documents fetch loading ─── */}
        {docsLoading && (
          <div style={{ ...S.card, textAlign: "center", padding: 24, marginBottom: 12 }}>
            <p style={{ fontSize: 12, color: "var(--text)", fontFamily: "'IBM Plex Mono', monospace" }}>
              Loading documents...
            </p>
          </div>
        )}

        {/* ═══════ PROCESSING ═══════ */}
        {processing && (
          <div style={{ padding: "40px 20px", animation: "fadeUp 0.4s ease" }}>
            <div style={{ maxWidth: 500, margin: "0 auto", textAlign: "center" }}>
              <div style={{ ...S.logoGem, width: 52, height: 52, fontSize: 24, margin: "0 auto 18px", animation: "glow 2s ease infinite" }}>R</div>
              <h3 style={{ fontSize: 16, fontWeight: 700, color: "var(--text-bright)", marginBottom: 6 }}>Processing Pipeline</h3>
              <p style={{ fontSize: 13, color: "var(--primary)", fontFamily: "'IBM Plex Mono', monospace", marginBottom: 16 }}>{processStep}</p>
              <div style={S.progressTrack}>
                <div style={{ ...S.progressFill, width: `${processProgress}%` }} />
              </div>
              <p style={{ fontSize: 11, color: "var(--text)", marginTop: 6 }}>{processProgress}%</p>
              <div style={{ marginTop: 20, background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 8, padding: "12px 14px", textAlign: "left", maxHeight: 180, overflowY: "auto" }}>
                <div style={{ fontSize: 10, fontWeight: 700, color: "var(--primary)", marginBottom: 8, textTransform: "uppercase", fontFamily: "'IBM Plex Mono', monospace" }}>
                  Pipeline Log
                </div>
                {processLog.map((l, i) => (
                  <div key={i} style={{ fontSize: 11, color: l.msg.startsWith("ERROR") ? "var(--danger)" : "var(--text)", marginBottom: 3, fontFamily: "'IBM Plex Mono', monospace", lineHeight: 1.4 }}>
                    <span style={{ color: "var(--border2)", marginRight: 6 }}>{l.time}</span>{l.msg}
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* ═══════ UPLOAD ═══════ */}
        {view === VIEWS.UPLOAD && !processing && (
          <div style={{ animation: "fadeUp 0.5s ease", maxWidth: 660, margin: "36px auto" }}>
            <div style={{ textAlign: "center", marginBottom: 32 }}>
              <h1 style={S.hero}><span style={{ color: "var(--primary)" }}>RAG</span> Pipeline <span style={{ fontSize: 16, color: "var(--accent2)" }}>v2</span></h1>
              <p style={S.heroSub}>
                Full RAG — PDF · TXT · MD · CSV · Hybrid BM25+Vector Search · Supabase pgvector · Streaming Generation
              </p>
            </div>

            {documentsLoadErrorCard}

            {!docsError && !docsLoading && documents.length === 0 && (
              <p
                style={{
                  textAlign: "center",
                  fontSize: 14,
                  color: "var(--text)",
                  marginBottom: 20,
                  fontFamily: "'IBM Plex Mono', monospace",
                }}
              >
                📂 No documents yet — upload to begin
              </p>
            )}

            <input
              ref={fileInputRef}
              type="file"
              accept=".txt,.md,.csv,.pdf"
              multiple
              onChange={handleFileUpload}
              style={{ display: "none" }}
            />

            <div
              className="drop-t"
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              style={S.dropZone}
            >
              <div style={{ fontSize: 36, color: "var(--primary)", marginBottom: 10, opacity: 0.5 }}>◈</div>
              <h3 style={{ fontSize: 16, fontWeight: 700, color: "var(--text-bright)", marginBottom: 4 }}>Drop documents here</h3>
              <p style={{ fontSize: 13, color: "var(--text)" }}>.pdf · .txt · .md · .csv · Multiple files supported</p>
              <div style={{ marginTop: 12, display: "flex", justifyContent: "center", gap: 6 }}>
                {["📕 PDF", "📄 TXT", "📝 MD", "📊 CSV"].map((t) => (
                  <span key={t} style={{ fontSize: 10, padding: "2px 8px", borderRadius: 4, background: "var(--surface2)", color: "var(--text)", fontFamily: "'IBM Plex Mono', monospace" }}>{t}</span>
                ))}
              </div>
            </div>

            <div style={{ textAlign: "center", margin: "14px 0" }}>
              <button onClick={handlePasteText} className="hov" style={{ ...S.btnOut, padding: "10px 24px" }}>📋 Paste Text</button>
            </div>

            {/* Architecture overview */}
            <div style={{ ...S.card, marginTop: 24 }}>
              <h3 style={S.cardTitle}>Architecture v2</h3>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8 }}>
                {[
                  { icon: "📕", title: "PDF + Text", desc: "pdf.js extraction, TXT/MD/CSV. Multi-file batch." },
                  { icon: "✂️", title: "Chunk", desc: `Sentence-aware splitting (~${chunkSize} chars, ${chunkOverlap} overlap)` },
                  { icon: "🧮", title: "Embed", desc: "768-dim vectors via POST /api/embed" },
                  { icon: "🗄️", title: "Index + Persist", desc: "Supabase pgvector + BM25 (server-side hybrid)" },
                  { icon: "⚡", title: "Hybrid Search", desc: "BM25 + Cosine via Reciprocal Rank Fusion" },
                  { icon: "🌊", title: "Stream Generate", desc: "Live streaming answers with [Source N] citations" },
                ].map((s, i) => (
                  <div key={i} style={{ display: "flex", gap: 8, padding: "10px", background: "var(--surface2)", borderRadius: 6 }}>
                    <span style={{ fontSize: 18 }}>{s.icon}</span>
                    <div>
                      <div style={{ fontSize: 11, fontWeight: 700, color: "var(--text-bright)", marginBottom: 2 }}>{s.title}</div>
                      <div style={{ fontSize: 10, color: "var(--text)", lineHeight: 1.4 }}>{s.desc}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* ═══════ DOCUMENTS ═══════ */}
        {view === VIEWS.DOCUMENTS && (
          <div style={{ animation: "fadeUp 0.4s ease" }}>
            <div style={S.viewHead}>
              <div>
                <h2 style={S.viewTitle}>Knowledge Base</h2>
                <p style={S.viewDesc}>
                  {documents.length} documents · {totalChunks} vectors ·{" "}
                  <span style={{ color: dbSynced ? "var(--accent)" : "var(--text)" }}>
                    {dbSynced ? "✓ persisted" : "not saved"}
                  </span>
                </p>
              </div>
              <div style={{ display: "flex", gap: 8 }}>
                <button onClick={() => docFileInputRef.current?.click()} className="hov" style={S.btnPri}>+ Add</button>
                <input ref={docFileInputRef} type="file" accept=".txt,.md,.csv,.pdf" multiple onChange={handleFileUpload} style={{ display: "none" }} />
                {documents.length > 0 && (
                  <button onClick={clearAll} className="hov" style={S.btnDanger}>Clear All + DB</button>
                )}
              </div>
            </div>

            {documents.length === 0 ? (
              docsError ? (
                documentsLoadErrorCard
              ) : (
                <div style={{ ...S.card, textAlign: "center", padding: 50 }}>
                  <p style={{ fontSize: 14, color: "var(--text)" }}>📂 No documents yet — upload to begin</p>
                </div>
              )
            ) : (
              <div style={{ display: "grid", gap: 10 }}>
                {documents.map((doc, i) => (
                  <div key={doc.id} style={{ ...S.card, animation: `slideIn 0.3s ease ${i * 0.04}s both` }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                      <div style={{ flex: 1 }}>
                        <h3 style={{ fontSize: 14, fontWeight: 700, color: "var(--text-bright)", marginBottom: 4 }}>
                          {fileTypeIcon(doc.fileType)} {doc.name}
                          {doc.pages && (
                            <span style={{ marginLeft: 8, fontSize: 10, color: "var(--accent2)", fontFamily: "'IBM Plex Mono', monospace" }}>
                              {doc.pages} pages
                            </span>
                          )}
                        </h3>
                        <div style={{ display: "flex", gap: 10, fontSize: 11, color: "var(--text)", flexWrap: "wrap" }}>
                          <span>{doc.charCount.toLocaleString()} chars</span>
                          <span>·</span>
                          <span>{doc.wordCount.toLocaleString()} words</span>
                          <span>·</span>
                          <span>{doc.chunks} chunks</span>
                          <span>·</span>
                          <span>avg {doc.avgChunkSize} chars/chunk</span>
                          <span>·</span>
                          <span style={{ color: "var(--text)" }}>{doc.uploadedAt}</span>
                        </div>
                      </div>
                      <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                        {!docSummaries[doc.id] && (
                          <button
                            onClick={() => generateDocSummary(doc.id)}
                            className="hov"
                            disabled={analyticsLoading}
                            style={{ ...S.btnOut, padding: "4px 10px", fontSize: 11 }}
                          >
                            {analyticsLoading ? "..." : "✨ Summarize"}
                          </button>
                        )}
                        <button onClick={() => removeDocument(doc.id)} className="hov" style={{ ...S.btnDanger, padding: "4px 10px", fontSize: 11 }}>
                          Remove
                        </button>
                      </div>
                    </div>

                    {docSummaries[doc.id] && (
                      <div style={{ marginTop: 10, padding: "12px 14px", background: "var(--surface2)", borderRadius: 6, borderLeft: "2px solid var(--primary)" }}>
                        <p style={{ fontSize: 12, color: "var(--text-bright)", lineHeight: 1.5, marginBottom: 8 }}>
                          {docSummaries[doc.id].summary}
                        </p>
                        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                          {docSummaries[doc.id].topics?.map((t, ti) => (
                            <span key={ti} style={{ fontSize: 10, padding: "2px 8px", borderRadius: 4, background: "var(--primary)" + "18", color: "var(--primary)", fontFamily: "'IBM Plex Mono', monospace" }}>{t}</span>
                          ))}
                          <span style={{ fontSize: 10, padding: "2px 8px", borderRadius: 4, background: "var(--accent2)" + "18", color: "var(--accent2)", fontFamily: "'IBM Plex Mono', monospace" }}>
                            {docSummaries[doc.id].complexity}
                          </span>
                          <span style={{ fontSize: 10, padding: "2px 8px", borderRadius: 4, background: "var(--accent)" + "18", color: "var(--accent)", fontFamily: "'IBM Plex Mono', monospace" }}>
                            {docSummaries[doc.id].sentiment}
                          </span>
                        </div>
                      </div>
                    )}

                    <div style={{ marginTop: 8, padding: "8px 10px", background: "var(--surface2)", borderRadius: 4, fontSize: 11, color: "var(--text)", fontFamily: "'IBM Plex Mono', monospace", maxHeight: 60, overflow: "hidden", lineHeight: 1.5 }}>
                      {doc.text.substring(0, 250)}…
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* ═══════ CHAT ═══════ */}
        {view === VIEWS.CHAT && (
          <div style={{ animation: "fadeUp 0.4s ease", display: "flex", flexDirection: "column", height: "calc(100vh - 40px)" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 14, paddingBottom: 12, borderBottom: "1px solid var(--border)" }}>
              <div>
                <h2 style={S.viewTitle}>Knowledge Q&A</h2>
                <div style={{ display: "flex", gap: 8, alignItems: "center", marginTop: 2 }}>
                  <p style={S.viewDesc}>{documents.length} docs · {totalChunks} chunks</p>
                  <SearchModeBadge mode={useHybridSearch ? "hybrid" : "vector"} />
                  <span style={{ fontSize: 10, color: "var(--accent)", fontFamily: "'IBM Plex Mono', monospace" }}>⟳ Streaming</span>
                </div>
              </div>
              <div style={{ display: "flex", gap: 6 }}>
                <button
                  onClick={() => setShowRetrieval(!showRetrieval)}
                  className="hov"
                  style={{ ...S.btnOut, padding: "5px 12px", fontSize: 11, borderColor: showRetrieval ? "var(--primary)" : "var(--border)" }}
                >
                  {showRetrieval ? "Hide" : "Show"} Retrieval
                </button>
                <button onClick={() => setChatMessages([])} className="hov" style={{ ...S.btnOut, padding: "5px 12px", fontSize: 11 }}>Clear</button>
              </div>
            </div>

            <div style={{ flex: 1, display: "flex", gap: 14, minHeight: 0 }}>
              {/* Chat Column */}
              <div style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0 }}>
                <div style={{ flex: 1, overflowY: "auto", display: "flex", flexDirection: "column", gap: 12, paddingBottom: 10 }}>
                  {chatMessages.length === 0 && (
                    <div style={{ textAlign: "center", padding: "40px 20px", opacity: 0.5 }}>
                      <div style={{ fontSize: 36, marginBottom: 8 }}>◈</div>
                      <p style={{ fontSize: 13, color: "var(--text)" }}>Ask anything about your documents. Answers stream live with source citations.</p>
                    </div>
                  )}

                  {chatMessages.map((msg, mi) => (
                    <div key={mi} style={{
                      display: "flex",
                      justifyContent: msg.role === "user" ? "flex-end" : "flex-start",
                      animation: "fadeUp 0.3s ease",
                    }}>
                      {msg.role === "user" ? (
                        <div style={{ maxWidth: "72%", padding: "10px 14px", borderRadius: "10px 10px 2px 10px", background: "linear-gradient(135deg, var(--primary), var(--primary2))", color: "white" }}>
                          <p style={{ fontSize: 13, lineHeight: 1.5 }}>{msg.text}</p>
                          <p style={{ fontSize: 9, opacity: 0.6, marginTop: 4, textAlign: "right", fontFamily: "'IBM Plex Mono', monospace" }}>{msg.timestamp}</p>
                        </div>
                      ) : msg.streaming ? (
                        /* ─── Streaming message ─── */
                        <div style={{ maxWidth: "82%", padding: "12px 16px", borderRadius: "10px 10px 10px 2px", background: "var(--surface2)", border: "1px solid var(--primary)" + "33" }}>
                          <div style={{ fontSize: 9, color: "var(--primary)", fontFamily: "'IBM Plex Mono', monospace", marginBottom: 6, display: "flex", alignItems: "center", gap: 6 }}>
                            <span style={{ animation: "pulse 1s infinite" }}>⟳</span> Generating…
                          </div>
                          <p style={{ fontSize: 13, color: "var(--text-bright)", lineHeight: 1.6, whiteSpace: "pre-wrap" }}>
                            {msg.streamText || " "}
                            <span className="streaming-cursor" />
                          </p>
                        </div>
                      ) : (
                        /* ─── Final assistant message ─── */
                        <div style={{ maxWidth: "82%", padding: "12px 16px", borderRadius: "10px 10px 10px 2px", background: "var(--surface2)", border: "1px solid var(--border)" }}>
                          <div style={{ display: "flex", gap: 6, alignItems: "center", marginBottom: 8, flexWrap: "wrap" }}>
                            <ConfBadge level={msg.confidence} />
                            <SearchModeBadge mode={msg.searchMode} />
                            <span style={{ fontSize: 9, color: "var(--text)", fontFamily: "'IBM Plex Mono', monospace" }}>{msg.timestamp}</span>
                          </div>
                          <p style={{ fontSize: 13, color: "var(--text-bright)", lineHeight: 1.7, marginBottom: 10, whiteSpace: "pre-wrap" }}>
                            {msg.text}
                          </p>
                          {msg.retrievedChunks.length > 0 && (
                            <div style={{ marginBottom: 10 }}>
                              <p style={{ fontSize: 10, color: "var(--text)", marginBottom: 6, fontFamily: "'IBM Plex Mono', monospace", textTransform: "uppercase", letterSpacing: "0.04em" }}>
                                Sources
                              </p>
                              <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                                {msg.retrievedChunks.map((s) => (
                                  <button
                                    key={s.id}
                                    type="button"
                                    onClick={() => goToRetrievedSource(s)}
                                    className="hov"
                                    style={{
                                      textAlign: "left",
                                      padding: "6px 10px",
                                      borderRadius: 6,
                                      border: "1px solid var(--border)",
                                      background: "var(--surface)",
                                      color: "var(--text-bright)",
                                      fontSize: 11,
                                      cursor: "pointer",
                                      fontFamily: "'IBM Plex Mono', monospace",
                                      lineHeight: 1.4,
                                    }}
                                  >
                                    📄 {s.documentName} — chunk {s.chunkIndex}
                                    <span style={{ display: "block", fontSize: 10, color: "var(--text)", marginTop: 4, fontWeight: 400 }}>
                                      {s.contentSnippet}
                                    </span>
                                  </button>
                                ))}
                              </div>
                            </div>
                          )}
                          {msg.keyEntities?.length > 0 && (
                            <div style={{ display: "flex", gap: 4, flexWrap: "wrap", marginBottom: 8 }}>
                              {msg.keyEntities.map((e, ei) => (
                                <span key={ei} style={{ fontSize: 10, padding: "2px 7px", borderRadius: 4, background: "var(--accent)" + "15", color: "var(--accent)", fontFamily: "'IBM Plex Mono', monospace" }}>{e}</span>
                              ))}
                            </div>
                          )}
                          {msg.followUps?.length > 0 && (
                            <div style={{ marginTop: 8, paddingTop: 8, borderTop: "1px solid var(--border)" }}>
                              <p style={{ fontSize: 10, color: "var(--text)", marginBottom: 4, fontFamily: "'IBM Plex Mono', monospace", textTransform: "uppercase" }}>Follow-ups</p>
                              {msg.followUps.map((q, qi) => (
                                <button
                                  key={qi}
                                  onClick={() => { setChatInput(q); chatInputRef.current?.focus(); }}
                                  className="hov"
                                  style={{ display: "block", width: "100%", textAlign: "left", padding: "4px 8px", marginBottom: 3, background: "transparent", border: "1px solid var(--border)", borderRadius: 4, color: "var(--primary)", fontSize: 11, cursor: "pointer", fontFamily: "'Plus Jakarta Sans', sans-serif" }}
                                >
                                  → {q}
                                </button>
                              ))}
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  ))}

                  {chatLoading &&
                    !chatMessages.some(
                      (m) => m.role === "assistant" && "streaming" in m && m.streaming
                    ) && (
                    <div style={{ display: "flex", justifyContent: "flex-start" }}>
                      <div style={{ padding: "12px 16px", borderRadius: 10, background: "var(--surface2)", border: "1px solid var(--border)" }}>
                        <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
                          {[0, 1, 2].map((i) => (
                            <div key={i} style={{ width: 5, height: 5, borderRadius: "50%", background: "var(--primary)", animation: `pulse 1s ease ${i * 0.15}s infinite` }} />
                          ))}
                          <span style={{ fontSize: 11, color: "var(--text)", fontFamily: "'IBM Plex Mono', monospace", marginLeft: 6 }}>
                            Embedding → {useHybridSearch ? "BM25+Vector" : "Vector"} search → Streaming…
                          </span>
                        </div>
                      </div>
                    </div>
                  )}
                  <div ref={chatEndRef} />
                </div>

                <div style={{ display: "flex", gap: 8, padding: "12px 0", borderTop: "1px solid var(--border)" }}>
                  <input
                    ref={chatInputRef}
                    type="text"
                    placeholder="Ask about your documents…"
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && submitQuestion()}
                    disabled={chatLoading}
                    style={{ ...S.input, flex: 1, padding: "11px 14px", fontSize: 13 }}
                  />
                  <button
                    onClick={submitQuestion}
                    disabled={chatLoading || !chatInput.trim()}
                    className="hov"
                    style={{ ...S.btnPri, opacity: chatLoading || !chatInput.trim() ? 0.4 : 1 }}
                  >
                    Send →
                  </button>
                </div>
              </div>

              {/* Retrieval Panel */}
              {showRetrieval && lastRetrieved.length > 0 && (
                <div style={{ width: 290, flexShrink: 0, background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 8, padding: "14px", overflowY: "auto", maxHeight: "calc(100vh - 140px)" }}>
                  <h4 style={{ fontSize: 11, fontWeight: 700, color: "var(--primary)", textTransform: "uppercase", fontFamily: "'IBM Plex Mono', monospace", marginBottom: 4 }}>
                    Retrieved Chunks ({lastRetrieved.length})
                  </h4>
                  <p style={{ fontSize: 9, color: "var(--text)", fontFamily: "'IBM Plex Mono', monospace", marginBottom: 10 }}>
                    {useHybridSearch ? "BM25 + Vector → RRF scores" : "Cosine similarity scores"}
                  </p>
                  {lastRetrieved.map((c, i) => (
                    <div key={c.id} style={{ padding: "8px", marginBottom: 6, background: "var(--surface2)", borderRadius: 4, borderLeft: `2px solid ${i === 0 ? "var(--primary)" : "var(--border)"}` }}>
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                        <span style={{ fontSize: 10, color: "var(--accent2)", fontFamily: "'IBM Plex Mono', monospace" }}>
                          {c.documentName.length > 18 ? c.documentName.substring(0, 18) + "…" : c.documentName}
                          <span style={{ color: "var(--text)" }}> · #{c.chunkIndex}</span>
                        </span>
                      </div>
                      {useHybridSearch ? (
                        <div style={{ fontSize: 9, fontFamily: "'IBM Plex Mono', monospace", color: "var(--text)", marginBottom: 4 }}>
                          <span style={{ color: "var(--primary)" }}>vec {((c.vectorScore ?? 0) * 100).toFixed(0)}%</span>
                          {" · "}
                          <span style={{ color: "var(--accent2)" }}>bm25 {c.bm25Score?.toFixed(2)}</span>
                        </div>
                      ) : (
                        <ScoreBar score={c.score ?? 0} />
                      )}
                      <div style={{ marginTop: 4, fontSize: 10, color: "var(--text)", fontFamily: "'IBM Plex Mono', monospace", lineHeight: 1.4 }}>
                        {c.contentSnippet}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {/* ═══════ CHUNK INSPECTOR ═══════ */}
        {view === VIEWS.CHUNKS && (
          <div style={{ animation: "fadeUp 0.4s ease" }}>
            <div style={S.viewHead}>
              <div>
                <h2 style={S.viewTitle}>Chunk Inspector</h2>
                <p style={S.viewDesc}>Explore and semantically search the vector store · <SearchModeBadge mode={useHybridSearch ? "hybrid" : "vector"} /></p>
              </div>
            </div>

            <div style={{ ...S.card, marginBottom: 16 }}>
              <h3 style={S.cardTitle}>🔍 {useHybridSearch ? "Hybrid" : "Semantic"} Search</h3>
              <div style={{ display: "flex", gap: 8, marginBottom: 10 }}>
                <select
                  value={selectedDocForChunks || ""}
                  onChange={(e) => setSelectedDocForChunks(e.target.value || null)}
                  style={S.selectInput}
                >
                  <option value="">All Documents</option>
                  {documents.map((d) => (
                    <option key={d.id} value={d.id}>{d.name}</option>
                  ))}
                </select>
                <input
                  type="text"
                  placeholder="Search chunks…"
                  value={chunkSearchQuery}
                  onChange={(e) => setChunkSearchQuery(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && searchChunks()}
                  style={{ ...S.input, flex: 1 }}
                />
                <button
                  onClick={searchChunks}
                  disabled={chunkSearchLoading || !chunkSearchQuery.trim()}
                  className="hov"
                  style={{ ...S.btnPri, opacity: chunkSearchLoading || !chunkSearchQuery.trim() ? 0.4 : 1 }}
                >
                  {chunkSearchLoading ? "…" : "Search"}
                </button>
              </div>

              {chunkSearchResults && (
                <div>
                  <p style={{ fontSize: 12, color: "var(--text)", marginBottom: 10 }}>
                    {chunkSearchResults.length} chunks found
                    {useHybridSearch && " · scores shown as vector% + bm25"}
                  </p>
                  {chunkSearchResults.map((c, i) => (
                    <div key={c.id} style={{
                      padding: "10px 12px", marginBottom: 6,
                      background: "var(--surface2)", borderRadius: 6,
                      borderLeft: `3px solid ${(c.vectorScore ?? 0) >= 0.6 || (c.score ?? 0) >= 0.6 ? "var(--accent)" : (c.vectorScore ?? 0) >= 0.4 || (c.score ?? 0) >= 0.4 ? "var(--warn)" : "var(--border2)"}`,
                    }}>
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                        <span style={{ fontSize: 11, fontWeight: 600, color: "var(--accent2)" }}>
                          {c.documentName} · Chunk #{c.chunkIndex}
                        </span>
                        {useHybridSearch ? (
                          <span style={{ fontSize: 10, fontFamily: "'IBM Plex Mono', monospace", color: "var(--text)" }}>
                            <span style={{ color: "var(--primary)" }}>{((c.vectorScore ?? 0) * 100).toFixed(0)}%</span>
                            {" + "}
                            <span style={{ color: "var(--accent2)" }}>{c.bm25Score?.toFixed(2)}</span>
                          </span>
                        ) : (
                          <span
                            style={{
                              fontSize: 11,
                              fontWeight: 800,
                              fontFamily: "'IBM Plex Mono', monospace",
                              color:
                                (c.score ?? 0) >= 0.6
                                  ? "var(--accent)"
                                  : (c.score ?? 0) >= 0.4
                                    ? "var(--warn)"
                                    : "var(--text)",
                            }}
                          >
                            {((c.score ?? 0) * 100).toFixed(1)}%
                          </span>
                        )}
                      </div>
                      {!useHybridSearch && <ScoreBar score={c.score ?? 0} />}
                      <div style={{ marginTop: 6, fontSize: 12, color: "var(--text-bright)", lineHeight: 1.6, fontFamily: "'IBM Plex Mono', monospace" }}>{c.content}</div>
                      <div style={{ marginTop: 4, fontSize: 10, color: "var(--text)" }}>
                        {c.content.length} chars
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div style={S.card}>
              <h3 style={S.cardTitle}>📦 All Chunks by Document</h3>
              {documents.map((doc) => {
                const docChunks = chunksByDocId[doc.id] ?? [];
                return (
                  <details
                    key={doc.id}
                    open={chunksDocExpanded[doc.id] ?? false}
                    onToggle={(e) => {
                      setChunksDocExpanded((p) => ({ ...p, [doc.id]: e.currentTarget.open }));
                    }}
                    style={{ marginBottom: 8 }}
                  >
                    <summary style={{ cursor: "pointer", padding: "8px 10px", background: "var(--surface2)", borderRadius: 4, fontSize: 12, fontWeight: 600, color: "var(--text-bright)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                      <span>{fileTypeIcon(doc.fileType)} {doc.name}</span>
                      <span style={{ fontSize: 10, color: "var(--text)", fontFamily: "'IBM Plex Mono', monospace" }}>{docChunks.length} chunks</span>
                    </summary>
                    <div style={{ padding: "8px 0" }}>
                      {docChunks.map((c, ci) => (
                        <div
                          key={c.id ?? ci}
                          data-chunk-jump={c.id}
                          style={{ padding: "6px 10px", marginBottom: 3, borderLeft: "2px solid var(--border)", fontSize: 11, color: "var(--text)", fontFamily: "'IBM Plex Mono', monospace", lineHeight: 1.4 }}
                        >
                          <span style={{ color: "var(--primary)", fontWeight: 700 }}>#{c.metadata?.chunkIndex}</span>{" "}
                          {c.text.substring(0, 200)}{c.text.length > 200 ? "…" : ""}
                        </div>
                      ))}
                    </div>
                  </details>
                );
              })}
            </div>
          </div>
        )}

        {/* ═══════ ANALYTICS ═══════ */}
        {view === VIEWS.ANALYTICS && (
          <div style={{ animation: "fadeUp 0.4s ease" }}>
            <div style={S.viewHead}>
              <div>
                <h2 style={S.viewTitle}>Analytics</h2>
                <p style={S.viewDesc}>{queryLog.length} queries logged · {documents.length} documents</p>
              </div>
            </div>

            {/* KPI Row */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 10, marginBottom: 14 }}>
              {[
                { label: "Documents", value: documents.length, color: "var(--primary)" },
                { label: "Total Chunks", value: totalChunks, color: "var(--accent)" },
                { label: "Queries", value: queryLog.length, color: "var(--accent2)" },
                {
                  label: "Avg Top Score",
                  value: queryStats ? (queryStats.avgScore * 100).toFixed(1) + "%" : "—",
                  color: "var(--warn)",
                },
              ].map((kpi) => (
                <div key={kpi.label} style={{ ...S.card, textAlign: "center", marginBottom: 0 }}>
                  <div style={{ fontSize: 22, fontWeight: 800, color: kpi.color, fontFamily: "'IBM Plex Mono', monospace" }}>{kpi.value}</div>
                  <div style={{ fontSize: 10, color: "var(--text)", marginTop: 2, textTransform: "uppercase", letterSpacing: "0.05em" }}>{kpi.label}</div>
                </div>
              ))}
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
              {/* Chunks per document */}
              <div style={S.card}>
                <h3 style={S.cardTitle}>Chunks per Document</h3>
                {DEBUG_DISABLE_RECHARTS ? (
                  <div style={{ height: 160, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 12, color: "var(--text)" }}>
                    Charts disabled (DEBUG_DISABLE_RECHARTS)
                  </div>
                ) : (
                  <ResponsiveContainer width="100%" height={160}>
                    <BarChart data={chunkDistribution}>
                      <XAxis dataKey="name" tick={{ fontSize: 9, fill: "#8B92AB" }} />
                      <YAxis tick={{ fontSize: 9, fill: "#8B92AB" }} />
                      <Tooltip contentStyle={{ background: "#0F1118", border: "1px solid #252940", borderRadius: 6, fontSize: 11 }} />
                      <Bar dataKey="chunks" fill="#7C5CFC" radius={[2, 2, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                )}
              </div>

              {/* Confidence distribution */}
              <div style={S.card}>
                <h3 style={S.cardTitle}>Confidence Distribution</h3>
                {queryStats && queryStats.total > 0 ? (
                  DEBUG_DISABLE_RECHARTS ? (
                    <div style={{ height: 160, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 12, color: "var(--text)" }}>
                      Charts disabled (DEBUG_DISABLE_RECHARTS)
                    </div>
                  ) : (
                    <ResponsiveContainer width="100%" height={160}>
                      <PieChart>
                        <Pie
                          data={[
                            { name: "High", value: queryStats.confCounts.high },
                            { name: "Medium", value: queryStats.confCounts.medium },
                            { name: "Low", value: queryStats.confCounts.low },
                          ]}
                          cx="50%" cy="50%" outerRadius={60}
                          dataKey="value"
                          label={({ name, percent }: { name?: string; percent?: number }) =>
                            `${name ?? ""} ${percent != null ? (percent * 100).toFixed(0) : "0"}%`
                          }
                          labelLine={false}
                        >
                          {["#06D6A0", "#FFD166", "#EF476F"].map((c, i) => (
                            <Cell key={i} fill={c} />
                          ))}
                        </Pie>
                        <Tooltip contentStyle={{ background: "#0F1118", border: "1px solid #252940", borderRadius: 6, fontSize: 11 }} />
                      </PieChart>
                    </ResponsiveContainer>
                  )
                ) : (
                  <div style={{ height: 160, display: "flex", alignItems: "center", justifyContent: "center" }}>
                    <p style={{ fontSize: 12, color: "var(--text)" }}>No queries yet</p>
                  </div>
                )}
              </div>

              {/* Retrieval scores over time */}
              {queryLog.length > 0 && (
                <div style={{ ...S.card, gridColumn: "1 / -1" }}>
                  <h3 style={S.cardTitle}>Retrieval Score per Query</h3>
                  {DEBUG_DISABLE_RECHARTS ? (
                    <div style={{ height: 120, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 12, color: "var(--text)" }}>
                      Charts disabled (DEBUG_DISABLE_RECHARTS)
                    </div>
                  ) : (
                    <ResponsiveContainer width="100%" height={120}>
                      <LineChart data={queryLog.map((q, i) => ({ idx: i + 1, score: parseFloat((q.topScore * 100).toFixed(1)), mode: q.searchMode }))}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1F2333" />
                        <XAxis dataKey="idx" tick={{ fontSize: 9, fill: "#8B92AB" }} label={{ value: "Query #", position: "insideBottom", offset: -2, fontSize: 9, fill: "#8B92AB" }} />
                        <YAxis tick={{ fontSize: 9, fill: "#8B92AB" }} domain={[0, 100]} unit="%" />
                        <Tooltip
                          contentStyle={{ background: "#0F1118", border: "1px solid #252940", borderRadius: 6, fontSize: 11 }}
                          formatter={(v) => [v + "%", "Top Score"]}
                        />
                        <Line type="monotone" dataKey="score" stroke="#06D6A0" strokeWidth={2} dot={{ r: 3, fill: "#06D6A0" }} />
                      </LineChart>
                    </ResponsiveContainer>
                  )}
                </div>
              )}
            </div>

            {/* Query Log */}
            {queryLog.length > 0 && (
              <div style={{ ...S.card, marginTop: 12 }}>
                <h3 style={S.cardTitle}>Query Log</h3>
                <div style={{ overflowX: "auto" }}>
                  <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
                    <thead>
                      <tr>
                        {["#", "Question", "Mode", "Confidence", "Top Score", "Sources", "Time"].map((h) => (
                          <th key={h} style={{ padding: "6px 8px", textAlign: "left", color: "var(--text)", fontWeight: 700, fontSize: 10, textTransform: "uppercase", fontFamily: "'IBM Plex Mono', monospace", borderBottom: "1px solid var(--border)" }}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {queryLog.map((q, i) => (
                        <tr key={i} style={{ borderBottom: "1px solid var(--border)" + "55" }}>
                          <td style={{ padding: "6px 8px", color: "var(--text)", fontFamily: "'IBM Plex Mono', monospace" }}>{i + 1}</td>
                          <td style={{ padding: "6px 8px", color: "var(--text-bright)", maxWidth: 240, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{q.question}</td>
                          <td style={{ padding: "6px 8px" }}><SearchModeBadge mode={q.searchMode} /></td>
                          <td style={{ padding: "6px 8px" }}><ConfBadge level={q.confidence} /></td>
                          <td style={{ padding: "6px 8px", fontFamily: "'IBM Plex Mono', monospace", color: "var(--accent)" }}>{(q.topScore * 100).toFixed(1)}%</td>
                          <td style={{ padding: "6px 8px", fontFamily: "'IBM Plex Mono', monospace", color: "var(--text)" }}>{q.sourcesUsed}</td>
                          <td style={{ padding: "6px 8px", color: "var(--text)", fontSize: 10 }}>{q.timestamp}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ═══════ PIPELINE CONFIG ═══════ */}
        {view === VIEWS.PIPELINE && (
          <div style={{ animation: "fadeUp 0.4s ease" }}>
            <div style={S.viewHead}>
              <div>
                <h2 style={S.viewTitle}>Pipeline Config</h2>
                <p style={S.viewDesc}>Tune chunking, retrieval, and search strategy</p>
              </div>
            </div>

            {documentsLoadErrorCard}

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
              <div style={S.card}>
                <h3 style={S.cardTitle}>✂️ Chunking</h3>
                <div style={{ marginBottom: 14 }}>
                  <label style={S.cfgLabel}>Chunk Size (chars)</label>
                  <input type="range" min="200" max="1500" step="50" value={chunkSize}
                    onChange={(e) => setChunkSize(Number(e.target.value))} style={{ width: "100%", accentColor: "var(--primary)" }} />
                  <div style={S.cfgVal}>{chunkSize}</div>
                </div>
                <div>
                  <label style={S.cfgLabel}>Overlap (chars)</label>
                  <input type="range" min="0" max="300" step="25" value={chunkOverlap}
                    onChange={(e) => setChunkOverlap(Number(e.target.value))} style={{ width: "100%", accentColor: "var(--primary)" }} />
                  <div style={S.cfgVal}>{chunkOverlap}</div>
                </div>
                <div style={{ marginTop: 10, padding: "8px", background: "var(--surface2)", borderRadius: 4, fontSize: 10, color: "var(--text)", lineHeight: 1.5 }}>
                  Smaller chunks = precise retrieval, less context. Larger = more context, may dilute relevance.
                </div>
              </div>

              <div style={S.card}>
                <h3 style={S.cardTitle}>🔍 Retrieval</h3>
                <div style={{ marginBottom: 14 }}>
                  <label style={S.cfgLabel}>Top-K Results</label>
                  <input type="range" min="1" max="15" step="1" value={topK}
                    onChange={(e) => setTopK(Number(e.target.value))} style={{ width: "100%", accentColor: "var(--primary)" }} />
                  <div style={S.cfgVal}>{topK}</div>
                </div>
                <div style={{ marginBottom: 14 }}>
                  <label style={S.cfgLabel}>Vector Similarity Threshold</label>
                  <input type="range" min="0" max="0.8" step="0.05" value={similarityThreshold}
                    onChange={(e) => setSimilarityThreshold(Number(e.target.value))} style={{ width: "100%", accentColor: "var(--accent)" }} />
                  <div style={S.cfgVal}>{similarityThreshold.toFixed(2)}</div>
                </div>
                <div style={{ padding: "8px", background: "var(--surface2)", borderRadius: 4, fontSize: 10, color: "var(--text)", lineHeight: 1.5 }}>
                  Applied to vector score. Chunks below threshold excluded even in hybrid mode.
                </div>
              </div>

              {/* NEW: Search Strategy card */}
              <div style={{ ...S.card, gridColumn: "1 / -1", borderColor: "var(--accent2)" + "33" }}>
                <h3 style={{ ...S.cardTitle, color: "var(--accent2)" }}>⚡ Search Strategy</h3>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                  {[
                    {
                      label: "Hybrid (BM25 + Vector)",
                      value: true,
                      desc: "Reciprocal Rank Fusion combines BM25 keyword ranking with semantic vector similarity. Best for most queries — catches exact terms AND semantic meaning.",
                      best: "Recommended for production",
                    },
                    {
                      label: "Vector Only (Cosine)",
                      value: false,
                      desc: "Pure semantic search using cosine similarity on 768-dim embeddings. Better for highly conceptual queries where exact keyword matches don't matter.",
                      best: "Pure semantic queries",
                    },
                  ].map((opt) => (
                    <div
                      key={opt.label}
                      onClick={() => setUseHybridSearch(opt.value)}
                      style={{
                        padding: 12, borderRadius: 6, cursor: "pointer",
                        border: `1px solid ${useHybridSearch === opt.value ? "var(--accent2)" : "var(--border)"}`,
                        background: useHybridSearch === opt.value ? "var(--accent2)" + "08" : "var(--surface2)",
                        transition: "all 0.2s ease",
                      }}
                    >
                      <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 6 }}>
                        <div style={{
                          width: 12, height: 12, borderRadius: "50%",
                          border: `2px solid ${useHybridSearch === opt.value ? "var(--accent2)" : "var(--border2)"}`,
                          background: useHybridSearch === opt.value ? "var(--accent2)" : "transparent",
                          flexShrink: 0,
                        }} />
                        <span style={{ fontSize: 12, fontWeight: 700, color: "var(--text-bright)" }}>{opt.label}</span>
                      </div>
                      <p style={{ fontSize: 11, color: "var(--text)", lineHeight: 1.5, marginBottom: 6 }}>{opt.desc}</p>
                      <span style={{ fontSize: 10, fontFamily: "'IBM Plex Mono', monospace", color: "var(--accent2)" }}>{opt.best}</span>
                    </div>
                  ))}
                </div>
                {/* RRF explanation */}
                <div style={{ marginTop: 10, padding: "10px 12px", background: "var(--surface2)", borderRadius: 4, fontSize: 10, color: "var(--text)", lineHeight: 1.6, fontFamily: "'IBM Plex Mono', monospace" }}>
                  <span style={{ color: "var(--accent2)", fontWeight: 700 }}>RRF formula:</span>{" "}
                  score(d) = Σ 1/(k + rank_i(d)) where k=60 — ranks are fused across BM25 and vector rankings.
                  This is the same approach used by Elasticsearch&apos;s hybrid search.
                </div>
              </div>

              <div style={S.card}>
                <h3 style={S.cardTitle}>🧮 Embeddings</h3>
                {[
                  ["Endpoint", "POST /api/embed", "var(--primary)"],
                  ["Dimensions", "768", "var(--accent)"],
                  ["Batching", "Client → server", "var(--accent2)"],
                  ["Provider", "Your backend", "var(--warn)"],
                ].map(([k, v, c]) => (
                  <div key={k} style={{ display: "flex", justifyContent: "space-between", padding: "5px 0", borderBottom: "1px solid var(--border)", fontSize: 12 }}>
                    <span style={{ color: "var(--text)" }}>{k}</span>
                    <span style={{ color: c, fontFamily: "'IBM Plex Mono', monospace", fontWeight: 600 }}>{v}</span>
                  </div>
                ))}
              </div>

              <div style={S.card}>
                <h3 style={S.cardTitle}>✨ Generation</h3>
                {[
                  ["Endpoint", "POST /api/chat", "var(--primary)"],
                  ["Body", "{ prompt }", "var(--accent)"],
                  ["UI", "Simulated stream", "var(--accent2)"],
                  ["Secrets", "Server only", "var(--warn)"],
                  ["Features", "Citations, Entities, Follow-ups", "var(--text)"],
                ].map(([k, v, c]) => (
                  <div key={k} style={{ display: "flex", justifyContent: "space-between", padding: "5px 0", borderBottom: "1px solid var(--border)", fontSize: 12 }}>
                    <span style={{ color: "var(--text)" }}>{k}</span>
                    <span style={{ color: c, fontFamily: "'IBM Plex Mono', monospace", fontWeight: 600 }}>{v}</span>
                  </div>
                ))}
              </div>

              <div style={S.card}>
                <h3 style={S.cardTitle}>🗄️ Persistence</h3>
                {[
                  ["Storage", "Supabase", "var(--primary)"],
                  ["Scope", "Browser origin", "var(--accent)"],
                  ["Documents", `${documents.length} saved`, "var(--accent2)"],
                  ["Vectors", `${totalChunks} embeddings`, "var(--warn)"],
                  ["Status", dbSynced ? "✓ Synced" : "Not saved", dbSynced ? "var(--accent)" : "var(--text)"],
                ].map(([k, v, c]) => (
                  <div key={k} style={{ display: "flex", justifyContent: "space-between", padding: "5px 0", borderBottom: "1px solid var(--border)", fontSize: 12 }}>
                    <span style={{ color: "var(--text)" }}>{k}</span>
                    <span style={{ color: c, fontFamily: "'IBM Plex Mono', monospace", fontWeight: 600 }}>{v}</span>
                  </div>
                ))}
                <button onClick={clearAll} className="hov" style={{ ...S.btnDanger, width: "100%", marginTop: 10, textAlign: "center" }}>
                  Clear All + Wipe Supabase
                </button>
              </div>

              <div style={{ ...S.card, gridColumn: "1 / -1", borderColor: "var(--primary)" + "33" }}>
                <h3 style={S.cardTitle}>📐 Pipeline Flow v2</h3>
                <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, color: "var(--text)", lineHeight: 2.4, padding: "8px 0" }}>
                  <span style={{ color: "var(--accent2)" }}>PDF/TXT/MD/CSV</span>{" → "}
                  <span style={{ color: "var(--primary)" }}>Sentence Chunker</span>{" → "}
                  <span style={{ color: "var(--accent)" }}>Batch Embeddings (768d)</span>{" → "}
                  <span style={{ color: "var(--warn)" }}>Vector Store + BM25</span>{" → "}
                  <span style={{ color: "var(--accent2)" }}>Supabase</span>{" → "}
                  <span style={{ color: "var(--danger)" }}>Hybrid Search (RRF)</span>{" → "}
                  <span style={{ color: "var(--primary)" }}>Threshold Filter</span>{" → "}
                  <span style={{ color: "var(--accent)" }}>Context Assembly</span>{" → "}
                  <span style={{ color: "var(--accent2)" }}>Streaming Generation</span>
                </div>
              </div>
            </div>
          </div>
        )}

        <div style={S.footer}>
          <p>Built by <strong>Prasanna Warad</strong> · RAGBase v2 · Backend /api/embed + /api/chat + /api/retrieve · Hybrid BM25+Vector · Supabase pgvector</p>
        </div>
      </div>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════
const S: Record<string, CSSProperties> = {
  wrapper: { minHeight: "100vh", background: "var(--bg)", fontFamily: "'Plus Jakarta Sans', sans-serif", color: "var(--text)", display: "flex" },
  sidebar: { width: 216, flexShrink: 0, background: "var(--surface)", borderRight: "1px solid var(--border)", display: "flex", flexDirection: "column", position: "sticky", top: 0, height: "100vh" },
  sidebarTop: { padding: "16px 14px", borderBottom: "1px solid var(--border)" },
  logoRow: { display: "flex", alignItems: "center", gap: 8 },
  logoGem: { width: 30, height: 30, borderRadius: 7, background: "linear-gradient(135deg, var(--primary), var(--primary2))", display: "flex", alignItems: "center", justifyContent: "center", color: "white", fontWeight: 800, fontSize: 14, fontFamily: "'IBM Plex Mono', monospace" },
  logoTitle: { fontFamily: "'IBM Plex Mono', monospace", fontSize: 14, fontWeight: 800, color: "var(--text-bright)" },
  logoSub: { fontSize: 9, color: "var(--text)" },
  sidebarNav: { padding: "10px 6px", flex: 1, overflowY: "auto" },
  navItem: { width: "100%", display: "flex", alignItems: "center", gap: 8, padding: "8px 10px", borderRadius: 5, background: "transparent", border: "none", fontFamily: "'Plus Jakarta Sans', sans-serif", fontWeight: 600, color: "var(--text-bright)", cursor: "pointer", marginBottom: 1, textAlign: "left" },
  badge: { fontSize: 9, fontWeight: 700, padding: "1px 5px", borderRadius: 8, background: "var(--primary)" + "22", color: "var(--primary)", fontFamily: "'IBM Plex Mono', monospace" },
  sidebarStats: { padding: "10px 14px", borderTop: "1px solid var(--border)", display: "flex", justifyContent: "space-between" },
  statItem: { display: "flex", flexDirection: "column", alignItems: "center", gap: 2 },
  sidebarFoot: { padding: "10px 6px", borderTop: "1px solid var(--border)" },
  main: { flex: 1, padding: "18px 24px", minWidth: 0, overflowY: "auto" },
  input: { width: "100%", padding: "8px 12px", background: "var(--surface2)", border: "1px solid var(--border)", borderRadius: 5, color: "var(--text-bright)", fontSize: 12, fontFamily: "'Plus Jakarta Sans', sans-serif" },
  selectInput: { padding: "8px 10px", background: "var(--surface2)", border: "1px solid var(--border)", borderRadius: 5, color: "var(--text-bright)", fontSize: 12, fontFamily: "'IBM Plex Mono', monospace", minWidth: 140 },
  btnPri: { padding: "8px 18px", background: "linear-gradient(135deg, var(--primary), var(--primary2))", border: "none", borderRadius: 5, color: "white", fontSize: 12, fontWeight: 700, fontFamily: "'Plus Jakarta Sans', sans-serif", cursor: "pointer", whiteSpace: "nowrap" },
  btnOut: { padding: "8px 16px", background: "transparent", border: "1px solid var(--border)", borderRadius: 5, color: "var(--text-bright)", fontSize: 12, fontWeight: 600, fontFamily: "'Plus Jakarta Sans', sans-serif", cursor: "pointer" },
  btnDanger: { padding: "8px 14px", background: "var(--danger)" + "18", border: "1px solid var(--danger)" + "33", borderRadius: 5, color: "var(--danger)", fontSize: 11, fontWeight: 700, fontFamily: "'Plus Jakarta Sans', sans-serif", cursor: "pointer" },
  errorBar: { display: "flex", alignItems: "center", gap: 8, background: "var(--danger)" + "10", border: "1px solid var(--danger)" + "33", borderRadius: 6, padding: "8px 14px", marginBottom: 12, fontSize: 12, color: "var(--danger)" },
  hero: { fontFamily: "'IBM Plex Mono', monospace", fontSize: 30, fontWeight: 800, color: "var(--text-bright)", lineHeight: 1.2, marginBottom: 10 },
  heroSub: { fontSize: 12, color: "var(--text)", maxWidth: 520, margin: "0 auto", lineHeight: 1.7 },
  dropZone: { border: "2px dashed var(--border2)", borderRadius: 10, padding: "40px 20px", textAlign: "center", cursor: "pointer", marginBottom: 12 },
  card: { background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 8, padding: "16px 18px", marginBottom: 10 },
  cardTitle: { fontSize: 11, fontWeight: 700, color: "var(--primary)", textTransform: "uppercase", letterSpacing: "0.05em", fontFamily: "'IBM Plex Mono', monospace", marginBottom: 12 },
  viewHead: { display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 16, flexWrap: "wrap", gap: 10 },
  viewTitle: { fontFamily: "'IBM Plex Mono', monospace", fontSize: 18, fontWeight: 800, color: "var(--text-bright)", marginBottom: 3 },
  viewDesc: { fontSize: 12, color: "var(--text)" },
  progressTrack: { height: 4, background: "var(--surface3)", borderRadius: 2, overflow: "hidden", width: "100%" },
  progressFill: { height: "100%", borderRadius: 2, background: "linear-gradient(90deg, var(--primary), var(--accent))", transition: "width 0.5s ease", animation: "progressPulse 1.5s ease infinite" },
  cfgLabel: { display: "block", fontSize: 10, fontWeight: 700, color: "var(--text)", textTransform: "uppercase", letterSpacing: "0.05em", fontFamily: "'IBM Plex Mono', monospace", marginBottom: 4 },
  cfgVal: { textAlign: "right", fontSize: 13, fontWeight: 800, color: "var(--primary)", fontFamily: "'IBM Plex Mono', monospace", marginTop: 2 },
  footer: { textAlign: "center", marginTop: 28, padding: "12px", fontSize: 10, color: "#2D3346", borderTop: "1px solid var(--border)", fontFamily: "'IBM Plex Mono', monospace" },
};
