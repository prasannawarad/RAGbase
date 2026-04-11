import { useState, useRef, useEffect, useCallback, useMemo } from "react";

// ══════════════════════════════════════════════════════════════
// RAGBase v2 — Document Intelligence & RAG Platform
// ── New in v2 ──────────────────────────────────────────────
//   ✦ PDF support via pdf.js (dynamic CDN load)
//   ✦ Hybrid search: BM25 + Vector via Reciprocal Rank Fusion
//   ✦ IndexedDB persistence (KB survives page refresh)
//   ✦ Streaming generation (answer text streams live)
// ══════════════════════════════════════════════════════════════

import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line, CartesianGrid, Legend,
} from "recharts";

// ─── Text Utilities ──────────────────────────────────────────
function tokenize(text) {
  // Lowercase, strip punctuation, split on whitespace, filter short tokens
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((t) => t.length > 1);
}

function chunkText(text, { chunkSize = 500, overlap = 100 } = {}) {
  const sentences = text.match(/[^.!?\n]+[.!?\n]+|[^.!?\n]+$/g) || [text];
  const chunks = [];
  let current = "";
  let currentSentences = [];
  for (const sentence of sentences) {
    const trimmed = sentence.trim();
    if (!trimmed) continue;
    if ((current + " " + trimmed).length > chunkSize && current.length > 0) {
      chunks.push({ text: current.trim(), sentences: [...currentSentences] });
      const overlapText = currentSentences.slice(-2).join(" ");
      if (overlapText.length <= overlap) {
        current = overlapText + " " + trimmed;
        currentSentences = [...currentSentences.slice(-2), trimmed];
      } else {
        current = trimmed;
        currentSentences = [trimmed];
      }
    } else {
      current = current ? current + " " + trimmed : trimmed;
      currentSentences.push(trimmed);
    }
  }
  if (current.trim()) chunks.push({ text: current.trim(), sentences: currentSentences });
  return chunks.map((c, i) => ({
    id: `chunk_${i}`,
    text: c.text,
    index: i,
    charStart: text.indexOf(c.text.substring(0, 30)),
    length: c.text.length,
    wordCount: c.text.split(/\s+/).length,
  }));
}

// ─── BM25 ─────────────────────────────────────────────────────
// Okapi BM25: state-of-the-art keyword ranking used in Elasticsearch,
// Solr, and most production search engines. Combined with vector search
// via Reciprocal Rank Fusion for hybrid retrieval.
class BM25 {
  constructor(k1 = 1.5, b = 0.75) {
    this.k1 = k1; // term frequency saturation (1.2–2.0 typical)
    this.b = b;   // document length normalization (0.75 typical)
    this.docs = [];
    this.avgDL = 0;
    this.idf = {};
    this.tf = [];
  }

  build(docs) {
    this.docs = docs;
    if (!docs.length) return;
    this.avgDL = docs.reduce((s, d) => s + (d.tokens?.length || 0), 0) / docs.length;
    const N = docs.length;
    // Document frequency per term
    const df = {};
    docs.forEach((doc) => {
      const seen = new Set();
      (doc.tokens || []).forEach((t) => {
        if (!seen.has(t)) { df[t] = (df[t] || 0) + 1; seen.add(t); }
      });
    });
    // IDF with smoothing: log((N - df + 0.5) / (df + 0.5) + 1)
    this.idf = Object.fromEntries(
      Object.entries(df).map(([t, n]) => [t, Math.log((N - n + 0.5) / (n + 0.5) + 1)])
    );
    // Term frequencies per document
    this.tf = docs.map((doc) => {
      const freq = {};
      (doc.tokens || []).forEach((t) => { freq[t] = (freq[t] || 0) + 1; });
      return freq;
    });
  }

  scoreDoc(queryTokens, docIdx) {
    const tf = this.tf[docIdx];
    if (!tf) return 0;
    const dl = this.docs[docIdx]?.tokens?.length || 0;
    return queryTokens.reduce((s, t) => {
      if (!tf[t]) return s;
      const idf = this.idf[t] || 0;
      const tfNorm =
        (tf[t] * (this.k1 + 1)) /
        (tf[t] + this.k1 * (1 - this.b + (this.b * dl) / (this.avgDL || 1)));
      return s + idf * tfNorm;
    }, 0);
  }

  search(query, topK = 10) {
    const qt = tokenize(query);
    return this.docs
      .map((_, i) => ({ idx: i, score: this.scoreDoc(qt, i) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);
  }
}

// ─── Vector Store + Hybrid Search ────────────────────────────
function cosineSimilarity(a, b) {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i]; normA += a[i] * a[i]; normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB) + 1e-8);
}

class VectorStore {
  constructor() {
    this.documents = [];
    this.bm25 = new BM25();
  }

  add(id, text, embedding, metadata = {}) {
    this.documents.push({ id, text, embedding, metadata, tokens: tokenize(text) });
    this._rebuildBM25();
  }

  // Efficient batch insert: rebuilds BM25 only once at the end
  addBatch(entries) {
    entries.forEach(({ id, text, embedding, metadata }) => {
      this.documents.push({ id, text, embedding, metadata, tokens: tokenize(text) });
    });
    this._rebuildBM25();
  }

  _rebuildBM25() {
    this.bm25.build(this.documents);
  }

  // Pure vector search (cosine similarity)
  search(queryEmbedding, topK = 5, filter = null) {
    let docs = filter ? this.documents.filter(filter) : this.documents;
    return docs
      .map((doc) => ({ ...doc, score: cosineSimilarity(queryEmbedding, doc.embedding) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);
  }

  // Hybrid search: BM25 + Vector fused via Reciprocal Rank Fusion
  // RRF(d) = Σ 1/(k + rank_i(d))  — k=60 is standard default
  hybridSearch(query, queryEmbedding, topK = 5, filter = null, k = 60) {
    let docs = filter ? this.documents.filter(filter) : this.documents;
    if (!docs.length) return [];

    // Vector ranking
    const vectorRanked = docs
      .map((doc, i) => ({ i, vectorScore: cosineSimilarity(queryEmbedding, doc.embedding) }))
      .sort((a, b) => b.vectorScore - a.vectorScore);

    // BM25 ranking — score against each filtered doc directly
    const qt = tokenize(query);
    const bm25Ranked = docs
      .map((doc, i) => ({
        i,
        bm25Score: this.bm25.scoreDoc(qt, this.documents.indexOf(doc)),
      }))
      .sort((a, b) => b.bm25Score - a.bm25Score);

    // Compute RRF scores
    const rrfScores = {};
    vectorRanked.forEach(({ i }, rank) => {
      rrfScores[i] = (rrfScores[i] || 0) + 1 / (k + rank + 1);
    });
    bm25Ranked.forEach(({ i }, rank) => {
      rrfScores[i] = (rrfScores[i] || 0) + 1 / (k + rank + 1);
    });

    // Build results with both scores for display
    const vectorScoreMap = Object.fromEntries(vectorRanked.map(({ i, vectorScore }) => [i, vectorScore]));
    const bm25ScoreMap = Object.fromEntries(bm25Ranked.map(({ i, bm25Score }) => [i, bm25Score]));

    return docs
      .map((doc, i) => ({
        ...doc,
        score: rrfScores[i] || 0,
        vectorScore: vectorScoreMap[i] || 0,
        bm25Score: bm25ScoreMap[i] || 0,
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);
  }

  getByDocId(docId) { return this.documents.filter((d) => d.metadata.docId === docId); }
  clear() { this.documents = []; this._rebuildBM25(); }
  removeByDocId(docId) {
    this.documents = this.documents.filter((d) => d.metadata.docId !== docId);
    this._rebuildBM25();
  }
  get size() { return this.documents.length; }
  getStats() {
    const byDoc = {};
    this.documents.forEach((d) => {
      const name = d.metadata.docName || "Unknown";
      if (!byDoc[name]) byDoc[name] = { count: 0, totalLen: 0 };
      byDoc[name].count++;
      byDoc[name].totalLen += d.text.length;
    });
    return byDoc;
  }
}

// ─── PDF Extraction ───────────────────────────────────────────
// Dynamically loads pdf.js from CDN so we don't need a build step.
// Caches the loaded library in window.pdfjsLib after first load.
async function loadPDFJS() {
  if (window.pdfjsLib) return window.pdfjsLib;
  return new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js";
    script.onload = () => {
      window.pdfjsLib.GlobalWorkerOptions.workerSrc =
        "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";
      resolve(window.pdfjsLib);
    };
    script.onerror = () => reject(new Error("Failed to load PDF.js from CDN"));
    document.head.appendChild(script);
  });
}

async function extractPDFText(file) {
  const pdfjsLib = await loadPDFJS();
  const arrayBuffer = await file.arrayBuffer();
  const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
  const pageTexts = [];
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

// ─── IndexedDB Persistence ────────────────────────────────────
// Persists documents metadata + raw vector embeddings across sessions.
// DB schema: 'documents' store (doc metadata) + 'vectors' store (embeddings).
// Tokens are NOT persisted — regenerated on load from text (saves space).
const DB_NAME = "ragbase_v2";
const DB_VERSION = 1;

function openDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = (e) => {
      const db = e.target.result;
      if (!db.objectStoreNames.contains("documents"))
        db.createObjectStore("documents", { keyPath: "id" });
      if (!db.objectStoreNames.contains("vectors"))
        db.createObjectStore("vectors", { keyPath: "id" });
    };
    req.onsuccess = (e) => resolve(e.target.result);
    req.onerror = () => reject(req.error);
  });
}

async function dbSaveDocuments(docs) {
  const db = await openDB();
  const tx = db.transaction("documents", "readwrite");
  const store = tx.objectStore("documents");
  // Save only serializable metadata (no tokens, no text preview beyond what's in doc)
  docs.forEach((doc) => store.put(doc));
  return new Promise((res, rej) => { tx.oncomplete = res; tx.onerror = rej; });
}

async function dbLoadDocuments() {
  const db = await openDB();
  return new Promise((res, rej) => {
    const req = db.transaction("documents", "readonly").objectStore("documents").getAll();
    req.onsuccess = () => res(req.result || []);
    req.onerror = () => rej(req.error);
  });
}

async function dbSaveVectors(vectors) {
  const db = await openDB();
  const tx = db.transaction("vectors", "readwrite");
  const store = tx.objectStore("vectors");
  // Strip tokens before storing — they'll be recomputed on load
  vectors.forEach(({ id, text, embedding, metadata }) =>
    store.put({ id, text, embedding, metadata })
  );
  return new Promise((res, rej) => { tx.oncomplete = res; tx.onerror = rej; });
}

async function dbLoadVectors() {
  const db = await openDB();
  return new Promise((res, rej) => {
    const req = db.transaction("vectors", "readonly").objectStore("vectors").getAll();
    req.onsuccess = () => res(req.result || []);
    req.onerror = () => rej(req.error);
  });
}

async function dbDeleteDoc(docId) {
  const db = await openDB();
  // Delete doc metadata
  await new Promise((res, rej) => {
    const tx = db.transaction("documents", "readwrite");
    tx.objectStore("documents").delete(docId);
    tx.oncomplete = res; tx.onerror = rej;
  });
  // Delete associated vectors
  const db2 = await openDB();
  await new Promise((res, rej) => {
    const tx = db2.transaction("vectors", "readwrite");
    const store = tx.objectStore("vectors");
    const req = store.getAll();
    req.onsuccess = () => {
      req.result
        .filter((v) => v.metadata?.docId === docId)
        .forEach((v) => store.delete(v.id));
    };
    tx.oncomplete = res; tx.onerror = rej;
  });
}

async function dbClearAll() {
  const db = await openDB();
  return new Promise((res, rej) => {
    const tx = db.transaction(["documents", "vectors"], "readwrite");
    tx.objectStore("documents").clear();
    tx.objectStore("vectors").clear();
    tx.oncomplete = res; tx.onerror = rej;
  });
}

// ─── Gemini APIs ─────────────────────────────────────────────
async function getEmbeddings(apiKey, texts) {
  const results = [];
  const batchSize = 20;
  for (let i = 0; i < texts.length; i += batchSize) {
    const batch = texts.slice(i, i + batchSize);
    const requests = batch.map((text) => ({
      model: "models/text-embedding-004",
      content: { parts: [{ text }] },
    }));
    const res = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:batchEmbedContents?key=${apiKey}`,
      { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ requests }) }
    );
    const data = await res.json();
    if (data.error) throw new Error(data.error.message || "Embedding API error");
    results.push(...(data.embeddings?.map((e) => e.values) || []));
  }
  return results;
}

async function callGemini(apiKey, prompt, temp = 0.3) {
  const res = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${apiKey}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        contents: [{ parts: [{ text: prompt }] }],
        generationConfig: { temperature: temp, maxOutputTokens: 3000 },
      }),
    }
  );
  const data = await res.json();
  if (data.error) throw new Error(data.error.message || "API error");
  return data?.candidates?.[0]?.content?.parts?.[0]?.text || "";
}

async function callGeminiJSON(apiKey, prompt, temp = 0.3) {
  const raw = await callGemini(apiKey, prompt, temp);
  return JSON.parse(raw.replace(/```json|```/g, "").trim());
}

// Streaming generation via Server-Sent Events.
// The prompt format:
//   - Write the answer naturally first (visible to user during stream)
//   - Append metadata in a <metadata>{...json...}</metadata> block at the end
// During streaming: answer text is shown live, metadata block is hidden until parsed.
async function callGeminiStream(apiKey, prompt, onChunk, temp = 0.3) {
  const res = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:streamGenerateContent?key=${apiKey}&alt=sse`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        contents: [{ parts: [{ text: prompt }] }],
        generationConfig: { temperature: temp, maxOutputTokens: 3000 },
      }),
    }
  );
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.error?.message || "Streaming API error");
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let fullText = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const lines = decoder.decode(value, { stream: true }).split("\n");
    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      try {
        const data = JSON.parse(line.slice(6));
        const chunk = data?.candidates?.[0]?.content?.parts?.[0]?.text || "";
        if (chunk) {
          fullText += chunk;
          onChunk(fullText);
        }
      } catch (_) { /* partial SSE frames — safe to ignore */ }
    }
  }
  return fullText;
}

// Parse streamed response into answer + metadata
function parseStreamedResponse(fullText) {
  const metaMatch = fullText.match(/<metadata>([\s\S]*?)<\/metadata>/);
  const answer = fullText.replace(/<metadata>[\s\S]*?<\/metadata>/, "").trim();
  let metadata = {};
  if (metaMatch) {
    try { metadata = JSON.parse(metaMatch[1].trim()); } catch (_) {}
  }
  return { answer, metadata };
}

// ─── Constants ───────────────────────────────────────────────
const VIEWS = {
  UPLOAD: "upload", DOCUMENTS: "documents", CHAT: "chat",
  CHUNKS: "chunks", ANALYTICS: "analytics", PIPELINE: "pipeline",
};
const CHART_COLORS = [
  "#7C5CFC","#06D6A0","#3EDBF0","#FFD166","#EF476F",
  "#FF8C42","#A78BFA","#34D399","#F472B6","#60A5FA",
];

// ══════════════════════════════════════════════════════════════
export default function RAGBase() {
  // ─── Core State ─────────────────────────────────────────────
  const [apiKey, setApiKey] = useState("");
  const [keySaved, setKeySaved] = useState(false);
  const [showKeyPanel, setShowKeyPanel] = useState(true);
  const [view, setView] = useState(VIEWS.UPLOAD);
  const [documents, setDocuments] = useState([]);
  const [vectorStore] = useState(() => new VectorStore());
  const [totalChunks, setTotalChunks] = useState(0);

  // ─── Persistence State ──────────────────────────────────────
  const [dbLoading, setDbLoading] = useState(true);  // loading from IndexedDB on mount
  const [dbSynced, setDbSynced] = useState(false);   // at least one successful save

  // ─── Processing ─────────────────────────────────────────────
  const [processing, setProcessing] = useState(false);
  const [processStep, setProcessStep] = useState("");
  const [processProgress, setProcessProgress] = useState(0);
  const [processLog, setProcessLog] = useState([]);

  // ─── Chat ───────────────────────────────────────────────────
  const [chatMessages, setChatMessages] = useState([]);
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const [lastRetrieved, setLastRetrieved] = useState([]);
  const [showRetrieval, setShowRetrieval] = useState(false);

  // ─── Chunk Inspector ────────────────────────────────────────
  const [selectedDocForChunks, setSelectedDocForChunks] = useState(null);
  const [chunkSearchQuery, setChunkSearchQuery] = useState("");
  const [chunkSearchResults, setChunkSearchResults] = useState(null);
  const [chunkSearchLoading, setChunkSearchLoading] = useState(false);

  // ─── Analytics / Summaries ──────────────────────────────────
  const [docSummaries, setDocSummaries] = useState({});
  const [analyticsLoading, setAnalyticsLoading] = useState(false);
  const [queryLog, setQueryLog] = useState([]);

  // ─── Settings ───────────────────────────────────────────────
  const [chunkSize, setChunkSize] = useState(500);
  const [chunkOverlap, setChunkOverlap] = useState(100);
  const [topK, setTopK] = useState(5);
  const [similarityThreshold, setSimilarityThreshold] = useState(0.3);
  const [useHybridSearch, setUseHybridSearch] = useState(true);  // NEW: BM25 + Vector

  const [error, setError] = useState("");
  const chatEndRef = useRef(null);
  const chatInputRef = useRef(null);
  const fileInputRef = useRef(null);
  const docFileInputRef = useRef(null);

  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: "smooth" }); }, [chatMessages]);
  useEffect(() => {
    if (!chatLoading && view === VIEWS.CHAT) chatInputRef.current?.focus();
  }, [chatLoading, view]);

  // ─── Load from IndexedDB on Mount ───────────────────────────
  useEffect(() => {
    async function restoreFromDB() {
      try {
        const [savedDocs, savedVectors] = await Promise.all([
          dbLoadDocuments(),
          dbLoadVectors(),
        ]);
        if (savedDocs.length && savedVectors.length) {
          // Restore vector store (tokens recomputed from text automatically in addBatch)
          vectorStore.addBatch(savedVectors);
          setDocuments(savedDocs);
          setTotalChunks(vectorStore.size);
          setDbSynced(true);
          setView(VIEWS.CHAT); // jump to chat if knowledge base exists
        }
      } catch (err) {
        console.warn("IndexedDB restore failed:", err);
      } finally {
        setDbLoading(false);
      }
    }
    restoreFromDB();
  }, []);

  const saveKey = () => {
    if (apiKey.trim()) { setKeySaved(true); setShowKeyPanel(false); setError(""); }
  };
  const addLog = (msg) =>
    setProcessLog((prev) => [...prev, { time: new Date().toLocaleTimeString(), msg }]);

  // ─── Document Processing Pipeline ───────────────────────────
  const processDocuments = async (texts) => {
    if (!apiKey.trim()) { setShowKeyPanel(true); setError("Add your API key first."); return; }
    setProcessing(true); setError(""); setProcessLog([]);

    try {
      addLog("Starting document processing pipeline...");
      setProcessStep("Parsing documents..."); setProcessProgress(5);

      const allChunks = [];
      const newDocs = [];

      texts.forEach((doc, di) => {
        const chunks = chunkText(doc.text, { chunkSize, overlap: chunkOverlap });
        const docEntry = {
          id: `doc_${Date.now()}_${di}`,
          name: doc.name,
          text: doc.text,
          chunks: chunks.length,
          charCount: doc.text.length,
          wordCount: doc.text.split(/\s+/).length,
          avgChunkSize: Math.round(chunks.reduce((s, c) => s + c.length, 0) / chunks.length),
          pages: doc.pages || null,  // for PDFs
          fileType: doc.fileType || "text",
          status: "chunked",
          uploadedAt: new Date().toLocaleString(),
        };
        newDocs.push(docEntry);
        chunks.forEach((c) =>
          allChunks.push({ ...c, docName: doc.name, docId: docEntry.id })
        );
        addLog(
          `Chunked "${doc.name}": ${chunks.length} chunks (avg ${docEntry.avgChunkSize} chars)` +
          (doc.pages ? ` · ${doc.pages} pages` : "")
        );
      });

      setProcessProgress(20);
      setProcessStep(`Generating embeddings for ${allChunks.length} chunks...`);
      addLog(`Sending ${allChunks.length} chunks to Gemini text-embedding-004...`);

      const chunkTexts = allChunks.map((c) => c.text);
      const embeddings = await getEmbeddings(apiKey, chunkTexts);

      setProcessProgress(70);
      addLog(`Received ${embeddings.length} embeddings (768 dimensions each)`);
      setProcessStep("Indexing vectors + building BM25...");

      // Batch add to vector store (single BM25 rebuild at the end)
      const batchEntries = allChunks
        .map((chunk, i) =>
          embeddings[i]
            ? {
                id: chunk.id,
                text: chunk.text,
                embedding: embeddings[i],
                metadata: {
                  docName: chunk.docName,
                  docId: chunk.docId,
                  chunkIndex: chunk.index,
                  wordCount: chunk.wordCount,
                },
              }
            : null
        )
        .filter(Boolean);
      vectorStore.addBatch(batchEntries);

      setProcessProgress(85);
      addLog(`BM25 index built over ${vectorStore.size} chunks`);
      setProcessStep("Persisting to IndexedDB...");

      // Persist documents and vectors
      newDocs.forEach((d) => { d.status = "indexed"; });
      const updatedDocs = [...documents, ...newDocs];
      await dbSaveDocuments(updatedDocs);
      await dbSaveVectors(batchEntries);
      setDbSynced(true);
      addLog("Saved to IndexedDB — knowledge base persists on refresh");

      setDocuments(updatedDocs);
      setTotalChunks(vectorStore.size);
      setProcessProgress(100);
      setProcessStep("Pipeline complete!");
      addLog(`Knowledge base ready: ${vectorStore.size} vectors indexed`);

      setTimeout(() => {
        setView(VIEWS.CHAT);
        setProcessing(false);
        setProcessStep("");
        setProcessProgress(0);
      }, 1000);
    } catch (err) {
      console.error(err);
      addLog(`ERROR: ${err.message}`);
      setError("Processing failed: " + err.message);
      setProcessing(false);
    }
  };

  // ─── File Handlers ───────────────────────────────────────────
  const handleFileUpload = async (e) => {
    const files = Array.from(e.target.files || []);
    if (!files.length) return;
    const texts = [];

    for (const file of files) {
      const ext = file.name.split(".").pop().toLowerCase();
      if (ext === "pdf") {
        addLog(`Extracting text from PDF: ${file.name}...`);
        try {
          const { text, pages } = await extractPDFText(file);
          if (!text.trim()) throw new Error("No extractable text found in PDF");
          texts.push({ name: file.name, text, pages, fileType: "pdf" });
        } catch (err) {
          setError(`PDF extraction failed for "${file.name}": ${err.message}`);
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

  const handleDrop = (e) => {
    e.preventDefault();
    const files = Array.from(e.dataTransfer.files).filter((f) =>
      /\.(txt|md|csv|pdf)$/i.test(f.name)
    );
    if (files.length) {
      const dt = new DataTransfer();
      files.forEach((f) => dt.items.add(f));
      handleFileUpload({ target: { files: dt.files } });
    }
  };

  const removeDocument = async (docId) => {
    vectorStore.removeByDocId(docId);
    const updated = documents.filter((d) => d.id !== docId);
    setDocuments(updated);
    setTotalChunks(vectorStore.size);
    try { await dbDeleteDoc(docId); } catch (err) { console.warn("DB delete failed:", err); }
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
      const [queryEmbedding] = await getEmbeddings(apiKey, [question]);

      // Choose retrieval strategy
      let results;
      if (useHybridSearch) {
        results = vectorStore.hybridSearch(question, queryEmbedding, topK);
        // For hybrid results: filter by vectorScore (not RRF score) vs threshold
        results = results.filter((r) => r.vectorScore >= similarityThreshold);
      } else {
        results = vectorStore.search(queryEmbedding, topK);
        results = results.filter((r) => r.score >= similarityThreshold);
      }
      setLastRetrieved(results);

      const contextBlock = results
        .map(
          (c, i) =>
            `[Source ${i + 1}] (from: ${c.metadata.docName}, ` +
            (useHybridSearch
              ? `vector: ${(c.vectorScore * 100).toFixed(1)}%, bm25: ${c.bm25Score?.toFixed(2) || "N/A"}`
              : `relevance: ${(c.score * 100).toFixed(1)}%`) +
            `)\n${c.text}`
        )
        .join("\n\n");

      const history = chatMessages.slice(-6);
      const historyBlock = history.length
        ? "\nRecent conversation:\n" +
          history.map((m) => `${m.role}: ${m.text || m.streamText || ""}`).join("\n") + "\n"
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

      // Stream: show answer text live, hide metadata block while building
      let fullText = "";
      await callGeminiStream(apiKey, prompt, (partial) => {
        fullText = partial;
        // Hide the <metadata> block from live display
        const displayText = partial.replace(/<metadata>[\s\S]*$/s, "").trim();
        setChatMessages((prev) =>
          prev.map((m) =>
            m.id === streamId ? { ...m, streamText: displayText } : m
          )
        );
      });

      // Parse final response
      const { answer, metadata } = parseStreamedResponse(fullText);

      const finalEntry = {
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
        prev.map((m) => (m.id === streamId ? finalEntry : m))
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
    } catch (err) {
      console.error(err);
      setChatMessages((prev) =>
        prev.map((m) =>
          m.streaming
            ? {
                ...m,
                streaming: false,
                text: "Error processing your question. Please try again.",
                confidence: "low",
                streamText: "",
              }
            : m
        )
      );
      setError(err.message);
    } finally {
      setChatLoading(false);
    }
  };

  // ─── Chunk Search (Hybrid) ────────────────────────────────
  const searchChunks = async () => {
    if (!chunkSearchQuery.trim() || !apiKey.trim()) return;
    setChunkSearchLoading(true);
    try {
      const [qEmbed] = await getEmbeddings(apiKey, [chunkSearchQuery]);
      const filter = selectedDocForChunks
        ? (d) => d.metadata.docId === selectedDocForChunks
        : null;
      const results = useHybridSearch
        ? vectorStore.hybridSearch(chunkSearchQuery, qEmbed, 20, filter)
        : vectorStore.search(qEmbed, 20, filter);
      setChunkSearchResults(results);
    } catch (err) {
      setError(err.message);
    } finally {
      setChunkSearchLoading(false);
    }
  };

  // ─── Doc Summary ─────────────────────────────────────────
  const generateDocSummary = async (docId) => {
    const doc = documents.find((d) => d.id === docId);
    if (!doc || docSummaries[docId]) return;
    setAnalyticsLoading(true);
    try {
      const parsed = await callGeminiJSON(
        apiKey,
        `Summarize this document concisely. Respond ONLY in JSON:
{"summary": "2-3 sentence summary", "topics": ["topic1","topic2","topic3"], "key_entities": ["entity1","entity2"], "sentiment": "positive|neutral|negative", "complexity": "basic|intermediate|advanced"}

Document (first 3000 chars):
${doc.text.substring(0, 3000)}`,
        0.4
      );
      setDocSummaries((prev) => ({ ...prev, [docId]: parsed }));
    } catch (err) {
      setError(err.message);
    } finally {
      setAnalyticsLoading(false);
    }
  };

  // ─── Analytics ───────────────────────────────────────────
  const storeStats = useMemo(() => vectorStore.getStats(), [totalChunks]);
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
    queryLog.forEach((q) => confCounts[q.confidence]++);
    return { avgScore, confCounts, total: queryLog.length };
  }, [queryLog]);

  // ─── Clear All ───────────────────────────────────────────
  const clearAll = async () => {
    if (!confirm("Clear all documents, vectors, and IndexedDB?")) return;
    vectorStore.clear();
    setDocuments([]); setTotalChunks(0);
    setChatMessages([]); setLastRetrieved([]); setQueryLog([]);
    setDocSummaries({}); setChunkSearchResults(null);
    setView(VIEWS.UPLOAD);
    try { await dbClearAll(); setDbSynced(false); } catch (err) { console.warn(err); }
  };

  // ─── Sub-Components ──────────────────────────────────────
  const ConfBadge = ({ level }) => {
    const c = { high: "#06D6A0", medium: "#FFD166", low: "#EF476F" };
    return (
      <span style={{
        fontSize: 10, fontWeight: 800, padding: "2px 8px", borderRadius: 4,
        textTransform: "uppercase", letterSpacing: "0.05em",
        background: (c[level] || c.low) + "18", color: c[level] || c.low,
        fontFamily: "'IBM Plex Mono', monospace",
      }}>{level}</span>
    );
  };

  const SearchModeBadge = ({ mode }) => (
    <span style={{
      fontSize: 9, fontWeight: 700, padding: "1px 6px", borderRadius: 4,
      background: mode === "hybrid" ? "var(--accent2)" + "18" : "var(--primary)" + "18",
      color: mode === "hybrid" ? "var(--accent2)" : "var(--primary)",
      fontFamily: "'IBM Plex Mono', monospace", textTransform: "uppercase",
    }}>{mode === "hybrid" ? "⚡ Hybrid" : "◈ Vector"}</span>
  );

  const ScoreBar = ({ score, max = 1 }) => (
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

  const fileTypeIcon = (type) => {
    if (type === "pdf") return "📕";
    if (type === "md") return "📝";
    if (type === "csv") return "📊";
    return "📄";
  };

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
              {dbLoading ? "loading…" : dbSynced ? "DB synced" : "not persisted"}
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
              {item.count > 0 && <span style={S.badge}>{item.count}</span>}
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
          <button
            onClick={() => setShowKeyPanel(true)}
            className="hov"
            style={{ ...S.keyBtn, borderColor: keySaved ? "var(--accent)" + "44" : "var(--danger)" + "44" }}
          >
            {keySaved ? "🔑 ✓ Saved" : "🔑 Setup"}
          </button>
        </div>
      </div>

      {/* ─── Main ─── */}
      <div style={S.main}>
        {/* Key Panel */}
        {showKeyPanel && (
          <div style={S.keyCard}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
              <h3 style={{ fontSize: 14, fontWeight: 700, color: "var(--text-bright)" }}>🔑 Gemini API Key</h3>
              {keySaved && (
                <button onClick={() => setShowKeyPanel(false)} style={{ background: "none", border: "none", color: "var(--text)", cursor: "pointer", fontSize: 16 }}>✕</button>
              )}
            </div>
            <p style={{ fontSize: 12, color: "var(--text)", marginBottom: 10, lineHeight: 1.5 }}>
              Free from{" "}
              <a href="https://aistudio.google.com/apikey" target="_blank" rel="noreferrer" style={{ color: "var(--primary)", textDecoration: "underline" }}>
                Google AI Studio
              </a>. Powers embeddings, generation, and streaming.
            </p>
            <div style={{ display: "flex", gap: 8 }}>
              <input
                type="password"
                placeholder="Paste key..."
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && saveKey()}
                style={{ ...S.input, flex: 1 }}
              />
              <button
                onClick={saveKey}
                disabled={!apiKey.trim()}
                className="hov"
                style={{ ...S.btnPri, opacity: apiKey.trim() ? 1 : 0.4 }}
              >
                Save
              </button>
            </div>
          </div>
        )}

        {error && (
          <div style={S.errorBar}>
            <span>⚠️</span>
            <span style={{ flex: 1 }}>{error}</span>
            <button onClick={() => setError("")} style={{ background: "none", border: "none", color: "var(--danger)", cursor: "pointer" }}>✕</button>
          </div>
        )}

        {/* ─── DB Loading Indicator ─── */}
        {dbLoading && (
          <div style={{ ...S.card, textAlign: "center", padding: 24, marginBottom: 12 }}>
            <p style={{ fontSize: 12, color: "var(--text)", fontFamily: "'IBM Plex Mono', monospace" }}>
              ◈ Loading knowledge base from IndexedDB…
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
                Full RAG — PDF · TXT · MD · CSV · Hybrid BM25+Vector Search · IndexedDB Persistence · Streaming Generation
              </p>
            </div>

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
                  { icon: "🧮", title: "Embed", desc: "768-dim vectors via text-embedding-004" },
                  { icon: "🗄️", title: "Index + Persist", desc: "Vector store + BM25 + IndexedDB (survives refresh)" },
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
              <div style={{ ...S.card, textAlign: "center", padding: 50 }}>
                <p style={{ fontSize: 14, color: "var(--text)" }}>No documents yet. Upload files to build your knowledge base.</p>
              </div>
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

                  {chatLoading && !chatMessages.some((m) => m.streaming) && (
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
                    <div key={i} style={{ padding: "8px", marginBottom: 6, background: "var(--surface2)", borderRadius: 4, borderLeft: `2px solid ${i === 0 ? "var(--primary)" : "var(--border)"}` }}>
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                        <span style={{ fontSize: 10, color: "var(--accent2)", fontFamily: "'IBM Plex Mono', monospace" }}>
                          {c.metadata?.docName?.length > 18 ? c.metadata.docName.substring(0, 18) + "…" : c.metadata?.docName}
                        </span>
                      </div>
                      {useHybridSearch ? (
                        <div style={{ fontSize: 9, fontFamily: "'IBM Plex Mono', monospace", color: "var(--text)", marginBottom: 4 }}>
                          <span style={{ color: "var(--primary)" }}>vec {(c.vectorScore * 100).toFixed(0)}%</span>
                          {" · "}
                          <span style={{ color: "var(--accent2)" }}>bm25 {c.bm25Score?.toFixed(2)}</span>
                        </div>
                      ) : (
                        <ScoreBar score={c.score} />
                      )}
                      <div style={{ marginTop: 4, fontSize: 10, color: "var(--text)", fontFamily: "'IBM Plex Mono', monospace", lineHeight: 1.4 }}>
                        {c.text.substring(0, 140)}…
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
                    <div key={i} style={{
                      padding: "10px 12px", marginBottom: 6,
                      background: "var(--surface2)", borderRadius: 6,
                      borderLeft: `3px solid ${c.vectorScore >= 0.6 || c.score >= 0.6 ? "var(--accent)" : c.vectorScore >= 0.4 || c.score >= 0.4 ? "var(--warn)" : "var(--border2)"}`,
                    }}>
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                        <span style={{ fontSize: 11, fontWeight: 600, color: "var(--accent2)" }}>
                          {c.metadata?.docName} · Chunk #{c.metadata?.chunkIndex}
                        </span>
                        {useHybridSearch ? (
                          <span style={{ fontSize: 10, fontFamily: "'IBM Plex Mono', monospace", color: "var(--text)" }}>
                            <span style={{ color: "var(--primary)" }}>{(c.vectorScore * 100).toFixed(0)}%</span>
                            {" + "}
                            <span style={{ color: "var(--accent2)" }}>{c.bm25Score?.toFixed(2)}</span>
                          </span>
                        ) : (
                          <span style={{ fontSize: 11, fontWeight: 800, fontFamily: "'IBM Plex Mono', monospace", color: c.score >= 0.6 ? "var(--accent)" : c.score >= 0.4 ? "var(--warn)" : "var(--text)" }}>
                            {(c.score * 100).toFixed(1)}%
                          </span>
                        )}
                      </div>
                      {!useHybridSearch && <ScoreBar score={c.score} />}
                      <div style={{ marginTop: 6, fontSize: 12, color: "var(--text-bright)", lineHeight: 1.6, fontFamily: "'IBM Plex Mono', monospace" }}>{c.text}</div>
                      <div style={{ marginTop: 4, fontSize: 10, color: "var(--text)" }}>
                        {c.text.length} chars · {c.metadata?.wordCount || "?"} words
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div style={S.card}>
              <h3 style={S.cardTitle}>📦 All Chunks by Document</h3>
              {documents.map((doc) => {
                const docChunks = vectorStore.getByDocId(doc.id);
                return (
                  <details key={doc.id} style={{ marginBottom: 8 }}>
                    <summary style={{ cursor: "pointer", padding: "8px 10px", background: "var(--surface2)", borderRadius: 4, fontSize: 12, fontWeight: 600, color: "var(--text-bright)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                      <span>{fileTypeIcon(doc.fileType)} {doc.name}</span>
                      <span style={{ fontSize: 10, color: "var(--text)", fontFamily: "'IBM Plex Mono', monospace" }}>{docChunks.length} chunks</span>
                    </summary>
                    <div style={{ padding: "8px 0" }}>
                      {docChunks.map((c, ci) => (
                        <div key={ci} style={{ padding: "6px 10px", marginBottom: 3, borderLeft: "2px solid var(--border)", fontSize: 11, color: "var(--text)", fontFamily: "'IBM Plex Mono', monospace", lineHeight: 1.4 }}>
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
                <ResponsiveContainer width="100%" height={160}>
                  <BarChart data={chunkDistribution}>
                    <XAxis dataKey="name" tick={{ fontSize: 9, fill: "#8B92AB" }} />
                    <YAxis tick={{ fontSize: 9, fill: "#8B92AB" }} />
                    <Tooltip contentStyle={{ background: "#0F1118", border: "1px solid #252940", borderRadius: 6, fontSize: 11 }} />
                    <Bar dataKey="chunks" fill="#7C5CFC" radius={[2, 2, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Confidence distribution */}
              <div style={S.card}>
                <h3 style={S.cardTitle}>Confidence Distribution</h3>
                {queryStats && queryStats.total > 0 ? (
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
                        label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                        labelLine={false}
                      >
                        {["#06D6A0", "#FFD166", "#EF476F"].map((c, i) => (
                          <Cell key={i} fill={c} />
                        ))}
                      </Pie>
                      <Tooltip contentStyle={{ background: "#0F1118", border: "1px solid #252940", borderRadius: 6, fontSize: 11 }} />
                    </PieChart>
                  </ResponsiveContainer>
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
                  This is the same approach used by Elasticsearch's hybrid search.
                </div>
              </div>

              <div style={S.card}>
                <h3 style={S.cardTitle}>🧮 Embeddings</h3>
                {[
                  ["Model", "text-embedding-004", "var(--primary)"],
                  ["Dimensions", "768", "var(--accent)"],
                  ["Batch Size", "20", "var(--accent2)"],
                  ["Provider", "Google (Free)", "var(--warn)"],
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
                  ["Model", "gemini-2.0-flash", "var(--primary)"],
                  ["Temperature", "0.3", "var(--accent)"],
                  ["Max Tokens", "3000", "var(--accent2)"],
                  ["Mode", "Streaming SSE", "var(--warn)"],
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
                  ["Storage", "IndexedDB", "var(--primary)"],
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
                  Clear All + Wipe IndexedDB
                </button>
              </div>

              <div style={{ ...S.card, gridColumn: "1 / -1", borderColor: "var(--primary)" + "33" }}>
                <h3 style={S.cardTitle}>📐 Pipeline Flow v2</h3>
                <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, color: "var(--text)", lineHeight: 2.4, padding: "8px 0" }}>
                  <span style={{ color: "var(--accent2)" }}>PDF/TXT/MD/CSV</span>{" → "}
                  <span style={{ color: "var(--primary)" }}>Sentence Chunker</span>{" → "}
                  <span style={{ color: "var(--accent)" }}>Batch Embeddings (768d)</span>{" → "}
                  <span style={{ color: "var(--warn)" }}>Vector Store + BM25</span>{" → "}
                  <span style={{ color: "var(--accent2)" }}>IndexedDB</span>{" → "}
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
          <p>Built by <strong>Prasanna Warad</strong> · RAGBase v2 · Gemini Embedding + Flash · Hybrid BM25+Vector · 100% in-browser</p>
        </div>
      </div>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════
const S = {
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
  keyBtn: { width: "100%", padding: "6px 10px", borderRadius: 5, background: "var(--surface2)", border: "1px solid var(--border)", color: "var(--text-bright)", fontSize: 11, fontWeight: 600, fontFamily: "'Plus Jakarta Sans', sans-serif", cursor: "pointer", textAlign: "center" },
  main: { flex: 1, padding: "18px 24px", minWidth: 0, overflowY: "auto" },
  keyCard: { background: "var(--surface)", border: "1px solid var(--primary)" + "33", borderRadius: 8, padding: "16px 18px", marginBottom: 14, animation: "fadeUp 0.3s ease" },
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
