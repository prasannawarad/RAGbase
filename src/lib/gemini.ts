const GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta";

function getApiKey(): string {
  const key = process.env.GEMINI_API_KEY;
  if (!key?.trim()) {
    throw new Error("GEMINI_API_KEY is not configured");
  }
  return key.trim();
}

const EMBEDDING_MODEL = "gemini-embedding-001";
export const EMBEDDING_DIMS = 768;

// gemini-2.0-flash has zero free-tier generateContent quota (July 2026);
// flash-lite is the tier that still serves free requests.
export const GEMINI_TEXT_MODEL = "gemini-2.5-flash-lite";

function l2Normalize(values: number[]): number[] {
  const norm = Math.sqrt(values.reduce((sum, v) => sum + v * v, 0));
  if (!norm) return values;
  return values.map((v) => v / norm);
}

async function embedOne(text: string, apiKey: string, retries = 3): Promise<number[]> {
  for (let attempt = 0; ; attempt++) {
    const res = await fetch(
      `${GEMINI_API_BASE}/models/${EMBEDDING_MODEL}:embedContent?key=${encodeURIComponent(apiKey)}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          content: { parts: [{ text }] },
          outputDimensionality: EMBEDDING_DIMS,
        }),
      }
    );

    const data = (await res.json()) as {
      error?: { message?: string };
      embedding?: { values: number[] };
    };

    if (res.status === 429 && attempt < retries) {
      // Free tier allows 100 embed requests/min; respect the suggested delay.
      const suggested = data.error?.message?.match(/retry in ([\d.]+)s/i);
      const delayMs = suggested ? Math.ceil(parseFloat(suggested[1]) * 1000) + 1000 : 45_000;
      console.warn(`[gemini] embed rate-limited, retrying in ${Math.round(delayMs / 1000)}s`);
      await new Promise((r) => setTimeout(r, delayMs));
      continue;
    }

    if (!res.ok || data.error) {
      throw new Error(data.error?.message || "Embedding API error");
    }
    const values = data.embedding?.values;
    if (!values || values.length !== EMBEDDING_DIMS) {
      throw new Error("Embedding API returned unexpected dimensions");
    }
    // Truncated (non-3072) gemini-embedding-001 vectors are not unit length;
    // normalize so L2 distance ranking in pgvector stays meaningful.
    return l2Normalize(values);
  }
}

/**
 * Embedding via gemini-embedding-001 truncated to 768 dims (matches the
 * pgvector vector(768) column). The model has no batch endpoint, so requests
 * run with limited concurrency.
 */
export async function embedTexts(texts: string[]): Promise<number[][]> {
  const apiKey = getApiKey();
  const results: number[][] = new Array(texts.length);
  const concurrency = 8;

  for (let i = 0; i < texts.length; i += concurrency) {
    const batch = texts.slice(i, i + concurrency);
    const vectors = await Promise.all(batch.map((text) => embedOne(text, apiKey)));
    vectors.forEach((v, j) => {
      results[i + j] = v;
    });
  }

  return results;
}

/**
 * Text generation via Gemini (non-streaming).
 */
export async function generateText(
  prompt: string,
  options?: { temperature?: number; maxOutputTokens?: number }
): Promise<string> {
  const apiKey = getApiKey();

  const res = await fetch(
    `${GEMINI_API_BASE}/models/${GEMINI_TEXT_MODEL}:generateContent?key=${encodeURIComponent(apiKey)}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        contents: [{ parts: [{ text: prompt }] }],
        generationConfig: {
          temperature: options?.temperature ?? 0.3,
          maxOutputTokens: options?.maxOutputTokens ?? 3000,
        },
      }),
    }
  );

  const data = (await res.json()) as {
    error?: { message?: string };
    candidates?: { content?: { parts?: { text?: string }[] } }[];
  };

  if (!res.ok || data.error) {
    throw new Error(data.error?.message || "GenerateContent API error");
  }

  return data?.candidates?.[0]?.content?.parts?.[0]?.text ?? "";
}

function extractDeltaTextFromSseJson(obj: unknown): string {
  const o = obj as {
    candidates?: { content?: { parts?: { text?: string }[] } }[];
  };
  const parts = o?.candidates?.[0]?.content?.parts;
  if (!parts?.length) return "";
  return parts.map((p) => p.text ?? "").join("");
}

function createSseBodyToUtf8TextStream(body: ReadableStream<Uint8Array>): ReadableStream<Uint8Array> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  const encoder = new TextEncoder();
  let lineBuffer = "";

  return new ReadableStream<Uint8Array>({
    async start(controller) {
      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          lineBuffer += decoder.decode(value, { stream: true });

          const lines = lineBuffer.split("\n");
          lineBuffer = lines.pop() ?? "";

          for (const rawLine of lines) {
            const line = rawLine.trim();
            if (!line || line.startsWith(":")) continue;
            if (!line.startsWith("data:")) continue;
            const payload = line.slice(5).trim();
            if (payload === "[DONE]") continue;
            try {
              const json = JSON.parse(payload) as unknown;
              const delta = extractDeltaTextFromSseJson(json);
              if (delta) controller.enqueue(encoder.encode(delta));
            } catch {
              /* skip malformed line */
            }
          }
        }

        const tail = lineBuffer.trim();
        if (tail.startsWith("data:")) {
          const payload = tail.slice(5).trim();
          if (payload && payload !== "[DONE]") {
            try {
              const json = JSON.parse(payload) as unknown;
              const delta = extractDeltaTextFromSseJson(json);
              if (delta) controller.enqueue(encoder.encode(delta));
            } catch {
              /* ignore */
            }
          }
        }

        controller.close();
      } catch (e) {
        controller.error(e instanceof Error ? e : new Error(String(e)));
      }
    },
  });
}

/**
 * Opens Gemini streamGenerateContent (?alt=sse), parses SSE, and returns a stream of UTF-8 text deltas.
 * Throws if the HTTP request fails so callers can fall back to {@link generateText}.
 */
export async function openGeminiChatTextStream(
  prompt: string,
  options?: { temperature?: number; maxOutputTokens?: number }
): Promise<ReadableStream<Uint8Array>> {
  const apiKey = getApiKey();
  const url = `${GEMINI_API_BASE}/models/${GEMINI_TEXT_MODEL}:streamGenerateContent?alt=sse&key=${encodeURIComponent(apiKey)}`;

  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      contents: [{ parts: [{ text: prompt }] }],
      generationConfig: {
        temperature: options?.temperature ?? 0.3,
        maxOutputTokens: options?.maxOutputTokens ?? 3000,
      },
    }),
  });

  if (!res.ok) {
    let msg = `Gemini stream HTTP ${res.status}`;
    try {
      const err = (await res.json()) as { error?: { message?: string } };
      msg = err.error?.message ?? msg;
    } catch {
      /* ignore */
    }
    throw new Error(msg);
  }

  if (!res.body) {
    throw new Error("Empty stream body");
  }

  return createSseBodyToUtf8TextStream(res.body);
}
