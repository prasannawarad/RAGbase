const GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta";

function getApiKey(): string {
  const key = process.env.GEMINI_API_KEY;
  if (!key?.trim()) {
    throw new Error("GEMINI_API_KEY is not configured");
  }
  return key.trim();
}

/**
 * Batch embedding via Gemini text-embedding-004 (768 dimensions per vector).
 * Batches of 20 to match API limits.
 */
export async function embedTexts(texts: string[]): Promise<number[][]> {
  const apiKey = getApiKey();
  const results: number[][] = [];
  const batchSize = 20;

  for (let i = 0; i < texts.length; i += batchSize) {
    const batch = texts.slice(i, i + batchSize);
    const requests = batch.map((text) => ({
      model: "models/text-embedding-004",
      content: { parts: [{ text }] },
    }));

    const res = await fetch(
      `${GEMINI_API_BASE}/models/text-embedding-004:batchEmbedContents?key=${encodeURIComponent(apiKey)}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ requests }),
      }
    );

    const data = (await res.json()) as {
      error?: { message?: string };
      embeddings?: { values: number[] }[];
    };

    if (!res.ok || data.error) {
      throw new Error(data.error?.message || "Embedding API error");
    }

    const batchEmb = data.embeddings?.map((e) => e.values) ?? [];
    if (batchEmb.length !== batch.length) {
      throw new Error("Embedding API returned incomplete batch");
    }
    results.push(...batchEmb);
  }

  return results;
}

/**
 * Text generation via Gemini gemini-2.0-flash (non-streaming).
 */
export async function generateText(
  prompt: string,
  options?: { temperature?: number; maxOutputTokens?: number }
): Promise<string> {
  const apiKey = getApiKey();

  const res = await fetch(
    `${GEMINI_API_BASE}/models/gemini-2.0-flash:generateContent?key=${encodeURIComponent(apiKey)}`,
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
  const url = `${GEMINI_API_BASE}/models/gemini-2.0-flash:streamGenerateContent?alt=sse&key=${encodeURIComponent(apiKey)}`;

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
