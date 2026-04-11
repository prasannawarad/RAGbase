import Groq from "groq-sdk";

const MODEL = "llama-3.3-70b-versatile";

function getApiKey(): string {
  const key = process.env.GROQ_API_KEY;
  if (!key?.trim()) {
    throw new Error("GROQ_API_KEY is not configured");
  }
  return key.trim();
}

function getClient(): Groq {
  return new Groq({ apiKey: getApiKey() });
}

/**
 * Text generation via Groq (Llama 3.3 70B).
 */
export async function generateText(
  prompt: string,
  options?: { temperature?: number; maxOutputTokens?: number }
): Promise<string> {
  const groq = getClient();
  const completion = await groq.chat.completions.create({
    model: MODEL,
    messages: [{ role: "user", content: prompt }],
    temperature: options?.temperature ?? 0.3,
    max_completion_tokens: options?.maxOutputTokens ?? 3000,
  });
  return completion.choices[0]?.message?.content ?? "";
}

/**
 * Streaming chat: returns UTF-8 text delta chunks (same shape as {@link openGeminiChatTextStream}).
 */
export async function openGroqChatTextStream(
  prompt: string,
  options?: { temperature?: number; maxOutputTokens?: number }
): Promise<ReadableStream<Uint8Array>> {
  const groq = getClient();
  const stream = await groq.chat.completions.create({
    model: MODEL,
    messages: [{ role: "user", content: prompt }],
    temperature: options?.temperature ?? 0.3,
    max_completion_tokens: options?.maxOutputTokens ?? 3000,
    stream: true,
  });

  const encoder = new TextEncoder();
  return new ReadableStream<Uint8Array>({
    async start(controller) {
      try {
        for await (const chunk of stream) {
          const delta = chunk.choices[0]?.delta?.content ?? "";
          if (delta) controller.enqueue(encoder.encode(delta));
        }
        controller.close();
      } catch (e) {
        controller.error(e instanceof Error ? e : new Error(String(e)));
      }
    },
  });
}
