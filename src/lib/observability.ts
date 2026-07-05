import { Langfuse } from "langfuse";
import type { LangfuseTraceClient } from "langfuse";

/**
 * Optional Langfuse tracing. Enabled only when LANGFUSE_SECRET_KEY and
 * LANGFUSE_PUBLIC_KEY are set — otherwise every helper is a no-op so the
 * app runs unchanged without an account.
 *
 * Server-side only (keys are secret); never import from "use client" modules.
 */

let client: Langfuse | null | undefined;

export function isTracingEnabled(): boolean {
  return Boolean(process.env.LANGFUSE_SECRET_KEY && process.env.LANGFUSE_PUBLIC_KEY);
}

export function getLangfuse(): Langfuse | null {
  if (client !== undefined) return client;
  if (!isTracingEnabled()) {
    client = null;
    return client;
  }
  client = new Langfuse({
    secretKey: process.env.LANGFUSE_SECRET_KEY,
    publicKey: process.env.LANGFUSE_PUBLIC_KEY,
    baseUrl: process.env.LANGFUSE_BASEURL || "https://cloud.langfuse.com",
  });
  return client;
}

export function startTrace(
  name: string,
  opts?: { input?: unknown; metadata?: Record<string, unknown>; sessionId?: string; tags?: string[] }
): LangfuseTraceClient | null {
  const lf = getLangfuse();
  if (!lf) return null;
  return lf.trace({ name, ...opts });
}

/** Batched events are lost on serverless unless flushed before the response ends. */
export async function flushTraces(): Promise<void> {
  const lf = getLangfuse();
  if (!lf) return;
  try {
    await lf.flushAsync();
  } catch (e) {
    console.warn("[observability] flush failed:", e);
  }
}

/**
 * Pass-through wrapper for a UTF-8 text stream that accumulates the full text
 * and invokes `onComplete` once the source closes. Response bytes are
 * forwarded untouched.
 */
export function tapTextStream(
  source: ReadableStream<Uint8Array>,
  onComplete: (fullText: string) => Promise<void> | void
): ReadableStream<Uint8Array> {
  const decoder = new TextDecoder();
  let full = "";
  const transform = new TransformStream<Uint8Array, Uint8Array>({
    transform(chunk, controller) {
      full += decoder.decode(chunk, { stream: true });
      controller.enqueue(chunk);
    },
    async flush() {
      full += decoder.decode();
      try {
        await onComplete(full);
      } catch (e) {
        console.warn("[observability] stream onComplete failed:", e);
      }
    },
  });
  return source.pipeThrough(transform);
}
