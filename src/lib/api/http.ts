import { NextResponse } from "next/server";
import type { ZodType } from "zod";
import { LIMITS } from "@/lib/api/schemas";

/**
 * Shared HTTP helpers for API routes.
 *
 * Error contract (every non-2xx JSON response):
 *   { error: string, code: string, ...extra }
 * `error` stays a human-readable string so existing clients (`data.error`) keep working;
 * `code` is a stable machine-readable value clients and tests can branch on.
 */

export type ApiErrorCode =
  | "bad_request"
  | "validation_failed"
  | "payload_too_large"
  | "unauthorized"
  | "forbidden"
  | "not_found"
  | "rate_limited"
  | "not_configured"
  | "internal_error";

export function apiError(
  status: number,
  code: ApiErrorCode,
  message: string,
  extra?: Record<string, unknown>
): NextResponse {
  return NextResponse.json({ error: message, code, ...extra }, { status });
}

/** Diagnostic detail is only safe to show outside production. */
export function devDetails(details: unknown): Record<string, unknown> {
  return process.env.NODE_ENV === "production" || details == null ? {} : { details };
}

export type ParseResult<T> = { ok: true; data: T } | { ok: false; response: NextResponse };

/**
 * Read, size-cap, JSON-parse and schema-validate a request body in one step.
 * Never trusts Content-Length alone: the body text length is checked after reading.
 */
export async function parseBody<T>(
  req: Request,
  schema: ZodType<T>,
  maxBytes: number = LIMITS.bodyBytesDefault
): Promise<ParseResult<T>> {
  const declared = Number(req.headers.get("content-length"));
  if (Number.isFinite(declared) && declared > maxBytes) {
    return { ok: false, response: apiError(413, "payload_too_large", "Request body too large") };
  }

  let text: string;
  try {
    text = await req.text();
  } catch {
    return { ok: false, response: apiError(400, "bad_request", "Could not read request body") };
  }
  if (text.length > maxBytes) {
    return { ok: false, response: apiError(413, "payload_too_large", "Request body too large") };
  }

  let json: unknown;
  try {
    json = JSON.parse(text);
  } catch {
    return { ok: false, response: apiError(400, "bad_request", "Invalid JSON body") };
  }

  return validate(schema, json);
}

/** Validate an already-parsed value (e.g. route params). */
export function validate<T>(schema: ZodType<T>, value: unknown): ParseResult<T> {
  const parsed = schema.safeParse(value);
  if (parsed.success) return { ok: true, data: parsed.data };
  const issues = parsed.error.issues.map((i) => ({
    path: i.path.join("."),
    message: i.message,
  }));
  const first = issues[0];
  const message = first?.path ? `Invalid "${first.path}": ${first.message}` : "Invalid request";
  return { ok: false, response: apiError(400, "validation_failed", message, { issues }) };
}
