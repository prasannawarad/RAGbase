import { describe, expect, it } from "vitest";
import { z } from "zod";
import { apiError, parseBody, validate } from "@/lib/api/http";

const schema = z.object({ n: z.number() });

function post(body: string, headers: Record<string, string> = {}) {
  return new Request("http://localhost/api/x", { method: "POST", body, headers });
}

describe("apiError", () => {
  it("returns the { error, code } contract with the given status", async () => {
    const res = apiError(418, "bad_request", "nope", { extra: 1 });
    expect(res.status).toBe(418);
    expect(await res.json()).toEqual({ error: "nope", code: "bad_request", extra: 1 });
  });
});

describe("parseBody", () => {
  it("parses and validates a good body", async () => {
    const r = await parseBody(post('{"n":3}'), schema);
    expect(r).toEqual({ ok: true, data: { n: 3 } });
  });

  it("returns 400 bad_request on malformed JSON", async () => {
    const r = await parseBody(post("{nope"), schema);
    expect(r.ok).toBe(false);
    if (!r.ok) {
      expect(r.response.status).toBe(400);
      expect((await r.response.json()).code).toBe("bad_request");
    }
  });

  it("returns 400 validation_failed with issues on schema mismatch", async () => {
    const r = await parseBody(post('{"n":"x"}'), schema);
    expect(r.ok).toBe(false);
    if (!r.ok) {
      const body = await r.response.json();
      expect(r.response.status).toBe(400);
      expect(body.code).toBe("validation_failed");
      expect(body.issues[0].path).toBe("n");
    }
  });

  it("returns 413 when the declared Content-Length exceeds the cap", async () => {
    const r = await parseBody(post('{"n":1}', { "content-length": "999999" }), schema, 100);
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.response.status).toBe(413);
  });

  it("returns 413 when the actual body exceeds the cap (lying/absent Content-Length)", async () => {
    const r = await parseBody(post(JSON.stringify({ n: 1, pad: "x".repeat(500) })), schema, 100);
    expect(r.ok).toBe(false);
    if (!r.ok) expect((await r.response.json()).code).toBe("payload_too_large");
  });
});

describe("validate", () => {
  it("validates non-body values such as route params", () => {
    expect(validate(schema, { n: 1 }).ok).toBe(true);
    expect(validate(schema, {}).ok).toBe(false);
  });
});
