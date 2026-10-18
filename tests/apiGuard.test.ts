import { beforeEach, describe, expect, it } from "vitest";
import { __resetGuardForTests, guardRequest } from "@/lib/apiGuard";

function req(headers: Record<string, string> = {}) {
  return new Request("http://localhost/api/x", { method: "POST", headers });
}

beforeEach(() => __resetGuardForTests());

describe("guardRequest: rate limit", () => {
  it("allows up to the limit then returns 429 with Retry-After", async () => {
    const h = { "x-forwarded-for": "1.1.1.1" };
    for (let i = 0; i < 3; i++) {
      expect(guardRequest(req(h), { name: "t", limit: 3 })).toBeNull();
    }
    const blocked = guardRequest(req(h), { name: "t", limit: 3 });
    expect(blocked?.status).toBe(429);
    expect(Number(blocked?.headers.get("Retry-After"))).toBeGreaterThan(0);
    expect((await blocked!.json()).code).toBe("rate_limited");
  });

  it("tracks IPs and route names independently", () => {
    const a = { "x-forwarded-for": "1.1.1.1" };
    const b = { "x-forwarded-for": "2.2.2.2" };
    expect(guardRequest(req(a), { name: "t", limit: 1 })).toBeNull();
    expect(guardRequest(req(a), { name: "t", limit: 1 })?.status).toBe(429);
    expect(guardRequest(req(b), { name: "t", limit: 1 })).toBeNull();
    expect(guardRequest(req(a), { name: "other", limit: 1 })).toBeNull();
  });

  it("uses the first x-forwarded-for hop as the client IP", () => {
    const h = { "x-forwarded-for": "3.3.3.3, 10.0.0.1" };
    expect(guardRequest(req(h), { name: "t", limit: 1 })).toBeNull();
    expect(
      guardRequest(req({ "x-forwarded-for": "3.3.3.3, 99.9.9.9" }), { name: "t", limit: 1 })?.status
    ).toBe(429);
  });
});

describe("guardRequest: origin check", () => {
  it("lets requests without Origin/Referer through (curl, server-to-server)", () => {
    expect(guardRequest(req(), { name: "o", limit: 5 })).toBeNull();
  });

  it("allows localhost origins outside production", () => {
    expect(guardRequest(req({ origin: "http://localhost:3000" }), { name: "o", limit: 5 })).toBeNull();
  });

  it("rejects a foreign browser origin with 403", async () => {
    const res = guardRequest(req({ origin: "https://evil.example" }), { name: "o", limit: 5 });
    expect(res?.status).toBe(403);
    expect((await res!.json()).code).toBe("forbidden");
  });
});
