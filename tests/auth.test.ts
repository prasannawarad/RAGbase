import { beforeEach, describe, expect, it, vi } from "vitest";

const getUserCookie = vi.fn();
const getUserBearer = vi.fn();
const createClientSpy = vi.fn();

vi.mock("@/lib/supabase/request", () => ({
  createRequestClient: async () => ({ auth: { getUser: getUserCookie } }),
}));
vi.mock("@supabase/supabase-js", () => ({
  createClient: (...args: unknown[]) => {
    createClientSpy(...args);
    return { auth: { getUser: getUserBearer } };
  },
}));

import { requireUser } from "@/lib/auth";

const req = (headers: Record<string, string> = {}) => new Request("http://localhost/api/x", { headers });

beforeEach(() => {
  vi.clearAllMocks();
  process.env.SUPABASE_URL = "https://proj.supabase.co";
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY = "anon-key";
});

describe("requireUser", () => {
  it("returns 503 not_configured when Supabase env is missing", async () => {
    delete process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
    const r = await requireUser(req());
    expect(r.ok).toBe(false);
    if (!r.ok) {
      expect(r.response.status).toBe(503);
      expect((await r.response.json()).code).toBe("not_configured");
    }
  });

  it("returns 401 when the cookie session has no user", async () => {
    getUserCookie.mockResolvedValue({ data: { user: null }, error: null });
    const r = await requireUser(req());
    expect(r.ok).toBe(false);
    if (!r.ok) {
      expect(r.response.status).toBe(401);
      expect((await r.response.json()).code).toBe("unauthorized");
    }
  });

  it("returns 401 when Supabase rejects the token", async () => {
    getUserCookie.mockResolvedValue({ data: { user: null }, error: { message: "bad jwt" } });
    const r = await requireUser(req());
    expect(r.ok).toBe(false);
  });

  it("authenticates via the session cookie", async () => {
    getUserCookie.mockResolvedValue({ data: { user: { id: "u1" } }, error: null });
    const r = await requireUser(req());
    expect(r.ok).toBe(true);
    if (r.ok) expect(r.user.id).toBe("u1");
    expect(createClientSpy).not.toHaveBeenCalled();
  });

  it("authenticates via a Bearer token and scopes the client to it (RLS runs as that user)", async () => {
    getUserBearer.mockResolvedValue({ data: { user: { id: "u2" } }, error: null });
    const r = await requireUser(req({ authorization: "Bearer tok123" }));
    expect(r.ok).toBe(true);
    if (r.ok) expect(r.user.id).toBe("u2");
    expect(getUserBearer).toHaveBeenCalledWith("tok123");
    expect(createClientSpy.mock.calls[0]![2]).toMatchObject({
      global: { headers: { Authorization: "Bearer tok123" } },
    });
    expect(getUserCookie).not.toHaveBeenCalled();
  });

  it("returns 401 for an invalid Bearer token without falling back to cookies", async () => {
    getUserBearer.mockResolvedValue({ data: { user: null }, error: { message: "invalid" } });
    const r = await requireUser(req({ authorization: "Bearer nope" }));
    expect(r.ok).toBe(false);
    expect(getUserCookie).not.toHaveBeenCalled();
  });
});
