import "server-only";

import type { NextResponse } from "next/server";
import { createClient, type SupabaseClient, type User } from "@supabase/supabase-js";
import { apiError } from "@/lib/api/http";
import { getSupabaseAnonKey, getSupabaseProjectUrl, isSupabaseAuthConfigured } from "@/lib/supabase/env";
import { createRequestClient } from "@/lib/supabase/request";

export type AuthContext = {
  /** Supabase client scoped to this user — all queries run under RLS. */
  supabase: SupabaseClient;
  user: User;
};

export type AuthResult = ({ ok: true } & AuthContext) | { ok: false; response: NextResponse };

function bearerToken(req: Request): string | null {
  const header = req.headers.get("authorization");
  const match = header?.match(/^Bearer\s+(.+)$/i);
  return match ? match[1]!.trim() : null;
}

/**
 * Authenticate an API request. Call near the top of every data route:
 *
 *   const auth = await requireUser(req);
 *   if (!auth.ok) return auth.response;
 *   const { supabase, user } = auth;
 *
 * Two credential sources, same result (a client acting as the user, under RLS):
 *   1. Session cookie — the browser app.
 *   2. `Authorization: Bearer <access_token>` — CLI tools such as `npm run eval`.
 *
 * Uses auth.getUser(), which validates the JWT with the Supabase Auth server (unlike
 * getSession(), which only trusts the cookie). Returns 503 when Supabase is not configured,
 * 401 when there is no valid session.
 */
export async function requireUser(req: Request): Promise<AuthResult> {
  if (!isSupabaseAuthConfigured()) {
    return { ok: false, response: apiError(503, "not_configured", "Supabase is not configured") };
  }

  const token = bearerToken(req);
  const supabase = token
    ? createClient(getSupabaseProjectUrl()!, getSupabaseAnonKey()!, {
        global: { headers: { Authorization: `Bearer ${token}` } },
        auth: { persistSession: false, autoRefreshToken: false },
      })
    : await createRequestClient();

  const { data, error } = token ? await supabase.auth.getUser(token) : await supabase.auth.getUser();
  if (error || !data.user) {
    return { ok: false, response: apiError(401, "unauthorized", "Authentication required") };
  }
  return { ok: true, supabase, user: data.user };
}
