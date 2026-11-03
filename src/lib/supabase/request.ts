import "server-only";

import { createServerClient } from "@supabase/ssr";
import type { SupabaseClient } from "@supabase/supabase-js";
import { cookies } from "next/headers";
import { getSupabaseAnonKey, getSupabaseProjectUrl } from "@/lib/supabase/env";

/**
 * Per-request Supabase client acting AS THE LOGGED-IN USER (anon key + the session JWT from
 * the request cookies). Every query it makes is subject to Row Level Security, so tenant
 * isolation is enforced by the database, not by remembering a WHERE clause.
 *
 * Do not cache this client across requests — it is bound to one request's cookies.
 * For privileged/admin work use the service-role client in ./server (scripts only).
 */
export async function createRequestClient(): Promise<SupabaseClient> {
  const url = getSupabaseProjectUrl();
  const anon = getSupabaseAnonKey();
  if (!url || !anon) {
    throw new Error("Missing SUPABASE_URL (or NEXT_PUBLIC_SUPABASE_URL) or NEXT_PUBLIC_SUPABASE_ANON_KEY");
  }
  const cookieStore = await cookies();
  return createServerClient(url, anon, {
    cookies: {
      getAll: () => cookieStore.getAll(),
      setAll: (list) => {
        try {
          for (const { name, value, options } of list) cookieStore.set(name, value, options);
        } catch {
          // Called from a context where cookies are read-only (e.g. a Server Component).
          // Safe to ignore: middleware refreshes the session cookies.
        }
      },
    },
  });
}
