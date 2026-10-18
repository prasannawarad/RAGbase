import { createBrowserClient } from "@supabase/ssr";
import type { SupabaseClient } from "@supabase/supabase-js";

/**
 * Browser-safe Supabase client (anon key only).
 *
 * Uses ONLY:
 * - NEXT_PUBLIC_SUPABASE_URL
 * - NEXT_PUBLIC_SUPABASE_ANON_KEY
 *
 * createBrowserClient (not plain createClient) stores the session in cookies, which is what
 * lets middleware and API routes see the logged-in user. RLS makes the anon key safe to ship.
 *
 * Never put SUPABASE_SERVICE_ROLE_KEY in client code or NEXT_PUBLIC_* variables.
 * Privileged operations belong in admin scripts (service role); request handling uses @/lib/auth.
 */

let _browser: SupabaseClient | null = null;

export function getSupabaseBrowser(): SupabaseClient {
  if (typeof window === "undefined") {
    throw new Error(
      "[supabase/client] getSupabaseBrowser() is for the browser only. Use requireUser() from @/lib/auth in server/API code."
    );
  }
  if (_browser) return _browser;
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const anon = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
  if (!url || !anon) {
    throw new Error("Missing NEXT_PUBLIC_SUPABASE_URL or NEXT_PUBLIC_SUPABASE_ANON_KEY");
  }
  _browser = createBrowserClient(url, anon);
  return _browser;
}

export function isSupabaseBrowserConfigured(): boolean {
  return Boolean(
    typeof window !== "undefined" &&
      process.env.NEXT_PUBLIC_SUPABASE_URL &&
      process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY
  );
}
