import "server-only";

import { createClient, type SupabaseClient } from "@supabase/supabase-js";
import { getSupabaseProjectUrl, isSupabaseServerConfigured } from "@/lib/supabase/env";

/**
 * SERVER / API ROUTES ONLY — Supabase admin client using the service role key.
 *
 * NEVER import this module from:
 * - "use client" components
 * - browser-only code
 * - shared modules that are also imported by the client
 *
 * The service role key bypasses Row Level Security and must never be bundled or
 * exposed to the frontend. Use `src/lib/supabase/client.ts` with
 * NEXT_PUBLIC_SUPABASE_ANON_KEY for any future browser-side Supabase usage.
 */

let _admin: SupabaseClient | null = null;

function assertNotBrowser(): void {
  if (typeof window !== "undefined") {
    throw new Error(
      "[supabase/server] Refusing to run in the browser: SUPABASE_SERVICE_ROLE_KEY must never be exposed to the client. " +
        "Use @/lib/supabase/client with NEXT_PUBLIC_SUPABASE_URL + NEXT_PUBLIC_SUPABASE_ANON_KEY only."
    );
  }
}

/** Server-only Supabase client (service role). */
export function getSupabaseAdmin(): SupabaseClient {
  assertNotBrowser();
  if (_admin) return _admin;
  const url = getSupabaseProjectUrl();
  const key = process.env.SUPABASE_SERVICE_ROLE_KEY?.trim();
  if (!url || !key) {
    throw new Error(
      "Missing SUPABASE_URL (or NEXT_PUBLIC_SUPABASE_URL) or SUPABASE_SERVICE_ROLE_KEY"
    );
  }
  _admin = createClient(url, key, {
    auth: { persistSession: false, autoRefreshToken: false },
  });
  return _admin;
}

export function isSupabaseConfigured(): boolean {
  assertNotBrowser();
  return isSupabaseServerConfigured();
}
