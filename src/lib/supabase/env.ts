import "server-only";
import { normalizeSupabaseUrl } from "@/lib/supabase/url";

/**
 * Prefer SUPABASE_URL for server routes; fall back to NEXT_PUBLIC_SUPABASE_URL
 * (same project URL in typical setups) so local .env mistakes do not break ingest.
 */
export function getSupabaseProjectUrl(): string | undefined {
  return normalizeSupabaseUrl(process.env.SUPABASE_URL) ?? normalizeSupabaseUrl(process.env.NEXT_PUBLIC_SUPABASE_URL);
}

export function getSupabaseAnonKey(): string | undefined {
  return process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY?.trim() || undefined;
}

/** True when API routes can build a user-scoped (anon key + session cookie) client. */
export function isSupabaseAuthConfigured(): boolean {
  return Boolean(getSupabaseProjectUrl() && getSupabaseAnonKey());
}
