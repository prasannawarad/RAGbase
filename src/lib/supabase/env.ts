import "server-only";

/**
 * Prefer SUPABASE_URL for server routes; fall back to NEXT_PUBLIC_SUPABASE_URL
 * (same project URL in typical setups) so local .env mistakes do not break ingest.
 */
export function getSupabaseProjectUrl(): string | undefined {
  const direct = process.env.SUPABASE_URL?.trim();
  if (direct) return direct;
  return process.env.NEXT_PUBLIC_SUPABASE_URL?.trim();
}

export function isSupabaseServerConfigured(): boolean {
  return Boolean(getSupabaseProjectUrl() && process.env.SUPABASE_SERVICE_ROLE_KEY?.trim());
}
