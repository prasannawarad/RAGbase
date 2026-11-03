/**
 * Reduce whatever was pasted into a Supabase URL variable to the bare project origin.
 * The dashboard shows several URLs (".../rest/v1/", ".../auth/v1", trailing slashes); the client
 * libraries want only "https://<ref>.supabase.co" and append their own paths, so an extra path
 * silently breaks sign-in with URLs like ".../rest/v1//auth/v1/...".
 * Safe for both server and browser (no server-only imports).
 */
export function normalizeSupabaseUrl(value: string | undefined | null): string | undefined {
  const raw = value?.trim();
  if (!raw) return undefined;
  try {
    return new URL(raw).origin;
  } catch {
    return raw; // unparseable: pass through so the library reports the real error
  }
}
