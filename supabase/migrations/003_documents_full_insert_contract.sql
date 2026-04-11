-- Align `documents` with /api/ingest insert payload (older or minimal tables may only have name + timestamps).
-- Safe to re-run: IF NOT EXISTS on each column.

alter table public.documents
  add column if not exists raw_text text,
  add column if not exists char_count int,
  add column if not exists word_count int,
  add column if not exists chunk_count int not null default 0,
  add column if not exists avg_chunk_size int,
  add column if not exists pages int,
  add column if not exists file_type text,
  add column if not exists status text,
  add column if not exists uploaded_at timestamptz not null default now();

-- Note: API uses SUPABASE_SERVICE_ROLE_KEY, which bypasses RLS. If you use the anon key for inserts, add policies instead of relying on this migration.
