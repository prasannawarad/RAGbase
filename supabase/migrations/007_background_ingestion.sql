-- 007: Resumable background ingestion (Stage 4)
--
-- Before: POST /api/ingest embedded every chunk inside one request. The Gemini free tier allows
-- ~100 embed calls/minute and serverless requests have a time limit, so large documents
-- timed out or failed halfway, and a failure fell back to storing mock embeddings.
--
-- After:
--   1. /api/ingest only validates + stores the chunk TEXT in `ingest_chunks` (staging) and
--      creates the document with status 'queued' (fast, no external calls).
--   2. /api/ingest/[id]/process claims small batches from staging, embeds them, writes real
--      rows to `chunks`, and advances the document queued -> processing -> indexed | failed.
--   3. A lease (`claimed_at`) stops two workers from embedding the same chunk; an expired lease
--      makes a crashed batch claimable again; `attempts` caps retries.
--   4. content_hash makes uploads idempotent: re-uploading the same text per user is a no-op.
--
-- Requires 005 (RLS pattern). Safe to re-run.

-- 1. Document progress + dedup columns ---------------------------------------------------
alter table public.documents
  add column if not exists content_hash text,
  add column if not exists chunks_done int not null default 0,
  add column if not exists error text,
  add column if not exists updated_at timestamptz not null default now();

-- Existing documents were ingested synchronously: they are complete.
update public.documents
   set chunks_done = chunk_count
 where status = 'indexed' and chunks_done = 0 and chunk_count > 0;

-- One logical copy of a given text per user (NULL hashes = legacy rows, unconstrained).
create unique index if not exists documents_user_hash_uidx
  on public.documents (user_id, content_hash)
  where content_hash is not null;

-- Makes writing a chunk idempotent: a retry after a half-finished batch upserts, never duplicates.
create unique index if not exists chunks_document_chunk_uidx
  on public.chunks (document_id, chunk_index);

-- 2. Staging table -----------------------------------------------------------------------
create table if not exists public.ingest_chunks (
  id uuid primary key default gen_random_uuid(),
  document_id uuid not null references public.documents (id) on delete cascade,
  user_id uuid not null references auth.users (id) on delete cascade,
  chunk_index int not null,
  content text not null,
  attempts int not null default 0,
  claimed_at timestamptz,
  unique (document_id, chunk_index)
);

alter table public.ingest_chunks enable row level security;

drop policy if exists ingest_chunks_owner_all on public.ingest_chunks;
create policy ingest_chunks_owner_all on public.ingest_chunks
  for all to authenticated
  using (user_id = (select auth.uid()))
  with check (
    user_id = (select auth.uid())
    and exists (
      select 1 from public.documents d
      where d.id = document_id and d.user_id = (select auth.uid())
    )
  );

revoke all on public.ingest_chunks from anon;
grant select, insert, update, delete on public.ingest_chunks to authenticated;

-- 3. Atomic batch claim ------------------------------------------------------------------
-- Claims up to p_limit unclaimed (or lease-expired) staging rows for one document and bumps
-- their attempt counter. FOR UPDATE SKIP LOCKED means concurrent callers receive DISJOINT rows
-- instead of blocking on or duplicating each other. SECURITY INVOKER: RLS limits it to the
-- caller's own rows.
create or replace function public.claim_ingest_batch(
  p_document_id uuid,
  p_limit int default 8,
  p_lease_seconds int default 120
)
returns setof public.ingest_chunks
language sql
volatile
set search_path = public
as $$
  update public.ingest_chunks
     set claimed_at = now(),
         attempts = attempts + 1
   where id in (
     select id
       from public.ingest_chunks
      where document_id = p_document_id
        and (claimed_at is null or claimed_at < now() - make_interval(secs => p_lease_seconds))
      order by chunk_index
      limit greatest(p_limit, 1)
      for update skip locked
   )
  returning *;
$$;

revoke execute on function public.claim_ingest_batch(uuid, int, int) from public, anon;
grant execute on function public.claim_ingest_batch(uuid, int, int) to authenticated, service_role;

notify pgrst, 'reload schema';
