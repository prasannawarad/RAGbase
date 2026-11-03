-- 006: Scalable search (Stage 2)
--
-- Before: vector search was an exact sequential scan, and hybrid search shipped EVERY chunk
-- (full distance ordering + all row text) to the API server to rank with in-process BM25.
-- Cost grew linearly with the corpus on every query.
--
-- After: both retrieval legs are index-backed inside Postgres and return only the top
-- candidates:
--   semantic leg  -> HNSW index on the embedding column
--   keyword leg   -> GIN index on a generated tsvector (Postgres full-text search)
--   fusion        -> Reciprocal Rank Fusion (k = 60), same formula as before, done in SQL
--
-- All functions are SECURITY INVOKER (the default), so Row Level Security from migration 005
-- applies automatically: a user can only ever retrieve their own chunks.
-- Requires 005. Safe to re-run.

-- 1. Indexes -----------------------------------------------------------------------------
-- HNSW (approximate nearest neighbour). Chosen over ivfflat: no training step, so it works on
-- an empty table and keeps good recall as rows are added. Operator class matches the `<->`
-- (L2) operator the query functions use.
create index if not exists chunks_embedding_hnsw_idx
  on public.chunks using hnsw (embedding vector_l2_ops)
  with (m = 16, ef_construction = 64);

-- Keyword search column maintained by Postgres itself (no app code to keep it in sync).
alter table public.chunks
  add column if not exists fts tsvector
  generated always as (to_tsvector('english', content)) stored;

create index if not exists chunks_fts_idx on public.chunks using gin (fts);

-- 2. Vector search (replaces the 2-arg match_chunks; now supports a document filter) --------
drop function if exists public.match_chunks(vector, int);
drop function if exists public.chunk_vector_distances(vector, uuid);

create or replace function public.match_chunks(
  query_embedding vector(768),
  match_count int default 5,
  filter_document_id uuid default null
)
returns table (
  id uuid,
  content text,
  document_id uuid,
  chunk_index int,
  distance double precision,
  word_count int,
  metadata jsonb
)
language sql
stable
set search_path = public, extensions
as $$
  select
    c.id,
    c.content,
    c.document_id,
    c.chunk_index,
    (c.embedding <-> query_embedding)::double precision as distance,
    c.word_count,
    c.metadata
  from public.chunks c
  where filter_document_id is null or c.document_id = filter_document_id
  order by c.embedding <-> query_embedding
  limit greatest(match_count, 1);
$$;

-- 3. Hybrid search: HNSW candidates + full-text candidates, fused with RRF -------------------
create or replace function public.hybrid_search_chunks(
  query_text text,
  query_embedding vector(768),
  match_count int default 5,
  filter_document_id uuid default null,
  candidate_count int default 100
)
returns table (
  id uuid,
  content text,
  document_id uuid,
  chunk_index int,
  distance double precision,
  keyword_score double precision,
  rrf_score double precision,
  word_count int,
  metadata jsonb
)
language sql
stable
set search_path = public, extensions
as $$
  with q as (
    -- plainto_tsquery ANDs every term, which almost never matches a natural-language question.
    -- Swap AND for OR so any shared keyword counts; ts_rank_cd then favours chunks that match more.
    select nullif(replace(plainto_tsquery('english', query_text)::text, '&', '|'), '')::tsquery as tsq
  ),
  semantic as (
    select
      c.id,
      row_number() over (order by c.embedding <-> query_embedding) as rnk
    from public.chunks c
    where filter_document_id is null or c.document_id = filter_document_id
    order by c.embedding <-> query_embedding
    limit candidate_count
  ),
  keyword as (
    select
      c.id,
      ts_rank_cd(c.fts, q.tsq)::double precision as kw,
      row_number() over (order by ts_rank_cd(c.fts, q.tsq) desc) as rnk
    from public.chunks c, q
    where q.tsq is not null
      and c.fts @@ q.tsq
      and (filter_document_id is null or c.document_id = filter_document_id)
    order by ts_rank_cd(c.fts, q.tsq) desc
    limit candidate_count
  ),
  fused as (
    select
      coalesce(s.id, k.id) as id,
      k.kw,
      coalesce(1.0 / (60 + s.rnk), 0) + coalesce(1.0 / (60 + k.rnk), 0) as rrf
    from semantic s
    full outer join keyword k on k.id = s.id
  )
  select
    c.id,
    c.content,
    c.document_id,
    c.chunk_index,
    (c.embedding <-> query_embedding)::double precision as distance,
    coalesce(f.kw, 0)::double precision as keyword_score,
    f.rrf::double precision as rrf_score,
    c.word_count,
    c.metadata
  from fused f
  join public.chunks c on c.id = f.id
  order by f.rrf desc
  limit greatest(match_count, 1);
$$;

-- 4. Privileges --------------------------------------------------------------------------
revoke execute on function public.match_chunks(vector, int, uuid) from public, anon;
revoke execute on function public.hybrid_search_chunks(text, vector, int, uuid, int) from public, anon;
grant execute on function public.match_chunks(vector, int, uuid) to authenticated, service_role;
grant execute on function public.hybrid_search_chunks(text, vector, int, uuid, int) to authenticated, service_role;

-- Keep the planner's statistics fresh so it picks the new indexes.
analyze public.chunks;

notify pgrst, 'reload schema';
