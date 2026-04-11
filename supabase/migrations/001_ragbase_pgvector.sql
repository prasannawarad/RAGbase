-- RAGBase: documents + chunks with pgvector (768-dim)
-- Run in Supabase SQL editor or via supabase db push

create extension if not exists vector;

create table if not exists public.documents (
  id uuid primary key default gen_random_uuid(),
  name text not null,
  raw_text text,
  char_count int,
  word_count int,
  chunk_count int not null default 0,
  avg_chunk_size int,
  pages int,
  file_type text,
  status text,
  uploaded_at timestamptz not null default now()
);

create table if not exists public.chunks (
  id uuid primary key default gen_random_uuid(),
  document_id uuid not null references public.documents (id) on delete cascade,
  chunk_index int not null,
  content text not null,
  embedding vector(768) not null,
  word_count int,
  metadata jsonb default '{}'::jsonb
);

create index if not exists chunks_document_id_idx on public.chunks (document_id);

-- Optional: create after you have enough rows for ivfflat to be useful
-- create index if not exists chunks_embedding_ivfflat_idx
--   on public.chunks using ivfflat (embedding vector_l2_ops) with (lists = 100);

-- Nearest-neighbor search (L2 distance)
create or replace function public.match_chunks(
  query_embedding vector(768),
  match_count int default 5
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
  order by c.embedding <-> query_embedding
  limit greatest(match_count, 1);
$$;

-- Full ordering by vector distance (for hybrid RRF ranking)
create or replace function public.chunk_vector_distances(
  query_embedding vector(768),
  filter_document_id uuid default null
)
returns table (
  id uuid,
  distance double precision
)
language sql
stable
as $$
  select
    c.id,
    (c.embedding <-> query_embedding)::double precision as distance
  from public.chunks c
  where filter_document_id is null or c.document_id = filter_document_id
  order by c.embedding <-> query_embedding;
$$;

grant usage on schema public to service_role;
grant all on public.documents to service_role;
grant all on public.chunks to service_role;
grant execute on function public.match_chunks(vector, int) to service_role;
grant execute on function public.chunk_vector_distances(vector, uuid) to service_role;
