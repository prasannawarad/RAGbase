-- Idempotent fix for documents metadata columns (remote DBs may predate 001 or omit columns)
alter table public.documents
  add column if not exists char_count int,
  add column if not exists word_count int,
  add column if not exists avg_chunk_size int;
