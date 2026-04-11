-- Grants + PostgREST schema cache refresh (fixes stale "column not found" after DDL in SQL editor)
-- Safe to re-run.

grant usage on schema public to service_role;

grant all on table public.documents to service_role;
grant all on table public.chunks to service_role;

grant all on all sequences in schema public to service_role;

notify pgrst, 'reload schema';
