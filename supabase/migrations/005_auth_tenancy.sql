-- 005: Authentication + multi-tenancy (Stage 1)
--
-- Every document and chunk is owned by one auth.users row. Row Level Security makes the
-- database itself enforce isolation: API routes now query with the *user's* JWT (anon key +
-- session cookie), so a missing WHERE clause in app code cannot leak another user's data.
-- The service-role key bypasses RLS and is now reserved for admin scripts (npm run reembed).
--
-- Safe to re-run. Existing rows have no owner and become invisible to users until you
-- backfill them — see the BACKFILL section below.

-- 1. Ownership columns -------------------------------------------------------------------
alter table public.documents
  add column if not exists user_id uuid references auth.users (id) on delete cascade;
alter table public.chunks
  add column if not exists user_id uuid references auth.users (id) on delete cascade;

-- "List my documents, newest first" and "my chunks" are the hot paths for RLS filters.
create index if not exists documents_user_uploaded_idx
  on public.documents (user_id, uploaded_at desc);
create index if not exists chunks_user_id_idx
  on public.chunks (user_id);

-- 2. Row Level Security ------------------------------------------------------------------
alter table public.documents enable row level security;
alter table public.chunks enable row level security;

-- (select auth.uid()) is evaluated once per statement instead of once per row.
drop policy if exists documents_owner_all on public.documents;
create policy documents_owner_all on public.documents
  for all to authenticated
  using (user_id = (select auth.uid()))
  with check (user_id = (select auth.uid()));

-- A chunk must be owned by the caller AND attached to a document the caller owns, so a user
-- cannot plant chunks inside someone else's document.
drop policy if exists chunks_owner_all on public.chunks;
create policy chunks_owner_all on public.chunks
  for all to authenticated
  using (user_id = (select auth.uid()))
  with check (
    user_id = (select auth.uid())
    and exists (
      select 1 from public.documents d
      where d.id = document_id and d.user_id = (select auth.uid())
    )
  );

-- 3. Privileges --------------------------------------------------------------------------
-- No anonymous-role access at all. (Supabase "anonymous sign-ins" users still use the
-- `authenticated` role, so a guest session works under the policies above.)
revoke all on public.documents from anon;
revoke all on public.chunks from anon;
grant select, insert, update, delete on public.documents to authenticated;
grant select, insert, update, delete on public.chunks to authenticated;

-- 4. BACKFILL (manual, once) -------------------------------------------------------------
-- Rows created before this migration have user_id = NULL. To keep them, assign them to your
-- account (find your id under Authentication > Users after signing up), then re-run this file
-- so the NOT NULL constraints below apply:
--
--   update public.documents set user_id = '<your-user-uuid>' where user_id is null;
--   update public.chunks    set user_id = '<your-user-uuid>' where user_id is null;
--
-- Or discard old demo data instead:  delete from public.documents where user_id is null;

do $$
begin
  if not exists (select 1 from public.documents where user_id is null) then
    alter table public.documents alter column user_id set not null;
  end if;
  if not exists (select 1 from public.chunks where user_id is null) then
    alter table public.chunks alter column user_id set not null;
  end if;
end $$;

notify pgrst, 'reload schema';
