-- Minimal stand-ins for the parts of Supabase that migrations 005/006 depend on, so the
-- migrations and RLS policies can be tested on a throwaway local Postgres + pgvector.
-- NEVER run this against a real Supabase project (it would try to create auth.users etc.).
--
--   createdb ragtest
--   psql ragtest -f supabase/tests/local_stubs.sql
--   for f in supabase/migrations/00*.sql; do psql ragtest -f $f; done
--   psql ragtest -f supabase/tests/rls_isolation.sql
create extension if not exists vector;
create role anon nologin;
create role authenticated nologin;
create role service_role nologin bypassrls;
create schema auth;
create table auth.users (id uuid primary key default gen_random_uuid(), email text);
create function auth.uid() returns uuid language sql stable as $$
  select nullif(current_setting('request.jwt.claim.sub', true), '')::uuid
$$;
grant usage on schema public, auth to anon, authenticated, service_role;
grant execute on function auth.uid() to anon, authenticated, service_role;
-- Supabase grants new public tables/functions to all API roles by default; mirror that so the
-- revokes in migration 005/006 are actually exercised.
alter default privileges in schema public grant all on tables to anon, authenticated, service_role;
alter default privileges in schema public grant execute on functions to anon, authenticated, service_role;
