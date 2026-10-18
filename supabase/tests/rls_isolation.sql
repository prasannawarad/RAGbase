-- Tenant-isolation and search checks for migrations 005/006. Run on a scratch DB prepared with
-- local_stubs.sql (see that file). Each section prints its expectation; compare by eye.
-- Expected: T1 1 row; T2/T3 RLS errors; T5/T6 only Bob's chunk; T7 backup chunk first;
-- T8 2 rows; T9 3 rows; T10 two 'permission denied'; T11 count 0.
\set ON_ERROR_STOP off
\pset pager off
-- users
insert into auth.users (id, email) values
 ('aaaaaaaa-0000-0000-0000-000000000001','alice@x.io'),
 ('bbbbbbbb-0000-0000-0000-000000000002','bob@x.io');

-- helper: random unit-ish vector
create or replace function pg_temp.rv(seed int) returns vector language sql as $$
  select (select array_agg(sin(seed * g + g * 0.37))::vector(768) from generate_series(1,768) g)
$$;

-- Alice (authenticated) ingests a doc + chunks
set role authenticated; set request.jwt.claim.sub = 'aaaaaaaa-0000-0000-0000-000000000001';
insert into documents (id, user_id, name, chunk_count) values ('d1000000-0000-0000-0000-000000000001', auth.uid(), 'alice-ops.md', 3);
insert into chunks (user_id, document_id, chunk_index, content, embedding) values
 (auth.uid(), 'd1000000-0000-0000-0000-000000000001', 0, 'Database backups run nightly at 02:00 UTC and are retained for 30 days.', pg_temp.rv(1)),
 (auth.uid(), 'd1000000-0000-0000-0000-000000000001', 1, 'Rate limiting is applied per user with a sliding window.',               pg_temp.rv(2)),
 (auth.uid(), 'd1000000-0000-0000-0000-000000000001', 2, 'The weather in the office is usually sunny.',                          pg_temp.rv(3));
reset role;

-- Bob (authenticated) ingests his own doc
set role authenticated; set request.jwt.claim.sub = 'bbbbbbbb-0000-0000-0000-000000000002';
insert into documents (id, user_id, name, chunk_count) values ('d2000000-0000-0000-0000-000000000002', auth.uid(), 'bob-secret.md', 1);
insert into chunks (user_id, document_id, chunk_index, content, embedding) values
 (auth.uid(), 'd2000000-0000-0000-0000-000000000002', 0, 'Bob confidential: the backup password is hunter2.', pg_temp.rv(1));
reset role;

\echo '--- T1: Bob sees only his own documents (expect 1: bob-secret.md)'
set role authenticated; set request.jwt.claim.sub = 'bbbbbbbb-0000-0000-0000-000000000002';
select name from documents;

\echo '--- T2: Bob tries to plant a chunk inside ALICE''s document (expect RLS error)'
insert into chunks (user_id, document_id, chunk_index, content, embedding)
 values (auth.uid(), 'd1000000-0000-0000-0000-000000000001', 9, 'injected', pg_temp.rv(5));

\echo '--- T3: Bob tries to claim Alice as owner (expect RLS error)'
insert into documents (user_id, name) values ('aaaaaaaa-0000-0000-0000-000000000001', 'forged');

\echo '--- T4: Bob deletes Alice''s doc by id (expect DELETE 0)'
delete from documents where id = 'd1000000-0000-0000-0000-000000000001';

\echo '--- T5: Bob''s hybrid search for "backup" must return ONLY Bob''s chunk'
select left(content, 40) as content, document_id from hybrid_search_chunks('backup password', pg_temp.rv(1), 5, null);
\echo '--- T6: Bob''s vector search must return ONLY Bob''s chunk (1 row)'
select left(content, 40) as content from match_chunks(pg_temp.rv(1), 5, null);
reset role;

\echo '--- T7: Alice hybrid search "backups nightly" (keyword leg: expect backup chunk first)'
set role authenticated; set request.jwt.claim.sub = 'aaaaaaaa-0000-0000-0000-000000000001';
select left(content, 50) as content, round(keyword_score::numeric,3) kw, round(rrf_score::numeric,4) rrf from hybrid_search_chunks('When do backups run nightly?', pg_temp.rv(3), 3, null);
\echo '--- T8: Alice vector search filtered to her doc, topK=2 (expect 2 rows)'
select left(content, 40), round(distance::numeric,3) from match_chunks(pg_temp.rv(2), 2, 'd1000000-0000-0000-0000-000000000001');
\echo '--- T9: stopword-only keyword query still works (vector leg only, no error)'
select count(*) from hybrid_search_chunks('the and of', pg_temp.rv(1), 3, null);
reset role;

\echo '--- T10: anon role has no table or function access (expect permission denied x2)'
set role anon;
select count(*) from documents;
select count(*) from match_chunks(pg_temp.rv(1), 1, null);
reset role;

\echo '--- T11: no-session authenticated user (auth.uid() null) sees nothing (expect 0)'
set role authenticated; reset request.jwt.claim.sub;
select count(*) from documents;
reset role;
