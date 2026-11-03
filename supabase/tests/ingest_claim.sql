-- Checks for migration 007 (background ingestion). Run on a scratch DB prepared with
-- local_stubs.sql + migrations 001-007. Comments state the expected result of each step.
\set ON_ERROR_STOP off
\pset pager off
insert into auth.users (id, email) values
 ('aaaaaaaa-1111-0000-0000-000000000001','ing-alice@x.io'),
 ('bbbbbbbb-1111-0000-0000-000000000002','ing-bob@x.io') on conflict do nothing;

-- Alice queues a 10-chunk document
set role authenticated; set request.jwt.claim.sub = 'aaaaaaaa-1111-0000-0000-000000000001';
insert into documents (id, user_id, name, chunk_count, status, content_hash)
 values ('d4000000-0000-0000-0000-00000000000a', auth.uid(), 'big.txt', 10, 'queued', 'hash-1');
insert into ingest_chunks (document_id, user_id, chunk_index, content)
 select 'd4000000-0000-0000-0000-00000000000a', auth.uid(), g, 'chunk ' || g from generate_series(0, 9) g;

\echo '--- T1: first claim of 4 returns chunk_index 0..3 with attempts=1'
select chunk_index, attempts, claimed_at is not null as claimed from claim_ingest_batch('d4000000-0000-0000-0000-00000000000a', 4);

\echo '--- T2: second claim returns the NEXT rows 4..7 (claimed rows are skipped)'
select chunk_index from claim_ingest_batch('d4000000-0000-0000-0000-00000000000a', 4);

\echo '--- T3: third claim returns only the 2 remaining rows 8,9'
select chunk_index from claim_ingest_batch('d4000000-0000-0000-0000-00000000000a', 4);

\echo '--- T4: fourth claim returns nothing while all leases are live (expect 0 rows)'
select count(*) from claim_ingest_batch('d4000000-0000-0000-0000-00000000000a', 4);

\echo '--- T5: expire the lease on rows 0-3 (simulated crashed worker); they are claimable again with attempts=2'
update ingest_chunks set claimed_at = now() - interval '5 minutes' where chunk_index < 4;
select chunk_index, attempts from claim_ingest_batch('d4000000-0000-0000-0000-00000000000a', 10) order by chunk_index;

\echo '--- T6: duplicate content hash for the same user is rejected (expect unique violation)'
insert into documents (user_id, name, chunk_count, status, content_hash)
 values (auth.uid(), 'dup.txt', 1, 'queued', 'hash-1');

\echo '--- T7: idempotent chunk write: same (document, chunk_index) twice = 1 row via upsert'
insert into chunks (user_id, document_id, chunk_index, content, embedding)
 values (auth.uid(), 'd4000000-0000-0000-0000-00000000000a', 0, 'c0', (select array_agg(0.1)::vector(768) from generate_series(1,768)))
 on conflict (document_id, chunk_index) do nothing;
insert into chunks (user_id, document_id, chunk_index, content, embedding)
 values (auth.uid(), 'd4000000-0000-0000-0000-00000000000a', 0, 'c0-again', (select array_agg(0.1)::vector(768) from generate_series(1,768)))
 on conflict (document_id, chunk_index) do nothing;
select count(*) as rows_for_chunk_0, max(content) from chunks where document_id='d4000000-0000-0000-0000-00000000000a' and chunk_index=0;
reset role;

\echo '--- T8: Bob cannot see, claim, or plant staging rows in Alice''s document'
set role authenticated; set request.jwt.claim.sub = 'bbbbbbbb-1111-0000-0000-000000000002';
select count(*) as bob_sees_staging from ingest_chunks;
select count(*) as bob_claims from claim_ingest_batch('d4000000-0000-0000-0000-00000000000a', 10);
insert into ingest_chunks (document_id, user_id, chunk_index, content) values ('d4000000-0000-0000-0000-00000000000a', auth.uid(), 99, 'planted');
reset role;

\echo '--- T9: anon has no access to the staging table or claim function (expect 2 x permission denied)'
set role anon;
select count(*) from ingest_chunks;
select count(*) from claim_ingest_batch('d4000000-0000-0000-0000-00000000000a', 1);
reset role;

\echo '--- T10: deleting the document cascades to its staging rows (expect 0)'
set role authenticated; set request.jwt.claim.sub = 'aaaaaaaa-1111-0000-0000-000000000001';
delete from documents where id = 'd4000000-0000-0000-0000-00000000000a';
select count(*) as staging_left from ingest_chunks where document_id = 'd4000000-0000-0000-0000-00000000000a';
reset role;
