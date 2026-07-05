/**
 * Re-embed every chunk in Supabase with the current embedding model.
 *
 * Needed whenever the embedding model changes (e.g. the forced migration off
 * the retired text-embedding-004) or after chunks were ingested while the
 * embedder was falling back to mock vectors — query and chunk vectors must
 * come from the same model or retrieval is meaningless.
 *
 * Usage: npm run reembed            # all chunks
 *        npm run reembed -- --dry   # count only, no writes
 */

import fs from "node:fs";
import path from "node:path";
import { createClient } from "@supabase/supabase-js";

function loadEnvLocal() {
  const p = path.join(process.cwd(), ".env.local");
  if (!fs.existsSync(p)) return;
  for (const line of fs.readFileSync(p, "utf8").split("\n")) {
    const m = line.match(/^\s*([A-Z0-9_]+)\s*=\s*(.*)\s*$/);
    if (m && !(m[1] in process.env)) {
      process.env[m[1]] = m[2].replace(/^["']|["']$/g, "");
    }
  }
}
loadEnvLocal();

const DRY_RUN = process.argv.includes("--dry");
const PAGE_SIZE = 100;

async function main() {
  const { embedTexts } = await import("../src/lib/gemini");

  const url = process.env.SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
  const key = process.env.SUPABASE_SERVICE_ROLE_KEY;
  if (!url || !key) {
    console.error("Missing SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY in .env.local");
    process.exit(1);
  }
  if (!process.env.GEMINI_API_KEY) {
    console.error("Missing GEMINI_API_KEY in .env.local");
    process.exit(1);
  }
  const supabase = createClient(url, key, {
    auth: { persistSession: false, autoRefreshToken: false },
  });

  const { count, error: countError } = await supabase
    .from("chunks")
    .select("id", { count: "exact", head: true });
  if (countError) throw countError;
  console.log(`${count} chunks to re-embed${DRY_RUN ? " (dry run, no writes)" : ""}`);
  if (DRY_RUN || !count) return;

  let done = 0;
  let failed = 0;
  for (let offset = 0; offset < count; offset += PAGE_SIZE) {
    const { data: rows, error } = await supabase
      .from("chunks")
      .select("id, content")
      .order("id")
      .range(offset, offset + PAGE_SIZE - 1);
    if (error) throw error;
    if (!rows?.length) break;

    const vectors = await embedTexts(rows.map((r) => r.content));

    for (let i = 0; i < rows.length; i++) {
      const { error: updateError } = await supabase
        .from("chunks")
        .update({ embedding: vectors[i] })
        .eq("id", rows[i].id);
      if (updateError) {
        failed++;
        console.error(`  chunk ${rows[i].id}: ${updateError.message}`);
      } else {
        done++;
      }
    }
    console.log(`  ${done}/${count} re-embedded${failed ? ` (${failed} failed)` : ""}`);
  }

  console.log(failed ? `Done with ${failed} failures.` : "Done. All chunks re-embedded.");
  if (failed) process.exit(1);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
