import { NextResponse } from "next/server";
import { createRequestClient } from "@/lib/supabase/request";

/** POST-only so a link or <img> on another site cannot log the user out (CSRF). */
export async function POST(request: Request) {
  const supabase = await createRequestClient();
  await supabase.auth.signOut();
  // 303 turns the POST into a GET on /login.
  return NextResponse.redirect(new URL("/login", request.url), 303);
}
