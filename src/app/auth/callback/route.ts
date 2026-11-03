import { NextResponse } from "next/server";
import { createRequestClient } from "@/lib/supabase/request";

/** Landing point for email-confirmation / magic links: swaps the ?code for a session cookie. */
export async function GET(request: Request) {
  const { searchParams, origin } = new URL(request.url);
  const code = searchParams.get("code");
  if (code) {
    const supabase = await createRequestClient();
    const { error } = await supabase.auth.exchangeCodeForSession(code);
    if (!error) return NextResponse.redirect(`${origin}/`);
  }
  return NextResponse.redirect(`${origin}/login?error=confirmation_failed`);
}
