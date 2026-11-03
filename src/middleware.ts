import { createServerClient } from "@supabase/ssr";
import { NextResponse, type NextRequest } from "next/server";
import { normalizeSupabaseUrl } from "@/lib/supabase/url";

/**
 * Runs on every page request (not static assets):
 *  1. Refreshes the Supabase session so cookies don't expire mid-use.
 *  2. Sends signed-out visitors to /login.
 *
 * /api/* is deliberately NOT redirected — API routes answer 401 JSON themselves via
 * requireUser(), because a redirect is the wrong response for fetch() callers.
 */

const PUBLIC_PREFIXES = ["/login", "/auth/"];

export async function middleware(request: NextRequest) {
  const url = normalizeSupabaseUrl(process.env.SUPABASE_URL) ?? normalizeSupabaseUrl(process.env.NEXT_PUBLIC_SUPABASE_URL);
  const anon = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY?.trim();
  // Not configured (fresh clone): let the app render; the UI surfaces the config error.
  if (!url || !anon) return NextResponse.next({ request });

  let response = NextResponse.next({ request });
  const supabase = createServerClient(url, anon, {
    cookies: {
      getAll: () => request.cookies.getAll(),
      setAll: (list) => {
        for (const { name, value } of list) request.cookies.set(name, value);
        response = NextResponse.next({ request });
        for (const { name, value, options } of list) response.cookies.set(name, value, options);
      },
    },
  });

  // Do not run code between createServerClient and getUser() — it refreshes the cookies.
  const { data } = await supabase.auth.getUser();

  const { pathname } = request.nextUrl;
  const isApi = pathname.startsWith("/api/");
  const isPublic = PUBLIC_PREFIXES.some((p) => pathname === p || pathname.startsWith(p));

  if (!data.user && !isApi && !isPublic) {
    const login = request.nextUrl.clone();
    login.pathname = "/login";
    login.search = "";
    return NextResponse.redirect(login);
  }
  if (data.user && pathname === "/login" && !data.user.is_anonymous) {
    const home = request.nextUrl.clone();
    home.pathname = "/";
    home.search = "";
    return NextResponse.redirect(home);
  }

  return response;
}

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.svg|icon.svg|.*\\.(?:svg|png|jpg|jpeg|gif|webp|ico)$).*)"],
};
