"use client";

import { useState, type FormEvent } from "react";
import { useRouter } from "next/navigation";
import { getSupabaseBrowser } from "@/lib/supabase/client";

type Mode = "signin" | "signup";

const field =
  "w-full rounded-xl border border-white/10 bg-white/5 px-3.5 py-2.5 text-sm text-[var(--text-bright)] outline-none transition focus:border-[var(--focus-ring)]";

export default function LoginPage() {
  const router = useRouter();
  const [mode, setMode] = useState<Mode>("signin");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  async function run(action: () => Promise<{ error: { message: string } | null }>, onDone?: () => void) {
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const { error: err } = await action();
      if (err) {
        setError(err.message);
        return;
      }
      onDone ? onDone() : (router.replace("/"), router.refresh());
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong");
    } finally {
      setBusy(false);
    }
  }

  function onSubmit(e: FormEvent) {
    e.preventDefault();
    const supabase = getSupabaseBrowser();
    if (mode === "signin") {
      void run(() => supabase.auth.signInWithPassword({ email, password }));
    } else {
      void run(
        () =>
          supabase.auth.signUp({
            email,
            password,
            options: { emailRedirectTo: `${window.location.origin}/auth/callback` },
          }),
        () => setNotice("Check your inbox to confirm your email, then sign in.")
      );
    }
  }

  function continueAsGuest() {
    void run(() => getSupabaseBrowser().auth.signInAnonymously());
  }

  return (
    <main className="flex h-full w-full items-center justify-center overflow-y-auto p-4">
      <div
        className="w-full max-w-sm rounded-2xl border border-white/10 p-6"
        style={{ background: "var(--surface)" }}
      >
        <h1 className="text-xl font-semibold text-[var(--text-bright)]">RAGBase</h1>
        <p className="mt-1 text-sm text-[var(--text)]">
          {mode === "signin" ? "Sign in to your document workspace." : "Create an account to get started."}
        </p>

        <form onSubmit={onSubmit} className="mt-5 space-y-3">
          <label className="block">
            <span className="mb-1 block text-xs text-[var(--text)]">Email</span>
            <input
              type="email"
              required
              autoComplete="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className={field}
            />
          </label>
          <label className="block">
            <span className="mb-1 block text-xs text-[var(--text)]">Password</span>
            <input
              type="password"
              required
              minLength={8}
              autoComplete={mode === "signin" ? "current-password" : "new-password"}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className={field}
            />
          </label>

          {error && (
            <p role="alert" className="text-sm" style={{ color: "var(--danger)" }}>
              {error}
            </p>
          )}
          {notice && (
            <p role="status" className="text-sm" style={{ color: "var(--accent)" }}>
              {notice}
            </p>
          )}

          <button
            type="submit"
            disabled={busy}
            className="w-full rounded-xl px-4 py-2.5 text-sm font-semibold text-black transition disabled:opacity-50"
            style={{ background: "var(--primary)" }}
          >
            {busy ? "Please wait…" : mode === "signin" ? "Sign in" : "Create account"}
          </button>
        </form>

        <button
          type="button"
          onClick={() => {
            setMode(mode === "signin" ? "signup" : "signin");
            setError(null);
            setNotice(null);
          }}
          className="mt-3 w-full text-center text-xs text-[var(--text)] underline-offset-2 hover:underline"
        >
          {mode === "signin" ? "No account? Create one" : "Already have an account? Sign in"}
        </button>

        <div className="my-4 flex items-center gap-3 text-xs text-[var(--text)]">
          <span className="h-px flex-1 bg-white/10" />
          or
          <span className="h-px flex-1 bg-white/10" />
        </div>

        <button
          type="button"
          onClick={continueAsGuest}
          disabled={busy}
          className="w-full rounded-xl border border-white/10 px-4 py-2.5 text-sm text-[var(--text-bright)] transition hover:bg-white/5 disabled:opacity-50"
        >
          Continue as guest
        </button>
        <p className="mt-2 text-center text-[11px] text-[var(--text)]">
          Guest data is private to this browser and may be lost if you clear cookies.
        </p>
      </div>
    </main>
  );
}
