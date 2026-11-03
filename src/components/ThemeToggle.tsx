"use client";

import { useEffect, useState } from "react";
import { THEME_KEY, applyTheme, type Theme } from "@/lib/theme";

/** Dark is the default; the choice is stored per browser. The no-flash script in layout.tsx applies it before paint (the <html> element persists across client navigations). */
export default function ThemeToggle({ className = "" }: { className?: string }) {
  const [theme, setTheme] = useState<Theme>("dark");

  // Follow the real attribute, so several toggles on one page (phone header + sidebar) can never
  // leave an icon out of step.
  useEffect(() => {
    const read = () => setTheme(document.documentElement.dataset.theme === "light" ? "light" : "dark");
    read();
    const observer = new MutationObserver(read);
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    return () => observer.disconnect();
  }, []);

  function toggle() {
    const next: Theme = theme === "dark" ? "light" : "dark";
    applyTheme(next);
    try {
      localStorage.setItem(THEME_KEY, next);
    } catch {
      /* storage blocked: the choice just won't persist */
    }
  }

  const isDark = theme === "dark";
  return (
    <button
      type="button"
      onClick={toggle}
      aria-label={isDark ? "Switch to light mode" : "Switch to dark mode"}
      title={isDark ? "Light mode" : "Dark mode"}
      className={`inline-flex h-10 w-10 items-center justify-center rounded-xl border border-[var(--border)] bg-[var(--surface)] text-[var(--text-bright)] transition hover:bg-[var(--panel-hover)] ${className}`}
    >
      <span aria-hidden="true" className="text-base leading-none">
        {isDark ? "☀" : "☾"}
      </span>
    </button>
  );
}
