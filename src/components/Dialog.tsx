"use client";

import { useEffect, useRef, type ReactNode } from "react";

/**
 * In-app modal replacing window.prompt/confirm (which block the page and look dated on phones).
 * Bottom sheet below `sm`, centered card above. Esc and backdrop click close it.
 */
export default function Dialog({
  open,
  title,
  description,
  onClose,
  children,
  actions,
}: {
  open: boolean;
  title: string;
  description?: string;
  onClose: () => void;
  children?: ReactNode;
  actions: ReactNode;
}) {
  const panelRef = useRef<HTMLDivElement>(null);
  // Held in a ref so a new inline onClose each render doesn't re-run the effect and steal focus mid-typing.
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;

  useEffect(() => {
    if (!open) return;
    const previous = document.activeElement as HTMLElement | null;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onCloseRef.current();
    };
    document.addEventListener("keydown", onKey);
    // Focus the first field (or the panel) so keyboard and screen-reader users land inside.
    const first = panelRef.current?.querySelector<HTMLElement>("textarea, input, button");
    (first ?? panelRef.current)?.focus();
    return () => {
      document.removeEventListener("keydown", onKey);
      previous?.focus?.();
    };
  }, [open]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-[80] flex items-end justify-center sm:items-center sm:p-6"
      style={{ background: "var(--overlay)", animation: "fadeIn 0.15s ease" }}
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        ref={panelRef}
        role="dialog"
        aria-modal="true"
        aria-label={title}
        tabIndex={-1}
        className="sheet-bottom flex max-h-[90dvh] w-full flex-col gap-4 overflow-y-auto rounded-t-2xl border border-[var(--border)] bg-[var(--surface)] p-5 outline-none sm:max-w-lg sm:rounded-2xl sm:p-6"
        style={{ boxShadow: "var(--shadow-lg)", animation: "fadeUp 0.2s ease" }}
      >
        <div>
          <h2 className="font-display text-xl font-semibold tracking-tight text-[var(--text-bright)]">{title}</h2>
          {description && <p className="mt-1 text-sm text-[var(--text)]">{description}</p>}
        </div>
        {children}
        <div className="flex flex-col-reverse gap-2 sm:flex-row sm:justify-end">{actions}</div>
      </div>
    </div>
  );
}
