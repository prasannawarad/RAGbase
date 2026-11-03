"use client";

import { useEffect } from "react";

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error("[app/error] Route segment error:", error);
    if (error.stack) console.error("[app/error] stack:", error.stack);
  }, [error]);

  return (
    <div
      style={{
        minHeight: "100dvh",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        padding: 24,
        background: "var(--bg)",
        color: "var(--text-bright)",
        fontFamily: "system-ui, sans-serif",
      }}
    >
      <h2 style={{ fontSize: 18, marginBottom: 8 }}>Something went wrong</h2>
      <pre
        style={{
          color: "var(--danger)",
          fontSize: 13,
          maxWidth: 560,
          whiteSpace: "pre-wrap",
          marginBottom: 16,
        }}
      >
        {error.message}
      </pre>
      <button
        type="button"
        onClick={reset}
        style={{
          padding: "8px 16px",
          borderRadius: 6,
          border: "1px solid var(--border2)",
          background: "var(--surface)",
          color: "var(--text-bright)",
          cursor: "pointer",
        }}
      >
        Try again
      </button>
    </div>
  );
}
