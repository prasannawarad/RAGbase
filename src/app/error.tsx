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
        minHeight: "100vh",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        padding: 24,
        background: "#08090E",
        color: "#E4E7F0",
        fontFamily: "system-ui, sans-serif",
      }}
    >
      <h2 style={{ fontSize: 18, marginBottom: 8 }}>Something went wrong</h2>
      <pre
        style={{
          color: "#EF476F",
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
          border: "1px solid #252940",
          background: "#171A26",
          color: "#E4E7F0",
          cursor: "pointer",
        }}
      >
        Try again
      </button>
    </div>
  );
}
