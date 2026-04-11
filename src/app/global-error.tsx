"use client";

import { useEffect } from "react";

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error("[app/global-error] Root error:", error);
    if (error.stack) console.error("[app/global-error] stack:", error.stack);
  }, [error]);

  return (
    <html lang="en">
      <body style={{ margin: 0, background: "#08090E", color: "#E4E7F0", fontFamily: "system-ui, sans-serif" }}>
        <div style={{ padding: 24, minHeight: "100vh" }}>
          <h2>Application error</h2>
          <pre style={{ color: "#EF476F", whiteSpace: "pre-wrap" }}>{error.message}</pre>
          <button type="button" onClick={reset} style={{ marginTop: 16, padding: "8px 16px", cursor: "pointer" }}>
            Try again
          </button>
        </div>
      </body>
    </html>
  );
}
