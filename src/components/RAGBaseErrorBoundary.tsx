"use client";

import { Component, type ErrorInfo, type ReactNode } from "react";

type Props = { children: ReactNode };

type State = { hasError: boolean; message: string };

/**
 * Catches render/runtime errors inside RAGBase so the tab doesn’t stay blank without feedback.
 */
export class RAGBaseErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false, message: "" };

  static getDerivedStateFromError(err: Error): State {
    console.error("[RAGBaseErrorBoundary] getDerivedStateFromError:", err);
    return { hasError: true, message: err.message || "Unknown error" };
  }

  componentDidCatch(err: Error, info: ErrorInfo) {
    console.error("[RAGBaseErrorBoundary] componentDidCatch:", err);
    console.error("[RAGBaseErrorBoundary] componentStack:", info.componentStack);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div
          style={{
            padding: 24,
            color: "var(--text-bright)",
            background: "var(--bg)",
            minHeight: "100dvh",
            fontFamily: "'Outfit', system-ui, -apple-system, Segoe UI, Roboto, sans-serif",
          }}
        >
          <h1 style={{ fontSize: 18, marginBottom: 12, letterSpacing: "-0.01em" }}>
            RAGBase — runtime error
          </h1>
          <p style={{ color: "var(--text)", marginBottom: 8 }}>
            Check the browser console for full details.
          </p>
          <pre
            style={{
              whiteSpace: "pre-wrap",
              color: "var(--danger)",
              fontSize: 13,
              padding: 12,
              background: "var(--surface2)",
              borderRadius: 14,
              border: "1px solid var(--border)",
            }}
          >
            {this.state.message}
          </pre>
          <button
            type="button"
            onClick={() => this.setState({ hasError: false, message: "" })}
            style={{
              marginTop: 16,
              padding: "8px 16px",
              borderRadius: 12,
              border: "1px solid var(--border)",
              cursor: "pointer",
              background: "var(--btn-bg)",
              color: "var(--btn-fg)",
              fontWeight: 800,
            }}
          >
            Try again
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
