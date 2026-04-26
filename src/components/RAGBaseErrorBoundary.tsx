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
            color: "rgba(243,246,255,0.94)",
            background: "#07080c",
            minHeight: "100vh",
            fontFamily: "'Outfit', system-ui, -apple-system, Segoe UI, Roboto, sans-serif",
          }}
        >
          <h1 style={{ fontSize: 18, marginBottom: 12, letterSpacing: "-0.01em" }}>
            RAGBase — runtime error
          </h1>
          <p style={{ color: "rgba(223,230,242,0.68)", marginBottom: 8 }}>
            Check the browser console for full details.
          </p>
          <pre
            style={{
              whiteSpace: "pre-wrap",
              color: "rgba(251,113,133,0.95)",
              fontSize: 13,
              padding: 12,
              background: "rgba(21,23,33,0.82)",
              borderRadius: 14,
              border: "1px solid rgba(255,255,255,0.10)",
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
              border: "1px solid rgba(255,255,255,0.10)",
              cursor: "pointer",
              background: "linear-gradient(135deg, rgba(125,211,252,0.95), rgba(56,189,248,0.85))",
              color: "rgba(8,10,14,0.92)",
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
