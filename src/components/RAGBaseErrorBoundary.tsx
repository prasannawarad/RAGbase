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
            color: "#E4E7F0",
            background: "#08090E",
            minHeight: "100vh",
            fontFamily: "system-ui, sans-serif",
          }}
        >
          <h1 style={{ fontSize: 18, marginBottom: 12 }}>RAGBase — runtime error</h1>
          <p style={{ color: "#8B92AB", marginBottom: 8 }}>Check the browser console for full details.</p>
          <pre
            style={{
              whiteSpace: "pre-wrap",
              color: "#EF476F",
              fontSize: 13,
              padding: 12,
              background: "#171A26",
              borderRadius: 8,
              border: "1px solid #252940",
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
              borderRadius: 6,
              border: "none",
              cursor: "pointer",
              background: "#7C5CFC",
              color: "#fff",
              fontWeight: 600,
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
