import RAGBase from "@/components/RAGBase";
import { RAGBaseErrorBoundary } from "@/components/RAGBaseErrorBoundary";

/**
 * Hard isolation for blank-screen diagnosis:
 * - 1 → only "Hello world" (confirms App Router + layout render)
 * - 2 → shell without RAGBase (confirms not RAGBase import side effects)
 * - 3 → "Shell loaded" + SafeRender + RAGBase (bisect inside component)
 */
const DEBUG_PAGE_STEP = 3 as 1 | 2 | 3;

function SafeRender() {
  try {
    return <RAGBase />;
  } catch (e) {
    console.error("RAGBase crash:", e);
    return (
      <div style={{ color: "#EF476F", padding: 20 }}>
        RAGBase crashed (sync errors only — render errors use the error boundary)
      </div>
    );
  }
}

export default function Page() {
  if (DEBUG_PAGE_STEP === 1) {
    return <div style={{ color: "white", padding: 20 }}>Hello world</div>;
  }

  if (DEBUG_PAGE_STEP === 2) {
    return (
      <div style={{ color: "white", padding: 20 }}>
        <div>Hello world</div>
        {/* <RAGBase /> */}
      </div>
    );
  }

  return (
    <div style={{ color: "white", padding: 20 }}>
      <div>Shell loaded</div>
      <RAGBaseErrorBoundary>
        <SafeRender />
      </RAGBaseErrorBoundary>
    </div>
  );
}
