import type { Trace, TraceBranch } from "@/lib/trace";

export default function TracePanel({ trace }: { trace: Trace }) {
  const hasStarted = trace.branches.length > 0 || !!trace.reasoning;

  return (
    <aside className="trace-panel">
      <div className="trace-title">Orchestration Trace</div>

      {!hasStarted && <p className="trace-empty">Send a message to see how the agents coordinate.</p>}

      {hasStarted && (
        <div className="trace-graph">
          <TraceNode label="Orchestrator" sublabel={trace.reasoning} status="done" />
          <div className="trace-connector" />

          <div className="trace-branches">
            {trace.branches.map((b) => (
              <BranchColumn key={b.agent} branch={b} />
            ))}
          </div>

          <div className="trace-connector" />
          <TraceNode label="Orchestrator" status={trace.finished ? "done" : "running"} />

          {trace.error && <div className="trace-error">{trace.error}</div>}
          {trace.finished && !trace.error && (
            <>
              <div className="trace-connector" />
              <TraceNode label="Final Answer" status="done" />
            </>
          )}
        </div>
      )}
    </aside>
  );
}

function BranchColumn({ branch }: { branch: TraceBranch }) {
  return (
    <div className="trace-branch">
      <TraceNode label={branch.agent} status={branch.status} compact />
      {branch.steps.map((step, i) => (
        <div key={i} className="trace-step">
          <div className="trace-connector trace-connector-small" />
          <div className={`trace-step-box ${step.status}`}>
            <span className="trace-step-tool">{step.tool}</span>
            {step.callDetail && <span className="trace-step-detail">{step.callDetail}</span>}
            {step.status === "done" && step.resultDetail && (
              <span className="trace-step-result">{step.resultDetail}</span>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}

function TraceNode({
  label,
  sublabel,
  status,
  compact,
}: {
  label: string;
  sublabel?: string;
  status: StepStatusLike;
  compact?: boolean;
}) {
  return (
    <div className={`trace-node ${status} ${compact ? "compact" : ""}`}>
      <span className="trace-node-label">
        {label}
        {status === "running" && <span className="trace-node-spinner" aria-hidden="true" />}
      </span>
      {sublabel && <span className="trace-node-sublabel">{sublabel}</span>}
    </div>
  );
}

type StepStatusLike = "running" | "done";
