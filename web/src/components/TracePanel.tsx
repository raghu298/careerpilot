import type { TraceEvent } from "@/lib/types";

const DOT: Record<string, string> = {
  routing_decided: "route",
  agent_started: "agent",
  agent_finished: "agent",
  tool_called: "tool",
  tool_result: "tool",
  staged: "hitl",
  pending_action: "hitl",
  done: "done",
  error: "err",
};

function label(e: TraceEvent): string {
  switch (e.type) {
    case "routing_decided":
      return `Routed → ${(e.agents || []).join(", ") || "none"}`;
    case "agent_started":
      return `${e.agent} started`;
    case "agent_finished":
      return `${e.agent} finished`;
    case "tool_called":
      return `${e.agent}: ${e.tool}()`;
    case "tool_result":
      return `${e.agent}: ${e.tool} → result`;
    case "staged":
      return `${e.agent} staged a draft`;
    case "pending_action":
      return `Awaiting your confirmation (${e.kind})`;
    case "done":
      return "Done";
    case "error":
      return `Error: ${e.message}`;
    default:
      return e.type;
  }
}

export default function TracePanel({ events }: { events: TraceEvent[] }) {
  const routing = events.find((e) => e.type === "routing_decided");
  return (
    <aside className="trace">
      <h3>Orchestration Trace</h3>
      {events.length === 0 ? (
        <div className="trace-empty">Send a message to see how the agents coordinate.</div>
      ) : (
        <>
          {routing?.reasoning && (
            <div className="reasoning">
              <div className="reasoning-head">Orchestrator</div>
              {routing.reasoning}
            </div>
          )}
          {events.map((e, i) => (
            <div className="evt" key={i}>
              <span className={`dot ${DOT[e.type] || "done"}`} />
              <span>{label(e)}</span>
            </div>
          ))}
        </>
      )}
    </aside>
  );
}
