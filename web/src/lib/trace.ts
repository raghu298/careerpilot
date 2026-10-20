import type { TraceEvent } from "./types";

export type StepStatus = "running" | "done";

export type TraceStep = {
  tool: string;
  callDetail?: string;
  resultDetail?: string;
  status: StepStatus;
};

export type TraceBranch = {
  agent: string;
  status: StepStatus;
  steps: TraceStep[];
};

export type Trace = {
  reasoning: string;
  branches: TraceBranch[];
  finished: boolean;
  error?: string;
};

export const EMPTY_TRACE: Trace = { reasoning: "", branches: [], finished: false };

function argSummary(args?: Record<string, unknown>): string | undefined {
  if (!args) return undefined;
  const parts = Object.entries(args)
    .filter(([, v]) => v !== "" && v != null)
    .map(([k, v]) => `${k}: ${typeof v === "string" ? v : JSON.stringify(v)}`);
  const s = parts.join(", ");
  return s.length > 80 ? s.slice(0, 77) + "…" : s || undefined;
}

/** Fold the flat SSE event stream into the branch/step graph the panel renders. */
export function buildTrace(events: TraceEvent[]): Trace {
  const trace: Trace = { reasoning: "", branches: [], finished: false };
  const branchOf = (agent?: string) => trace.branches.find((b) => b.agent === agent);

  for (const e of events) {
    switch (e.type) {
      case "routing_decided":
        trace.reasoning = e.reasoning || "";
        for (const agent of e.agents || []) {
          if (!branchOf(agent)) trace.branches.push({ agent, status: "running", steps: [] });
        }
        break;
      case "agent_started":
        if (e.agent && !branchOf(e.agent)) trace.branches.push({ agent: e.agent, status: "running", steps: [] });
        break;
      case "tool_called": {
        const b = branchOf(e.agent);
        b?.steps.push({ tool: `${e.tool}()`, callDetail: argSummary(e.args), status: "running" });
        break;
      }
      case "tool_result": {
        const b = branchOf(e.agent);
        const step = [...(b?.steps || [])].reverse().find((s) => s.tool === `${e.tool}()` && s.status === "running");
        if (step) {
          step.status = "done";
          step.resultDetail = "result received";
        }
        break;
      }
      case "staged":
      case "pending_action": {
        const b = branchOf(e.agent);
        if (b) b.steps.push({ tool: "staged draft", status: "done", resultDetail: "awaiting confirmation" });
        break;
      }
      case "agent_finished": {
        const b = branchOf(e.agent);
        if (b) b.status = "done";
        break;
      }
      case "done":
        trace.finished = true;
        break;
      case "error":
        trace.error = e.message;
        trace.finished = true;
        break;
    }
  }
  return trace;
}
