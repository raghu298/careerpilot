import { useState } from "react";
import type { Pending } from "@/lib/types";

export default function PendingActionCard({
  pending,
  busy,
  onResolve,
}: {
  pending: Pending;
  busy: boolean;
  onResolve: (action: "confirm" | "cancel" | "revise", edit?: string) => void;
}) {
  const [editing, setEditing] = useState(false);
  const [edit, setEdit] = useState("");

  return (
    <div className="pending">
      <div className="tag">human-in-the-loop · {pending.kind}</div>
      <div style={{ margin: "6px 0", color: "var(--muted)", fontSize: 13 }}>
        Nothing is sent until you confirm.
      </div>
      <pre>{pending.preview}</pre>
      {editing && (
        <textarea
          className="composer"
          style={{ width: "100%", marginTop: 8, minHeight: 60 }}
          placeholder="How should I revise it? e.g. 'make it shorter and mention the LangGraph project'"
          value={edit}
          onChange={(e) => setEdit(e.target.value)}
        />
      )}
      <div className="pending-actions">
        <button className="btn primary" disabled={busy} onClick={() => onResolve("confirm")}>
          Send
        </button>
        {editing ? (
          <button className="btn" disabled={busy || !edit.trim()} onClick={() => onResolve("revise", edit)}>
            Apply edit
          </button>
        ) : (
          <button className="btn" disabled={busy} onClick={() => setEditing(true)}>
            Edit
          </button>
        )}
        <button className="btn danger" disabled={busy} onClick={() => onResolve("cancel")}>
          Cancel
        </button>
      </div>
    </div>
  );
}
