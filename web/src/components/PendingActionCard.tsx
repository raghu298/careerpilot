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
    <div className="pending-card">
      <span className="message-label">{pending.agent} · awaiting your confirmation</span>
      <div style={{ color: "var(--text-muted)", fontSize: 13, marginBottom: 2 }}>
        Nothing is sent until you confirm.
      </div>
      <pre className="pending-preview">{pending.preview}</pre>

      {editing && (
        <div className="revise-row">
          <input
            placeholder="How should I revise it? e.g. 'shorter, mention my LangGraph project'"
            value={edit}
            onChange={(e) => setEdit(e.target.value)}
          />
          <button disabled={busy || !edit.trim()} onClick={() => onResolve("revise", edit)}>
            Apply
          </button>
        </div>
      )}

      <div className="pending-actions">
        <button className="send-button" disabled={busy} onClick={() => onResolve("confirm")}>
          Send
        </button>
        {!editing && (
          <button disabled={busy} onClick={() => setEditing(true)}>
            Edit
          </button>
        )}
        <button className="cancel-button" disabled={busy} onClick={() => onResolve("cancel")}>
          Cancel
        </button>
      </div>
    </div>
  );
}
