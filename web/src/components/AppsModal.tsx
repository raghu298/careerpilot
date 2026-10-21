import { useEffect, useState } from "react";
import { ApplicationRow, getApplications, setApplications } from "@/lib/api";

const STATUSES = ["Applied", "Screening", "Interviewing", "Offer", "Rejected", "Ghosted"];

const EMPTY: ApplicationRow = { company: "", role: "", status: "Applied", applied_on: "", location: "" };

export default function AppsModal({
  token,
  onClose,
  onSaved,
}: {
  token: string;
  onClose: () => void;
  onSaved: () => void;
}) {
  const [rows, setRows] = useState<ApplicationRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    getApplications(token)
      .then((r) => setRows(r.length ? r : [{ ...EMPTY }]))
      .catch(() => setRows([{ ...EMPTY }]))
      .finally(() => setLoading(false));
  }, [token]);

  function update(i: number, patch: Partial<ApplicationRow>) {
    setRows((rs) => rs.map((r, j) => (j === i ? { ...r, ...patch } : r)));
  }
  function addRow() {
    setRows((rs) => [...rs, { ...EMPTY }]);
  }
  function removeRow(i: number) {
    setRows((rs) => rs.filter((_, j) => j !== i));
  }

  async function save() {
    setSaving(true);
    try {
      const clean = rows.filter((r) => r.company.trim() && r.role.trim());
      await setApplications(token, clean);
      onSaved();
      onClose();
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal modal-wide" onClick={(e) => e.stopPropagation()}>
        <h2>My applications</h2>
        <p className="modal-sub">
          Add the jobs you&apos;ve actually applied to. The Analytics agent answers by running <em>real
          read-only SQL</em> over these rows — ask things like &quot;how many are still pending?&quot; or
          &quot;which have I been ghosted on?&quot;. Stays in this session only.
        </p>

        {loading ? (
          <div className="apps-empty">Loading…</div>
        ) : (
          <div className="apps-table">
            <div className="apps-head">
              <span>Company</span>
              <span>Role</span>
              <span>Status</span>
              <span>Applied on</span>
              <span></span>
            </div>
            {rows.map((r, i) => (
              <div className="apps-row" key={i}>
                <input value={r.company} placeholder="Acme AI" onChange={(e) => update(i, { company: e.target.value })} />
                <input value={r.role} placeholder="Senior GenAI Engineer" onChange={(e) => update(i, { role: e.target.value })} />
                <select value={r.status} onChange={(e) => update(i, { status: e.target.value })}>
                  {STATUSES.map((s) => (
                    <option key={s} value={s}>{s}</option>
                  ))}
                </select>
                <input type="date" value={r.applied_on || ""} onChange={(e) => update(i, { applied_on: e.target.value })} />
                <button className="apps-del" onClick={() => removeRow(i)} aria-label="Remove">×</button>
              </div>
            ))}
            <button className="apps-add" onClick={addRow}>+ Add application</button>
          </div>
        )}

        <div className="modal-actions">
          <button className="btn-ghost" onClick={onClose} disabled={saving}>Cancel</button>
          <button className="btn-primary" onClick={save} disabled={saving || loading}>
            {saving ? "Saving…" : "Save my applications"}
          </button>
        </div>
      </div>
    </div>
  );
}
