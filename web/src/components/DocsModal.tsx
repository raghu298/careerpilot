import { useState } from "react";

export default function DocsModal({
  onClose,
  onSave,
}: {
  onClose: () => void;
  onSave: (resume: string, jd: string) => Promise<void>;
}) {
  const [resume, setResume] = useState("");
  const [jd, setJd] = useState("");
  const [saving, setSaving] = useState(false);

  async function save() {
    setSaving(true);
    try {
      await onSave(resume, jd);
      onClose();
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <h2>Use your own resume</h2>
        <p className="modal-sub">
          Paste your résumé and a target job description. The Knowledge agent will analyse <em>your</em>
          data — match and gaps — instead of the demo profile. Stays in this session only.
        </p>
        <label className="modal-label">Your résumé</label>
        <textarea value={resume} onChange={(e) => setResume(e.target.value)} placeholder="Paste your résumé text…" rows={7} />
        <label className="modal-label">Target job description</label>
        <textarea value={jd} onChange={(e) => setJd(e.target.value)} placeholder="Paste the job description…" rows={6} />
        <div className="modal-actions">
          <button className="btn-ghost" onClick={onClose} disabled={saving}>Cancel</button>
          <button className="btn-primary" onClick={save} disabled={saving || (!resume.trim() && !jd.trim())}>
            {saving ? "Saving…" : "Use these"}
          </button>
        </div>
      </div>
    </div>
  );
}
