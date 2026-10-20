"use client";

import { useEffect, useState } from "react";
import { createSession, getPersonas } from "@/lib/api";
import type { Persona } from "@/lib/types";

export default function Landing() {
  const [personas, setPersonas] = useState<Persona[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getPersonas()
      .then(setPersonas)
      .catch(() => setError("Couldn't reach the CareerPilot API — the backend may be waking up. Refresh in ~30s."));
  }, []);

  async function choose(p: Persona) {
    setBusy(true);
    try {
      const token = await createSession(p.id);
      localStorage.setItem("cp_session", token);
      localStorage.setItem("cp_persona", JSON.stringify(p));
      window.location.href = "/chat";
    } catch {
      setError("Couldn't start a session. Try again in a moment.");
      setBusy(false);
    }
  }

  return (
    <div className="chooser-page">
      <div className="chooser-left">
        <p className="chooser-brand">CareerPilot</p>
        <h1 className="chooser-welcome">Your multi-agent job-hunt copilot.</h1>
        <p className="chooser-subtitle">
          Ask one question. A router sends it to the right specialist agents — resume/JD analysis (RAG),
          live job search (MCP), your application tracker (SQL), and a confirm-before-send outreach drafter —
          running in parallel with a live orchestration trace.
        </p>
      </div>
      <div className="chooser-right">
        <h2 className="chooser-title">Pick a demo profile — no sign-up</h2>
        {error && <p className="chooser-error">{error}</p>}
        {personas.map((p) => (
          <button key={p.id} className="persona-card" disabled={busy} onClick={() => choose(p)}>
            <div className="pname">{p.name}</div>
            <div className="prole">
              {p.id === "riya" ? "Active job seeker · software engineer" : "Senior career switcher · premium guides"}
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
