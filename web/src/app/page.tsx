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
      .catch(() => setError("Couldn't reach the CareerPilot API. Is the backend running?"));
  }, []);

  async function choose(p: Persona) {
    setBusy(true);
    try {
      const token = await createSession(p.id);
      localStorage.setItem("cp_session", token);
      localStorage.setItem("cp_persona", JSON.stringify(p));
      window.location.href = "/chat";
    } catch {
      setError("Couldn't start a session.");
      setBusy(false);
    }
  }

  return (
    <div className="center">
      <div className="card">
        <h1>CareerPilot</h1>
        <p>
          A multi-agent job-hunt copilot. Ask one question; a router sends it to the right
          specialist agents — resume/JD analysis, live job search over MCP, your application
          tracker, and a confirm-before-send outreach drafter — running in parallel with a
          live orchestration trace.
        </p>
        <p style={{ marginTop: 10 }}>Pick a demo profile to begin — no sign-up:</p>
        {error && <p style={{ color: "var(--danger)" }}>{error}</p>}
        <div className="persona-grid">
          {personas.map((p) => (
            <button key={p.id} className="persona-pick" disabled={busy} onClick={() => choose(p)}>
              <div>{p.name}</div>
              <div className="role">
                {p.id === "riya" ? "Active job seeker · software engineer" : "Senior career switcher · premium guides"}
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
