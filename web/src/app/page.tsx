"use client";

import { useEffect, useState } from "react";
import { createSession, getPersonas, warmBackend } from "@/lib/api";
import type { Persona } from "@/lib/types";

export default function Landing() {
  const [personas, setPersonas] = useState<Persona[]>([]);
  const [busy, setBusy] = useState(false);
  const [waking, setWaking] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    warmBackend(); // start the dyno booting immediately
    getPersonas(() => setWaking(true))
      .then((ps) => {
        setPersonas(ps);
        setWaking(false);
      })
      .catch(() => {
        setWaking(false);
        setError("Couldn't reach the CareerPilot API. It may be waking up — refresh in ~30s.");
      });
  }, []);

  async function choose(p: Persona) {
    setBusy(true);
    setError(null);
    try {
      const token = await createSession(p.id, () => setWaking(true));
      setWaking(false);
      localStorage.setItem("cp_session", token);
      localStorage.setItem("cp_persona", JSON.stringify(p));
      window.location.href = "/chat";
    } catch {
      setWaking(false);
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
        {waking && !error && (
          <p className="chooser-waking">
            Waking the backend… Render&apos;s free tier sleeps when idle, so the first load can take ~30–60s. Hang tight.
          </p>
        )}
        {error && <p className="chooser-error">{error}</p>}
        {!personas.length && !waking && !error && <p className="chooser-waking">Loading…</p>}
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
