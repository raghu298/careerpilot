"use client";

import { useEffect, useRef, useState } from "react";
import { streamPost } from "@/lib/api";
import type { ChatResult, Message, Pending, Persona, TraceEvent } from "@/lib/types";
import TracePanel from "@/components/TracePanel";
import PendingActionCard from "@/components/PendingActionCard";

const STARTERS = [
  "Does my resume fit the Senior GenAI JD? What are my gaps?",
  "Find senior GenAI roles in Bangalore",
  "How many of my applications got a response, and which are stale?",
  "Find a role I haven't applied to yet and draft a cover letter for it",
];

export default function ChatPage() {
  const [persona, setPersona] = useState<Persona | null>(null);
  const [session, setSession] = useState<string>("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [events, setEvents] = useState<TraceEvent[]>([]);
  const [pending, setPending] = useState<Pending | null>(null);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const tok = localStorage.getItem("cp_session");
    const p = localStorage.getItem("cp_persona");
    if (!tok) {
      window.location.href = "/";
      return;
    }
    setSession(tok);
    if (p) setPersona(JSON.parse(p));
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollTo(0, scrollRef.current.scrollHeight);
  }, [messages, pending]);

  function applyResult(result: ChatResult | null) {
    if (!result) return;
    if (result.replies.length) {
      setMessages((m) => [
        ...m,
        { role: "assistant", replies: result.replies, routing_reasoning: result.routing_reasoning },
      ]);
    }
    setPending(result.pending);
  }

  async function send(text: string) {
    if (!text.trim() || busy || !session) return;
    setInput("");
    setMessages((m) => [...m, { role: "user", content: text }]);
    setEvents([]);
    setBusy(true);
    try {
      const result = await streamPost("/chat", { session, message: text }, (e) =>
        setEvents((prev) => [...prev, e]),
      );
      applyResult(result);
    } catch {
      setEvents((prev) => [
        ...prev,
        { type: "error", message: "Request failed — the backend may be waking up. Try again in a moment." },
      ]);
      setMessages((m) => [
        ...m,
        { role: "assistant", replies: [{ agent: "system", content: "Couldn't reach the backend. Render's free tier sleeps when idle — wait ~30s and resend.", metadata: {} }], routing_reasoning: "" },
      ]);
    } finally {
      setBusy(false);
    }
  }

  async function resolve(action: "confirm" | "cancel" | "revise", edit?: string) {
    if (!session) return;
    setBusy(true);
    try {
      const result = await streamPost("/pending", { session, action, edit }, (e) =>
        setEvents((prev) => [...prev, e]),
      );
      applyResult(result);
    } finally {
      setBusy(false);
    }
  }

  function switchProfile() {
    localStorage.removeItem("cp_session");
    localStorage.removeItem("cp_persona");
    window.location.href = "/";
  }

  const empty = messages.length === 0;

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="sidebar-head">
          <span>Conversations</span>
          <button className="icon-btn" onClick={() => setMessages([])} aria-label="New conversation">
            +
          </button>
        </div>
        <div className="conv active">Current session</div>
      </aside>

      <main className="main">
        <header className="header">
          <div className="brand">
            <span className="logo">C</span>
            <div>
              CareerPilot
              <div className="persona">{persona ? persona.name : "…"}</div>
            </div>
          </div>
          <div className="header-right">
            <a href="/outbox">Demo Outbox</a>
            <button className="text-btn" onClick={switchProfile}>
              Switch profile
            </button>
          </div>
        </header>

        <div className="banner">
          <b>Demo mode</b> — drafts are captured in the Demo Outbox, never actually sent. Inactive
          history expires after 24 hours.
        </div>

        <div className="chat" ref={scrollRef}>
          {empty && (
            <div className="empty">
              <p>Ask about your resume, live roles, your application pipeline, or draft outreach.</p>
              <div className="starters">
                {STARTERS.map((s) => (
                  <button key={s} className="starter" onClick={() => setInput(s)}>
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((m, i) =>
            m.role === "user" ? (
              <div className="msg user" key={i}>
                <div className="who">You</div>
                {m.content}
              </div>
            ) : (
              <div className="msg" key={i}>
                {m.replies.map((r, j) => (
                  <div className="reply" key={j}>
                    <span className="tag">{r.agent}</span>
                    <pre>{r.content}</pre>
                    {Array.isArray(r.metadata?.citations) && (r.metadata.citations as string[]).length > 0 && (
                      <div className="cites">sources: {(r.metadata.citations as string[]).join(", ")}</div>
                    )}
                  </div>
                ))}
              </div>
            ),
          )}

          {busy && <div className="msg"><div className="reply">Thinking…</div></div>}
          {pending && <PendingActionCard pending={pending} busy={busy} onResolve={resolve} />}
        </div>

        <form
          className="composer"
          onSubmit={(e) => {
            e.preventDefault();
            send(input);
          }}
        >
          <input
            placeholder={pending ? "Resolve the pending draft above first…" : "Ask CareerPilot…"}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={busy || !!pending}
          />
          <button className="btn primary" type="submit" disabled={busy || !!pending || !input.trim()}>
            Send
          </button>
        </form>
      </main>

      <TracePanel events={events} />
    </div>
  );
}
