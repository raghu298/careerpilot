"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { streamPost } from "@/lib/api";
import { buildTrace } from "@/lib/trace";
import type { ChatResult, Message, Pending, Persona, TraceEvent } from "@/lib/types";
import TracePanel from "@/components/TracePanel";
import PendingActionCard from "@/components/PendingActionCard";
import StarterPrompts from "@/components/StarterPrompts";

export default function ChatPage() {
  const [persona, setPersona] = useState<Persona | null>(null);
  const [session, setSession] = useState<string>("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [events, setEvents] = useState<TraceEvent[]>([]);
  const [pending, setPending] = useState<Pending | null>(null);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  const trace = useMemo(() => buildTrace(events), [events]);

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
  }, [messages, pending, busy]);

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
      const result = await streamPost("/chat", { session, message: text }, (e) => setEvents((p) => [...p, e]));
      applyResult(result);
    } catch {
      setEvents((p) => [...p, { type: "error", message: "Backend may be waking up — try again in ~30s." }]);
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
      const result = await streamPost("/pending", { session, action, edit }, (e) => setEvents((p) => [...p, e]));
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
    <div className="chat-page">
      <aside className="conversation-sidebar">
        <div className="conversation-sidebar-header">
          <span>Conversations</span>
          <button onClick={() => { setMessages([]); setEvents([]); setPending(null); }} aria-label="New conversation">
            +
          </button>
        </div>
        <div className="conversation-list">
          <div className="conversation-row active">
            <button className="conversation-open">Current session</button>
          </div>
        </div>
        <div className="sidebar-foot">Multi-agent · LangGraph · MCP · RAG · SQL · HITL</div>
      </aside>

      <main className="chat-main">
        <header className="chat-header">
          <div className="brand">
            <span className="brand-logo">C</span>
            <div>
              <div className="brand-name">CareerPilot</div>
              <div className="brand-persona">{persona ? persona.name : "…"}</div>
            </div>
          </div>
          <div className="header-center">Job-Hunt Copilot</div>
          <div className="header-right">
            <a href="/outbox">Demo Outbox</a>
            <button className="text-button" onClick={switchProfile}>Switch profile</button>
          </div>
        </header>

        <div className="demo-banner">
          <b>Demo mode</b> — drafts are captured in the Demo Outbox, never actually sent. Inactive history expires after 24 hours.
        </div>

        <div className="chat-body" ref={scrollRef}>
          <div className="chat-scroll-inner">
            {empty && <StarterPrompts onPick={(q) => send(q)} />}

            {messages.map((m, i) =>
              m.role === "user" ? (
                <div className="user-message" key={i}>
                  <span className="message-label">You</span>
                  {m.content}
                </div>
              ) : (
                m.replies.map((r, j) => (
                  <div className="agent-reply" key={`${i}-${j}`}>
                    <span className="message-label">{r.agent}</span>
                    <div className="body">{r.content}</div>
                    {Array.isArray(r.metadata?.citations) && (r.metadata.citations as string[]).length > 0 && (
                      <div className="citations">
                        sources
                        <ul>
                          {(r.metadata.citations as string[]).map((c, k) => (
                            <li key={k}>{c}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                ))
              ),
            )}

            {busy && <div className="agent-reply thinking">Thinking…</div>}
            {pending && <PendingActionCard pending={pending} busy={busy} onResolve={resolve} />}
          </div>
        </div>

        <form className="chat-input-form" onSubmit={(e) => { e.preventDefault(); send(input); }}>
          <div className="chat-input-row">
            <input
              placeholder={pending ? "Resolve the pending draft above first…" : "Ask CareerPilot…"}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={busy || !!pending}
            />
            <button type="submit" disabled={busy || !!pending || !input.trim()}>Send</button>
          </div>
        </form>
      </main>

      <TracePanel trace={trace} />
    </div>
  );
}
